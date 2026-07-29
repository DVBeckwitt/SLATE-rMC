"""Interactively view Monte Carlo detector-native pixel mass."""

from __future__ import annotations

import argparse
import math
import queue
import threading
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from painted_ewald import enumerate_rods_within_ewald_sphere
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry import (
    AxisRotation,
    axis_rotation_transform,
    build_incident_states,
    compose_intrinsic_xy_rotation,
)
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.pipeline.configured_simulation import (
    AxisRotationConfiguration,
    ConfiguredSimulationInputs,
    SimulationConfiguration,
    build_configured_simulation_inputs,
    build_source_averaged_detector,
    load_simulation_config,
)
from rasim_next.pipeline.source_averaged_detector import (
    MonteCarloDetectorPixelMass,
    SourceAveragedDetectorEwaldMeasure,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"
DEFAULT_DRAWS_PER_SOURCE_STATE = 49
DEFAULT_DETECTOR_SEED = 20260728

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GeometryDeltas:
    """Zero-based mechanical and effective corrections using manuscript names."""

    detector_pitch_offset_deg: float = 0.0
    detector_yaw_offset_deg: float = 0.0
    detector_in_plane_rotation_offset_deg: float = 0.0
    detector_column_translation_mm: float = 0.0
    detector_row_translation_mm: float = 0.0
    detector_distance_offset_mm: float = 0.0
    goniometer_axis_pitch_offset_deg: float = 0.0
    goniometer_axis_yaw_offset_deg: float = 0.0
    effective_incidence_angle_offset_deg: float = 0.0
    effective_sample_tilt_offset_deg: float = 0.0
    sample_in_plane_rotation_offset_deg: float = 0.0
    sample_in_plane_x_translation_mm: float = 0.0
    sample_in_plane_y_translation_mm: float = 0.0
    sample_normal_translation_mm: float = 0.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            object.__setattr__(self, item.name, value)

    @classmethod
    def zero(cls) -> GeometryDeltas:
        return cls()


def _intrinsic_xyz_rotation(
    x_deg: float,
    y_deg: float,
    z_deg: float,
) -> FloatArray:
    xy_rotation = compose_intrinsic_xy_rotation(
        np.eye(3),
        math.radians(x_deg),
        math.radians(y_deg),
    )
    z_rad = math.radians(z_deg)
    cosine = math.cos(z_rad)
    sine = math.sin(z_rad)
    about_current_z = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    return np.asarray(xy_rotation @ about_current_z, dtype=np.float64)


def _controlled_pose(
    base: RigidTransform,
    *,
    local_rotation: FloatArray,
    local_offset_mm: tuple[float, float, float],
) -> RigidTransform:
    local_offset_m = 1.0e-3 * np.asarray(local_offset_mm, dtype=np.float64)
    return RigidTransform(
        rotation=base.rotation @ local_rotation,
        translation_m=base.translation_m + base.rotation @ local_offset_m,
        source_frame=base.source_frame,
        target_frame=base.target_frame,
    )


def _corrected_goniometer_axis_rotations(
    configured_axis_rotations: tuple[AxisRotationConfiguration, ...],
    deltas: GeometryDeltas,
) -> tuple[AxisRotationConfiguration, ...]:
    rotations = tuple(configured_axis_rotations)
    pitch_offset_deg = deltas.goniometer_axis_pitch_offset_deg
    yaw_offset_deg = deltas.goniometer_axis_yaw_offset_deg
    if pitch_offset_deg == 0.0 and yaw_offset_deg == 0.0:
        return rotations
    if len(rotations) != 1:
        raise ValueError("goniometer-axis offsets require exactly one configured goniometer axis")

    configured = rotations[0]
    base_rotation = AxisRotation(
        axis_lab=np.asarray(configured.axis_lab, dtype=np.float64),
        angle_rad=math.radians(configured.angle_deg),
        pivot_lab_m=np.asarray(configured.pivot_lab_m, dtype=np.float64),
    )
    x_axis, y_axis, z_axis = base_rotation.axis_lab
    horizontal = math.hypot(x_axis, y_axis)
    if horizontal <= 1.0e-12:
        raise ValueError("goniometer-axis pitch/yaw offsets require a nonvertical configured axis")
    base_pitch_rad = math.atan2(z_axis, horizontal)
    base_yaw_rad = math.atan2(-y_axis, x_axis)
    pitch_rad = base_pitch_rad + math.radians(pitch_offset_deg)
    yaw_rad = base_yaw_rad + math.radians(yaw_offset_deg)
    cos_pitch = math.cos(pitch_rad)
    corrected_axis = (
        math.cos(yaw_rad) * cos_pitch,
        -math.sin(yaw_rad) * cos_pitch,
        math.sin(pitch_rad),
    )
    return (replace(configured, axis_lab=corrected_axis),)


def _axis_rotation(configuration: AxisRotationConfiguration) -> AxisRotation:
    return AxisRotation(
        axis_lab=np.asarray(configuration.axis_lab, dtype=np.float64),
        angle_rad=math.radians(configuration.angle_deg),
        pivot_lab_m=np.asarray(configuration.pivot_lab_m, dtype=np.float64),
    )


def apply_geometry_deltas(
    instrument: CompiledInstrument,
    deltas: GeometryDeltas,
    *,
    configured_axis_rotations: tuple[AxisRotationConfiguration, ...] = (),
) -> CompiledInstrument:
    """Apply mechanical goniometer-axis and effective local-pose corrections.

    Goniometer pitch/yaw reorient the configured commanded axis about its declared LAB pivot.
    The remaining sample controls are effective end-pose corrections in the sample-local frame.
    """

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(deltas, GeometryDeltas):
        raise TypeError("deltas must be GeometryDeltas")
    corrected_axis_rotations = _corrected_goniometer_axis_rotations(
        configured_axis_rotations,
        deltas,
    )
    sample_base = instrument.lab_from_sample
    if corrected_axis_rotations != tuple(configured_axis_rotations):
        base_motion = axis_rotation_transform(_axis_rotation(configured_axis_rotations[0]))
        corrected_motion = axis_rotation_transform(_axis_rotation(corrected_axis_rotations[0]))
        sample_base = corrected_motion.compose(base_motion.inverse()).compose(sample_base)
    detector_pose = _controlled_pose(
        instrument.lab_from_detector,
        local_rotation=_intrinsic_xyz_rotation(
            deltas.detector_pitch_offset_deg,
            deltas.detector_yaw_offset_deg,
            deltas.detector_in_plane_rotation_offset_deg,
        ),
        local_offset_mm=(
            deltas.detector_column_translation_mm,
            deltas.detector_row_translation_mm,
            deltas.detector_distance_offset_mm,
        ),
    )
    sample_pose = _controlled_pose(
        sample_base,
        local_rotation=_intrinsic_xyz_rotation(
            deltas.effective_incidence_angle_offset_deg,
            deltas.effective_sample_tilt_offset_deg,
            deltas.sample_in_plane_rotation_offset_deg,
        ),
        local_offset_mm=(
            deltas.sample_in_plane_x_translation_mm,
            deltas.sample_in_plane_y_translation_mm,
            deltas.sample_normal_translation_mm,
        ),
    )
    return replace(
        instrument,
        lab_from_detector=detector_pose,
        lab_from_sample=sample_pose,
    )


@dataclass(frozen=True, slots=True)
class DetectorRaster:
    """Display wrapper around one native-pixel Monte Carlo estimate."""

    estimate: MonteCarloDetectorPixelMass
    wall_time_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_time_s) or self.wall_time_s < 0.0:
            raise ValueError("wall_time_s must be finite and nonnegative")


def sample_detector_raster(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    draws_per_source_state: int,
    seed: int,
) -> DetectorRaster:
    """Sample the mosaic law and sum weighted roots into native pixels."""

    start = perf_counter()
    estimate = detector.sample_native_pixel_mass(
        draws_per_source_state=draws_per_source_state,
        seed=seed,
    )
    return DetectorRaster(
        estimate=estimate,
        wall_time_s=perf_counter() - start,
    )


@dataclass(frozen=True, slots=True)
class _DetectorBundle:
    inputs: ConfiguredSimulationInputs
    detector: SourceAveragedDetectorEwaldMeasure


def _build_bundle(
    config: SimulationConfiguration,
    source_sample_count: int,
) -> _DetectorBundle:
    if not config.bragg.include_detector_visible_m0:
        raise ValueError(
            "interactive all-m display requires bragg.include_detector_visible_m0=true"
        )
    configured = replace(
        config,
        source=replace(config.source, sample_count=source_sample_count),
    )
    inputs = build_configured_simulation_inputs(configured)
    return _DetectorBundle(inputs, build_source_averaged_detector(inputs))


def _evaluate_bundle(
    bundle: _DetectorBundle,
    deltas: GeometryDeltas,
    *,
    draws_per_source_state: int,
    seed: int,
) -> DetectorRaster:
    if deltas == GeometryDeltas.zero():
        detector = bundle.detector
    else:
        configured_axis_rotations = bundle.inputs.config.instrument.axis_rotations
        corrected_axis_rotations = _corrected_goniometer_axis_rotations(
            configured_axis_rotations,
            deltas,
        )
        instrument = apply_geometry_deltas(
            bundle.inputs.instrument,
            deltas,
            configured_axis_rotations=configured_axis_rotations,
        )
        incident = build_incident_states(bundle.inputs.samples, bundle.inputs.material, instrument)
        if not np.any(incident.states.valid):
            raise ValueError("geometry produced no valid incident state")
        if np.array_equal(incident.states.valid, bundle.inputs.incident.states.valid):
            detector = bundle.detector.rebind_geometry(incident=incident, instrument=instrument)
        else:
            valid_index = np.flatnonzero(incident.states.valid)
            maximum_air_k_Ainv = (
                2.0 * np.pi / float(np.min(incident.states.wavelength_A[valid_index]))
            )
            rods = enumerate_rods_within_ewald_sphere(
                reciprocal_basis_Ainv=bundle.inputs.reciprocal.basis_Ainv,
                k_norm_Ainv=maximum_air_k_Ainv,
                population=bundle.inputs.config.bragg.rod_population,
            )
            changed_inputs = replace(
                bundle.inputs,
                config=replace(
                    bundle.inputs.config,
                    instrument=replace(
                        bundle.inputs.config.instrument,
                        axis_rotations=corrected_axis_rotations,
                    ),
                ),
                incident=incident,
                instrument=instrument,
                rods=rods,
            )
            detector = build_source_averaged_detector(changed_inputs)
    return sample_detector_raster(
        detector,
        draws_per_source_state=draws_per_source_state,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class _RenderRequest:
    revision: int
    source_sample_count: int
    draws_per_source_state: int
    deltas: GeometryDeltas


@dataclass(frozen=True, slots=True)
class _RenderOutcome:
    request: _RenderRequest
    raster: DetectorRaster | None
    bundle: _DetectorBundle | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _ControlSpec:
    field_name: str
    label: str
    minimum: float
    maximum: float


_CONTROL_SPECS = (
    _ControlSpec(
        "detector_pitch_offset_deg",
        r"detector pitch $-\Delta\gamma_{\rm RA}$ (deg)",
        -10.0,
        10.0,
    ),
    _ControlSpec(
        "detector_yaw_offset_deg",
        r"detector yaw $\Delta\Gamma_{\rm RA}$ (deg)",
        -10.0,
        10.0,
    ),
    _ControlSpec(
        "detector_in_plane_rotation_offset_deg",
        r"detector in-plane $\Delta\chi_D$ (deg)",
        -10.0,
        10.0,
    ),
    _ControlSpec(
        "detector_column_translation_mm",
        r"detector column $\Delta x_D$ [$x_0$-coupled] (mm)",
        -20.0,
        20.0,
    ),
    _ControlSpec(
        "detector_row_translation_mm",
        r"detector row $\Delta y_D$ [$y_0$-coupled] (mm)",
        -20.0,
        20.0,
    ),
    _ControlSpec(
        "detector_distance_offset_mm",
        r"detector-normal distance $\Delta D_n$ (mm)",
        -50.0,
        50.0,
    ),
    _ControlSpec(
        "goniometer_axis_pitch_offset_deg",
        r"goniometer-axis pitch $\Delta\alpha$ [RA-SIM cor_angle] (deg)",
        -5.0,
        5.0,
    ),
    _ControlSpec(
        "goniometer_axis_yaw_offset_deg",
        r"goniometer-axis yaw $\Delta\psi_g$ [RA-SIM psi_z] (deg)",
        -5.0,
        5.0,
    ),
    _ControlSpec(
        "effective_incidence_angle_offset_deg",
        r"effective incidence $\Delta\theta_i$ (deg)",
        -5.0,
        5.0,
    ),
    _ControlSpec(
        "effective_sample_tilt_offset_deg",
        r"effective sample tilt $\Delta\delta$ [RA-SIM $\chi$] (deg)",
        -5.0,
        5.0,
    ),
    _ControlSpec(
        "sample_in_plane_rotation_offset_deg",
        r"sample in-plane $\Delta\chi_S$ [RA-SIM $-\Delta\psi$] (deg)",
        -10.0,
        10.0,
    ),
    _ControlSpec(
        "sample_in_plane_x_translation_mm",
        r"sample in-plane $\Delta x_S$ (mm)",
        -2.0,
        2.0,
    ),
    _ControlSpec(
        "sample_in_plane_y_translation_mm",
        r"sample in-plane $\Delta y_S$ (mm)",
        -2.0,
        2.0,
    ),
    _ControlSpec(
        "sample_normal_translation_mm",
        r"sample normal $\Delta n_S=-\Delta z_S$ (mm)",
        -2.0,
        2.0,
    ),
)


class InteractiveDetectorViewer:
    """Matplotlib controller for settled Monte Carlo native-pixel mass."""

    def __init__(
        self,
        config: SimulationConfiguration,
        *,
        draws_per_source_state: int,
        initial_source_sample_count: int,
        detector_seed: int,
    ) -> None:
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.widgets import Button, Slider

        self._config = config
        self._initial_draws_per_source_state = draws_per_source_state
        self._initial_source_sample_count = initial_source_sample_count
        self._detector_seed = detector_seed
        self._revision = 0
        self._closed = False
        self._active_thread: threading.Thread | None = None
        self._pending_request: _RenderRequest | None = None
        self._outcomes: queue.SimpleQueue[_RenderOutcome] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._cached_bundle: _DetectorBundle | None = None
        self._suspend_updates = False
        self._slider_dirty = False
        self._image_background: object | None = None
        self._status_background: object | None = None
        self._slider_artists: dict[int, tuple[object, tuple[object, ...]]] = {}
        self._slider_backgrounds: dict[int, tuple[object, object]] = {}

        self.figure = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
        self._image_axis = self.figure.add_axes((0.055, 0.08, 0.59, 0.84))
        rows, columns = config.instrument.detector_shape_rc
        self._cmap = plt.get_cmap("magma").copy()
        self._cmap.set_bad("#111217")
        self._image = self._image_axis.imshow(
            np.ma.masked_all((1, 1), dtype=np.float64),
            origin="upper",
            extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
            interpolation="nearest",
            rasterized=True,
            cmap=self._cmap,
            norm=LogNorm(vmin=1.0e-8, vmax=1.0, clip=True),
            aspect="equal",
        )
        self._image.set_animated(self.figure.canvas.supports_blit)
        self._image_axis.title.set_animated(self.figure.canvas.supports_blit)
        self._image_axis.set_title("MONTE CARLO NATIVE PIXEL MASS: rendering requested pose")
        self._image_axis.set_xlabel("detector column (native pixel)")
        self._image_axis.set_ylabel("detector row (native pixel)")
        self._colorbar = self.figure.colorbar(
            self._image,
            ax=self._image_axis,
            pad=0.02,
            label=r"weighted raw detector-pixel mass estimate ($\AA^2$; display log scale)",
        )

        self._sliders: dict[str, Slider] = {}
        control_top = 0.91
        control_step = 0.044
        slider_height = 0.012
        slider_label_offset = 0.016
        for index, spec in enumerate(_CONTROL_SPECS):
            axis_y = control_top - index * control_step
            self.figure.text(
                0.70,
                axis_y + slider_label_offset,
                spec.label,
                ha="left",
                va="bottom",
                fontsize=9,
            )
            axis = self.figure.add_axes((0.70, axis_y, 0.265, slider_height))
            slider = Slider(
                axis,
                "",
                spec.minimum,
                spec.maximum,
                valinit=0.0,
            )
            self._register_slider(slider)
            slider.on_changed(
                lambda value, active=slider: self._on_render_setting_change(value, active)
            )
            self._sliders[spec.field_name] = slider
        source_samples_y = 0.235
        self.figure.text(
            0.70,
            source_samples_y + slider_label_offset,
            r"incident-ray samples $N_{\rm ray}$",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        source_samples_axis = self.figure.add_axes((0.70, source_samples_y, 0.265, slider_height))
        self._source_sample_slider = Slider(
            source_samples_axis,
            "",
            1,
            max(1000, initial_source_sample_count * 4),
            valinit=initial_source_sample_count,
            valstep=1,
        )
        self._register_slider(self._source_sample_slider)
        self._source_sample_slider.on_changed(
            lambda value, active=self._source_sample_slider: self._on_render_setting_change(
                value, active
            )
        )
        detector_draws_y = 0.285
        self.figure.text(
            0.70,
            detector_draws_y + slider_label_offset,
            r"mosaic draws per $k_i$ state $M$",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        detector_draws_axis = self.figure.add_axes((0.70, detector_draws_y, 0.265, slider_height))
        self._detector_draw_slider = Slider(
            detector_draws_axis,
            "",
            1,
            max(256, draws_per_source_state * 4),
            valinit=draws_per_source_state,
            valstep=1,
        )
        self._register_slider(self._detector_draw_slider)
        self._detector_draw_slider.on_changed(
            lambda value, active=self._detector_draw_slider: self._on_render_setting_change(
                value, active
            )
        )

        render_axis = self.figure.add_axes((0.70, 0.165, 0.125, 0.04))
        reset_axis = self.figure.add_axes((0.84, 0.165, 0.125, 0.04))
        self._render_button = Button(render_axis, "Render requested")
        self._reset_button = Button(reset_axis, "Reset all")
        self._render_button.on_clicked(self._request_render)
        self._reset_button.on_clicked(self._reset)
        self._status_axis = self.figure.add_axes((0.70, 0.085, 0.265, 0.06))
        self._status_axis.set_axis_off()
        self._status = self._status_axis.text(
            0.0,
            1.0,
            "",
            transform=self._status_axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            wrap=True,
        )
        self._status.set_animated(self.figure.canvas.supports_blit)
        self.figure.text(
            0.70,
            0.012,
            "Weighted roots sum directly into native pixels; no detector-coordinate quadrature.\n"
            "Drag keeps the settled image. Release renders the latest requested pose.\n"
            "All deltas are relative to the configured pose.\n"
            "Keys: R render, 0 reset, Q close.",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#343434",
        )
        self.figure.canvas.mpl_connect("button_release_event", self._on_button_release)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self.figure.canvas.mpl_connect("resize_event", self._on_resize)
        self.figure.canvas.mpl_connect("draw_event", self._on_draw)
        self._poll_timer = self.figure.canvas.new_timer(interval=100)
        self._poll_timer.add_callback(self._poll_render)
        self._poll_timer.start()
        self._set_status("ready; rendering the configured Monte Carlo pixel image")
        self._redraw_and_cache()
        self._request_render()

    def _deltas(self) -> GeometryDeltas:
        return GeometryDeltas(**{name: float(slider.val) for name, slider in self._sliders.items()})

    def _current_request(self) -> _RenderRequest:
        return _RenderRequest(
            revision=self._revision,
            source_sample_count=round(self._source_sample_slider.val),
            draws_per_source_state=round(self._detector_draw_slider.val),
            deltas=self._deltas(),
        )

    @staticmethod
    def _display_values(raster: DetectorRaster) -> tuple[np.ma.MaskedArray, float, float]:
        image_A2 = raster.estimate.image_A2
        positive = image_A2[image_A2 > 0.0]
        if not positive.size:
            return np.ma.masked_all(image_A2.shape), 1.0e-8, 1.0
        high = float(np.max(positive))
        low = max(float(np.min(positive)), high * 1.0e-8)
        if low >= high:
            low = max(np.finfo(np.float64).tiny, 0.1 * high)
        return np.ma.array(image_A2, mask=image_A2 <= 0.0), low, high

    def _cache_backgrounds(self) -> None:
        from matplotlib.transforms import Bbox

        if not self.figure.canvas.supports_blit:
            self._image_background = None
            self._status_background = None
            self._slider_backgrounds.clear()
            return
        self._image_background = self.figure.canvas.copy_from_bbox(self._image_axis.bbox)
        self._status_background = self.figure.canvas.copy_from_bbox(self._status_axis.bbox)
        renderer = self.figure.canvas.get_renderer()
        self._slider_backgrounds = {}
        for key, (slider, artists) in self._slider_artists.items():
            artist_bbox = Bbox.union(
                [slider.ax.bbox, *(artist.get_window_extent(renderer) for artist in artists)]
            ).padded(3.0)
            blit_bbox = Bbox.from_extents(
                max(self.figure.bbox.x0, artist_bbox.x0),
                max(self.figure.bbox.y0, artist_bbox.y0),
                self.figure.bbox.x1,
                min(self.figure.bbox.y1, artist_bbox.y1),
            )
            self._slider_backgrounds[key] = (
                blit_bbox,
                self.figure.canvas.copy_from_bbox(blit_bbox),
            )

    def _register_slider(self, slider: object) -> None:
        if not self.figure.canvas.supports_blit:
            return
        slider.drawon = False
        artists = (slider.poly, slider._handle, slider.valtext)
        for artist in artists:
            artist.set_animated(True)
        self._slider_artists[id(slider)] = (slider, artists)

    def _draw_slider(self, slider: object) -> None:
        registered = self._slider_artists.get(id(slider))
        if registered is None:
            return
        active, artists = registered
        cached = self._slider_backgrounds.get(id(active))
        if cached is None:
            self.figure.canvas.draw_idle()
            return
        blit_bbox, background = cached
        self._blit_axis(
            active.ax,
            background,
            *artists,
            blit_bbox=blit_bbox,
        )

    def _redraw_and_cache(self) -> None:
        self.figure.canvas.draw()

    def _on_draw(self, _event: object) -> None:
        if self._closed or not self.figure.canvas.supports_blit:
            return
        self._cache_backgrounds()
        self._blit_axis(
            self._image_axis,
            self._image_background,
            self._image,
            self._image_axis.title,
        )
        self._blit_axis(self._status_axis, self._status_background, self._status)
        for slider, _ in self._slider_artists.values():
            self._draw_slider(slider)

    def _blit_axis(
        self,
        axis: object,
        background: object | None,
        *artists: object,
        blit_bbox: object | None = None,
    ) -> None:
        if background is None:
            self.figure.canvas.draw_idle()
            return
        self.figure.canvas.restore_region(background)
        for artist in artists:
            axis.draw_artist(artist)
        self.figure.canvas.blit(axis.bbox if blit_bbox is None else blit_bbox)

    def _show_raster(self, raster: DetectorRaster) -> None:
        from matplotlib.colors import LogNorm

        display, low, high = self._display_values(raster)
        self._image.set_data(display)
        self._image.set_norm(LogNorm(vmin=low, vmax=high, clip=True))
        self._colorbar.update_normal(self._image)
        estimate = raster.estimate
        self._image_axis.set_title(
            f"MONTE CARLO NATIVE PIXEL MASS: {estimate.source_state_count} $k_i$ states x "
            f"{estimate.draws_per_source_state} draws, {len(estimate.rods)} physical rods\n"
            f"{estimate.visible_hit_count:,} visible root deposits; seed {estimate.seed}; "
            f"{raster.wall_time_s:.3f} s; total {estimate.total_detector_mass_A2:.6g} $\\AA^2$"
        )
        self._redraw_and_cache()

    def _set_status(self, text: str) -> None:
        self._status.set_text(text)
        self._blit_axis(self._status_axis, self._status_background, self._status)

    def _on_render_setting_change(
        self,
        _value: float,
        slider: object | None = None,
    ) -> None:
        if self._suspend_updates or self._closed:
            return
        if slider is not None:
            self._draw_slider(slider)
        self._revision += 1
        self._slider_dirty = True
        request = self._current_request()
        self._set_status(
            f"release to render {request.source_sample_count} $k_i$ states x "
            f"{request.draws_per_source_state} mosaic draws; settled image unchanged"
        )

    def _on_button_release(self, _event: object) -> None:
        if not self._suspend_updates and self._slider_dirty:
            self._request_render()

    def _request_render(self, _event: object | None = None) -> None:
        if self._closed:
            return
        self._slider_dirty = False
        request = self._current_request()
        if self._active_thread is not None:
            self._pending_request = request
            self._set_status(
                f"render running; queued latest {request.source_sample_count} $k_i$ states x "
                f"{request.draws_per_source_state} draws"
            )
            return
        self._start_render(request)

    def _start_render(self, request: _RenderRequest) -> None:
        cached = self._cached_bundle
        config = self._config
        seed = self._detector_seed
        stop_event = self._stop_event
        outcomes = self._outcomes

        def run() -> None:
            bundle: _DetectorBundle | None = cached
            try:
                if stop_event.is_set():
                    return
                if (
                    bundle is None
                    or bundle.inputs.config.source.sample_count != request.source_sample_count
                ):
                    bundle = _build_bundle(config, request.source_sample_count)
                if stop_event.is_set():
                    return
                raster = _evaluate_bundle(
                    bundle,
                    request.deltas,
                    draws_per_source_state=request.draws_per_source_state,
                    seed=seed,
                )
                outcome = _RenderOutcome(request, raster, bundle, None)
            except Exception as error:
                outcome = _RenderOutcome(request, None, bundle, str(error))
            if not stop_event.is_set():
                outcomes.put(outcome)

        self._set_status(
            f"rendering {request.source_sample_count} $k_i$ states x "
            f"{request.draws_per_source_state} mosaic draws with seed {seed}; "
            "the latest released pose will be queued"
        )
        self._active_thread = threading.Thread(
            target=run,
            name="detector-render",
            daemon=True,
        )
        self._active_thread.start()

    def _poll_render(self) -> None:
        if self._closed:
            return
        try:
            outcome = self._outcomes.get_nowait()
        except queue.Empty:
            return
        self._active_thread = None
        if outcome.bundle is not None:
            self._cached_bundle = outcome.bundle
        current = self._current_request()
        if outcome.error is not None and outcome.request == current:
            self._set_status(f"requested render failed: {outcome.error}")
        elif outcome.raster is not None and outcome.request == current:
            self._show_raster(outcome.raster)
            self._set_status(
                f"settled {current.source_sample_count} $k_i$ x "
                f"{current.draws_per_source_state} draws; pixel sampling "
                f"{outcome.raster.wall_time_s:.3f} s"
            )
        else:
            self._set_status("discarded stale render; current settled image remains visible")
        pending = self._pending_request
        self._pending_request = None
        if (
            pending is not None
            and pending != outcome.request
            and pending == self._current_request()
        ):
            self._start_render(pending)

    def _reset(self, _event: object | None = None) -> None:
        self._suspend_updates = True
        try:
            for slider in self._sliders.values():
                slider.set_val(0.0)
            self._source_sample_slider.set_val(self._initial_source_sample_count)
            self._detector_draw_slider.set_val(self._initial_draws_per_source_state)
        finally:
            self._suspend_updates = False
        self._redraw_and_cache()
        self._on_render_setting_change(0.0)
        self._request_render()

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key in {"r", "R"}:
            self._request_render()
        elif key == "0":
            self._reset()
        elif key in {"q", "Q"}:
            from matplotlib import pyplot as plt

            plt.close(self.figure)

    def _on_close(self, _event: object) -> None:
        self._closed = True
        self._pending_request = None
        self._stop_event.set()
        self._poll_timer.stop()

    def _on_resize(self, _event: object) -> None:
        self._redraw_and_cache()

    def show(self) -> None:
        from matplotlib import pyplot as plt

        plt.show()


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _detector_seed(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= result < 2**64:
        raise argparse.ArgumentTypeError("seed must lie in [0, 2**64)")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively sample the mosaic distribution and sum weighted roots into native "
            "detector pixels."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-samples",
        "--ki-samples",
        dest="source_sample_count",
        type=_positive_integer,
        help="incident-wavevector states (default: configured source count)",
    )
    parser.add_argument(
        "--draws-per-ki",
        "--draws-per-source-state",
        dest="draws_per_source_state",
        type=_positive_integer,
        default=DEFAULT_DRAWS_PER_SOURCE_STATE,
        help="Monte Carlo mosaic draws for each fixed ki state (default: 49)",
    )
    parser.add_argument(
        "--seed",
        type=_detector_seed,
        default=DEFAULT_DETECTOR_SEED,
        help="detector Monte Carlo seed, separate from the configured source seed",
    )
    args = parser.parse_args(argv)
    config = load_simulation_config(args.config.resolve(), repository_root=ROOT)
    source_sample_count = (
        config.source.sample_count if args.source_sample_count is None else args.source_sample_count
    )
    viewer = InteractiveDetectorViewer(
        config,
        draws_per_source_state=args.draws_per_source_state,
        initial_source_sample_count=source_sample_count,
        detector_seed=args.seed,
    )
    viewer.show()


if __name__ == "__main__":
    main()
