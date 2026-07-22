"""Interactively view the continuous detector-coordinate density."""

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
from rasim_next.pipeline.source_averaged_detector import SourceAveragedDetectorEwaldMeasure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"
DEFAULT_SOURCE_SAMPLE_COUNT = 25
LIVE_DISPLAY_SAMPLES_PER_AXIS = 32

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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
    """Display-only center samples of the continuous detector density."""

    density_A2_per_px2: FloatArray
    valid: BoolArray
    caustic: BoolArray
    column_centers_px: FloatArray
    row_centers_px: FloatArray
    source_state_count: int
    physical_rod_count: int
    root_policy: str
    execution_backend: str
    execution_device: str | None
    wall_time_s: float
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"

    def __post_init__(self) -> None:
        density = np.array(self.density_A2_per_px2, dtype=np.float64, copy=True, order="C")
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        column = np.array(self.column_centers_px, dtype=np.float64, copy=True, order="C")
        row = np.array(self.row_centers_px, dtype=np.float64, copy=True, order="C")
        expected_shape = (row.size, column.size)
        if density.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError("detector raster arrays must use [row, column] ordering")
        if caustic.shape != expected_shape or np.any(np.isnan(density)):
            raise ValueError("detector raster must have aligned caustic flags and no NaN")
        if np.any(density < 0.0) or not np.all(np.isfinite(column)) or not np.all(np.isfinite(row)):
            raise ValueError(
                "detector raster coordinates and density must be nonnegative and finite"
            )
        if self.source_state_count < 1 or self.physical_rod_count < 1:
            raise ValueError("detector raster requires source states and physical rods")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("detector raster must include every retained root")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector raster measure")
        if not math.isfinite(self.wall_time_s) or self.wall_time_s < 0.0:
            raise ValueError("wall_time_s must be finite and nonnegative")
        for value in (density, valid, caustic, column, row):
            value.setflags(write=False)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "column_centers_px", column)
        object.__setattr__(self, "row_centers_px", row)


def sample_detector_raster(
    detector: SourceAveragedDetectorEwaldMeasure,
    *,
    detector_shape_rc: tuple[int, int],
    display_samples_per_axis: int,
    execution_backend: str,
) -> DetectorRaster:
    """Sample active-panel coordinates after the complete source/rod/root reduction."""

    column_centers, row_centers, column_grid, row_grid = _detector_sample_grid(
        detector_shape_rc,
        display_samples_per_axis,
    )
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    start = perf_counter()
    evaluated = detector.evaluate_detector_density_all_roots(
        column_grid,
        row_grid,
        execution_backend=execution_backend,
    )
    wall_time_s = perf_counter() - start
    return DetectorRaster(
        density_A2_per_px2=evaluated.density_A2_per_px2,
        valid=evaluated.valid_source_count > 0,
        caustic=evaluated.caustic,
        column_centers_px=column_centers,
        row_centers_px=row_centers,
        source_state_count=detector.source_state_count,
        physical_rod_count=len(detector.rods),
        root_policy=evaluated.root_policy,
        execution_backend=evaluated.execution_backend,
        execution_device=evaluated.execution_device,
        wall_time_s=wall_time_s,
    )


def _detector_sample_grid(
    detector_shape_rc: tuple[int, int],
    display_samples_per_axis: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if type(display_samples_per_axis) is not int or display_samples_per_axis < 2:
        raise ValueError("display_samples_per_axis must be an integer of at least two")
    rows, columns = detector_shape_rc
    if type(rows) is not int or type(columns) is not int or rows < 1 or columns < 1:
        raise ValueError("detector_shape_rc must contain positive integers")
    column_edges = np.linspace(-0.5, columns - 0.5, display_samples_per_axis + 1)
    row_edges = np.linspace(-0.5, rows - 0.5, display_samples_per_axis + 1)
    column_centers = 0.5 * (column_edges[:-1] + column_edges[1:])
    row_centers = 0.5 * (row_edges[:-1] + row_edges[1:])
    column_grid, row_grid = np.meshgrid(column_centers, row_centers)
    return column_centers, row_centers, column_grid, row_grid


def _zero_detector_raster(
    *,
    detector_shape_rc: tuple[int, int],
    display_samples_per_axis: int,
    source_state_count: int,
    physical_rod_count: int,
    execution_backend: str,
) -> DetectorRaster:
    column_centers, row_centers, column_grid, _ = _detector_sample_grid(
        detector_shape_rc,
        display_samples_per_axis,
    )
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    shape = column_grid.shape
    return DetectorRaster(
        density_A2_per_px2=np.zeros(shape, dtype=np.float64),
        valid=np.zeros(shape, dtype=np.bool_),
        caustic=np.zeros(shape, dtype=np.bool_),
        column_centers_px=column_centers,
        row_centers_px=row_centers,
        source_state_count=source_state_count,
        physical_rod_count=physical_rod_count,
        root_policy="all_retained_roots.v1",
        execution_backend=execution_backend,
        execution_device=None,
        wall_time_s=0.0,
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
    display_samples_per_axis: int,
    execution_backend: str,
) -> DetectorRaster:
    if deltas == GeometryDeltas.zero():
        detector = bundle.detector
        instrument = bundle.inputs.instrument
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
            return _zero_detector_raster(
                detector_shape_rc=instrument.detector_shape_rc,
                display_samples_per_axis=display_samples_per_axis,
                source_state_count=incident.states.incident_state_id.size,
                physical_rod_count=len(bundle.detector.rods),
                execution_backend=execution_backend,
            )
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
        detector_shape_rc=instrument.detector_shape_rc,
        display_samples_per_axis=display_samples_per_axis,
        execution_backend=execution_backend,
    )


@dataclass(frozen=True, slots=True)
class _RenderRequest:
    revision: int
    source_sample_count: int
    display_samples_per_axis: int
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
    """Matplotlib controller with a fast one-state preview and settled total density."""

    def __init__(
        self,
        config: SimulationConfiguration,
        *,
        display_samples_per_axis: int,
        initial_source_sample_count: int,
        render_backend: str,
    ) -> None:
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.widgets import Button, Slider

        self._config = config
        self._initial_display_samples_per_axis = display_samples_per_axis
        self._initial_source_sample_count = initial_source_sample_count
        self._render_backend = render_backend
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

        build_start = perf_counter()
        self._preview_bundle = _build_bundle(config, 1)
        initial_raster = _evaluate_bundle(
            self._preview_bundle,
            GeometryDeltas.zero(),
            display_samples_per_axis=LIVE_DISPLAY_SAMPLES_PER_AXIS,
            execution_backend="cpu",
        )
        build_time = perf_counter() - build_start

        self.figure = plt.figure(figsize=(15.5, 9.0), constrained_layout=False)
        self._image_axis = self.figure.add_axes((0.055, 0.08, 0.59, 0.84))
        rows, columns = self._preview_bundle.inputs.instrument.detector_shape_rc
        self._cmap = plt.get_cmap("magma").copy()
        self._cmap.set_bad("#111217")
        initial_display, low, high = self._display_values(initial_raster)
        self._image = self._image_axis.imshow(
            initial_display,
            origin="upper",
            extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
            interpolation="bilinear",
            rasterized=True,
            cmap=self._cmap,
            norm=LogNorm(vmin=low, vmax=high),
            aspect="equal",
        )
        self._image.set_animated(self.figure.canvas.supports_blit)
        self._image_axis.title.set_animated(self.figure.canvas.supports_blit)
        self._image_axis.set_xlabel("detector column (continuous native coordinate, px)")
        self._image_axis.set_ylabel("detector row (continuous native coordinate, px)")
        self._colorbar = self.figure.colorbar(
            self._image,
            ax=self._image_axis,
            pad=0.02,
            label=r"total raw detector density ($\AA^2$/px$^2$; display log scale)",
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
            slider.on_changed(lambda value, active=slider: self._on_geometry_change(value, active))
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
        display_samples_y = 0.285
        self.figure.text(
            0.70,
            display_samples_y + slider_label_offset,
            r"display samples $N_{\rm disp}$ / axis",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        display_samples_axis = self.figure.add_axes((0.70, display_samples_y, 0.265, slider_height))
        self._display_sample_slider = Slider(
            display_samples_axis,
            "",
            32,
            256,
            valinit=display_samples_per_axis,
            valstep=16,
        )
        self._register_slider(self._display_sample_slider)
        self._display_sample_slider.on_changed(
            lambda value, active=self._display_sample_slider: self._on_render_setting_change(
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
            "Total source/rod/root density at each coordinate; no pixels integrated.\n"
            f"Drag: {LIVE_DISPLAY_SAMPLES_PER_AXIS}x"
            f"{LIVE_DISPLAY_SAMPLES_PER_AXIS} one-state preview. Release: requested render.\n"
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
        self._show_raster(initial_raster, requested_render=False)
        self._set_status(
            f"ready; initial build + warm preview {build_time:.2f} s; "
            f"requested-render backend={render_backend}"
        )
        self._redraw_and_cache()
        self._request_render()

    def _deltas(self) -> GeometryDeltas:
        return GeometryDeltas(**{name: float(slider.val) for name, slider in self._sliders.items()})

    def _current_request(self) -> _RenderRequest:
        return _RenderRequest(
            revision=self._revision,
            source_sample_count=round(self._source_sample_slider.val),
            display_samples_per_axis=round(self._display_sample_slider.val),
            deltas=self._deltas(),
        )

    @staticmethod
    def _display_values(raster: DetectorRaster) -> tuple[np.ma.MaskedArray, float, float]:
        density = raster.density_A2_per_px2
        finite_positive = density[np.isfinite(density) & (density > 0.0) & raster.valid]
        high = float(np.max(finite_positive)) if finite_positive.size else 1.0
        low = max(
            float(np.quantile(finite_positive, 0.001)) if finite_positive.size else high * 1.0e-8,
            high * 1.0e-8,
        )
        display = np.where(np.isposinf(density), high, density)
        masked = np.ma.array(display, mask=~raster.valid | (display < low))
        return masked, low, high

    def _display_values_on_current_scale(self, raster: DetectorRaster) -> np.ma.MaskedArray:
        high = float(self._image.norm.vmax)
        display = np.where(np.isposinf(raster.density_A2_per_px2), high, raster.density_A2_per_px2)
        return np.ma.array(display, mask=~raster.valid | (display <= 0.0))

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

    def _show_raster(self, raster: DetectorRaster, *, requested_render: bool) -> None:
        from matplotlib.colors import LogNorm

        if requested_render:
            display, low, high = self._display_values(raster)
        else:
            display = self._display_values_on_current_scale(raster)
            low = high = 0.0
        self._image.set_data(display)
        if requested_render:
            self._image.set_norm(LogNorm(vmin=low, vmax=high))
            self._colorbar.update_normal(self._image)
        state = "SETTLED TOTAL DENSITY" if requested_render else "LIVE 1-STATE TOTAL DENSITY"
        device = f" on {raster.execution_device}" if raster.execution_device else ""
        self._image_axis.set_title(
            f"{state}: {raster.source_state_count} incident-ray states, "
            f"{raster.physical_rod_count} physical rods (all m), all retained roots\n"
            f"{raster.column_centers_px.size}x{raster.row_centers_px.size} continuous samples; "
            f"{raster.execution_backend}{device}; density eval {raster.wall_time_s:.3f} s"
        )
        if requested_render:
            self._redraw_and_cache()
        else:
            self._blit_axis(
                self._image_axis,
                self._image_background,
                self._image,
                self._image_axis.title,
            )

    def _set_status(self, text: str) -> None:
        self._status.set_text(text)
        self._blit_axis(self._status_axis, self._status_background, self._status)

    def _on_geometry_change(self, _value: float, slider: object | None = None) -> None:
        if self._suspend_updates or self._closed:
            return
        if slider is not None:
            self._draw_slider(slider)
        self._revision += 1
        self._slider_dirty = True
        request = self._current_request()
        self._set_status(
            f"updating {LIVE_DISPLAY_SAMPLES_PER_AXIS}x"
            f"{LIVE_DISPLAY_SAMPLES_PER_AXIS} live one-state preview; "
            f"requested render={request.source_sample_count} incident rays"
        )
        try:
            raster = _evaluate_bundle(
                self._preview_bundle,
                request.deltas,
                display_samples_per_axis=LIVE_DISPLAY_SAMPLES_PER_AXIS,
                execution_backend="cpu",
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._set_status(f"preview rejected: {error}")
            return
        self._show_raster(raster, requested_render=False)
        self._set_status(
            f"live preview density evaluation {raster.wall_time_s:.3f} s; "
            f"release to render {request.source_sample_count} incident rays"
        )

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
            f"release to render {request.source_sample_count} incident rays at "
            f"{request.display_samples_per_axis}x{request.display_samples_per_axis}; "
            "live field unchanged"
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
                "render running; queued latest "
                f"{request.display_samples_per_axis}x{request.display_samples_per_axis} "
                f"pose with {request.source_sample_count} incident rays"
            )
            return
        self._start_render(request)

    def _start_render(self, request: _RenderRequest) -> None:
        cached = self._cached_bundle
        config = self._config
        backend = self._render_backend
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
                    display_samples_per_axis=request.display_samples_per_axis,
                    execution_backend=backend,
                )
                outcome = _RenderOutcome(request, raster, bundle, None)
            except Exception as error:
                outcome = _RenderOutcome(request, None, bundle, str(error))
            if not stop_event.is_set():
                outcomes.put(outcome)

        self._set_status(
            f"rendering {request.source_sample_count} incident rays at "
            f"{request.display_samples_per_axis}x{request.display_samples_per_axis} "
            f"on {backend}; the latest released pose will be queued"
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
            self._show_raster(outcome.raster, requested_render=True)
            self._set_status(
                f"settled {current.source_sample_count}-state total density; "
                f"density evaluation {outcome.raster.wall_time_s:.3f} s"
            )
        else:
            self._set_status("discarded stale render; current pose remains in live preview")
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
            self._display_sample_slider.set_val(self._initial_display_samples_per_axis)
        finally:
            self._suspend_updates = False
        self._redraw_and_cache()
        self._on_geometry_change(0.0)
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively sample the all-root continuous detector function without pixel "
            "integration."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--source-samples",
        "--ki-samples",
        dest="source_sample_count",
        type=_positive_integer,
        default=DEFAULT_SOURCE_SAMPLE_COUNT,
        help="incident-ray phase-space samples (default: 25)",
    )
    parser.add_argument(
        "--display-samples-per-axis",
        "--raster-size",
        dest="display_samples_per_axis",
        type=_positive_integer,
        default=128,
        help="settled continuous display samples per detector axis (default: 128)",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if not 32 <= args.display_samples_per_axis <= 256 or args.display_samples_per_axis % 16:
        parser.error("--display-samples-per-axis must be a multiple of 16 from 32 through 256")
    config = load_simulation_config(args.config.resolve(), repository_root=ROOT)
    backend = config.numerics.detector_execution_backend if args.backend is None else args.backend
    if backend == "cuda":
        from rasim_next.pipeline._continuous_detector_cuda import require_cuda_available

        require_cuda_available()
    viewer = InteractiveDetectorViewer(
        config,
        display_samples_per_axis=args.display_samples_per_axis,
        initial_source_sample_count=args.source_sample_count,
        render_backend=backend,
    )
    viewer.show()


if __name__ == "__main__":
    main()
