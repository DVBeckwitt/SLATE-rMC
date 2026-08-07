"""Interactive full-native Monte Carlo detector viewer."""
# ruff: noqa: E402  # Direct execution bootstraps the repository source tree below.

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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
    CompiledMonteCarloDetectorSampler,
    MonteCarloDetectorPixelMass,
    MonteCarloDetectorPresentation,
    MonteCarloSamplingCancelled,
    SourceAveragedDetectorEwaldMeasure,
)

DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"
DEFAULT_DRAWS_PER_SOURCE_STATE = 49
DEFAULT_DETECTOR_SEED = 20260728
_INCIDENCE_CONTROL_FIELD = "effective_incidence_angle_offset_deg"
_INCIDENCE_CONTROL_MINIMUM_DEG = 0.0
_INCIDENCE_CONTROL_MAXIMUM_DEG = 20.0

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


def _control_values_to_geometry_deltas(
    control_values: dict[str, float],
    *,
    configured_incidence_angle_deg: float,
) -> GeometryDeltas:
    delta_values = {name: float(value) for name, value in control_values.items()}
    incidence_angle_deg = delta_values[_INCIDENCE_CONTROL_FIELD]
    delta_values[_INCIDENCE_CONTROL_FIELD] = incidence_angle_deg - float(
        configured_incidence_angle_deg
    )
    return GeometryDeltas(**delta_values)


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

    estimate: MonteCarloDetectorPixelMass | MonteCarloDetectorPresentation
    wall_time_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.wall_time_s) or self.wall_time_s < 0.0:
            raise ValueError("wall_time_s must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class _FullNativeTextureFrame:
    """Presentation-only single-channel texture upload and logarithmic bounds."""

    image_A2: NDArray[np.float32]
    low_A2: float
    high_A2: float

    def __post_init__(self) -> None:
        image = np.asarray(self.image_A2)
        if image.dtype != np.float32 or image.ndim != 2 or not image.flags.c_contiguous:
            raise ValueError("texture image must be a contiguous two-dimensional float32 array")
        low = float(self.low_A2)
        high = float(self.high_A2)
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high < low:
            raise ValueError("texture logarithmic bounds must be finite, positive, and ordered")
        object.__setattr__(self, "low_A2", low)
        object.__setattr__(self, "high_A2", high)


_FULL_SCREEN_TEXTURE_XY_UV = np.asarray(
    (
        (-1.0, -1.0, 0.0, 1.0),
        (3.0, -1.0, 2.0, 1.0),
        (-1.0, 3.0, 0.0, -1.0),
    ),
    dtype=np.float32,
)
_FULL_SCREEN_TEXTURE_XY_UV.setflags(write=False)


def _full_screen_vertex_shader() -> str:
    positions = ", ".join(f"vec2({x:.1f}, {y:.1f})" for x, y in _FULL_SCREEN_TEXTURE_XY_UV[:, :2])
    coordinates = ", ".join(f"vec2({u:.1f}, {v:.1f})" for u, v in _FULL_SCREEN_TEXTURE_XY_UV[:, 2:])
    return (
        "#version 330 core\n"
        "out vec2 texture_coordinate;\n"
        f"const vec2 positions[3] = vec2[3]({positions});\n"
        f"const vec2 coordinates[3] = vec2[3]({coordinates});\n"
        "void main() {\n"
        "    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);\n"
        "    texture_coordinate = coordinates[gl_VertexID];\n"
        "}\n"
    )


def _prepare_full_native_texture(image_A2: NDArray[np.generic]) -> _FullNativeTextureFrame:
    """Preserve native row/column ownership while preparing one R32F upload."""

    supplied = np.asarray(image_A2)
    if supplied.ndim != 2:
        raise ValueError("detector presentation image must be two-dimensional")
    image = np.ascontiguousarray(supplied, dtype=np.float32)
    high = float(np.max(image, initial=np.float32(0.0)))
    if high <= 0.0:
        return _FullNativeTextureFrame(image, 1.0e-8, 1.0)
    low = max(float(np.finfo(np.float32).tiny), high * 1.0e-8)
    return _FullNativeTextureFrame(image, low, high)


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


def _instrument_for_deltas(
    bundle: _DetectorBundle,
    deltas: GeometryDeltas,
) -> CompiledInstrument:
    configured_axis_rotations = bundle.inputs.config.instrument.axis_rotations
    return apply_geometry_deltas(
        bundle.inputs.instrument,
        deltas,
        configured_axis_rotations=configured_axis_rotations,
    )


def _detector_for_deltas(
    bundle: _DetectorBundle,
    deltas: GeometryDeltas,
) -> SourceAveragedDetectorEwaldMeasure:
    if deltas == GeometryDeltas.zero():
        detector = bundle.detector
    else:
        configured_axis_rotations = bundle.inputs.config.instrument.axis_rotations
        instrument = _instrument_for_deltas(bundle, deltas)
        incident = bundle.inputs.incident
        if _requires_incident_rebuild(deltas):
            incident = build_incident_states(
                bundle.inputs.samples,
                bundle.inputs.material,
                instrument,
            )
        if not np.any(incident.states.valid):
            raise ValueError("geometry produced no valid incident state")
        if np.array_equal(incident.states.valid, bundle.inputs.incident.states.valid):
            detector = bundle.detector.rebind_geometry(incident=incident, instrument=instrument)
        else:
            corrected_axis_rotations = _corrected_goniometer_axis_rotations(
                configured_axis_rotations,
                deltas,
            )
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
    return detector


def _evaluate_bundle(
    bundle: _DetectorBundle,
    deltas: GeometryDeltas,
    *,
    draws_per_source_state: int,
    seed: int,
) -> DetectorRaster:
    return sample_detector_raster(
        _detector_for_deltas(bundle, deltas),
        draws_per_source_state=draws_per_source_state,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class _RenderRequest:
    revision: int
    source_sample_count: int
    draws_per_source_state: int
    deltas: GeometryDeltas


class _RenderCancellation:
    """Thread-safe cancellation state for one render stage."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


@dataclass(frozen=True, slots=True)
class _ScheduledRender:
    request: _RenderRequest
    draws_per_source_state: int
    materialize_result: bool
    cancellation: _RenderCancellation


def _progressive_draw_counts(requested_draw_count: int, *, settled: bool) -> tuple[int, ...]:
    requested = int(requested_draw_count)
    if requested < 1:
        raise ValueError("requested_draw_count must be positive")
    candidates = (1, 4, 8, requested) if settled else (1, 4, 8)
    return tuple(dict.fromkeys(min(candidate, requested) for candidate in candidates))


class _ProgressiveRenderScheduler:
    """Latest-only state machine for cancellable progressive full-native renders."""

    __slots__ = (
        "_active",
        "_completed_draw_count",
        "_completed_materialized",
        "_latest_request",
        "_pending_stages",
    )

    def __init__(self) -> None:
        self._active: _ScheduledRender | None = None
        self._completed_draw_count = 0
        self._completed_materialized = False
        self._latest_request: _RenderRequest | None = None
        self._pending_stages: list[tuple[int, bool]] = []

    def submit(self, request: _RenderRequest, *, settled: bool) -> None:
        if not isinstance(request, _RenderRequest):
            raise TypeError("request must be a _RenderRequest")
        same_request = self._latest_request == request
        if self._active is not None and self._active.request != request:
            self._active.cancellation.cancel()
        if not same_request:
            self._completed_draw_count = 0
            self._completed_materialized = False
        self._latest_request = request
        active_is_reusable = (
            self._active is not None
            and self._active.request == request
            and not self._active.cancellation.cancelled
        )
        active_draw_count = self._active.draws_per_source_state if active_is_reusable else 0
        represented_draw_count = max(self._completed_draw_count, active_draw_count)
        stages = [
            (
                draw_count,
                settled and draw_count == request.draws_per_source_state,
            )
            for draw_count in _progressive_draw_counts(
                request.draws_per_source_state,
                settled=settled,
            )
        ]
        self._pending_stages = [
            (draw_count, materialize)
            for draw_count, materialize in stages
            if draw_count > represented_draw_count
            or (
                materialize
                and not self._completed_materialized
                and not (active_is_reusable and self._active.materialize_result)
            )
        ]

    def start_next(self) -> _ScheduledRender | None:
        if self._active is not None or not self._pending_stages:
            return None
        if self._latest_request is None:
            raise RuntimeError("pending render stages require a latest request")
        draw_count, materialize = self._pending_stages.pop(0)
        stage = _ScheduledRender(
            request=self._latest_request,
            draws_per_source_state=draw_count,
            materialize_result=materialize,
            cancellation=_RenderCancellation(),
        )
        self._active = stage
        return stage

    def complete(
        self,
        stage: _ScheduledRender,
        *,
        completed_draw_count: int | None = None,
    ) -> bool:
        if stage is not self._active:
            raise ValueError("only the active render stage can complete")
        self._active = None
        accepted = not stage.cancellation.cancelled and stage.request == self._latest_request
        if accepted:
            completed = (
                stage.draws_per_source_state
                if completed_draw_count is None
                else int(completed_draw_count)
            )
            if completed < stage.draws_per_source_state:
                raise ValueError("completed draw count cannot precede the scheduled stage")
            self._completed_draw_count = max(self._completed_draw_count, completed)
            self._completed_materialized = self._completed_materialized or stage.materialize_result
            self._pending_stages = [
                (draw_count, materialize)
                for draw_count, materialize in self._pending_stages
                if draw_count > self._completed_draw_count
                or (materialize and not self._completed_materialized)
            ]
        return accepted

    def discard_pending(self) -> None:
        self._pending_stages.clear()

    def fail(self, stage: _ScheduledRender) -> bool:
        if stage is not self._active:
            raise ValueError("only the active render stage can fail")
        self._active = None
        accepted = not stage.cancellation.cancelled and stage.request == self._latest_request
        if accepted:
            self.reset_latest()
        return accepted

    def reset_latest(self) -> None:
        self._completed_draw_count = 0
        self._completed_materialized = False
        self._pending_stages.clear()

    def cancel(self) -> None:
        if self._active is not None:
            self._active.cancellation.cancel()
        self._pending_stages.clear()


@dataclass(frozen=True, slots=True)
class _RenderOutcome:
    stage: _ScheduledRender
    raster: DetectorRaster | None
    texture: _FullNativeTextureFrame | None
    error: str | None
    cancelled: bool = False


class _DetectorRenderSession:
    """Thread-confined bundle and progressive sampler state."""

    def __init__(
        self,
        config: SimulationConfiguration,
        *,
        detector_seed: int,
        execution_backend: str,
        prepare_texture: bool,
    ) -> None:
        self._config = config
        self._detector_seed = detector_seed
        self._execution_backend = execution_backend
        self._prepare_texture = prepare_texture
        self._bundle: _DetectorBundle | None = None
        self._bound_deltas: GeometryDeltas | None = None
        self._sampler: CompiledMonteCarloDetectorSampler | None = None

    def render(
        self,
        stage: _ScheduledRender,
        *,
        stop_requested: threading.Event,
    ) -> tuple[DetectorRaster, _FullNativeTextureFrame | None]:
        def cancel_requested() -> bool:
            return stop_requested.is_set() or stage.cancellation.cancelled

        if cancel_requested():
            raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
        request = stage.request
        if (
            self._bundle is None
            or self._bundle.inputs.config.source.sample_count != request.source_sample_count
        ):
            self._bundle = _build_bundle(self._config, request.source_sample_count)
            self._bound_deltas = None
            self._sampler = None
        if cancel_requested():
            raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
        if self._bound_deltas != request.deltas or self._sampler is None:
            changed_fields = _changed_delta_fields(self._bound_deltas, request.deltas)
            detector_pose_only = (
                self._sampler is not None
                and self._bound_deltas is not None
                and bool(changed_fields)
                and changed_fields <= _DETECTOR_ONLY_DELTA_FIELDS
            )
            if detector_pose_only:
                instrument = _instrument_for_deltas(self._bundle, request.deltas)
                if cancel_requested():
                    raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
                self._sampler.rebind_detector_pose(instrument)
            else:
                detector = _detector_for_deltas(self._bundle, request.deltas)
                if cancel_requested():
                    raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
                if self._sampler is None:
                    self._sampler = detector.compile_monte_carlo_sampler(
                        execution_backend=self._execution_backend,
                        seed=self._detector_seed,
                    )
                else:
                    try:
                        self._sampler.rebind_geometry(detector)
                    except ValueError:
                        self._sampler = detector.compile_monte_carlo_sampler(
                            execution_backend=self._execution_backend,
                            seed=self._detector_seed,
                        )
            self._bound_deltas = request.deltas
            if cancel_requested():
                raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
        if request.draws_per_source_state < self._sampler.draws_completed:
            self._sampler.reset()
        target_draw_count = max(
            stage.draws_per_source_state,
            self._sampler.draws_completed,
        )
        start = perf_counter()
        if stage.materialize_result or not self._prepare_texture:
            estimate: MonteCarloDetectorPixelMass | MonteCarloDetectorPresentation = (
                self._sampler.advance_to(
                    target_draw_count,
                    cancel_requested=cancel_requested,
                )
            )
        else:
            estimate = self._sampler.advance_preview_to(
                target_draw_count,
                cancel_requested=cancel_requested,
            )
        raster = DetectorRaster(estimate=estimate, wall_time_s=perf_counter() - start)
        if cancel_requested():
            self._sampler.reset()
            raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
        texture = _prepare_full_native_texture(estimate.image_A2) if self._prepare_texture else None
        return raster, texture

    def reset_after_cancellation(self) -> None:
        if self._sampler is not None and self._sampler.draws_completed > 0:
            self._sampler.reset()


class _DetectorRenderWorker:
    """One long-lived worker that owns the CUDA context and compiled sampler."""

    def __init__(
        self,
        config: SimulationConfiguration,
        *,
        detector_seed: int,
        execution_backend: str,
        prepare_texture: bool,
        outcomes: queue.SimpleQueue[_RenderOutcome],
    ) -> None:
        self._commands: queue.SimpleQueue[_ScheduledRender | None] = queue.SimpleQueue()
        self._outcomes = outcomes
        self._stop_event = threading.Event()
        self._session_arguments = (
            config,
            detector_seed,
            execution_backend,
            prepare_texture,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="detector-render",
            daemon=True,
        )
        self._thread.start()

    def submit(self, stage: _ScheduledRender) -> None:
        self._commands.put(stage)

    def close(self) -> None:
        self._stop_event.set()
        self._commands.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _run(self) -> None:
        config, detector_seed, execution_backend, prepare_texture = self._session_arguments
        session: _DetectorRenderSession | None = None
        while True:
            stage = self._commands.get()
            if stage is None:
                return
            try:
                if session is None:
                    session = _DetectorRenderSession(
                        config,
                        detector_seed=detector_seed,
                        execution_backend=execution_backend,
                        prepare_texture=prepare_texture,
                    )
                raster, texture = session.render(stage, stop_requested=self._stop_event)
                outcome = _RenderOutcome(stage, raster, texture, None)
            except MonteCarloSamplingCancelled:
                try:
                    if session is None:
                        raise RuntimeError("render session was not initialized")
                    session.reset_after_cancellation()
                except Exception as error:
                    session = None
                    outcome = _RenderOutcome(
                        stage,
                        None,
                        None,
                        f"could not reset cancelled sampler: {error}",
                    )
                else:
                    outcome = _RenderOutcome(stage, None, None, None, cancelled=True)
            except Exception as error:
                session = None
                outcome = _RenderOutcome(stage, None, None, str(error))
            self._outcomes.put(outcome)


@dataclass(frozen=True, slots=True)
class _ControlSpec:
    field_name: str
    label: str
    minimum: float
    maximum: float
    invalidation_scope: str

    def __post_init__(self) -> None:
        if self.invalidation_scope not in {"detector", "incident"}:
            raise ValueError("invalidation_scope must be detector or incident")


_DETECTOR_ONLY_DELTA_FIELDS = frozenset(
    {
        "detector_pitch_offset_deg",
        "detector_yaw_offset_deg",
        "detector_in_plane_rotation_offset_deg",
        "detector_column_translation_mm",
        "detector_row_translation_mm",
        "detector_distance_offset_mm",
    }
)


def _changed_delta_fields(
    previous: GeometryDeltas | None,
    current: GeometryDeltas,
) -> frozenset[str]:
    if previous is None:
        return frozenset(item.name for item in fields(current))
    return frozenset(
        item.name
        for item in fields(current)
        if getattr(previous, item.name) != getattr(current, item.name)
    )


def _requires_incident_rebuild(deltas: GeometryDeltas) -> bool:
    return any(
        getattr(deltas, item.name) != 0.0
        for item in fields(deltas)
        if item.name not in _DETECTOR_ONLY_DELTA_FIELDS
    )


_CONTROL_SPECS = (
    _ControlSpec(
        "detector_pitch_offset_deg",
        r"detector pitch $-\Delta\gamma_{\rm RA}$ (deg)",
        -10.0,
        10.0,
        "detector",
    ),
    _ControlSpec(
        "detector_yaw_offset_deg",
        r"detector yaw $\Delta\Gamma_{\rm RA}$ (deg)",
        -10.0,
        10.0,
        "detector",
    ),
    _ControlSpec(
        "detector_in_plane_rotation_offset_deg",
        r"detector in-plane $\Delta\chi_D$ (deg)",
        -10.0,
        10.0,
        "detector",
    ),
    _ControlSpec(
        "detector_column_translation_mm",
        r"detector column $\Delta x_D$ [$x_0$-coupled] (mm)",
        -20.0,
        20.0,
        "detector",
    ),
    _ControlSpec(
        "detector_row_translation_mm",
        r"detector row $\Delta y_D$ [$y_0$-coupled] (mm)",
        -20.0,
        20.0,
        "detector",
    ),
    _ControlSpec(
        "detector_distance_offset_mm",
        r"detector-normal distance $\Delta D_n$ (mm)",
        -50.0,
        50.0,
        "detector",
    ),
    _ControlSpec(
        "goniometer_axis_pitch_offset_deg",
        r"goniometer-axis pitch $\Delta\alpha$ [RA-SIM cor_angle] (deg)",
        -5.0,
        5.0,
        "incident",
    ),
    _ControlSpec(
        "goniometer_axis_yaw_offset_deg",
        r"goniometer-axis yaw $\Delta\psi_g$ [RA-SIM psi_z] (deg)",
        -5.0,
        5.0,
        "incident",
    ),
    _ControlSpec(
        _INCIDENCE_CONTROL_FIELD,
        r"effective incidence $\theta_i$ (deg)",
        _INCIDENCE_CONTROL_MINIMUM_DEG,
        _INCIDENCE_CONTROL_MAXIMUM_DEG,
        "incident",
    ),
    _ControlSpec(
        "effective_sample_tilt_offset_deg",
        r"effective sample tilt $\Delta\delta$ [RA-SIM $\chi$] (deg)",
        -5.0,
        5.0,
        "incident",
    ),
    _ControlSpec(
        "sample_in_plane_rotation_offset_deg",
        r"sample in-plane $\Delta\chi_S$ [RA-SIM $-\Delta\psi$] (deg)",
        -10.0,
        10.0,
        "incident",
    ),
    _ControlSpec(
        "sample_in_plane_x_translation_mm",
        r"sample in-plane $\Delta x_S$ (mm)",
        -2.0,
        2.0,
        "incident",
    ),
    _ControlSpec(
        "sample_in_plane_y_translation_mm",
        r"sample in-plane $\Delta y_S$ (mm)",
        -2.0,
        2.0,
        "incident",
    ),
    _ControlSpec(
        "sample_normal_translation_mm",
        r"sample normal $\Delta n_S=-\Delta z_S$ (mm)",
        -2.0,
        2.0,
        "incident",
    ),
)


class _OpenGLDetectorPresenter:
    """Persistent R32F texture overlay for the Matplotlib Qt canvas."""

    backend_id = "qt_opengl_r32f.v1"

    def __init__(
        self,
        canvas: object,
        *,
        detector_shape_rc: tuple[int, int],
        magma_rgba_u8: NDArray[np.uint8],
    ) -> None:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtOpenGL import (
                QOpenGLShader,
                QOpenGLShaderProgram,
                QOpenGLTexture,
                QOpenGLVertexArrayObject,
            )
            from PySide6.QtOpenGLWidgets import QOpenGLWidget
            from PySide6.QtWidgets import QWidget
            from shiboken6 import VoidPtr
        except ImportError as error:
            raise RuntimeError(
                "OpenGL presentation requires the visualization extra; install it or pass "
                "--presentation-backend matplotlib"
            ) from error
        if not isinstance(canvas, QWidget):
            raise RuntimeError(
                "OpenGL presentation requires Matplotlib's QtAgg canvas; pass "
                "--presentation-backend matplotlib for the software fallback"
            )
        rows, columns = detector_shape_rc
        lut = np.ascontiguousarray(magma_rgba_u8, dtype=np.uint8)
        if lut.shape != (256, 4):
            raise ValueError("magma_rgba_u8 must have shape (256, 4)")

        class DetectorTextureWidget(QOpenGLWidget):
            def __init__(self, parent: QWidget) -> None:
                super().__init__(parent)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                self.setAutoFillBackground(False)
                self._detector_texture: QOpenGLTexture | None = None
                self._lut_texture: QOpenGLTexture | None = None
                self._program: QOpenGLShaderProgram | None = None
                self._vertex_array: QOpenGLVertexArrayObject | None = None
                self._pending_frame: _FullNativeTextureFrame | None = None
                self._low_A2 = 1.0e-8
                self._high_A2 = 1.0
                self._initialization_error: str | None = None
                self._resources_released = False
                self._context_generation = 0

            def initializeGL(self) -> None:
                try:
                    self._resources_released = False
                    self._initialization_error = None
                    program = QOpenGLShaderProgram(self)
                    vertex_source = _full_screen_vertex_shader()
                    fragment_source = """
                        #version 330 core
                        in vec2 texture_coordinate;
                        out vec4 fragment_color;
                        uniform sampler2D detector_texture;
                        uniform sampler1D magma_texture;
                        uniform float log_low_A2;
                        uniform float log_high_A2;
                        void main() {
                            float value_A2 = texture(detector_texture, texture_coordinate).r;
                            if (value_A2 <= 0.0) {
                                fragment_color = vec4(0.0667, 0.0706, 0.0902, 1.0);
                                return;
                            }
                            float denominator = max(log_high_A2 - log_low_A2, 1.0e-12);
                            float coordinate = clamp(
                                (log(value_A2) - log_low_A2) / denominator,
                                0.0,
                                1.0
                            );
                            fragment_color = texture(magma_texture, coordinate);
                        }
                    """
                    if not program.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Vertex,
                        vertex_source,
                    ) or not program.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Fragment,
                        fragment_source,
                    ):
                        raise RuntimeError(program.log())
                    if not program.link():
                        raise RuntimeError(program.log())
                    vertex_array = QOpenGLVertexArrayObject(self)
                    if not vertex_array.create():
                        raise RuntimeError("could not create the OpenGL vertex array")
                    detector_texture = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
                    detector_texture.setFormat(QOpenGLTexture.TextureFormat.R32F)
                    detector_texture.setSize(columns, rows)
                    detector_texture.allocateStorage(
                        QOpenGLTexture.PixelFormat.Red,
                        QOpenGLTexture.PixelType.Float32,
                    )
                    detector_texture.setMinMagFilters(
                        QOpenGLTexture.Filter.Nearest,
                        QOpenGLTexture.Filter.Nearest,
                    )
                    detector_texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
                    lut_texture = QOpenGLTexture(QOpenGLTexture.Target.Target1D)
                    lut_texture.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
                    lut_texture.setSize(256)
                    lut_texture.allocateStorage(
                        QOpenGLTexture.PixelFormat.RGBA,
                        QOpenGLTexture.PixelType.UInt8,
                    )
                    lut_texture.setMinMagFilters(
                        QOpenGLTexture.Filter.Linear,
                        QOpenGLTexture.Filter.Linear,
                    )
                    lut_texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
                    lut_texture.setData(
                        QOpenGLTexture.PixelFormat.RGBA,
                        QOpenGLTexture.PixelType.UInt8,
                        VoidPtr(lut.ctypes.data, lut.nbytes, False),
                    )
                    self._program = program
                    self._vertex_array = vertex_array
                    self._detector_texture = detector_texture
                    self._lut_texture = lut_texture
                    self._context_generation += 1
                    self.context().aboutToBeDestroyed.connect(self.release_resources)
                    if self._pending_frame is not None:
                        self._upload(self._pending_frame)
                        self._pending_frame = None
                except Exception as error:
                    self._initialization_error = str(error)

            def _upload(self, frame: _FullNativeTextureFrame) -> None:
                if self._detector_texture is None:
                    raise RuntimeError("the OpenGL detector texture is not initialized")
                self._detector_texture.setData(
                    QOpenGLTexture.PixelFormat.Red,
                    QOpenGLTexture.PixelType.Float32,
                    VoidPtr(
                        frame.image_A2.ctypes.data,
                        frame.image_A2.nbytes,
                        False,
                    ),
                )
                self._low_A2 = frame.low_A2
                self._high_A2 = frame.high_A2

            def present(self, frame: _FullNativeTextureFrame) -> None:
                if frame.image_A2.shape != (rows, columns):
                    raise ValueError("texture frame does not match the native detector shape")
                if self._initialization_error is not None:
                    raise RuntimeError(
                        f"OpenGL detector initialization failed: {self._initialization_error}"
                    )
                if self._detector_texture is None:
                    self._pending_frame = _FullNativeTextureFrame(
                        np.array(frame.image_A2, copy=True, order="C"),
                        frame.low_A2,
                        frame.high_A2,
                    )
                    self.update()
                    return
                self.makeCurrent()
                try:
                    self._upload(frame)
                finally:
                    self.doneCurrent()
                self.update()

            def paintGL(self) -> None:
                functions = self.context().functions()
                functions.glClearColor(0.0667, 0.0706, 0.0902, 1.0)
                functions.glClear(0x00004000)
                if (
                    self._program is None
                    or self._vertex_array is None
                    or self._detector_texture is None
                    or self._lut_texture is None
                ):
                    return
                self._program.bind()
                self._vertex_array.bind()
                self._detector_texture.bind(0)
                self._lut_texture.bind(1)
                functions.glUniform1i(self._program.uniformLocation("detector_texture"), 0)
                functions.glUniform1i(self._program.uniformLocation("magma_texture"), 1)
                functions.glUniform1f(
                    self._program.uniformLocation("log_low_A2"),
                    math.log(self._low_A2),
                )
                functions.glUniform1f(
                    self._program.uniformLocation("log_high_A2"),
                    math.log(self._high_A2),
                )
                functions.glDrawArrays(0x0004, 0, 3)
                self._lut_texture.release()
                self._detector_texture.release()
                self._vertex_array.release()
                self._program.release()

            def release_resources(self) -> None:
                if self._resources_released:
                    return
                self._resources_released = True
                if not self.isValid():
                    self._detector_texture = None
                    self._lut_texture = None
                    self._vertex_array = None
                    self._program = None
                    return
                self.makeCurrent()
                try:
                    for texture in (self._detector_texture, self._lut_texture):
                        if texture is not None:
                            texture.destroy()
                    if self._vertex_array is not None:
                        self._vertex_array.destroy()
                finally:
                    self._detector_texture = None
                    self._lut_texture = None
                    self._vertex_array = None
                    self._program = None
                    self.doneCurrent()

        self._canvas = canvas
        self._widget = DetectorTextureWidget(canvas)
        self._widget.show()
        self._widget.raise_()

    def sync_to_axes(self, axes: object) -> None:
        ratio = float(getattr(self._canvas, "device_pixel_ratio", 1.0))
        bbox = axes.bbox
        figure_height = float(self._canvas.figure.bbox.height)
        self._widget.setGeometry(
            round(float(bbox.x0) / ratio),
            round((figure_height - float(bbox.y1)) / ratio),
            max(1, round(float(bbox.width) / ratio)),
            max(1, round(float(bbox.height) / ratio)),
        )
        self._widget.raise_()

    def present(self, frame: _FullNativeTextureFrame) -> None:
        self._widget.present(frame)

    @property
    def initialization_error(self) -> str | None:
        return self._widget._initialization_error

    @property
    def context_generation(self) -> int:
        return self._widget._context_generation

    def close(self) -> None:
        self._widget.release_resources()
        self._widget.close()
        self._widget.deleteLater()


class InteractiveDetectorViewer:
    """Continuously scheduled full-native Monte Carlo detector viewer."""

    def __init__(
        self,
        config: SimulationConfiguration,
        *,
        draws_per_source_state: int,
        initial_source_sample_count: int,
        detector_seed: int,
        execution_backend: str,
        presentation_backend: str,
    ) -> None:
        from matplotlib import pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.widgets import Button, Slider

        self._config = config
        configured_axis_rotations = config.instrument.axis_rotations
        if len(configured_axis_rotations) != 1:
            raise ValueError("absolute theta_i control requires one configured incidence axis")
        self._configured_incidence_angle_deg = float(configured_axis_rotations[0].angle_deg)
        if not (
            _INCIDENCE_CONTROL_MINIMUM_DEG
            <= self._configured_incidence_angle_deg
            <= _INCIDENCE_CONTROL_MAXIMUM_DEG
        ):
            raise ValueError("configured incidence angle must be between 0 and 20 degrees")
        self._initial_draws_per_source_state = draws_per_source_state
        self._initial_source_sample_count = initial_source_sample_count
        self._detector_seed = detector_seed
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be cpu or cuda")
        if presentation_backend not in {"opengl", "matplotlib"}:
            raise ValueError("presentation_backend must be opengl or matplotlib")
        self._execution_backend = execution_backend
        self._presentation_backend = presentation_backend
        self._revision = 0
        self._closed = False
        self._outcomes: queue.SimpleQueue[_RenderOutcome] = queue.SimpleQueue()
        self._scheduler = _ProgressiveRenderScheduler()
        self._source_sample_count_committed = initial_source_sample_count
        self._suspend_updates = False
        self._slider_dirty = False
        self._preview_pending = False
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
        if presentation_backend == "opengl":
            self._colorbar.set_label(
                "relative weighted detector-pixel mass (8-decade GPU log scale)"
            )
            magma_rgba_u8 = np.rint(
                self._cmap(np.linspace(0.0, 1.0, 256, dtype=np.float64)) * 255.0
            ).astype(np.uint8)
            self._open_gl_presenter: _OpenGLDetectorPresenter | None = _OpenGLDetectorPresenter(
                self.figure.canvas,
                detector_shape_rc=(rows, columns),
                magma_rgba_u8=magma_rgba_u8,
            )
        else:
            self._open_gl_presenter = None
        self._open_gl_context_generation = (
            self._open_gl_presenter.context_generation if self._open_gl_presenter is not None else 0
        )
        self._open_gl_failed = False

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
                valinit=(
                    self._configured_incidence_angle_deg
                    if spec.field_name == _INCIDENCE_CONTROL_FIELD
                    else 0.0
                ),
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
            "Geometry and draw controls stream latest-only progressive previews while dragging.\n"
            "Source count commits on release; every frame keeps the full native grid.\n"
            r"$\theta_i$ is absolute; all other geometry controls are configured-pose deltas."
            "\n"
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
        self._worker = _DetectorRenderWorker(
            config,
            detector_seed=detector_seed,
            execution_backend=execution_backend,
            prepare_texture=presentation_backend == "opengl",
            outcomes=self._outcomes,
        )
        self._poll_timer = self.figure.canvas.new_timer(interval=5)
        self._poll_timer.add_callback(self._poll_render)
        self._poll_timer.start()
        self._set_status("ready; rendering the configured Monte Carlo pixel image")
        self._redraw_and_cache()
        self._request_render()

    def _deltas(self) -> GeometryDeltas:
        return _control_values_to_geometry_deltas(
            {name: float(slider.val) for name, slider in self._sliders.items()},
            configured_incidence_angle_deg=self._configured_incidence_angle_deg,
        )

    def _current_request(self) -> _RenderRequest:
        return _RenderRequest(
            revision=self._revision,
            source_sample_count=self._source_sample_count_committed,
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
        if self._closed:
            return
        if self._open_gl_presenter is not None:
            self._open_gl_presenter.sync_to_axes(self._image_axis)
        if not self.figure.canvas.supports_blit:
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

    def _show_raster(
        self,
        raster: DetectorRaster,
        texture: _FullNativeTextureFrame | None,
    ) -> None:
        from matplotlib.colors import LogNorm

        estimate = raster.estimate
        if self._open_gl_presenter is None:
            display, low, high = self._display_values(raster)
            self._image.set_data(display)
            self._image.set_norm(LogNorm(vmin=low, vmax=high, clip=True))
            self._colorbar.update_normal(self._image)
        else:
            if texture is None:
                raise RuntimeError("OpenGL presentation requires a prepared full-native texture")
            self._open_gl_presenter.present(texture)
        rod_count = (
            estimate.rod_count
            if isinstance(estimate, MonteCarloDetectorPresentation)
            else len(estimate.rods)
        )
        device = f" on {estimate.execution_device}" if estimate.execution_device else ""
        self._image_axis.set_title(
            f"MONTE CARLO NATIVE PIXEL MASS: {estimate.source_state_count} $k_i$ states x "
            f"{estimate.draws_per_source_state} draws, {rod_count} physical rods\n"
            f"{estimate.visible_hit_count:,} visible root deposits; seed {estimate.seed}; "
            f"{raster.wall_time_s:.3f} s; {estimate.execution_backend}{device}; "
            f"total {estimate.total_detector_mass_A2:.6g} $\\AA^2$"
        )
        if self._open_gl_presenter is None:
            self._redraw_and_cache()
        else:
            self._blit_axis(
                self._image_axis,
                self._image_background,
                self._image_axis.title,
            )

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
        if slider is self._source_sample_slider:
            self._scheduler.cancel()
            self._preview_pending = False
            requested_source_count = round(self._source_sample_slider.val)
            self._set_status(
                f"release to rebuild {requested_source_count} $k_i$ states; "
                "the full-native image remains visible"
            )
            return
        self._preview_pending = True
        request = self._current_request()
        self._set_status(
            f"tracking revision {request.revision}: latest-only full-native preview pending; "
            f"settled target {request.draws_per_source_state} draws"
        )

    def _on_button_release(self, _event: object) -> None:
        if not self._suspend_updates and self._slider_dirty:
            self._source_sample_count_committed = round(self._source_sample_slider.val)
            self._request_render()

    def _request_render(self, _event: object | None = None) -> None:
        if self._closed:
            return
        self._source_sample_count_committed = round(self._source_sample_slider.val)
        self._slider_dirty = False
        self._preview_pending = False
        self._scheduler.submit(self._current_request(), settled=True)
        self._dispatch_next()

    def _dispatch_next(self) -> None:
        stage = self._scheduler.start_next()
        if stage is None:
            return
        result_kind = "settled result" if stage.materialize_result else "preview"
        self._set_status(
            f"rendering {result_kind} for revision {stage.request.revision}: "
            f"{stage.request.source_sample_count} $k_i$ states x "
            f"{stage.draws_per_source_state} prefix draws; newer revisions cancel this stage"
        )
        self._worker.submit(stage)

    def _poll_render(self) -> None:
        if self._closed:
            return
        presentation_error = (
            self._open_gl_presenter.initialization_error
            if self._open_gl_presenter is not None
            else None
        )
        if self._open_gl_presenter is not None:
            context_generation = self._open_gl_presenter.context_generation
            if context_generation != self._open_gl_context_generation:
                rerender = self._open_gl_context_generation > 0 or self._open_gl_failed
                self._open_gl_context_generation = context_generation
                self._open_gl_failed = False
                if rerender:
                    self._scheduler.cancel()
                    self._scheduler.reset_latest()
                    self._scheduler.submit(self._current_request(), settled=True)
        if presentation_error is not None:
            self._open_gl_failed = True
            self._preview_pending = False
            self._scheduler.cancel()
        if self._preview_pending:
            self._preview_pending = False
            self._scheduler.submit(self._current_request(), settled=False)
        try:
            outcome = self._outcomes.get_nowait()
        except queue.Empty:
            if presentation_error is None:
                self._dispatch_next()
            else:
                self._set_status(
                    "OpenGL presentation failed: "
                    f"{presentation_error}; restart with --presentation-backend matplotlib"
                )
            return
        completed_draw_count = (
            outcome.raster.estimate.draws_per_source_state if outcome.raster is not None else None
        )
        if outcome.error is not None or outcome.cancelled:
            accepted = self._scheduler.fail(outcome.stage)
        else:
            accepted = self._scheduler.complete(
                outcome.stage,
                completed_draw_count=completed_draw_count,
            )
        current = self._current_request()
        if presentation_error is not None:
            self._set_status(
                "OpenGL presentation failed: "
                f"{presentation_error}; restart with --presentation-backend matplotlib"
            )
        elif accepted and outcome.error is not None and outcome.stage.request == current:
            self._scheduler.discard_pending()
            self._set_status(f"requested render failed: {outcome.error}")
        elif accepted and outcome.raster is not None and outcome.stage.request == current:
            try:
                self._show_raster(outcome.raster, outcome.texture)
            except Exception as error:
                self._scheduler.reset_latest()
                self._set_status(f"presentation failed: {error}")
            else:
                estimate = outcome.raster.estimate
                state = "settled" if outcome.stage.materialize_result else "preview"
                self._set_status(
                    f"{state} revision {current.revision}: {current.source_sample_count} $k_i$ x "
                    f"{estimate.draws_per_source_state} draws in "
                    f"{outcome.raster.wall_time_s:.3f} s"
                )
        elif outcome.cancelled or not accepted:
            self._set_status("discarded superseded work; rendering the latest revision")
        if presentation_error is None:
            self._dispatch_next()

    def _reset(self, _event: object | None = None) -> None:
        self._suspend_updates = True
        try:
            for name, slider in self._sliders.items():
                slider.set_val(
                    self._configured_incidence_angle_deg
                    if name == _INCIDENCE_CONTROL_FIELD
                    else 0.0
                )
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
        self._scheduler.cancel()
        self._worker.close()
        self._poll_timer.stop()
        if self._open_gl_presenter is not None:
            self._open_gl_presenter.close()

    def _on_resize(self, _event: object) -> None:
        self._redraw_and_cache()
        if self._open_gl_presenter is not None:
            self._open_gl_presenter.sync_to_axes(self._image_axis)

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
    parser.add_argument(
        "--execution-backend",
        choices=("cuda", "cpu"),
        default="cuda",
        help="explicit forward Monte Carlo backend (default: cuda; no automatic fallback)",
    )
    parser.add_argument(
        "--presentation-backend",
        choices=("opengl", "matplotlib"),
        default="opengl",
        help="full-native raster presenter (default: opengl; matplotlib is the explicit fallback)",
    )
    args = parser.parse_args(argv)
    if args.presentation_backend == "opengl":
        try:
            import matplotlib
            from PySide6.QtGui import QSurfaceFormat
        except ImportError as error:
            raise RuntimeError(
                "OpenGL presentation requires the visualization extra; install it or pass "
                "--presentation-backend matplotlib"
            ) from error
        surface_format = QSurfaceFormat()
        surface_format.setVersion(3, 3)
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(surface_format)
        matplotlib.use("qtagg", force=True)
    config = load_simulation_config(args.config.resolve(), repository_root=ROOT)
    source_sample_count = (
        config.source.sample_count if args.source_sample_count is None else args.source_sample_count
    )
    viewer = InteractiveDetectorViewer(
        config,
        draws_per_source_state=args.draws_per_source_state,
        initial_source_sample_count=source_sample_count,
        detector_seed=args.seed,
        execution_backend=args.execution_backend,
        presentation_backend=args.presentation_backend,
    )
    viewer.show()


if __name__ == "__main__":
    main()
