"""Workflow helpers for structure-factor pruning and solve-q controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import controllers as gui_controllers
from . import views as gui_views
from .runtime_values import resolve_runtime_value as _resolve_runtime_value
from .runtime_values import safe_var_get as _safe_var_get
from .runtime_values import safe_var_set as _safe_var_set
from .runtime_values import safe_var_trace_add as _safe_var_trace_add


@dataclass
class StructureFactorPruningRuntimeBindings:
    """Runtime callbacks and state used by the pruning / solve-q controls."""

    view_state: Any
    simulation_runtime_state: Any
    bragg_qr_manager_state: Any
    clip_prune_bias: Callable[[object], float]
    clip_solve_q_steps: Callable[[object], int]
    clip_solve_q_rel_tol: Callable[[object], float]
    normalize_solve_q_mode_label: Callable[[object], str]
    apply_filters: Callable[..., Mapping[str, object]] = gui_controllers.apply_bragg_qr_filters
    schedule_update: Callable[[], None] | None = None
    refresh_window: Callable[[], None] | None = None


@dataclass(frozen=True)
class RuntimeSolveQValues:
    """Normalized solve-q values derived from runtime GUI state or defaults."""

    steps: int
    rel_tol: float
    mode_label: str
    mode_flag: int


@dataclass(frozen=True)
class StructureFactorPruningControlDefaults:
    """Normalized pruning / solve-q defaults for GUI control setup."""

    prune_bias: float
    solve_q: RuntimeSolveQValues


def make_runtime_structure_factor_pruning_bindings_factory(
    *,
    view_state_factory: object,
    simulation_runtime_state,
    bragg_qr_manager_state,
    clip_prune_bias: Callable[[object], float],
    clip_solve_q_steps: Callable[[object], int],
    clip_solve_q_rel_tol: Callable[[object], float],
    normalize_solve_q_mode_label: Callable[[object], str],
    apply_filters: Callable[..., Mapping[str, object]] = gui_controllers.apply_bragg_qr_filters,
    schedule_update_factory: object | None = None,
    refresh_window_factory: object | None = None,
) -> Callable[[], StructureFactorPruningRuntimeBindings]:
    """Return a zero-arg factory for live pruning / solve-q runtime bindings."""

    def _build_bindings() -> StructureFactorPruningRuntimeBindings:
        return StructureFactorPruningRuntimeBindings(
            view_state=_resolve_runtime_value(view_state_factory),
            simulation_runtime_state=simulation_runtime_state,
            bragg_qr_manager_state=bragg_qr_manager_state,
            clip_prune_bias=clip_prune_bias,
            clip_solve_q_steps=clip_solve_q_steps,
            clip_solve_q_rel_tol=clip_solve_q_rel_tol,
            normalize_solve_q_mode_label=normalize_solve_q_mode_label,
            apply_filters=apply_filters,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            refresh_window=_resolve_runtime_value(refresh_window_factory),
        )

    return _build_bindings


def make_runtime_structure_factor_pruning_status_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[[], str]:
    """Return a zero-arg callback that refreshes pruning status from live bindings."""

    return lambda: update_runtime_structure_factor_pruning_status(bindings_factory())


def make_runtime_bragg_qr_filter_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., Mapping[str, object]]:
    """Return a callback that applies Bragg-Qr filters from live bindings."""

    def _apply_filters(*, trigger_update: bool = True) -> Mapping[str, object]:
        return apply_runtime_bragg_qr_filters(
            bindings_factory(),
            trigger_update=trigger_update,
        )

    return _apply_filters


def make_runtime_sf_prune_bias_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime SF-pruning bias updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_sf_prune_bias_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_steps_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q interval updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_steps_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_rel_tol_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q tolerance updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_rel_tol_change(bindings_factory())

    return _on_change


def make_runtime_solve_q_control_states_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[[], str]:
    """Return a zero-arg callback that syncs runtime solve-q control state."""

    return lambda: set_runtime_solve_q_control_states(bindings_factory())


def make_runtime_solve_q_mode_change_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[..., bool]:
    """Return a trace callback for runtime solve-q mode updates."""

    def _on_change(*_args) -> bool:
        return on_runtime_solve_q_mode_change(bindings_factory())

    return _on_change


def _var_float_matches(var: object, value: object) -> bool:
    try:
        raw_float = float(_safe_var_get(var))
    except (TypeError, ValueError):
        return False
    return bool(np.isclose(raw_float, float(value), rtol=0.0, atol=1e-12))


def clip_runtime_sf_prune_bias(
    value: object,
    *,
    fallback: object,
    minimum: float,
    maximum: float,
) -> float:
    """Clip one SF-pruning bias using the shared controller helper."""

    return float(
        gui_controllers.clip_structure_factor_prune_bias(
            value,
            fallback=fallback,
            minimum=minimum,
            maximum=maximum,
        )
    )


def clip_runtime_solve_q_steps(
    value: object,
    *,
    fallback: object,
    minimum: int,
    maximum: int,
) -> int:
    """Clip one solve-q step count using the shared controller helper."""

    return int(
        gui_controllers.clip_solve_q_steps(
            value,
            fallback=fallback,
            minimum=minimum,
            maximum=maximum,
        )
    )


def clip_runtime_solve_q_rel_tol(
    value: object,
    *,
    fallback: object,
    minimum: float,
    maximum: float,
) -> float:
    """Clip one solve-q relative tolerance using the shared controller helper."""

    return float(
        gui_controllers.clip_solve_q_rel_tol(
            value,
            fallback=fallback,
            minimum=minimum,
            maximum=maximum,
        )
    )


def normalize_runtime_solve_q_mode_label(value: object) -> str:
    """Normalize one solve-q mode label using the shared controller helper."""

    return str(gui_controllers.normalize_solve_q_mode_label(value))


def runtime_solve_q_mode_flag_from_label(
    label: str,
    *,
    uniform_flag: int,
    adaptive_flag: int,
) -> int:
    """Map one normalized solve-q label to the configured mode flag."""

    return int(
        gui_controllers.solve_q_mode_flag_from_label(
            label,
            uniform_flag=uniform_flag,
            adaptive_flag=adaptive_flag,
        )
    )


def build_runtime_solve_q_values(
    raw_steps: object,
    raw_rel_tol: object,
    raw_mode: object,
    *,
    steps_fallback: object,
    steps_minimum: int,
    steps_maximum: int,
    rel_tol_fallback: object,
    rel_tol_minimum: float,
    rel_tol_maximum: float,
    uniform_flag: int,
    adaptive_flag: int,
) -> RuntimeSolveQValues:
    """Normalize one solve-q value set from raw defaults or runtime state."""

    mode_label = normalize_runtime_solve_q_mode_label(raw_mode)
    return RuntimeSolveQValues(
        steps=clip_runtime_solve_q_steps(
            raw_steps,
            fallback=steps_fallback,
            minimum=steps_minimum,
            maximum=steps_maximum,
        ),
        rel_tol=clip_runtime_solve_q_rel_tol(
            raw_rel_tol,
            fallback=rel_tol_fallback,
            minimum=rel_tol_minimum,
            maximum=rel_tol_maximum,
        ),
        mode_label=mode_label,
        mode_flag=runtime_solve_q_mode_flag_from_label(
            mode_label,
            uniform_flag=uniform_flag,
            adaptive_flag=adaptive_flag,
        ),
    )


def build_runtime_structure_factor_pruning_defaults(
    raw_prune_bias: object,
    raw_solve_q_steps: object,
    raw_solve_q_rel_tol: object,
    raw_solve_q_mode: object,
    *,
    prune_bias_fallback: object,
    prune_bias_minimum: float,
    prune_bias_maximum: float,
    steps_fallback: object,
    steps_minimum: int,
    steps_maximum: int,
    rel_tol_fallback: object,
    rel_tol_minimum: float,
    rel_tol_maximum: float,
    uniform_flag: int,
    adaptive_flag: int,
) -> StructureFactorPruningControlDefaults:
    """Normalize pruning / solve-q defaults for GUI control setup."""

    return StructureFactorPruningControlDefaults(
        prune_bias=clip_runtime_sf_prune_bias(
            raw_prune_bias,
            fallback=prune_bias_fallback,
            minimum=prune_bias_minimum,
            maximum=prune_bias_maximum,
        ),
        solve_q=build_runtime_solve_q_values(
            raw_solve_q_steps,
            raw_solve_q_rel_tol,
            raw_solve_q_mode,
            steps_fallback=steps_fallback,
            steps_minimum=steps_minimum,
            steps_maximum=steps_maximum,
            rel_tol_fallback=rel_tol_fallback,
            rel_tol_minimum=rel_tol_minimum,
            rel_tol_maximum=rel_tol_maximum,
            uniform_flag=uniform_flag,
            adaptive_flag=adaptive_flag,
        ),
    )


def apply_runtime_structure_factor_pruning_defaults(
    view_state: Any,
    defaults: StructureFactorPruningControlDefaults,
) -> None:
    """Apply normalized pruning / solve-q defaults to one control view state."""

    if view_state is None:
        return
    _safe_var_set(getattr(view_state, "sf_prune_bias_var", None), defaults.prune_bias)
    _safe_var_set(
        getattr(view_state, "solve_q_mode_var", None),
        defaults.solve_q.mode_label,
    )
    _safe_var_set(
        getattr(view_state, "solve_q_steps_var", None),
        float(defaults.solve_q.steps),
    )
    _safe_var_set(
        getattr(view_state, "solve_q_rel_tol_var", None),
        float(defaults.solve_q.rel_tol),
    )


def initialize_runtime_structure_factor_pruning_controls(
    view_state: Any,
    *,
    on_sf_prune_bias_change: Callable[..., object],
    update_status_label: Callable[[], object],
    on_solve_q_steps_change: Callable[..., object],
    on_solve_q_rel_tol_change: Callable[..., object],
    on_solve_q_mode_change: Callable[..., object],
    set_solve_q_control_states: Callable[[], object],
) -> None:
    """Attach trace callbacks and sync one pruning control cluster."""

    if view_state is None:
        return
    _safe_var_trace_add(
        getattr(view_state, "sf_prune_bias_var", None),
        on_sf_prune_bias_change,
    )
    _safe_var_trace_add(
        getattr(view_state, "solve_q_steps_var", None),
        on_solve_q_steps_change,
    )
    _safe_var_trace_add(
        getattr(view_state, "solve_q_rel_tol_var", None),
        on_solve_q_rel_tol_change,
    )
    _safe_var_trace_add(
        getattr(view_state, "solve_q_mode_var", None),
        on_solve_q_mode_change,
    )
    update_status_label()
    set_solve_q_control_states()


def current_runtime_sf_prune_bias(
    bindings: StructureFactorPruningRuntimeBindings,
) -> float:
    """Return the current clipped SF-pruning bias from the bound view state."""

    return float(
        bindings.clip_prune_bias(
            _safe_var_get(getattr(bindings.view_state, "sf_prune_bias_var", None))
        )
    )


def current_runtime_solve_q_steps(
    bindings: StructureFactorPruningRuntimeBindings,
) -> int:
    """Return the current clipped solve-q interval count from the bound view state."""

    return int(
        bindings.clip_solve_q_steps(
            _safe_var_get(getattr(bindings.view_state, "solve_q_steps_var", None))
        )
    )


def current_runtime_solve_q_rel_tol(
    bindings: StructureFactorPruningRuntimeBindings,
) -> float:
    """Return the current clipped solve-q relative tolerance from the bound view state."""

    return float(
        bindings.clip_solve_q_rel_tol(
            _safe_var_get(getattr(bindings.view_state, "solve_q_rel_tol_var", None))
        )
    )


def current_runtime_solve_q_mode_label(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Return the normalized solve-q mode label from the bound view state."""

    return str(
        bindings.normalize_solve_q_mode_label(
            _safe_var_get(getattr(bindings.view_state, "solve_q_mode_var", None))
        )
    )


def current_runtime_solve_q_values(
    bindings: StructureFactorPruningRuntimeBindings,
    *,
    uniform_flag: int,
    adaptive_flag: int,
) -> RuntimeSolveQValues:
    """Return the normalized solve-q values from the bound runtime state."""

    mode_label = current_runtime_solve_q_mode_label(bindings)
    return RuntimeSolveQValues(
        steps=current_runtime_solve_q_steps(bindings),
        rel_tol=current_runtime_solve_q_rel_tol(bindings),
        mode_label=mode_label,
        mode_flag=runtime_solve_q_mode_flag_from_label(
            mode_label,
            uniform_flag=uniform_flag,
            adaptive_flag=adaptive_flag,
        ),
    )


def make_runtime_current_sf_prune_bias_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
) -> Callable[[], float]:
    """Return a zero-arg callback that reads the current pruning bias."""

    return lambda: current_runtime_sf_prune_bias(bindings_factory())


def make_runtime_current_solve_q_values_callback(
    bindings_factory: Callable[[], StructureFactorPruningRuntimeBindings],
    *,
    uniform_flag: int,
    adaptive_flag: int,
) -> Callable[[], RuntimeSolveQValues]:
    """Return a zero-arg callback that reads the current solve-q values."""

    return lambda: current_runtime_solve_q_values(
        bindings_factory(),
        uniform_flag=uniform_flag,
        adaptive_flag=adaptive_flag,
    )


def update_runtime_structure_factor_pruning_status(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Refresh the pruning-status label from the current runtime state."""

    text = gui_controllers.format_structure_factor_pruning_status(
        bindings.simulation_runtime_state.sf_prune_stats,
        prune_bias=current_runtime_sf_prune_bias(bindings),
    )
    if bindings.view_state is not None:
        gui_views.set_structure_factor_pruning_status_text(bindings.view_state, text)
    return str(text)


def invalidate_runtime_bragg_qr_filter_results(
    bindings: StructureFactorPruningRuntimeBindings,
    *,
    preserve_primary_cache: bool = False,
) -> None:
    """Clear cached simulation artifacts invalidated by Bragg-Qr filter changes."""

    bindings.simulation_runtime_state.last_sim_signature = None
    bindings.simulation_runtime_state.last_simulation_signature = None
    bindings.simulation_runtime_state.stored_max_positions_local = None
    bindings.simulation_runtime_state.stored_q_group_content_signature = None
    bindings.simulation_runtime_state.stored_sim_image = None
    bindings.simulation_runtime_state.stored_peak_table_lattice = None
    bindings.simulation_runtime_state.selected_peak_record = None
    if not preserve_primary_cache:
        bindings.simulation_runtime_state.primary_contribution_cache_signature = None
        bindings.simulation_runtime_state.primary_active_contribution_keys.clear()
        bindings.simulation_runtime_state.primary_hit_table_cache.clear()
        bindings.simulation_runtime_state.primary_intersection_cache_entry_cache.clear()
        bindings.simulation_runtime_state.primary_source_mode = "miller"
        bindings.simulation_runtime_state.primary_filter_signature = None


def apply_runtime_bragg_qr_filters(
    bindings: StructureFactorPruningRuntimeBindings,
    *,
    trigger_update: bool = True,
    preserve_primary_cache: bool = False,
) -> Mapping[str, object]:
    """Apply Bragg-Qr / SF-pruning filters and runtime side effects."""

    stats = bindings.apply_filters(
        bindings.simulation_runtime_state,
        bindings.bragg_qr_manager_state,
        prune_bias=current_runtime_sf_prune_bias(bindings),
    )
    update_runtime_structure_factor_pruning_status(bindings)
    invalidate_runtime_bragg_qr_filter_results(
        bindings,
        preserve_primary_cache=preserve_primary_cache,
    )
    if callable(bindings.refresh_window):
        bindings.refresh_window()
    if trigger_update and callable(bindings.schedule_update):
        bindings.schedule_update()
    return stats


def set_runtime_solve_q_control_states(
    bindings: StructureFactorPruningRuntimeBindings,
) -> str:
    """Sync the adaptive relative-tolerance enabled state from the current mode."""

    mode = current_runtime_solve_q_mode_label(bindings)
    if bindings.view_state is not None:
        gui_views.set_structure_factor_pruning_rel_tol_enabled(
            bindings.view_state,
            enabled=(mode != "uniform"),
        )
    return mode


def on_runtime_sf_prune_bias_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one SF-pruning bias variable change from the runtime GUI."""

    var = getattr(bindings.view_state, "sf_prune_bias_var", None)
    clipped_value = current_runtime_sf_prune_bias(bindings)
    if not _var_float_matches(var, clipped_value):
        _safe_var_set(var, clipped_value)
        return False

    apply_runtime_bragg_qr_filters(
        bindings,
        trigger_update=True,
        preserve_primary_cache=True,
    )
    return True


def on_runtime_solve_q_steps_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q interval-count change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_steps_var", None)
    clipped_value = current_runtime_solve_q_steps(bindings)
    if not _var_float_matches(var, clipped_value):
        _safe_var_set(var, float(clipped_value))
        return False

    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True


def on_runtime_solve_q_rel_tol_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q relative-tolerance change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_rel_tol_var", None)
    clipped_value = current_runtime_solve_q_rel_tol(bindings)
    if not _var_float_matches(var, clipped_value):
        _safe_var_set(var, float(clipped_value))
        return False

    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True


def on_runtime_solve_q_mode_change(
    bindings: StructureFactorPruningRuntimeBindings,
) -> bool:
    """Handle one solve-q mode change from the runtime GUI."""

    var = getattr(bindings.view_state, "solve_q_mode_var", None)
    raw_value = _safe_var_get(var)
    normalized = current_runtime_solve_q_mode_label(bindings)
    if raw_value != normalized:
        _safe_var_set(var, normalized)
        return False

    set_runtime_solve_q_control_states(bindings)
    invalidate_runtime_bragg_qr_filter_results(bindings)
    if callable(bindings.schedule_update):
        bindings.schedule_update()
    return True
