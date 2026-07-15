"""Frozen-frame caked-space objective for the first geometry-fit rung.

This module deliberately knows nothing about Tk, saved-state mutation, or the
least-squares implementation.  Callers freeze one exact native-to-caked
projector, lock the two measured branch targets, and supply a fresh native
prediction provider for every evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from ra_sim.fitting._numeric import finite_float_or_none as _finite_float
from ra_sim.utils import normalized_label_token as _normalized_frame_token


ACTIVE_PARAMETER_NAMES = ("gamma", "Gamma")
ACTIVE_PARAMETER_RUNGS = (
    ACTIVE_PARAMETER_NAMES,
    ("gamma", "Gamma", "chi"),
    ("gamma", "Gamma", "chi", "zb"),
    ("gamma", "Gamma", "chi", "zb", "zs"),
    ("gamma", "Gamma", "chi", "zb", "zs", "psi_z"),
    ("gamma", "Gamma", "chi", "zb", "zs", "psi_z", "cor_angle"),
)
RESIDUAL_UNITS = "deg"
TARGET_BACKGROUND_INDEX = 0
TARGET_BRANCHES = (0, 1)
CAKED_GEOMETRY_EQUIVALENCE_FRESH_OBJECTIVE_MARKER = (
    "_caked_geometry_equivalence_fresh_native_objective"
)

Point = tuple[float, float]
QGroupKey = tuple[object, ...]
NativeToCakedProjector = Callable[[float, float], object]
DisplayToNativeTransform = Callable[[float, float], object]
NativeToDisplayTransform = Callable[[float, float], object]
CandidateKey = tuple[object, ...]


class CakedGeometryObjectiveError(ValueError):
    """Fail-closed objective contract error with a stable reason code."""

    def __init__(self, reason: str, **details: object) -> None:
        self.reason = str(reason)
        self.details = dict(details)
        suffix = ""
        if self.details:
            suffix = ": " + ", ".join(
                f"{key}={value!r}" for key, value in sorted(self.details.items())
            )
        super().__init__(self.reason + suffix)


@dataclass(frozen=True, slots=True)
class CakedGeometryParameterNumerics:
    """Native-unit numerical policy for one physical fit parameter."""

    units: str
    sensitivity_step: float
    support_step: float
    jacobian_step: float
    jacobian_half_step: float
    x_scale: float
    numeric_bound_shell: float


ACTIVE_PARAMETER_NUMERICS: Mapping[str, CakedGeometryParameterNumerics] = (
    MappingProxyType(
        {
            name: CakedGeometryParameterNumerics(
                units="deg",
                sensitivity_step=0.01,
                support_step=5.0,
                jacobian_step=0.01,
                jacobian_half_step=0.005,
                x_scale=0.5,
                numeric_bound_shell=1.0e-8,
            )
            for name in ("gamma", "Gamma", "psi_z", "cor_angle")
        }
        | {
            # Keep the smallest prespecified pair.  The fitting projector is
            # continuous; cake-bin snapping is audited separately by the render
            # transform proof and must not quantize this derivative.
            "chi": CakedGeometryParameterNumerics(
                units="deg",
                sensitivity_step=0.01,
                # The configured physical chi domain is only +/-1 degree.
                # Keep the large diagnostic probe inside that domain so a
                # missing out-of-bounds branch cannot block the bounded solve.
                support_step=0.5,
                jacobian_step=0.01,
                jacobian_half_step=0.005,
                x_scale=0.5,
                numeric_bound_shell=1.0e-8,
            )
        }
        | {
            name: CakedGeometryParameterNumerics(
                units="m",
                sensitivity_step=1.0e-6,
                support_step=1.0e-4,
                jacobian_step=1.0e-6,
                jacobian_half_step=5.0e-7,
                x_scale=5.0e-5,
                # scipy.optimize.least_squares treats a point within a
                # relative 1e-10 of a bound as active and moves it before the
                # first residual call.  Keep the saved-state x0 genuinely
                # interior while retaining a physically negligible (1 nm)
                # allowance on the line-disfavored side.
                numeric_bound_shell=1.0e-9,
            )
            for name in ("zb", "zs")
        }
    )
)


def caked_geometry_parameter_numerics(
    active_parameter_names: Sequence[str],
) -> tuple[CakedGeometryParameterNumerics, ...]:
    """Return ordered native-unit numerics for one supported active block."""

    names = tuple(str(name) for name in active_parameter_names)
    if names not in ACTIVE_PARAMETER_RUNGS:
        raise CakedGeometryObjectiveError(
            "unsupported_caked_geometry_parameter_rung",
            names=list(names),
        )
    return tuple(ACTIVE_PARAMETER_NUMERICS[name] for name in names)


@dataclass(frozen=True, slots=True)
class DatasetFitSpaceFrame:
    """One immutable dataset frame with a captured exact projector."""

    background_index: int
    native_to_caked: NativeToCakedProjector = field(repr=False, compare=False)
    display_to_native: DisplayToNativeTransform = field(repr=False, compare=False)
    native_to_display: NativeToDisplayTransform = field(repr=False, compare=False)
    display_to_native_source: str
    native_to_display_source: str
    projector_signature_json: str
    projector_digest: str
    frame_digest: str
    roundtrip_tolerance_px: float = 1.0e-6
    projector_kind: str = "frozen_exact_caked"
    units: str = RESIDUAL_UNITS


@dataclass(frozen=True, slots=True)
class CoordinateRouteEvidence:
    """Complete, immutable detector-frame and frozen-caked coordinate route."""

    role: str
    authority: str
    detector_display_px: Point
    detector_display_frame: str
    detector_display_source: str
    detector_native_px: Point
    detector_native_frame: str
    detector_native_source: str
    display_to_native_source: str
    native_to_display_source: str
    native_to_caked_source: str
    frozen_caked_deg: Point
    native_to_caked_audit_deg: Point
    roundtrip_tolerance_px: float
    roundtrip_status: str
    roundtrip_display_error_px: float
    roundtrip_native_error_px: float
    same_frame_display_error_px: float
    same_frame_native_error_px: float
    display_to_native_px: Point
    display_to_native_to_display_px: Point
    native_to_display_px: Point
    native_to_display_to_native_px: Point


@dataclass(frozen=True, slots=True)
class LockedBranchTarget:
    """One fixed measured target and its physical branch identity."""

    background_index: int
    q_group_key: QGroupKey
    hkl: tuple[int, int, int]
    hkl_equivalence_key: tuple[int, int]
    branch_index: int
    measured_caked_deg: Point
    measured_native_px: Point
    measured_detector_display_px: Point
    measurement_origin: str
    saved_caked_audit_deg: Point | None
    coordinate_route: CoordinateRouteEvidence
    pair_id: str
    source_table_index: int | None = None
    source_row_index: int | None = None
    source_reflection_index: int | None = None

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            int(self.background_index),
            self.q_group_key,
            self.hkl_equivalence_key,
            int(self.branch_index),
        )


@dataclass(frozen=True, slots=True)
class CakedGeometryProblem:
    """The complete frozen first-rung objective contract."""

    frame: DatasetFitSpaceFrame
    targets: tuple[LockedBranchTarget, LockedBranchTarget]
    target_hkl_equivalent: tuple[int, int, int]
    hkl_equivalence_key: tuple[int, int]
    target_q_group_key: QGroupKey
    problem_digest: str
    active_parameter_names: tuple[str, ...] = ACTIVE_PARAMETER_NAMES
    residual_units: str = RESIDUAL_UNITS


@dataclass(frozen=True, slots=True)
class BranchSelectionEvidence:
    """Canonical fresh-candidate evidence for one locked physical branch."""

    selection_rule: str
    selection_metric: str
    candidate_count: int
    candidate_keys: tuple[CandidateKey, ...]
    candidate_weights: tuple[float | None, ...]
    candidate_identity_digest: str
    candidate_weight_digest: str
    selected_candidate_key: CandidateKey
    selected_candidate_digest: str
    selected_weight: float | None = None
    runner_up_weight: float | None = None
    margin_abs: float | None = None
    margin_relative: float | None = None
    tie_tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class TrialPrediction:
    """One fresh detector-native prediction returned by the simulator."""

    background_index: int
    q_group_key: QGroupKey
    hkl: tuple[int, int, int]
    branch_index: int
    native_pixel: Point
    source: str
    is_dynamic: bool
    cache_reused: bool
    hkl_equivalence_key: tuple[int, int] | None = None
    predicted_caked_deg: Point | None = None
    candidate_count: int = 1
    selection_rule: str = "unique_fresh_branch_row"
    selection_evidence: BranchSelectionEvidence | None = None
    detector_display_pixel: Point | None = None
    detector_native_frame: str = ""
    detector_display_frame: str = ""
    detector_native_source: str = ""
    detector_display_source: str = ""
    coordinate_route: CoordinateRouteEvidence | None = None


@dataclass(frozen=True, slots=True)
class BranchObjectiveEvaluation:
    """Native and caked diagnostics for one locked branch."""

    background_index: int
    q_group_key: QGroupKey
    hkl: tuple[int, int, int]
    hkl_equivalence_key: tuple[int, int]
    branch_index: int
    pair_id: str
    measurement_origin: str
    saved_caked_audit_deg: Point | None
    measured_native_px: Point
    measured_detector_display_px: Point
    measured_caked_deg: Point
    measured_coordinate_route: CoordinateRouteEvidence
    predicted_native_px: Point
    predicted_detector_display_px: Point
    predicted_caked_deg: Point
    predicted_coordinate_route: CoordinateRouteEvidence
    delta_two_theta_deg: float
    wrapped_delta_phi_deg: float
    prediction_source: str
    prediction_is_dynamic: bool
    prediction_cache_reused: bool
    prediction_candidate_count: int = 1
    prediction_selection_rule: str = "unique_fresh_branch_row"
    prediction_selection_evidence: BranchSelectionEvidence | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveEvaluation:
    """One deterministic four-component caked objective evaluation."""

    problem_digest: str
    trial_parameters: tuple[tuple[str, float], ...]
    predictions: tuple[TrialPrediction, TrialPrediction]
    rows: tuple[BranchObjectiveEvaluation, BranchObjectiveEvaluation]
    residuals_deg: tuple[float, float, float, float]
    row_identity_hash: str
    prediction_hash: str
    residual_hash: str
    evaluation_digest: str
    units: str = RESIDUAL_UNITS

    @property
    def residual_vector(self) -> tuple[float, float, float, float]:
        return self.residuals_deg


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return 0.0 if value == 0.0 else float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _digest(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_point(value: object) -> Point | None:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) < 2
    ):
        return None
    first = _finite_float(value[0])
    second = _finite_float(value[1])
    if first is None or second is None:
        return None
    return first, second


def _field_point(row: Mapping[str, object], x_key: str, y_key: str) -> Point | None:
    first = _finite_float(row.get(x_key))
    second = _finite_float(row.get(y_key))
    if first is None or second is None:
        return None
    return first, second


def _point_candidates(
    row: Mapping[str, object],
    *,
    tuple_keys: Sequence[str] = (),
    field_pairs: Sequence[tuple[str, str]] = (),
) -> list[tuple[Point, str]]:
    candidates: list[tuple[Point, str]] = []
    for key in tuple_keys:
        point = _finite_point(row.get(key))
        if point is not None:
            candidates.append((point, str(key)))
    for x_key, y_key in field_pairs:
        point = _field_point(row, x_key, y_key)
        if point is not None:
            candidates.append((point, f"{x_key}/{y_key}"))
    return candidates


def _choose_consistent_point(
    candidates: Sequence[tuple[Point, str]],
    *,
    reason: str,
    branch: int,
    tolerance: float = 1.0e-6,
) -> tuple[Point, str] | None:
    if not candidates:
        return None
    point, source = candidates[0]
    inconsistent = [
        (candidate_source, candidate)
        for candidate, candidate_source in candidates[1:]
        if math.hypot(candidate[0] - point[0], candidate[1] - point[1])
        > float(tolerance)
    ]
    if inconsistent:
        raise CakedGeometryObjectiveError(
            reason,
            branch=branch,
            selected_source=source,
            selected_point=point,
            conflicting_candidates=inconsistent[:8],
        )
    return point, source


_DETECTOR_NATIVE_FRAME_TOKENS = frozenset(
    {"detector_native", "detector_native_px", "native_detector", "background_detector"}
)
_DETECTOR_DISPLAY_FRAME_TOKENS = frozenset(
    {"detector", "detector_display", "detector_display_px", "display_detector", "fit_detector"}
)
_CAKED_FRAME_TOKENS = frozenset(
    {"caked", "caked_deg", "caked_2theta_phi", "caked_display", "two_theta_phi"}
)
_FORBIDDEN_DYNAMIC_SOURCE_TOKENS = (
    "saved",
    "ghost",
    "cache",
    "reuse",
    "fallback",
)


def _forbidden_dynamic_source_token(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    return next(
        (token for token in _FORBIDDEN_DYNAMIC_SOURCE_TOKENS if token in normalized),
        None,
    )


def _transform_point(
    transform: Callable[[float, float], object],
    point: Point,
    *,
    transform_name: str,
    role: str,
    branch: int,
) -> Point:
    try:
        result = transform(float(point[0]), float(point[1]))
    except Exception as exc:
        raise CakedGeometryObjectiveError(
            "coordinate_transform_failed",
            transform=transform_name,
            role=role,
            branch=branch,
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
    transformed = _finite_point(result)
    if transformed is None:
        raise CakedGeometryObjectiveError(
            "coordinate_transform_failed",
            transform=transform_name,
            role=role,
            branch=branch,
            result=result,
        )
    return transformed


def _point_error(first: Point, second: Point) -> float:
    return float(math.hypot(first[0] - second[0], first[1] - second[1]))


def _build_coordinate_route(
    *,
    role: str,
    authority: str,
    branch: int,
    detector_display_px: Point | None,
    detector_display_source: str | None,
    detector_native_px: Point | None,
    detector_native_source: str | None,
    frozen_display_to_native: DisplayToNativeTransform,
    frozen_native_to_display: NativeToDisplayTransform,
    frozen_native_to_caked: NativeToCakedProjector,
    display_to_native_source: str,
    native_to_display_source: str,
    native_to_caked_source: str,
    authoritative_caked_deg: Point | None = None,
    roundtrip_tolerance_px: float = 1.0e-6,
) -> CoordinateRouteEvidence:
    """Build and verify both detector directions before entering frozen caked space."""

    display_source = str(detector_display_source or "").strip()
    native_source = str(detector_native_source or "").strip()
    tolerance = _finite_float(roundtrip_tolerance_px)
    if tolerance is None or tolerance < 0.0:
        raise CakedGeometryObjectiveError(
            "invalid_coordinate_roundtrip_tolerance", value=roundtrip_tolerance_px
        )
    if detector_display_px is None and detector_native_px is None:
        raise CakedGeometryObjectiveError(
            "coordinate_same_frame_audit_unavailable", role=role, branch=branch
        )
    if detector_native_px is None:
        if detector_display_px is None:
            raise AssertionError("unreachable")
        detector_native_px = _transform_point(
            frozen_display_to_native,
            detector_display_px,
            transform_name="display_to_native",
            role=role,
            branch=branch,
        )
        native_source = f"{display_to_native_source}({display_source or 'detector_display_px'})"
    if detector_display_px is None:
        detector_display_px = _transform_point(
            frozen_native_to_display,
            detector_native_px,
            transform_name="native_to_display",
            role=role,
            branch=branch,
        )
        display_source = f"{native_to_display_source}({native_source or 'detector_native_px'})"
    if not display_source or not native_source:
        raise CakedGeometryObjectiveError(
            "coordinate_conversion_provenance_missing",
            role=role,
            branch=branch,
            display_source=display_source,
            native_source=native_source,
        )

    display_to_native_px = _transform_point(
        frozen_display_to_native,
        detector_display_px,
        transform_name="display_to_native",
        role=role,
        branch=branch,
    )
    display_to_native_to_display_px = _transform_point(
        frozen_native_to_display,
        display_to_native_px,
        transform_name="display_to_native_to_display",
        role=role,
        branch=branch,
    )
    native_to_display_px = _transform_point(
        frozen_native_to_display,
        detector_native_px,
        transform_name="native_to_display",
        role=role,
        branch=branch,
    )
    native_to_display_to_native_px = _transform_point(
        frozen_display_to_native,
        native_to_display_px,
        transform_name="native_to_display_to_native",
        role=role,
        branch=branch,
    )

    roundtrip_display_error = _point_error(
        display_to_native_to_display_px, detector_display_px
    )
    roundtrip_native_error = _point_error(
        native_to_display_to_native_px, detector_native_px
    )
    if max(roundtrip_display_error, roundtrip_native_error) > tolerance:
        raise CakedGeometryObjectiveError(
            "coordinate_roundtrip_failed",
            role=role,
            branch=branch,
            tolerance_px=tolerance,
            roundtrip_display_error_px=roundtrip_display_error,
            roundtrip_native_error_px=roundtrip_native_error,
        )

    same_frame_display_error = _point_error(native_to_display_px, detector_display_px)
    same_frame_native_error = _point_error(display_to_native_px, detector_native_px)
    if max(same_frame_display_error, same_frame_native_error) > tolerance:
        raise CakedGeometryObjectiveError(
            "coordinate_same_frame_audit_failed",
            role=role,
            branch=branch,
            tolerance_px=tolerance,
            detector_display_px=detector_display_px,
            detector_display_source=display_source,
            detector_native_px=detector_native_px,
            detector_native_source=native_source,
            same_frame_display_error_px=same_frame_display_error,
            same_frame_native_error_px=same_frame_native_error,
        )

    native_to_caked_audit_deg = _transform_point(
        frozen_native_to_caked,
        detector_native_px,
        transform_name="native_to_caked",
        role=role,
        branch=branch,
    )
    frozen_caked_deg = (
        authoritative_caked_deg
        if authoritative_caked_deg is not None
        else native_to_caked_audit_deg
    )
    return CoordinateRouteEvidence(
        role=str(role),
        authority=str(authority),
        detector_display_px=detector_display_px,
        detector_display_frame="detector_display",
        detector_display_source=display_source,
        detector_native_px=detector_native_px,
        detector_native_frame="detector_native",
        detector_native_source=native_source,
        display_to_native_source=str(display_to_native_source),
        native_to_display_source=str(native_to_display_source),
        native_to_caked_source=str(native_to_caked_source),
        frozen_caked_deg=frozen_caked_deg,
        native_to_caked_audit_deg=native_to_caked_audit_deg,
        roundtrip_tolerance_px=float(tolerance),
        roundtrip_status="passed",
        roundtrip_display_error_px=roundtrip_display_error,
        roundtrip_native_error_px=roundtrip_native_error,
        same_frame_display_error_px=same_frame_display_error,
        same_frame_native_error_px=same_frame_native_error,
        display_to_native_px=display_to_native_px,
        display_to_native_to_display_px=display_to_native_to_display_px,
        native_to_display_px=native_to_display_px,
        native_to_display_to_native_px=native_to_display_to_native_px,
    )


def _exact_int(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _nonnegative_int(value: object) -> int | None:
    index = _exact_int(value)
    if index is None:
        return None
    return index if index >= 0 else None


def _hkl(value: object) -> tuple[int, int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 3
    ):
        return None
    components = tuple(_exact_int(component) for component in value)
    if any(component is None for component in components):
        return None
    return int(components[0]), int(components[1]), int(components[2])


def _hkl_equivalence_key(value: tuple[int, int, int]) -> tuple[int, int]:
    h_val, k_val, l_val = value
    return h_val * h_val + h_val * k_val + k_val * k_val, l_val


def _q_group_key(value: object) -> QGroupKey | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        return None
    return tuple(value)


def _branch_index(value: object) -> int | None:
    branch = _nonnegative_int(value)
    return branch if branch in TARGET_BRANCHES else None


def _fixed_measured_caked_candidates(
    row: Mapping[str, object],
) -> list[tuple[Point, str]]:
    # Intentionally exclude every simulated/predicted caked alias.
    return _point_candidates(
        row,
        tuple_keys=(
            "manual_target_caked_deg",
            "fixed_measured_caked_deg",
            "background_caked_deg",
            "chosen_measured_caked_point",
        ),
        field_pairs=(
            ("background_two_theta_deg", "background_phi_deg"),
            ("manual_target_caked_x", "manual_target_caked_y"),
            ("caked_x", "caked_y"),
            ("measured_caked_two_theta", "measured_caked_phi"),
        ),
    )


def _require_applicable_coordinate_frame(
    row: Mapping[str, object],
    *,
    frame_keys: Sequence[str],
    allowed_frames: frozenset[str],
    invalid_reason: str,
    unproven_reason: str,
    branch: int | None,
    point_key: str,
) -> None:
    applicable = [key for key in frame_keys if row.get(key) is not None]
    if not applicable:
        raise CakedGeometryObjectiveError(
            unproven_reason,
            branch=branch,
            point_key=point_key,
            applicable_frame_keys=tuple(frame_keys),
        )
    for frame_key in applicable:
        frame = _normalized_frame_token(row.get(frame_key))
        if frame not in allowed_frames:
            raise CakedGeometryObjectiveError(
                invalid_reason if frame else unproven_reason,
                branch=branch,
                point_key=point_key,
                frame_key=frame_key,
                frame=frame,
            )


def _measured_native_candidates(
    row: Mapping[str, object],
    *,
    branch: int,
) -> list[tuple[Point, str]]:
    # Each native point must carry its own frame or an explicitly generic one.
    generic_frames = ("detector_native_frame", "native_pixel_frame")
    candidates: list[tuple[Point, str]] = []

    def add_tuple(point_key: str, *point_frames: str) -> None:
        point = _finite_point(row.get(point_key))
        if point is None:
            return
        _require_applicable_coordinate_frame(
            row,
            frame_keys=(*point_frames, *generic_frames),
            allowed_frames=_DETECTOR_NATIVE_FRAME_TOKENS,
            invalid_reason="measured_detector_native_frame_invalid",
            unproven_reason="measured_detector_native_frame_unproven",
            branch=branch,
            point_key=point_key,
        )
        candidates.append((point, point_key))

    def add_fields(
        x_key: str,
        y_key: str,
        *point_frames: str,
    ) -> None:
        point = _field_point(row, x_key, y_key)
        if point is None:
            return
        point_key = f"{x_key}/{y_key}"
        _require_applicable_coordinate_frame(
            row,
            frame_keys=(*point_frames, *generic_frames),
            allowed_frames=_DETECTOR_NATIVE_FRAME_TOKENS,
            invalid_reason="measured_detector_native_frame_invalid",
            unproven_reason="measured_detector_native_frame_unproven",
            branch=branch,
            point_key=point_key,
        )
        candidates.append((point, point_key))

    add_tuple(
        "manual_target_detector_native_px",
        "manual_target_detector_native_frame",
    )
    add_tuple("measured_detector_native_px", "measured_detector_native_frame")
    add_tuple("background_detector_native_px", "background_detector_native_frame")
    add_tuple("geometry_detector_native_px", "geometry_detector_native_frame")
    add_fields("measured_native_x", "measured_native_y", "measured_detector_native_frame")
    add_fields("detector_native_x", "detector_native_y")
    add_fields("native_col", "native_row")
    if _normalized_frame_token(row.get("background_detector_input_frame")) in (
        _DETECTOR_NATIVE_FRAME_TOKENS
    ):
        add_fields(
            "background_detector_x",
            "background_detector_y",
            "background_detector_input_frame",
        )
    if _normalized_frame_token(row.get("detector_input_frame")) in (
        _DETECTOR_NATIVE_FRAME_TOKENS
    ):
        add_fields("detector_x", "detector_y", "detector_input_frame")
    return candidates


def _measured_display_candidates(
    row: Mapping[str, object],
    *,
    branch: int,
) -> list[tuple[Point, str]]:
    generic_frames = ("detector_display_frame", "display_pixel_frame")
    candidates: list[tuple[Point, str]] = []

    def add_tuple(point_key: str, *point_frames: str) -> None:
        point = _finite_point(row.get(point_key))
        if point is None:
            return
        _require_applicable_coordinate_frame(
            row,
            frame_keys=(*point_frames, *generic_frames),
            allowed_frames=_DETECTOR_DISPLAY_FRAME_TOKENS,
            invalid_reason="measured_detector_display_frame_invalid",
            unproven_reason="measured_detector_display_frame_unproven",
            branch=branch,
            point_key=point_key,
        )
        candidates.append((point, point_key))

    def add_fields(x_key: str, y_key: str, *point_frames: str) -> None:
        point = _field_point(row, x_key, y_key)
        if point is None:
            return
        point_key = f"{x_key}/{y_key}"
        _require_applicable_coordinate_frame(
            row,
            frame_keys=(*point_frames, *generic_frames),
            allowed_frames=_DETECTOR_DISPLAY_FRAME_TOKENS,
            invalid_reason="measured_detector_display_frame_invalid",
            unproven_reason="measured_detector_display_frame_unproven",
            branch=branch,
            point_key=point_key,
        )
        candidates.append((point, point_key))

    add_tuple(
        "manual_target_detector_display_px",
        "manual_target_detector_display_frame",
    )
    add_tuple("measured_detector_display_px", "measured_detector_display_frame")
    add_tuple("background_detector_display_px", "background_detector_display_frame")
    add_fields("obs_detector_x", "obs_detector_y")
    add_fields("detector_display_x", "detector_display_y")
    add_fields("fit_detector_x", "fit_detector_y")
    add_fields("display_col", "display_row")
    add_fields("x", "y")
    if _normalized_frame_token(row.get("detector_input_frame")) in (
        _DETECTOR_DISPLAY_FRAME_TOKENS
    ):
        add_fields("detector_x", "detector_y", "detector_input_frame")
    return candidates


def _measurement_origin_and_frame(
    row: Mapping[str, object], *, branch: int
) -> tuple[str, str]:
    origin = _normalized_frame_token(row.get("manual_background_input_origin"))
    frame = _normalized_frame_token(row.get("manual_background_input_frame"))
    if not origin:
        raise CakedGeometryObjectiveError(
            "measurement_authority_missing", branch=branch
        )
    if origin in _CAKED_FRAME_TOKENS:
        normalized_origin = "caked"
    elif origin in _DETECTOR_NATIVE_FRAME_TOKENS or origin in _DETECTOR_DISPLAY_FRAME_TOKENS:
        normalized_origin = "detector"
    else:
        raise CakedGeometryObjectiveError(
            "measurement_authority_ambiguous", branch=branch, origin=origin
        )
    if not frame:
        raise CakedGeometryObjectiveError(
            "measurement_input_frame_missing", branch=branch, origin=normalized_origin
        )
    if normalized_origin == "caked" and frame not in _CAKED_FRAME_TOKENS:
        raise CakedGeometryObjectiveError(
            "measurement_authority_inconsistent",
            branch=branch,
            origin=normalized_origin,
            frame=frame,
        )
    if normalized_origin == "detector" and frame in _CAKED_FRAME_TOKENS:
        raise CakedGeometryObjectiveError(
            "measurement_authority_inconsistent",
            branch=branch,
            origin=normalized_origin,
            frame=frame,
        )
    if normalized_origin == "detector" and frame not in (
        _DETECTOR_NATIVE_FRAME_TOKENS | _DETECTOR_DISPLAY_FRAME_TOKENS
    ):
        raise CakedGeometryObjectiveError(
            "measurement_authority_ambiguous",
            branch=branch,
            origin=normalized_origin,
            frame=frame,
        )
    if origin in _DETECTOR_NATIVE_FRAME_TOKENS and frame not in (
        _DETECTOR_NATIVE_FRAME_TOKENS
    ):
        raise CakedGeometryObjectiveError(
            "measurement_authority_inconsistent",
            branch=branch,
            origin=origin,
            frame=frame,
        )
    if (
        origin in (_DETECTOR_DISPLAY_FRAME_TOKENS - {"detector"})
        and frame not in _DETECTOR_DISPLAY_FRAME_TOKENS
    ):
        raise CakedGeometryObjectiveError(
            "measurement_authority_inconsistent",
            branch=branch,
            origin=origin,
            frame=frame,
        )
    return normalized_origin, frame


def _target_identity_payload(target: LockedBranchTarget) -> dict[str, object]:
    return {
        "background_index": target.background_index,
        "q_group_key": target.q_group_key,
        "hkl_equivalence_key": target.hkl_equivalence_key,
        "branch_index": target.branch_index,
    }


def coordinate_route_to_jsonable(
    route: CoordinateRouteEvidence,
) -> dict[str, object]:
    return {
        "role": route.role,
        "authority": route.authority,
        "detector_display_px": _jsonable(route.detector_display_px),
        "detector_display_frame": route.detector_display_frame,
        "detector_display_source": route.detector_display_source,
        "detector_native_px": _jsonable(route.detector_native_px),
        "detector_native_frame": route.detector_native_frame,
        "detector_native_source": route.detector_native_source,
        "conversion_provenance": {
            "display_to_native": route.display_to_native_source,
            "native_to_display": route.native_to_display_source,
            "native_to_caked": route.native_to_caked_source,
        },
        "frozen_caked_deg": _jsonable(route.frozen_caked_deg),
        "native_to_caked_audit_deg": _jsonable(route.native_to_caked_audit_deg),
        "roundtrip_tolerance_px": route.roundtrip_tolerance_px,
        "roundtrip_status": route.roundtrip_status,
        "roundtrip_display_error_px": route.roundtrip_display_error_px,
        "roundtrip_native_error_px": route.roundtrip_native_error_px,
        "same_frame_display_error_px": route.same_frame_display_error_px,
        "same_frame_native_error_px": route.same_frame_native_error_px,
        "display_to_native_px": _jsonable(route.display_to_native_px),
        "display_to_native_to_display_px": _jsonable(
            route.display_to_native_to_display_px
        ),
        "native_to_display_px": _jsonable(route.native_to_display_px),
        "native_to_display_to_native_px": _jsonable(
            route.native_to_display_to_native_px
        ),
    }




def _target_contract_payload(target: LockedBranchTarget) -> dict[str, object]:
    return {
        **_target_identity_payload(target),
        "measurement_origin": target.measurement_origin,
        "measured_detector_display_px": target.measured_detector_display_px,
        "measured_caked_deg": target.measured_caked_deg,
        "measured_native_px": target.measured_native_px,
        "coordinate_route": coordinate_route_to_jsonable(target.coordinate_route),
    }


_TRIAL_IDENTITY_INDEX_KEYS = frozenset(
    {
        "background_index",
        "branch_index",
        "source_branch_index",
    }
)


def _without_provenance_indices(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_provenance_indices(item)
            for key, item in value.items()
            if not (
                str(key).endswith("_index")
                and str(key) not in _TRIAL_IDENTITY_INDEX_KEYS
            )
        }
    if isinstance(value, tuple):
        return tuple(_without_provenance_indices(item) for item in value)
    if isinstance(value, list):
        return [_without_provenance_indices(item) for item in value]
    return value


def caked_geometry_equivalence_trial_requirements(
    rows: Sequence[Mapping[str, object]],
    *,
    background_index: int | None = None,
) -> list[dict[str, object]]:
    """Copy trial requirements with provenance indices removed from identity.

    Reflection/table/row/sample indices describe how a saved row was produced;
    they must not select the current simulation reflection.  Background,
    Q-group/HKL equivalence, and physical branch remain authoritative.
    """

    requirements: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        stripped = _without_provenance_indices(dict(row))
        if not isinstance(stripped, Mapping):
            continue
        requirement = dict(stripped)
        if background_index is not None:
            requirement["background_index"] = int(background_index)
        requirement[CAKED_GEOMETRY_EQUIVALENCE_FRESH_OBJECTIVE_MARKER] = True
        requirements.append(requirement)
    return requirements


def build_caked_geometry_problem(
    rows: Sequence[Mapping[str, object]],
    *,
    frozen_native_to_caked: NativeToCakedProjector,
    frozen_display_to_native: DisplayToNativeTransform,
    frozen_native_to_display: NativeToDisplayTransform,
    display_to_native_source: str,
    native_to_display_source: str,
    projector_signature: object,
    projector_kind: str = "frozen_exact_caked",
    roundtrip_tolerance_px: float = 1.0e-6,
    target_hkl_equivalent: tuple[int, int, int] = (1, 0, 10),
    target_q_group_key: QGroupKey = ("q_group", "primary", 1, 10),
    active_parameter_names: Sequence[str] = ACTIVE_PARAMETER_NAMES,
) -> CakedGeometryProblem:
    """Lock the exact two-row bg0 problem and freeze its projector contract."""

    if not callable(frozen_native_to_caked):
        raise CakedGeometryObjectiveError("missing_frozen_caked_projector")
    if not callable(frozen_display_to_native):
        raise CakedGeometryObjectiveError("missing_frozen_display_to_native_transform")
    if not callable(frozen_native_to_display):
        raise CakedGeometryObjectiveError("missing_frozen_native_to_display_transform")
    normalized_display_to_native_source = str(display_to_native_source or "").strip()
    normalized_native_to_display_source = str(native_to_display_source or "").strip()
    if not normalized_display_to_native_source or normalized_display_to_native_source.lower() in {
        "unknown",
        "unavailable",
    }:
        raise CakedGeometryObjectiveError("missing_display_to_native_provenance")
    if not normalized_native_to_display_source or normalized_native_to_display_source.lower() in {
        "unknown",
        "unavailable",
    }:
        raise CakedGeometryObjectiveError("missing_native_to_display_provenance")
    normalized_roundtrip_tolerance = _finite_float(roundtrip_tolerance_px)
    if normalized_roundtrip_tolerance is None or normalized_roundtrip_tolerance < 0.0:
        raise CakedGeometryObjectiveError(
            "invalid_coordinate_roundtrip_tolerance", value=roundtrip_tolerance_px
        )
    if projector_signature is None:
        raise CakedGeometryObjectiveError("missing_projector_signature")
    normalized_projector_kind = str(projector_kind or "").strip()
    if not normalized_projector_kind:
        raise CakedGeometryObjectiveError("missing_projector_kind")
    normalized_active_names = tuple(str(name) for name in active_parameter_names)
    if normalized_active_names not in ACTIVE_PARAMETER_RUNGS:
        raise CakedGeometryObjectiveError(
            "unsupported_caked_geometry_parameter_rung",
            names=list(normalized_active_names),
        )

    normalized_target_hkl = _hkl(target_hkl_equivalent)
    normalized_target_q_group = _q_group_key(target_q_group_key)
    if normalized_target_hkl is None:
        raise CakedGeometryObjectiveError("invalid_target_hkl_equivalent")
    if normalized_target_q_group is None:
        raise CakedGeometryObjectiveError("invalid_target_q_group_key")
    target_equivalence = _hkl_equivalence_key(normalized_target_hkl)

    matching_by_branch: dict[int, list[Mapping[str, object]]] = {0: [], 1: []}
    invalid_branch_rows = 0
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        raw_background_index = raw_row.get("background_index")
        background_index = _nonnegative_int(raw_background_index)
        q_group = _q_group_key(raw_row.get("q_group_key"))
        raw_hkl = raw_row.get("hkl")
        row_hkl = _hkl(raw_hkl)
        if q_group == normalized_target_q_group:
            if background_index is None:
                raise CakedGeometryObjectiveError(
                    "invalid_background_index",
                    value=raw_background_index,
                )
            if background_index == TARGET_BACKGROUND_INDEX and row_hkl is None:
                raise CakedGeometryObjectiveError(
                    "invalid_locked_hkl",
                    value=raw_hkl,
                )
        if (
            background_index != TARGET_BACKGROUND_INDEX
            or q_group != normalized_target_q_group
            or row_hkl is None
            or _hkl_equivalence_key(row_hkl) != target_equivalence
        ):
            continue
        branch = _branch_index(raw_row.get("source_branch_index"))
        if branch is None:
            invalid_branch_rows += 1
            continue
        matching_by_branch[branch].append(raw_row)

    duplicate_branches = [
        branch for branch, branch_rows in matching_by_branch.items() if len(branch_rows) > 1
    ]
    if invalid_branch_rows:
        raise CakedGeometryObjectiveError(
            "invalid_branch",
            invalid_branch_rows=invalid_branch_rows,
        )
    if duplicate_branches:
        raise CakedGeometryObjectiveError(
            "duplicate_branch", branches=duplicate_branches
        )
    missing_branches = [
        branch for branch, branch_rows in matching_by_branch.items() if not branch_rows
    ]
    if missing_branches:
        raise CakedGeometryObjectiveError(
            "missing_branch",
            branches=missing_branches,
            invalid_branch_rows=invalid_branch_rows,
        )

    targets: list[LockedBranchTarget] = []
    for branch in TARGET_BRANCHES:
        row = matching_by_branch[branch][0]
        row_hkl = _hkl(row.get("hkl"))
        if row_hkl is None:
            raise CakedGeometryObjectiveError("missing_locked_hkl", branch=branch)
        measurement_origin, measurement_frame = _measurement_origin_and_frame(
            row, branch=branch
        )
        native_record = _choose_consistent_point(
            _measured_native_candidates(row, branch=branch),
            reason="measured_detector_native_candidates_inconsistent",
            branch=branch,
        )
        display_record = _choose_consistent_point(
            _measured_display_candidates(row, branch=branch),
            reason="measured_detector_display_candidates_inconsistent",
            branch=branch,
        )
        caked_candidates = _fixed_measured_caked_candidates(row)
        if measurement_origin == "caked":
            caked_record = _choose_consistent_point(
                caked_candidates,
                reason="measured_caked_candidates_inconsistent",
                branch=branch,
            )
            if caked_record is None:
                raise CakedGeometryObjectiveError(
                    "measured_caked_authority_point_missing", branch=branch
                )
        else:
            # Detector-origin saved caked fields are deliberately non-authoritative.
            caked_record = _choose_consistent_point(
                caked_candidates,
                reason="saved_caked_audit_candidates_inconsistent",
                branch=branch,
            )

        if measurement_origin == "detector":
            if native_record is None:
                raise CakedGeometryObjectiveError(
                    "measured_detector_native_point_missing", branch=branch
                )
            if measurement_frame in _DETECTOR_DISPLAY_FRAME_TOKENS and display_record is None:
                raise CakedGeometryObjectiveError(
                    "measured_detector_display_point_missing", branch=branch
                )
        if native_record is None and display_record is None:
            raise CakedGeometryObjectiveError(
                "coordinate_same_frame_audit_unavailable",
                role="measured",
                branch=branch,
            )
        route = _build_coordinate_route(
            role="measured",
            authority=("detector_native" if measurement_origin == "detector" else "caked_deg"),
            branch=branch,
            detector_display_px=(display_record[0] if display_record is not None else None),
            detector_display_source=(display_record[1] if display_record is not None else None),
            detector_native_px=(native_record[0] if native_record is not None else None),
            detector_native_source=(native_record[1] if native_record is not None else None),
            frozen_display_to_native=frozen_display_to_native,
            frozen_native_to_display=frozen_native_to_display,
            frozen_native_to_caked=frozen_native_to_caked,
            display_to_native_source=normalized_display_to_native_source,
            native_to_display_source=normalized_native_to_display_source,
            native_to_caked_source=f"frozen_native_to_caked:{normalized_projector_kind}",
            authoritative_caked_deg=(
                caked_record[0]
                if measurement_origin == "caked" and caked_record is not None
                else None
            ),
            roundtrip_tolerance_px=float(normalized_roundtrip_tolerance),
        )
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError("locked caked-geometry target lacks pair_id")
        targets.append(
            LockedBranchTarget(
                background_index=TARGET_BACKGROUND_INDEX,
                q_group_key=normalized_target_q_group,
                hkl=row_hkl,
                hkl_equivalence_key=target_equivalence,
                branch_index=branch,
                measured_caked_deg=route.frozen_caked_deg,
                measured_native_px=route.detector_native_px,
                measured_detector_display_px=route.detector_display_px,
                measurement_origin=measurement_origin,
                saved_caked_audit_deg=(
                    caked_record[0] if caked_record is not None else None
                ),
                coordinate_route=route,
                pair_id=pair_id,
                source_table_index=_nonnegative_int(row.get("source_table_index")),
                source_row_index=_nonnegative_int(row.get("source_row_index")),
                source_reflection_index=_nonnegative_int(
                    row.get("source_reflection_index")
                ),
            )
        )

    signature_json = json.dumps(
        _jsonable(projector_signature),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    projector_digest = _digest({"projector_signature": signature_json})
    frame_digest = _digest(
        {
            "background_index": TARGET_BACKGROUND_INDEX,
            "projector_kind": normalized_projector_kind,
            "projector_digest": projector_digest,
            "display_to_native_source": normalized_display_to_native_source,
            "native_to_display_source": normalized_native_to_display_source,
            "roundtrip_tolerance_px": normalized_roundtrip_tolerance,
            "units": RESIDUAL_UNITS,
        }
    )
    frame = DatasetFitSpaceFrame(
        background_index=TARGET_BACKGROUND_INDEX,
        native_to_caked=frozen_native_to_caked,
        display_to_native=frozen_display_to_native,
        native_to_display=frozen_native_to_display,
        display_to_native_source=normalized_display_to_native_source,
        native_to_display_source=normalized_native_to_display_source,
        projector_signature_json=signature_json,
        projector_digest=projector_digest,
        frame_digest=frame_digest,
        roundtrip_tolerance_px=float(normalized_roundtrip_tolerance),
        projector_kind=normalized_projector_kind,
    )
    target_tuple = (targets[0], targets[1])
    problem_digest = _digest(
        {
            "schema": "caked_geometry_problem_v2",
            "active_parameter_names": normalized_active_names,
            "residual_units": RESIDUAL_UNITS,
            "frame_digest": frame_digest,
            "target_hkl_equivalent": normalized_target_hkl,
            "hkl_equivalence_key": target_equivalence,
            "target_q_group_key": normalized_target_q_group,
            "targets": [_target_contract_payload(target) for target in target_tuple],
        }
    )
    return CakedGeometryProblem(
        frame=frame,
        targets=target_tuple,
        target_hkl_equivalent=normalized_target_hkl,
        hkl_equivalence_key=target_equivalence,
        target_q_group_key=normalized_target_q_group,
        problem_digest=problem_digest,
        active_parameter_names=normalized_active_names,
    )


def wrap_phi_delta_deg(delta: float) -> float:
    """Wrap one angular residual to ``[-180, 180)`` degrees."""

    numeric = _finite_float(delta)
    if numeric is None:
        raise CakedGeometryObjectiveError("nonfinite_phi_delta", value=delta)
    wrapped = (numeric + 180.0) % 360.0 - 180.0
    return 0.0 if wrapped == 0.0 else float(wrapped)


def _prediction_identity_payload(prediction: TrialPrediction) -> dict[str, object]:
    return {
        "background_index": prediction.background_index,
        "q_group_key": prediction.q_group_key,
        "hkl": prediction.hkl,
        "hkl_equivalence_key": (
            prediction.hkl_equivalence_key
            if prediction.hkl_equivalence_key is not None
            else _hkl_equivalence_key(prediction.hkl)
        ),
        "branch_index": prediction.branch_index,
    }


def branch_selection_evidence_to_jsonable(
    evidence: BranchSelectionEvidence | None,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    return {
        "selection_rule": evidence.selection_rule,
        "selection_metric": evidence.selection_metric,
        "candidate_count": int(evidence.candidate_count),
        "candidate_keys": _jsonable(evidence.candidate_keys),
        "candidate_weights": _jsonable(evidence.candidate_weights),
        "candidate_identity_digest": evidence.candidate_identity_digest,
        "candidate_weight_digest": evidence.candidate_weight_digest,
        "selected_candidate_key": _jsonable(evidence.selected_candidate_key),
        "selected_candidate_digest": evidence.selected_candidate_digest,
        "selected_weight": evidence.selected_weight,
        "runner_up_weight": evidence.runner_up_weight,
        "margin_abs": evidence.margin_abs,
        "margin_relative": evidence.margin_relative,
        "tie_tolerance": evidence.tie_tolerance,
    }


def _validated_trial_parameters(
    trial_parameters: Mapping[str, object],
    *,
    active_parameter_names: Sequence[str],
) -> tuple[dict[str, float], tuple[tuple[str, float], ...]]:
    expected_names = tuple(str(name) for name in active_parameter_names)
    if expected_names not in ACTIVE_PARAMETER_RUNGS:
        raise CakedGeometryObjectiveError(
            "unsupported_caked_geometry_parameter_rung",
            names=list(expected_names),
        )
    if set(str(key) for key in trial_parameters) != set(expected_names):
        raise CakedGeometryObjectiveError(
            "active_parameters_do_not_match_problem_rung",
            names=sorted(str(key) for key in trial_parameters),
            expected=list(expected_names),
        )
    values: dict[str, float] = {}
    for name in expected_names:
        numeric = _finite_float(trial_parameters.get(name))
        if numeric is None:
            raise CakedGeometryObjectiveError(
                "nonfinite_active_parameter", parameter=name
            )
        values[name] = numeric
    ordered = tuple((name, values[name]) for name in expected_names)
    return values, ordered


def _validate_prediction_identity(
    prediction: TrialPrediction,
    target: LockedBranchTarget,
) -> None:
    prediction_background = _nonnegative_int(prediction.background_index)
    if prediction_background != int(target.background_index):
        raise CakedGeometryObjectiveError(
            "prediction_identity_mismatch", field="background_index"
        )
    if _q_group_key(prediction.q_group_key) != target.q_group_key:
        raise CakedGeometryObjectiveError(
            "prediction_identity_mismatch", field="q_group_key"
        )
    normalized_hkl = _hkl(prediction.hkl)
    if normalized_hkl is None:
        raise CakedGeometryObjectiveError(
            "prediction_identity_mismatch", field="hkl"
        )
    prediction_equivalence = _hkl_equivalence_key(normalized_hkl)
    if prediction.hkl_equivalence_key is not None:
        raw_equivalence = prediction.hkl_equivalence_key
        if (
            not isinstance(raw_equivalence, Sequence)
            or isinstance(raw_equivalence, (str, bytes, bytearray))
            or len(raw_equivalence) != 2
        ):
            raise CakedGeometryObjectiveError(
                "prediction_identity_mismatch", field="hkl_equivalence_provenance"
            )
        normalized_equivalence = tuple(
            _exact_int(component) for component in raw_equivalence
        )
        if (
            any(component is None for component in normalized_equivalence)
            or normalized_equivalence != prediction_equivalence
        ):
            raise CakedGeometryObjectiveError(
                "prediction_identity_mismatch", field="hkl_equivalence_provenance"
            )
    if prediction_equivalence != target.hkl_equivalence_key:
        raise CakedGeometryObjectiveError(
            "prediction_identity_mismatch", field="hkl_equivalence"
        )
    if _branch_index(prediction.branch_index) != int(target.branch_index):
        raise CakedGeometryObjectiveError(
            "prediction_identity_mismatch", field="branch_index"
        )


def evaluate_caked_geometry_objective(
    problem: CakedGeometryProblem,
    trial_parameters: Mapping[str, object],
    *,
    predict_native: Callable[
        [Mapping[str, float], tuple[LockedBranchTarget, LockedBranchTarget]],
        Sequence[TrialPrediction],
    ],
) -> ObjectiveEvaluation:
    """Regenerate native predictions and assemble the fixed four-vector."""

    if not isinstance(problem, CakedGeometryProblem):
        raise CakedGeometryObjectiveError("invalid_caked_geometry_problem")
    if not callable(predict_native):
        raise CakedGeometryObjectiveError("missing_native_prediction_provider")
    params, ordered_params = _validated_trial_parameters(
        trial_parameters,
        active_parameter_names=problem.active_parameter_names,
    )
    try:
        raw_predictions = tuple(predict_native(dict(params), problem.targets))
    except CakedGeometryObjectiveError:
        raise
    except Exception as exc:
        raise CakedGeometryObjectiveError(
            "native_prediction_provider_failed",
            exception_type=type(exc).__name__,
            message=str(exc),
        ) from exc
    if any(not isinstance(item, TrialPrediction) for item in raw_predictions):
        raise CakedGeometryObjectiveError("invalid_trial_prediction_record")

    by_branch: dict[int, list[TrialPrediction]] = {0: [], 1: []}
    invalid_branches: list[object] = []
    for prediction in raw_predictions:
        branch = _branch_index(prediction.branch_index)
        if branch is None:
            invalid_branches.append(prediction.branch_index)
            continue
        by_branch[branch].append(prediction)
    duplicate_branches = [branch for branch, items in by_branch.items() if len(items) > 1]
    if duplicate_branches:
        raise CakedGeometryObjectiveError(
            "duplicate_branch", branches=duplicate_branches
        )
    missing_branches = [branch for branch, items in by_branch.items() if not items]
    if missing_branches:
        raise CakedGeometryObjectiveError(
            "missing_branch", branches=missing_branches, invalid_branches=invalid_branches
        )
    if len(raw_predictions) != 2:
        raise CakedGeometryObjectiveError(
            "unexpected_prediction_count", count=len(raw_predictions)
        )

    projected_predictions: list[TrialPrediction] = []
    rows: list[BranchObjectiveEvaluation] = []
    residuals: list[float] = []
    for branch, target in zip(TARGET_BRANCHES, problem.targets):
        prediction = by_branch[branch][0]
        _validate_prediction_identity(prediction, target)
        if not bool(prediction.is_dynamic):
            raise CakedGeometryObjectiveError(
                "prediction_not_dynamic", branch=branch
            )
        if bool(prediction.cache_reused):
            raise CakedGeometryObjectiveError(
                "prediction_cache_reused", branch=branch
            )
        source = str(prediction.source or "").strip()
        forbidden_source = next(
            (
                (field_name, field_value, token)
                for field_name, field_value in (
                    ("source", source),
                    ("detector_native_source", prediction.detector_native_source),
                    ("detector_display_source", prediction.detector_display_source),
                )
                if (token := _forbidden_dynamic_source_token(field_value)) is not None
            ),
            None,
        )
        if not source or forbidden_source is not None:
            raise CakedGeometryObjectiveError(
                "invalid_dynamic_prediction_source",
                branch=branch,
                source=source,
                offending_field=(
                    forbidden_source[0] if forbidden_source is not None else "source"
                ),
                forbidden_token=(
                    forbidden_source[2] if forbidden_source is not None else None
                ),
            )
        native_point = _finite_point(prediction.native_pixel)
        if native_point is None:
            raise CakedGeometryObjectiveError(
                "nonfinite_native_prediction", branch=branch
            )
        native_frame = _normalized_frame_token(prediction.detector_native_frame)
        if native_frame not in _DETECTOR_NATIVE_FRAME_TOKENS:
            raise CakedGeometryObjectiveError(
                "prediction_native_frame_invalid",
                branch=branch,
                frame=prediction.detector_native_frame,
            )
        display_frame = _normalized_frame_token(prediction.detector_display_frame)
        if display_frame not in _DETECTOR_DISPLAY_FRAME_TOKENS:
            raise CakedGeometryObjectiveError(
                "prediction_display_frame_invalid",
                branch=branch,
                frame=prediction.detector_display_frame,
            )
        display_point = _finite_point(prediction.detector_display_pixel)
        if display_point is None:
            raise CakedGeometryObjectiveError(
                "prediction_detector_display_point_missing", branch=branch
            )
        prediction_route = _build_coordinate_route(
            role="predicted",
            authority="detector_native",
            branch=branch,
            detector_display_px=display_point,
            detector_display_source=str(prediction.detector_display_source or ""),
            detector_native_px=native_point,
            detector_native_source=str(prediction.detector_native_source or ""),
            frozen_display_to_native=problem.frame.display_to_native,
            frozen_native_to_display=problem.frame.native_to_display,
            frozen_native_to_caked=problem.frame.native_to_caked,
            display_to_native_source=problem.frame.display_to_native_source,
            native_to_display_source=problem.frame.native_to_display_source,
            native_to_caked_source=(
                f"frozen_native_to_caked:{problem.frame.projector_kind}"
            ),
            roundtrip_tolerance_px=problem.frame.roundtrip_tolerance_px,
        )
        if (
            prediction.coordinate_route is not None
            and prediction.coordinate_route != prediction_route
        ):
            raise CakedGeometryObjectiveError(
                "prediction_coordinate_route_inconsistent", branch=branch
            )
        projected_caked = prediction_route.frozen_caked_deg
        delta_two_theta = float(projected_caked[0] - target.measured_caked_deg[0])
        delta_phi = wrap_phi_delta_deg(
            projected_caked[1] - target.measured_caked_deg[1]
        )
        projected_predictions.append(
            replace(
                prediction,
                native_pixel=native_point,
                detector_display_pixel=display_point,
                detector_native_frame="detector_native",
                detector_display_frame="detector_display",
                predicted_caked_deg=projected_caked,
                coordinate_route=prediction_route,
            )
        )
        rows.append(
            BranchObjectiveEvaluation(
                background_index=target.background_index,
                q_group_key=target.q_group_key,
                hkl=prediction.hkl,
                hkl_equivalence_key=target.hkl_equivalence_key,
                branch_index=branch,
                pair_id=target.pair_id,
                measurement_origin=target.measurement_origin,
                saved_caked_audit_deg=target.saved_caked_audit_deg,
                measured_native_px=target.measured_native_px,
                measured_detector_display_px=target.measured_detector_display_px,
                measured_caked_deg=target.measured_caked_deg,
                measured_coordinate_route=target.coordinate_route,
                predicted_native_px=native_point,
                predicted_detector_display_px=display_point,
                predicted_caked_deg=projected_caked,
                predicted_coordinate_route=prediction_route,
                delta_two_theta_deg=delta_two_theta,
                wrapped_delta_phi_deg=delta_phi,
                prediction_source=source,
                prediction_is_dynamic=True,
                prediction_cache_reused=False,
                prediction_candidate_count=int(prediction.candidate_count),
                prediction_selection_rule=str(prediction.selection_rule),
                prediction_selection_evidence=prediction.selection_evidence,
            )
        )
        residuals.extend((delta_two_theta, delta_phi))

    if len(residuals) != 4 or not all(math.isfinite(value) for value in residuals):
        raise CakedGeometryObjectiveError(
            "invalid_residual_vector", count=len(residuals)
        )
    prediction_tuple = (projected_predictions[0], projected_predictions[1])
    row_tuple = (rows[0], rows[1])
    residual_tuple = (residuals[0], residuals[1], residuals[2], residuals[3])
    row_identity_hash = _digest(
        [_target_identity_payload(target) for target in problem.targets]
    )
    prediction_hash = _digest(
        [
            {
                **_prediction_identity_payload(prediction),
                "native_pixel": prediction.native_pixel,
                "detector_display_pixel": prediction.detector_display_pixel,
                "predicted_caked_deg": prediction.predicted_caked_deg,
                "coordinate_route": (
                    coordinate_route_to_jsonable(prediction.coordinate_route)
                    if prediction.coordinate_route is not None
                    else None
                ),
                "source": prediction.source,
                "is_dynamic": prediction.is_dynamic,
                "cache_reused": prediction.cache_reused,
                "candidate_count": prediction.candidate_count,
                "selection_rule": prediction.selection_rule,
                "selection_evidence": branch_selection_evidence_to_jsonable(
                    prediction.selection_evidence
                ),
            }
            for prediction in prediction_tuple
        ]
    )
    residual_hash = _digest(residual_tuple)
    evaluation_digest = _digest(
        {
            "problem_digest": problem.problem_digest,
            "trial_parameters": ordered_params,
            "row_identity_hash": row_identity_hash,
            "prediction_hash": prediction_hash,
            "residual_hash": residual_hash,
        }
    )
    return ObjectiveEvaluation(
        problem_digest=problem.problem_digest,
        trial_parameters=ordered_params,
        predictions=prediction_tuple,
        rows=row_tuple,
        residuals_deg=residual_tuple,
        row_identity_hash=row_identity_hash,
        prediction_hash=prediction_hash,
        residual_hash=residual_hash,
        evaluation_digest=evaluation_digest,
    )


def trial_predictions_from_fresh_rows(
    problem: CakedGeometryProblem,
    rows: Sequence[Mapping[str, object]],
    *,
    source: str = "dynamic_trial_simulation:fresh_native_rows",
) -> tuple[TrialPrediction, TrialPrediction]:
    """Resolve exactly one fresh native row for each locked physical branch.

    Only fields emitted by the fresh hit-row builder are accepted.  Persisted
    ``refined_sim_*`` and all direct caked prediction aliases are ignored.
    """

    candidates: dict[int, list[tuple[Mapping[str, object], tuple[int, int, int]]]] = {
        0: [],
        1: [],
    }
    selection_evidence: dict[int, BranchSelectionEvidence] = {}
    candidate_inventory: list[dict[str, object]] = []
    invalid_target_identities: list[dict[str, object]] = []
    invalid_target_branches: list[dict[str, object]] = []

    def _fresh_native_record(
        row: Mapping[str, object], *, branch: int | None = None
    ) -> tuple[Point, str] | None:
        generic_frames = (
            "detector_native_frame",
            "native_pixel_frame",
            "fresh_prediction_native_frame",
        )
        candidates: list[tuple[Point, str]] = []

        def add_tuple(
            point_key: str,
            *point_frames: str,
            point_source_key: str | None = None,
        ) -> None:
            point = _finite_point(row.get(point_key))
            if point is None:
                return
            _require_applicable_coordinate_frame(
                row,
                frame_keys=(*point_frames, *generic_frames),
                allowed_frames=_DETECTOR_NATIVE_FRAME_TOKENS,
                invalid_reason="fresh_prediction_native_frame_invalid",
                unproven_reason="fresh_prediction_native_frame_unproven",
                branch=branch,
                point_key=point_key,
            )
            source_value = str(
                row.get(point_source_key) if point_source_key else ""
            ).strip()
            candidates.append((point, source_value or point_key))

        def add_fields(x_key: str, y_key: str) -> None:
            point = _field_point(row, x_key, y_key)
            if point is None:
                return
            point_key = f"{x_key}/{y_key}"
            _require_applicable_coordinate_frame(
                row,
                frame_keys=generic_frames,
                allowed_frames=_DETECTOR_NATIVE_FRAME_TOKENS,
                invalid_reason="fresh_prediction_native_frame_invalid",
                unproven_reason="fresh_prediction_native_frame_unproven",
                branch=branch,
                point_key=point_key,
            )
            candidates.append((point, point_key))

        add_tuple(
            "geometry_detector_native_px",
            "geometry_detector_native_frame",
            point_source_key="geometry_detector_native_source",
        )
        add_tuple(
            "raw_detector_native_px",
            "raw_detector_native_frame",
            "raw_detector_frame",
            point_source_key="raw_detector_native_source",
        )
        add_fields("native_col", "native_row")
        record = _choose_consistent_point(
            candidates,
            reason="fresh_prediction_native_candidates_inconsistent",
            branch=int(branch) if branch is not None else -1,
        )
        return record

    def _fresh_native_point(row: Mapping[str, object]) -> Point | None:
        record = _fresh_native_record(row)
        return record[0] if record is not None else None

    def _fresh_display_record(
        row: Mapping[str, object], *, branch: int
    ) -> tuple[Point, str] | None:
        generic_frames = (
            "detector_display_frame",
            "display_pixel_frame",
            "fresh_prediction_display_frame",
        )
        candidates: list[tuple[Point, str]] = []

        def add_tuple(
            point_key: str,
            *point_frames: str,
            point_source_key: str | None = None,
        ) -> None:
            point = _finite_point(row.get(point_key))
            if point is None:
                return
            _require_applicable_coordinate_frame(
                row,
                frame_keys=(*point_frames, *generic_frames),
                allowed_frames=_DETECTOR_DISPLAY_FRAME_TOKENS,
                invalid_reason="fresh_prediction_display_frame_invalid",
                unproven_reason="fresh_prediction_display_frame_unproven",
                branch=branch,
                point_key=point_key,
            )
            source_value = str(
                row.get(point_source_key) if point_source_key else ""
            ).strip()
            candidates.append((point, source_value or point_key))

        def add_fields(x_key: str, y_key: str) -> None:
            point = _field_point(row, x_key, y_key)
            if point is None:
                return
            point_key = f"{x_key}/{y_key}"
            _require_applicable_coordinate_frame(
                row,
                frame_keys=generic_frames,
                allowed_frames=_DETECTOR_DISPLAY_FRAME_TOKENS,
                invalid_reason="fresh_prediction_display_frame_invalid",
                unproven_reason="fresh_prediction_display_frame_unproven",
                branch=branch,
                point_key=point_key,
            )
            candidates.append((point, point_key))

        add_tuple(
            "geometry_detector_display_px",
            "geometry_detector_display_frame",
            point_source_key="geometry_detector_display_source",
        )
        add_tuple(
            "raw_detector_display_px",
            "raw_detector_display_frame",
            "raw_detector_frame",
            point_source_key="raw_detector_display_source",
        )
        add_tuple(
            "predicted_detector_display_px",
            "predicted_detector_display_frame",
            point_source_key="predicted_detector_display_source",
        )
        add_fields("display_col", "display_row")
        record = _choose_consistent_point(
            candidates,
            reason="fresh_prediction_display_candidates_inconsistent",
            branch=branch,
        )
        return record

    def _candidate_key(
        branch: int,
        hkl_value: tuple[int, int, int],
    ) -> CandidateKey:
        return (
            TARGET_BACKGROUND_INDEX,
            problem.target_q_group_key,
            problem.hkl_equivalence_key,
            int(branch),
            tuple(hkl_value),
        )

    def _candidate_sort_token(key: CandidateKey) -> str:
        return json.dumps(
            _jsonable(key),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def _selection_evidence(
        *,
        branch: int,
        items: Sequence[tuple[Mapping[str, object], tuple[int, int, int]]],
        selected_hkl: tuple[int, int, int],
        rule: str,
        metric: str,
        selected_weight: float | None = None,
        runner_up_weight: float | None = None,
        tie_tolerance: float | None = None,
    ) -> BranchSelectionEvidence:
        keyed_weights = sorted(
            [
                (
                _candidate_key(branch, hkl_value),
                _finite_float(row.get("weight")),
                )
                for row, hkl_value in items
            ],
            key=lambda item: (
                _candidate_sort_token(item[0]),
                float("inf") if item[1] is None else -float(item[1]),
            ),
        )
        candidate_keys = tuple(key for key, _weight in keyed_weights)
        margin_abs = (
            float(selected_weight - runner_up_weight)
            if selected_weight is not None and runner_up_weight is not None
            else None
        )
        margin_relative = (
            float(margin_abs / max(abs(selected_weight), 1.0e-300))
            if margin_abs is not None and selected_weight is not None
            else None
        )
        return BranchSelectionEvidence(
            selection_rule=str(rule),
            selection_metric=str(metric),
            candidate_count=int(len(items)),
            candidate_keys=candidate_keys,
            candidate_weights=tuple(weight for _key, weight in keyed_weights),
            candidate_identity_digest=_digest(candidate_keys),
            candidate_weight_digest=_digest(keyed_weights),
            selected_candidate_key=_candidate_key(branch, selected_hkl),
            selected_candidate_digest=_digest(
                (_candidate_key(branch, selected_hkl), selected_weight)
            ),
            selected_weight=selected_weight,
            runner_up_weight=runner_up_weight,
            margin_abs=margin_abs,
            margin_relative=margin_relative,
            tie_tolerance=tie_tolerance,
        )

    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            continue
        raw_background = raw_row.get("background_index")
        background = _nonnegative_int(raw_background)
        q_group = _q_group_key(raw_row.get("q_group_key"))
        raw_hkl = raw_row.get("hkl")
        row_hkl = _hkl(raw_hkl)
        branch = _branch_index(raw_row.get("source_branch_index"))
        native_point = _fresh_native_point(raw_row)
        candidate_inventory.append(
            {
                "background_index": background,
                "q_group_key": q_group,
                "hkl": row_hkl,
                "computed_hkl_equivalence_key": (
                    _hkl_equivalence_key(row_hkl) if row_hkl is not None else None
                ),
                "branch_index": branch,
                "has_geometry_detector_native_px": (
                    _finite_point(raw_row.get("geometry_detector_native_px")) is not None
                ),
                "has_raw_detector_native_px": (
                    _finite_point(raw_row.get("raw_detector_native_px")) is not None
                ),
                "has_native_col_row": (
                    _field_point(raw_row, "native_col", "native_row") is not None
                ),
                "native_point": native_point,
                "hkl_raw": raw_row.get("hkl_raw"),
                "phi": raw_row.get("phi"),
                "weight": raw_row.get("weight"),
                "mosaic_weight": raw_row.get("mosaic_weight"),
                "best_sample_index": raw_row.get("best_sample_index"),
                "mosaic_top_rank_key": raw_row.get("mosaic_top_rank_key"),
                "source_table_index": raw_row.get("source_table_index"),
                "source_row_index": raw_row.get("source_row_index"),
                "source_reflection_index": raw_row.get("source_reflection_index"),
            }
        )
        if q_group == problem.target_q_group_key:
            if background is None:
                invalid_target_identities.append(
                    {
                        "field": "background_index",
                        "value": raw_background,
                    }
                )
                continue
            if background == TARGET_BACKGROUND_INDEX and row_hkl is None:
                invalid_target_identities.append(
                    {
                        "field": "hkl",
                        "value": raw_hkl,
                    }
                )
                continue
        is_target_equivalent = bool(
            background == TARGET_BACKGROUND_INDEX
            and q_group == problem.target_q_group_key
            and row_hkl is not None
            and _hkl_equivalence_key(row_hkl) == problem.hkl_equivalence_key
        )
        if is_target_equivalent and branch is None:
            invalid_target_branches.append(
                {
                    "value": raw_row.get("source_branch_index"),
                    "hkl": row_hkl,
                }
            )
            continue
        if (
            background != TARGET_BACKGROUND_INDEX
            or q_group != problem.target_q_group_key
            or row_hkl is None
            or _hkl_equivalence_key(row_hkl) != problem.hkl_equivalence_key
            or branch is None
        ):
            continue
        _fresh_native_record(raw_row, branch=branch)
        _fresh_display_record(raw_row, branch=branch)
        candidates[branch].append((raw_row, row_hkl))

    if invalid_target_identities:
        raise CakedGeometryObjectiveError(
            "invalid_fresh_prediction_identity",
            invalid_rows=invalid_target_identities[:8],
            candidate_inventory=candidate_inventory[:16],
        )
    if invalid_target_branches:
        raise CakedGeometryObjectiveError(
            "invalid_fresh_prediction_branch",
            invalid_rows=invalid_target_branches[:8],
            candidate_inventory=candidate_inventory[:16],
        )

    duplicate: list[int] = []
    for branch, items in candidates.items():
        if len(items) == 1:
            row, hkl_value = items[0]
            selection_evidence[branch] = _selection_evidence(
                branch=branch,
                items=items,
                selected_hkl=hkl_value,
                rule="unique_fresh_branch_row",
                metric="single_fresh_candidate",
                selected_weight=_finite_float(row.get("weight")),
            )
            continue
        if not items:
            continue
        native_points = [_fresh_native_point(row) for row, _hkl_value in items]
        reference_point = native_points[0]
        coordinate_identical = bool(
            reference_point is not None
            and all(
                point is not None
                and np.allclose(
                    np.asarray(point, dtype=float),
                    np.asarray(reference_point, dtype=float),
                    rtol=0.0,
                    atol=1.0e-9,
                )
                for point in native_points[1:]
            )
        )
        if coordinate_identical:
            selected = sorted(
                items,
                key=lambda item: _candidate_sort_token(
                    _candidate_key(branch, item[1])
                ),
            )[0]
            candidates[branch] = [selected]
            selection_evidence[branch] = _selection_evidence(
                branch=branch,
                items=items,
                selected_hkl=selected[1],
                rule="coordinate_identical_duplicate_collapse",
                metric="identical_native_coordinate",
                selected_weight=_finite_float(selected[0].get("weight")),
            )
            continue
        weighted_items: list[
            tuple[float, Mapping[str, object], tuple[int, int, int]]
        ] = []
        for row, hkl_value in items:
            physical_weight = _finite_float(row.get("weight"))
            if physical_weight is None or physical_weight < 0.0:
                weighted_items = []
                break
            weighted_items.append((physical_weight, row, hkl_value))
        weighted_items.sort(
            key=lambda item: (
                -item[0],
                _candidate_sort_token(_candidate_key(branch, item[2])),
            )
        )
        if len(weighted_items) == len(items) and len(weighted_items) >= 2:
            best_weight = float(weighted_items[0][0])
            second_weight = float(weighted_items[1][0])
            tie_tolerance = max(1.0e-12, abs(best_weight) * 1.0e-12)
            if best_weight - second_weight > tie_tolerance:
                candidates[branch] = [
                    (weighted_items[0][1], weighted_items[0][2])
                ]
                selection_evidence[branch] = _selection_evidence(
                    branch=branch,
                    items=items,
                    selected_hkl=weighted_items[0][2],
                    rule="unique_max_fresh_physical_weight",
                    metric="fresh_hit_intensity_mass",
                    selected_weight=best_weight,
                    runner_up_weight=second_weight,
                    tie_tolerance=tie_tolerance,
                )
                continue
        duplicate.append(branch)
    if duplicate:
        raise CakedGeometryObjectiveError(
            "duplicate_fresh_prediction_branch",
            branches=duplicate,
            candidate_inventory=candidate_inventory[:16],
        )
    missing = [branch for branch, items in candidates.items() if not items]
    if missing:
        raise CakedGeometryObjectiveError(
            "missing_fresh_prediction_branch",
            branches=missing,
            candidate_inventory=candidate_inventory[:8],
        )

    predictions: list[TrialPrediction] = []
    for branch in TARGET_BRANCHES:
        row, row_hkl = candidates[branch][0]
        native_record = _fresh_native_record(row, branch=branch)
        if native_record is None:
            raise CakedGeometryObjectiveError(
                "fresh_prediction_native_point_missing", branch=branch
            )
        native_point, native_source = native_record
        display_record = _fresh_display_record(row, branch=branch)
        route = _build_coordinate_route(
            role="predicted",
            authority="detector_native",
            branch=branch,
            detector_display_px=(display_record[0] if display_record is not None else None),
            detector_display_source=(
                display_record[1] if display_record is not None else None
            ),
            detector_native_px=native_point,
            detector_native_source=native_source,
            frozen_display_to_native=problem.frame.display_to_native,
            frozen_native_to_display=problem.frame.native_to_display,
            frozen_native_to_caked=problem.frame.native_to_caked,
            display_to_native_source=problem.frame.display_to_native_source,
            native_to_display_source=problem.frame.native_to_display_source,
            native_to_caked_source=(
                f"frozen_native_to_caked:{problem.frame.projector_kind}"
            ),
            roundtrip_tolerance_px=problem.frame.roundtrip_tolerance_px,
        )
        predictions.append(
            TrialPrediction(
                background_index=TARGET_BACKGROUND_INDEX,
                q_group_key=problem.target_q_group_key,
                hkl=row_hkl,
                branch_index=branch,
                native_pixel=native_point,
                detector_display_pixel=route.detector_display_px,
                detector_native_frame="detector_native",
                detector_display_frame="detector_display",
                detector_native_source=native_source,
                detector_display_source=route.detector_display_source,
                source=str(source),
                is_dynamic=True,
                cache_reused=False,
                hkl_equivalence_key=problem.hkl_equivalence_key,
                candidate_count=int(selection_evidence[branch].candidate_count),
                selection_rule=str(selection_evidence[branch].selection_rule),
                selection_evidence=selection_evidence[branch],
                coordinate_route=route,
            )
        )
    return predictions[0], predictions[1]


def objective_evaluation_to_jsonable(evaluation: ObjectiveEvaluation) -> dict[str, object]:
    """Return a stable artifact payload without exposing the projector callable."""

    return {
        "schema": "caked_geometry_objective_evaluation_v2",
        "problem_digest": evaluation.problem_digest,
        "trial_parameters": _jsonable(evaluation.trial_parameters),
        "units": evaluation.units,
        "residuals_deg": _jsonable(evaluation.residuals_deg),
        "row_identity_hash": evaluation.row_identity_hash,
        "prediction_hash": evaluation.prediction_hash,
        "residual_hash": evaluation.residual_hash,
        "evaluation_digest": evaluation.evaluation_digest,
        "predictions": [
            {
                **_prediction_identity_payload(prediction),
                "native_pixel": _jsonable(prediction.native_pixel),
                "detector_native_px": _jsonable(prediction.native_pixel),
                "detector_native_frame": prediction.detector_native_frame,
                "detector_native_source": prediction.detector_native_source,
                "detector_display_px": _jsonable(prediction.detector_display_pixel),
                "detector_display_frame": prediction.detector_display_frame,
                "detector_display_source": prediction.detector_display_source,
                "predicted_caked_deg": _jsonable(prediction.predicted_caked_deg),
                "coordinate_route": (
                    coordinate_route_to_jsonable(prediction.coordinate_route)
                    if prediction.coordinate_route is not None
                    else None
                ),
                "source": prediction.source,
                "is_dynamic": prediction.is_dynamic,
                "cache_reused": prediction.cache_reused,
                "candidate_count": prediction.candidate_count,
                "selection_rule": prediction.selection_rule,
                "selection_evidence": branch_selection_evidence_to_jsonable(
                    prediction.selection_evidence
                ),
            }
            for prediction in evaluation.predictions
        ],
        "rows": [
            _jsonable(
                {
                    "background_index": row.background_index,
                    "q_group_key": row.q_group_key,
                    "hkl": row.hkl,
                    "hkl_equivalence_key": row.hkl_equivalence_key,
                    "branch_index": row.branch_index,
                    "pair_id": row.pair_id,
                    "measurement_origin": row.measurement_origin,
                    "saved_caked_audit_deg": row.saved_caked_audit_deg,
                    "measured_native_px": row.measured_native_px,
                    "measured_detector_display_px": row.measured_detector_display_px,
                    "measured_caked_deg": row.measured_caked_deg,
                    "measured_coordinate_route": coordinate_route_to_jsonable(
                        row.measured_coordinate_route
                    ),
                    "predicted_native_px": row.predicted_native_px,
                    "predicted_detector_display_px": row.predicted_detector_display_px,
                    "predicted_caked_deg": row.predicted_caked_deg,
                    "predicted_coordinate_route": coordinate_route_to_jsonable(
                        row.predicted_coordinate_route
                    ),
                    "delta_two_theta_deg": row.delta_two_theta_deg,
                    "wrapped_delta_phi_deg": row.wrapped_delta_phi_deg,
                    "prediction_source": row.prediction_source,
                    "prediction_is_dynamic": row.prediction_is_dynamic,
                    "prediction_cache_reused": row.prediction_cache_reused,
                    "prediction_candidate_count": row.prediction_candidate_count,
                    "prediction_selection_rule": row.prediction_selection_rule,
                    "prediction_selection_evidence": (
                        branch_selection_evidence_to_jsonable(
                            row.prediction_selection_evidence
                        )
                    ),
                }
            )
            for row in evaluation.rows
        ],
    }


__all__ = [
    "ACTIVE_PARAMETER_NAMES",
    "ACTIVE_PARAMETER_NUMERICS",
    "ACTIVE_PARAMETER_RUNGS",
    "BranchObjectiveEvaluation",
    "CakedGeometryObjectiveError",
    "CakedGeometryParameterNumerics",
    "CakedGeometryProblem",
    "CoordinateRouteEvidence",
    "DatasetFitSpaceFrame",
    "LockedBranchTarget",
    "ObjectiveEvaluation",
    "TrialPrediction",
    "build_caked_geometry_problem",
    "caked_geometry_parameter_numerics",
    "coordinate_route_to_jsonable",
    "evaluate_caked_geometry_objective",
    "objective_evaluation_to_jsonable",
    "trial_predictions_from_fresh_rows",
    "wrap_phi_delta_deg",
]
