"""Deterministic incidence-angle averages of detector-coordinate functions.

Incidence angle is an outer experimental probability measure, not an extra
sampled source coordinate. Every quadrature node therefore owns a complete
fixed-angle detector engine while all nodes retain one source realization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import Rod
from rasim_next.core.contracts import (
    canonical_revision_sha256,
    incidence_scan_calibration_binding_revision,
)
from rasim_next.pipeline.incidence_acquisition import IncidenceAngleQuadrature

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_MEASURE_ID = "raw_detector_coordinate_density_A2_per_px2.v1"
_ROOT_POLICY = "all_retained_roots.v1"
_REDUCTION_ID = "fixed_quadrature_estimate_of_incoherent_incidence_angle_probability_average.v1"
_RESULT_BUILDER_TOKEN = object()


class _DetailedCoordinateResult(Protocol):
    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    branch: int | None
    root_policy: str
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.integer]
    source_state_count: int
    source_revision: str
    measure_id: str
    execution_backend: str
    execution_device: str | None


class _RodReducedCoordinateResult(Protocol):
    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str | None
    branch: None
    root_policy: str
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.integer]
    source_state_count: int
    source_revision: str
    measure_id: str
    execution_backend: str
    execution_device: str | None


class _AllRootDetectorNode(Protocol):
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    source_revision: str
    source_state_count: int
    sample_geometry_revision: str
    incidence_axis_angle_rad: float | None
    incidence_angle_static_physics_revision: str
    detector_shape_rc: tuple[int, int]
    detector_panel_revision: str
    detector_visible_m0_q_gap_Ainv: float | None
    scan_calibration_binding_revision: str | None

    def restrict_rods(self, rods: tuple[Rod, ...]) -> _AllRootDetectorNode: ...

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> _DetailedCoordinateResult: ...


@runtime_checkable
class _RodReducedDetectorNode(Protocol):
    def evaluate_detector_density_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> _RodReducedCoordinateResult: ...


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"each detector must expose a nonempty {name}")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _source_revision(detector: _AllRootDetectorNode) -> str:
    return _required_text(detector.source_revision, "source_revision")


def _source_state_count(detector: _AllRootDetectorNode) -> int:
    supplied = detector.source_state_count
    if isinstance(supplied, bool) or not isinstance(supplied, (int, np.integer)):
        raise TypeError("each detector must expose an integer source_state_count")
    result = int(supplied)
    if result < 1:
        raise ValueError("source_state_count must be positive")
    return result


def _sample_geometry_revision(detector: _AllRootDetectorNode) -> str:
    return _required_text(detector.sample_geometry_revision, "sample_geometry_revision")


def _incidence_axis_angle_rad(detector: _AllRootDetectorNode) -> float:
    supplied = detector.incidence_axis_angle_rad
    if supplied is None:
        raise TypeError("each scan detector must expose incidence_axis_angle_rad")
    array = np.asarray(supplied)
    if np.iscomplexobj(array) and np.any(array.imag != 0.0):
        raise ValueError("incidence_axis_angle_rad must be real")
    result = float(array.real)
    if not math.isfinite(result):
        raise ValueError("incidence_axis_angle_rad must be finite")
    return result


def _detector_shape_rc(detector: _AllRootDetectorNode) -> tuple[int, int]:
    supplied = tuple(detector.detector_shape_rc)
    if len(supplied) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
        for value in supplied
    ):
        raise ValueError("each detector must expose a positive detector_shape_rc")
    return int(supplied[0]), int(supplied[1])


def _detector_panel_revision(detector: _AllRootDetectorNode) -> str:
    return _required_text(detector.detector_panel_revision, "detector_panel_revision")


def _static_physics_revision(detector: _AllRootDetectorNode) -> str:
    return _required_text(
        detector.incidence_angle_static_physics_revision,
        "incidence_angle_static_physics_revision",
    )


@dataclass(frozen=True, slots=True)
class _ValidatedResultFields:
    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_incident_state_fraction: FloatArray
    source_state_count: int
    component_sample_geometry_revision: tuple[str, ...]
    angle_evaluation_count: int
    detector_visible_m0_q_gap_Ainv: float | None


def _validated_result_fields(
    *,
    column_px: ArrayLike,
    row_px: ArrayLike,
    rods: tuple[Rod, ...],
    density_A2_per_px2: ArrayLike,
    caustic: ArrayLike,
    caustic_has_rod_axis: bool,
    valid_incident_state_fraction: ArrayLike,
    source_state_count: object,
    component_sample_geometry_revision: tuple[str, ...],
    angle_evaluation_count: object,
    detector_visible_m0_q_gap_Ainv: float | None,
    branch: object,
    root_policy: str,
    measure_id: str,
    reduction_id: str,
    execution_backend: str,
    component_execution_backend: str,
    execution_device: str | None,
    provenance: tuple[tuple[str, object], ...],
) -> _ValidatedResultFields:
    supplied_column = np.asarray(column_px)
    supplied_row = np.asarray(row_px)
    if (np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0)) or (
        np.iscomplexobj(supplied_row) and np.any(supplied_row.imag != 0.0)
    ):
        raise ValueError("detector coordinates must be real")
    column, row = np.broadcast_arrays(
        np.asarray(supplied_column.real, dtype=np.float64),
        np.asarray(supplied_row.real, dtype=np.float64),
    )
    column = np.array(column, copy=True, order="C")
    row = np.array(row, copy=True, order="C")
    shape = column.shape
    if not np.all(np.isfinite(column)) or not np.all(np.isfinite(row)):
        raise ValueError("detector coordinates must be finite")

    physical_rods = tuple(rods)
    if not physical_rods or any(not isinstance(rod, Rod) for rod in physical_rods):
        raise ValueError("rods must contain at least one physical Rod")
    if len({(rod.h, rod.k) for rod in physical_rods}) != len(physical_rods):
        raise ValueError("rods must not repeat a physical rod")

    supplied_density = np.asarray(density_A2_per_px2)
    if np.iscomplexobj(supplied_density) and np.any(supplied_density.imag != 0.0):
        raise ValueError("scan-averaged detector density must be real")
    density = np.array(supplied_density.real, dtype=np.float64, copy=True, order="C")
    if density.shape != shape or not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("scan-averaged detector density must be finite and nonnegative")
    supplied_caustic = np.asarray(caustic)
    if supplied_caustic.dtype.kind != "b":
        raise ValueError("caustic flags must have boolean dtype")
    caustic_flags = np.array(supplied_caustic, dtype=np.bool_, copy=True, order="C")
    expected_caustic_shape = (*shape, len(physical_rods)) if caustic_has_rod_axis else shape
    if caustic_flags.shape != expected_caustic_shape:
        raise ValueError("caustic flags have the wrong detector-coordinate shape")
    if np.any(caustic_flags):
        raise ValueError("fixed incidence-angle quadrature results must be non-caustic")

    supplied_valid_fraction = np.asarray(valid_incident_state_fraction)
    if np.iscomplexobj(supplied_valid_fraction) and np.any(supplied_valid_fraction.imag != 0.0):
        raise ValueError("valid incident state fraction must be real")
    valid_fraction = np.array(
        supplied_valid_fraction.real,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    tolerance = 64.0 * np.finfo(np.float64).eps
    if (
        valid_fraction.shape != shape
        or not np.all(np.isfinite(valid_fraction))
        or np.any(valid_fraction < 0.0)
        or np.any(valid_fraction > 1.0 + tolerance)
    ):
        raise ValueError("valid incident state fraction must lie in [0, 1]")
    np.minimum(valid_fraction, 1.0, out=valid_fraction)
    if np.any((valid_fraction == 0.0) & (density != 0.0)):
        raise ValueError("zero valid incident state fraction requires zero detector density")
    if branch is not None or root_policy != _ROOT_POLICY:
        raise ValueError("incidence-angle averages require all retained roots")
    if measure_id != _MEASURE_ID or reduction_id != _REDUCTION_ID:
        raise ValueError("unsupported incidence-angle detector reduction")
    if execution_backend not in {
        "incoherent_incidence_angle_cpu.v1",
        "incoherent_incidence_angle_cuda.v1",
    }:
        raise ValueError("unsupported incidence-angle execution backend")
    if execution_device is not None and (
        not isinstance(execution_device, str) or not execution_device
    ):
        raise ValueError("execution_device must be None or a nonempty string")
    if execution_backend.endswith("_cuda.v1") != (execution_device is not None):
        raise ValueError("CUDA scan results require an execution device")
    if component_execution_backend not in {
        "numba_cpu_source_averaged.v1",
        "numba_cuda_source_averaged.v1",
        "hybrid_cuda_cpu_local_m0.v1",
        "numpy_cpu_sparse_source_averaged.v1",
    }:
        raise ValueError("unsupported scan component execution backend")
    component_is_cuda = component_execution_backend in {
        "numba_cuda_source_averaged.v1",
        "hybrid_cuda_cpu_local_m0.v1",
    }
    if execution_backend.endswith("_cuda.v1") != component_is_cuda:
        raise ValueError("scan and component execution backends must use the same device class")

    state_count = _positive_integer(source_state_count, "source_state_count")
    evaluation_count = _positive_integer(angle_evaluation_count, "angle_evaluation_count")
    if isinstance(component_sample_geometry_revision, str):
        raise TypeError("component_sample_geometry_revision must be a sequence of revisions")
    component_revisions = tuple(component_sample_geometry_revision)
    if evaluation_count != len(component_revisions):
        raise ValueError("angle evaluation count must match component geometry provenance")
    for name, value in provenance:
        _required_text(value, name)
    for value in component_revisions:
        _required_text(value, "component_sample_geometry_revision")

    gap = detector_visible_m0_q_gap_Ainv
    has_m0 = any(rod.family_m == 0 for rod in physical_rods)
    if has_m0:
        supplied_gap = np.asarray(gap) if gap is not None else None
        if (
            supplied_gap is None
            or supplied_gap.ndim != 0
            or (np.iscomplexobj(supplied_gap) and supplied_gap.imag != 0.0)
        ):
            raise ValueError("scan-averaged m=0 requires a real support gap")
        real_gap = float(supplied_gap.real)
        if not math.isfinite(real_gap) or real_gap < 0.0:
            raise ValueError("scan-averaged m=0 requires a nonnegative support gap")
        gap = real_gap
    elif gap is not None:
        raise ValueError("an m=0 support gap requires an m=0 rod")

    for value in (column, row, density, caustic_flags, valid_fraction):
        value.setflags(write=False)
    return _ValidatedResultFields(
        column_px=column,
        row_px=row,
        rods=physical_rods,
        density_A2_per_px2=density,
        caustic=caustic_flags,
        valid_incident_state_fraction=valid_fraction,
        source_state_count=state_count,
        component_sample_geometry_revision=component_revisions,
        angle_evaluation_count=evaluation_count,
        detector_visible_m0_q_gap_Ainv=gap,
    )


@dataclass(frozen=True, slots=True)
class IncidenceAngleAveragedDetectorCoordinateIntensity:
    """Fixed-quadrature estimate of a normalized incidence-angle average."""

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_incident_state_fraction: FloatArray
    source_state_count: int
    source_revision: str
    quadrature_revision: str
    scan_revision: str
    detector_panel_revision: str
    component_sample_geometry_revision: tuple[str, ...]
    angle_evaluation_count: int
    _builder_token: InitVar[object | None] = None
    detector_visible_m0_q_gap_Ainv: float | None = None
    branch: None = field(init=False, default=None)
    root_policy: str = _ROOT_POLICY
    measure_id: str = _MEASURE_ID
    reduction_id: str = _REDUCTION_ID
    execution_backend: str = "incoherent_incidence_angle_cpu.v1"
    component_execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _RESULT_BUILDER_TOKEN:
            raise TypeError("incidence-angle results are built by their detector evaluator")
        fields = _validated_result_fields(
            column_px=self.column_px,
            row_px=self.row_px,
            rods=self.rods,
            density_A2_per_px2=self.density_A2_per_px2,
            caustic=self.caustic,
            caustic_has_rod_axis=True,
            valid_incident_state_fraction=self.valid_incident_state_fraction,
            source_state_count=self.source_state_count,
            component_sample_geometry_revision=self.component_sample_geometry_revision,
            angle_evaluation_count=self.angle_evaluation_count,
            detector_visible_m0_q_gap_Ainv=self.detector_visible_m0_q_gap_Ainv,
            branch=self.branch,
            root_policy=self.root_policy,
            measure_id=self.measure_id,
            reduction_id=self.reduction_id,
            execution_backend=self.execution_backend,
            component_execution_backend=self.component_execution_backend,
            execution_device=self.execution_device,
            provenance=(
                ("rod_catalog_revision", self.rod_catalog_revision),
                ("source_revision", self.source_revision),
                ("quadrature_revision", self.quadrature_revision),
                ("scan_revision", self.scan_revision),
                ("detector_panel_revision", self.detector_panel_revision),
            ),
        )
        supplied_per_rod = np.asarray(self.per_rod_density_A2_per_px2)
        if np.iscomplexobj(supplied_per_rod) and np.any(supplied_per_rod.imag != 0.0):
            raise ValueError("scan-averaged per-rod density must be real")
        per_rod = np.array(
            supplied_per_rod.real,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            per_rod.shape != (*fields.column_px.shape, len(fields.rods))
            or not np.all(np.isfinite(per_rod))
            or np.any(per_rod < 0.0)
        ):
            raise ValueError("scan-averaged per-rod density must be finite and nonnegative")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        zero_expected = expected == 0.0
        difference = np.abs(fields.density_A2_per_px2 - expected)
        nonzero_agrees = difference[~zero_expected] <= (
            1024.0 * np.finfo(np.float64).eps * np.abs(expected[~zero_expected])
        )
        if (
            not np.all(np.isfinite(expected))
            or np.any(fields.density_A2_per_px2[zero_expected] != 0.0)
            or not np.all(nonzero_agrees)
        ):
            raise ValueError("scan-averaged density must equal its physical rod sum")
        per_rod.setflags(write=False)
        object.__setattr__(self, "column_px", fields.column_px)
        object.__setattr__(self, "row_px", fields.row_px)
        object.__setattr__(self, "rods", fields.rods)
        object.__setattr__(self, "per_rod_density_A2_per_px2", per_rod)
        object.__setattr__(self, "density_A2_per_px2", fields.density_A2_per_px2)
        object.__setattr__(self, "caustic", fields.caustic)
        object.__setattr__(
            self,
            "valid_incident_state_fraction",
            fields.valid_incident_state_fraction,
        )
        object.__setattr__(self, "source_state_count", fields.source_state_count)
        object.__setattr__(
            self,
            "component_sample_geometry_revision",
            fields.component_sample_geometry_revision,
        )
        object.__setattr__(self, "angle_evaluation_count", fields.angle_evaluation_count)
        object.__setattr__(
            self,
            "detector_visible_m0_q_gap_Ainv",
            fields.detector_visible_m0_q_gap_Ainv,
        )


@dataclass(frozen=True, slots=True)
class IncidenceAngleAveragedDetectorCoordinateDensity:
    """Rod-reduced fixed-quadrature estimate on detector-native coordinates."""

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_incident_state_fraction: FloatArray
    source_state_count: int
    source_revision: str
    quadrature_revision: str
    scan_revision: str
    detector_panel_revision: str
    component_sample_geometry_revision: tuple[str, ...]
    angle_evaluation_count: int
    _builder_token: InitVar[object | None] = None
    detector_visible_m0_q_gap_Ainv: float | None = None
    branch: None = field(init=False, default=None)
    root_policy: str = _ROOT_POLICY
    measure_id: str = _MEASURE_ID
    reduction_id: str = _REDUCTION_ID
    execution_backend: str = "incoherent_incidence_angle_cpu.v1"
    component_execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _RESULT_BUILDER_TOKEN:
            raise TypeError("incidence-angle results are built by their detector evaluator")
        fields = _validated_result_fields(
            column_px=self.column_px,
            row_px=self.row_px,
            rods=self.rods,
            density_A2_per_px2=self.density_A2_per_px2,
            caustic=self.caustic,
            caustic_has_rod_axis=False,
            valid_incident_state_fraction=self.valid_incident_state_fraction,
            source_state_count=self.source_state_count,
            component_sample_geometry_revision=self.component_sample_geometry_revision,
            angle_evaluation_count=self.angle_evaluation_count,
            detector_visible_m0_q_gap_Ainv=self.detector_visible_m0_q_gap_Ainv,
            branch=None,
            root_policy=self.root_policy,
            measure_id=self.measure_id,
            reduction_id=self.reduction_id,
            execution_backend=self.execution_backend,
            component_execution_backend=self.component_execution_backend,
            execution_device=self.execution_device,
            provenance=(
                ("rod_catalog_revision", self.rod_catalog_revision),
                ("source_revision", self.source_revision),
                ("quadrature_revision", self.quadrature_revision),
                ("scan_revision", self.scan_revision),
                ("detector_panel_revision", self.detector_panel_revision),
            ),
        )
        object.__setattr__(self, "column_px", fields.column_px)
        object.__setattr__(self, "row_px", fields.row_px)
        object.__setattr__(self, "rods", fields.rods)
        object.__setattr__(self, "density_A2_per_px2", fields.density_A2_per_px2)
        object.__setattr__(self, "caustic", fields.caustic)
        object.__setattr__(
            self,
            "valid_incident_state_fraction",
            fields.valid_incident_state_fraction,
        )
        object.__setattr__(self, "source_state_count", fields.source_state_count)
        object.__setattr__(
            self,
            "component_sample_geometry_revision",
            fields.component_sample_geometry_revision,
        )
        object.__setattr__(self, "angle_evaluation_count", fields.angle_evaluation_count)
        object.__setattr__(
            self,
            "detector_visible_m0_q_gap_Ainv",
            fields.detector_visible_m0_q_gap_Ainv,
        )


def _validated_evaluation_request(
    column_px: ArrayLike,
    row_px: ArrayLike,
    *,
    execution_backend: str,
    cuda_coordinate_chunk_size: int | None,
) -> tuple[FloatArray, FloatArray, dict[str, object], str]:
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    if cuda_coordinate_chunk_size is not None and (
        execution_backend != "cuda"
        or isinstance(cuda_coordinate_chunk_size, bool)
        or not isinstance(cuda_coordinate_chunk_size, int)
        or cuda_coordinate_chunk_size < 1
    ):
        raise ValueError("cuda_coordinate_chunk_size requires a positive CUDA chunk size")
    supplied_column = np.asarray(column_px)
    supplied_row = np.asarray(row_px)
    if (np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0)) or (
        np.iscomplexobj(supplied_row) and np.any(supplied_row.imag != 0.0)
    ):
        raise ValueError("detector coordinates must be real")
    column, row = np.broadcast_arrays(
        np.asarray(supplied_column.real, dtype=np.float64),
        np.asarray(supplied_row.real, dtype=np.float64),
    )
    if not np.all(np.isfinite(column)) or not np.all(np.isfinite(row)):
        raise ValueError("detector coordinates must be finite")
    column = np.array(column, dtype=np.float64, copy=True, order="C")
    row = np.array(row, dtype=np.float64, copy=True, order="C")
    column.setflags(write=False)
    row.setflags(write=False)
    keyword: dict[str, object] = {"execution_backend": execution_backend}
    if cuda_coordinate_chunk_size is not None:
        keyword["cuda_coordinate_chunk_size"] = cuda_coordinate_chunk_size
    aggregate_backend = (
        "incoherent_incidence_angle_cuda.v1"
        if execution_backend == "cuda"
        else "incoherent_incidence_angle_cpu.v1"
    )
    return column, row, keyword, aggregate_backend


def _validated_component_source_count(
    result: object,
    coordinate_shape: tuple[int, ...],
    source_state_count: int,
) -> NDArray[np.integer]:
    valid_count = np.asarray(result.valid_source_count)
    if (
        valid_count.shape != coordinate_shape
        or valid_count.dtype.kind not in "iu"
        or np.any(valid_count < 0)
        or np.any(valid_count > source_state_count)
    ):
        raise RuntimeError("a scan component returned invalid source counts")
    return valid_count


def _component_execution_identity(result: object) -> tuple[str, str | None]:
    backend = _required_text(result.execution_backend, "execution_backend")
    device = result.execution_device
    if device is not None and (not isinstance(device, str) or not device):
        raise RuntimeError("a scan component returned an invalid execution device")
    return backend, device


def _consistent_component_execution(
    expected: tuple[str, str | None] | None,
    result: object,
    requested_execution_backend: str,
) -> tuple[str, str | None]:
    current = _component_execution_identity(result)
    backend, device = current
    cpu_backends = {
        "numba_cpu_source_averaged.v1",
        "numpy_cpu_sparse_source_averaged.v1",
    }
    cuda_backends = {
        "numba_cuda_source_averaged.v1",
        "hybrid_cuda_cpu_local_m0.v1",
    }
    if (
        requested_execution_backend == "cpu" and (backend not in cpu_backends or device is not None)
    ) or (
        requested_execution_backend == "cuda" and (backend not in cuda_backends or device is None)
    ):
        raise RuntimeError("a scan component contradicted the requested execution backend")
    if expected is not None and current != expected:
        raise RuntimeError("scan components used inconsistent execution backends")
    return current


@dataclass(frozen=True, slots=True)
class IncidenceAngleAveragedDetector:
    """Eager, immutable probability average of fixed-angle detector engines."""

    quadrature: IncidenceAngleQuadrature
    detectors: tuple[_AllRootDetectorNode, ...]
    scan_calibration_revision: str = field(init=False)
    scan_calibration_binding_revision: str = field(init=False)
    rods: tuple[Rod, ...] = field(init=False)
    rod_catalog_revision: str = field(init=False)
    source_revision: str = field(init=False)
    source_state_count: int = field(init=False)
    detector_shape_rc: tuple[int, int] = field(init=False)
    detector_panel_revision: str = field(init=False)
    static_physics_revision: str = field(init=False)
    component_sample_geometry_revision: tuple[str, ...] = field(init=False)
    detector_visible_m0_q_gap_Ainv: float | None = field(init=False)
    scan_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, IncidenceAngleQuadrature):
            raise TypeError("quadrature must be IncidenceAngleQuadrature")
        detectors = tuple(self.detectors)
        if len(detectors) != self.quadrature.incidence_angle_rad.size:
            raise ValueError("one fixed-angle detector is required per quadrature node")
        reference = detectors[0]
        rods = tuple(reference.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("each fixed-angle detector must expose physical rods")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("fixed-angle detector rods must not repeat a physical rod")
        rod_revision = _required_text(reference.rod_catalog_revision, "rod_catalog_revision")
        source_revision = _source_revision(reference)
        source_count = _source_state_count(reference)
        shape = _detector_shape_rc(reference)
        panel_revision = _detector_panel_revision(reference)
        static_revision = _static_physics_revision(reference)
        geometry_revisions = tuple(_sample_geometry_revision(item) for item in detectors)
        if len(set(geometry_revisions)) != len(geometry_revisions):
            raise ValueError("each incidence-angle node requires a distinct sample pose")
        component_angles = np.asarray(
            [_incidence_axis_angle_rad(item) for item in detectors],
            dtype=np.float64,
        )
        angle_tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * np.maximum(
                1.0,
                np.abs(self.quadrature.incidence_angle_rad),
            )
        )
        if np.any(np.abs(component_angles - self.quadrature.incidence_angle_rad) > angle_tolerance):
            raise ValueError("scan detector poses do not match their quadrature angles")
        calibration_revision = self.quadrature.incidence_angle_calibration_revision
        binding_revision = incidence_scan_calibration_binding_revision(
            scan_calibration_revision=calibration_revision,
            component_sample_geometry_revision=geometry_revisions,
            component_incidence_axis_angle_rad=component_angles,
            effective_incidence_angle_rad=self.quadrature.incidence_angle_rad,
            detector_panel_revision=panel_revision,
            source_revision=source_revision,
            source_state_count=source_count,
        )
        component_bindings = tuple(
            detector.scan_calibration_binding_revision for detector in detectors
        )
        if any(value != binding_revision for value in component_bindings):
            raise ValueError("scan detectors do not carry their calibrated scan binding")
        gaps: list[float] = []
        has_m0 = any(rod.family_m == 0 for rod in rods)

        for detector in detectors:
            if tuple(detector.rods) != rods:
                raise ValueError("incidence-angle detectors must share ordered physical rods")
            if (
                _required_text(detector.rod_catalog_revision, "rod_catalog_revision")
                != rod_revision
            ):
                raise ValueError("incidence-angle detectors must share one rod catalog")
            if (
                _source_revision(detector) != source_revision
                or _source_state_count(detector) != source_count
            ):
                raise ValueError("incidence-angle detectors must share one source realization")
            if (
                _detector_shape_rc(detector) != shape
                or _detector_panel_revision(detector) != panel_revision
            ):
                raise ValueError("incidence-angle detectors must share one detector-native chart")
            if _static_physics_revision(detector) != static_revision:
                raise ValueError("only incidence geometry may vary across scan detectors")
            gap = detector.detector_visible_m0_q_gap_Ainv
            if has_m0:
                supplied_gap = np.asarray(gap) if gap is not None else None
                if (
                    supplied_gap is None
                    or supplied_gap.ndim != 0
                    or (np.iscomplexobj(supplied_gap) and supplied_gap.imag != 0.0)
                ):
                    raise ValueError("each m=0 scan detector must expose a real support gap")
                real_gap = float(supplied_gap.real)
                if not math.isfinite(real_gap) or real_gap < 0.0:
                    raise ValueError("each m=0 scan detector must expose a support gap")
                gaps.append(real_gap)
            elif gap is not None:
                raise ValueError("an m=0 support gap requires an m=0 rod")

        scan_revision = canonical_revision_sha256(
            ("definition_id", "incidence_angle_averaged_detector.v1"),
            ("quadrature_revision", self.quadrature.quadrature_revision),
            ("scan_calibration_binding_revision", binding_revision),
            ("source_revision", source_revision),
            ("source_state_count", source_count),
            ("rod_catalog_revision", rod_revision),
            ("active_rod_h", np.asarray([rod.h for rod in rods], dtype=np.int64)),
            ("active_rod_k", np.asarray([rod.k for rod in rods], dtype=np.int64)),
            (
                "active_rod_population",
                np.asarray([rod.population for rod in rods], dtype=np.float64),
            ),
            ("detector_panel_revision", panel_revision),
            ("static_physics_revision", static_revision),
            ("component_sample_geometry_revision", geometry_revisions),
            ("component_incidence_axis_angle_rad", component_angles),
        )
        object.__setattr__(self, "detectors", detectors)
        object.__setattr__(self, "scan_calibration_revision", calibration_revision)
        object.__setattr__(
            self,
            "scan_calibration_binding_revision",
            binding_revision,
        )
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "rod_catalog_revision", rod_revision)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_state_count", source_count)
        object.__setattr__(self, "detector_shape_rc", shape)
        object.__setattr__(self, "detector_panel_revision", panel_revision)
        object.__setattr__(self, "static_physics_revision", static_revision)
        object.__setattr__(self, "component_sample_geometry_revision", geometry_revisions)
        object.__setattr__(
            self,
            "detector_visible_m0_q_gap_Ainv",
            min(gaps) if gaps else None,
        )
        object.__setattr__(self, "scan_revision", scan_revision)

    def restrict_rods(self, rods: tuple[Rod, ...]) -> IncidenceAngleAveragedDetector:
        """Apply one explicit signed-rod restriction to every angle node."""

        restricted = []
        for detector in self.detectors:
            restricted.append(detector.restrict_rods(rods))
        return IncidenceAngleAveragedDetector(self.quadrature, tuple(restricted))

    def _validate_component_provenance(
        self,
        result: object,
        column_px: FloatArray,
        row_px: FloatArray,
    ) -> None:
        if not np.array_equal(result.column_px, column_px) or not np.array_equal(
            result.row_px,
            row_px,
        ):
            raise RuntimeError("a scan component changed detector coordinates")
        if (
            tuple(result.rods) != self.rods
            or result.rod_catalog_revision != self.rod_catalog_revision
            or result.branch is not None
            or result.root_policy != _ROOT_POLICY
            or result.measure_id != _MEASURE_ID
            or result.source_revision != self.source_revision
            or _positive_integer(result.source_state_count, "component source_state_count")
            != self.source_state_count
        ):
            raise RuntimeError("a scan component violated shared detector provenance")

    def _evaluate_weighted_components(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        keyword: dict[str, object],
        *,
        preserve_rods: bool,
    ) -> tuple[FloatArray, FloatArray, str, str | None]:
        """Apply one fail-closed streaming reduction to either public observable."""

        coordinate_shape = column_px.shape
        component_shape = (*coordinate_shape, len(self.rods)) if preserve_rods else coordinate_shape
        aggregate = np.zeros(component_shape, dtype=np.float64)
        aggregate_compensation = np.zeros(component_shape, dtype=np.float64)
        valid_fraction = np.zeros(coordinate_shape, dtype=np.float64)
        valid_compensation = np.zeros(coordinate_shape, dtype=np.float64)
        execution_identity: tuple[str, str | None] | None = None
        for node_index, (angle, mass, detector) in enumerate(
            zip(
                self.quadrature.incidence_angle_rad,
                self.quadrature.exposure_probability_mass,
                self.detectors,
                strict=True,
            )
        ):
            if preserve_rods:
                result = detector.evaluate_detector_coordinates_all_roots(
                    column_px,
                    row_px,
                    **keyword,
                )
                supplied_component = np.asarray(result.per_rod_density_A2_per_px2)
                accepted_caustic_shapes = (component_shape,)
            else:
                total_method = (
                    detector.evaluate_detector_density_all_roots
                    if isinstance(detector, _RodReducedDetectorNode)
                    else None
                )
                if total_method is None:
                    result = detector.evaluate_detector_coordinates_all_roots(
                        column_px,
                        row_px,
                        **keyword,
                    )
                    supplied_per_rod = np.asarray(result.per_rod_density_A2_per_px2)
                    if np.iscomplexobj(supplied_per_rod) and np.any(supplied_per_rod.imag != 0.0):
                        raise RuntimeError("a scan component returned complex detector density")
                    per_rod = np.asarray(supplied_per_rod.real, dtype=np.float64)
                    if (
                        per_rod.shape != (*coordinate_shape, len(self.rods))
                        or not np.all(np.isfinite(per_rod))
                        or np.any(per_rod < 0.0)
                    ):
                        raise RuntimeError("a non-caustic scan component returned invalid density")
                    supplied_component = np.sum(per_rod, axis=-1, dtype=np.float64)
                    accepted_caustic_shapes = ((*coordinate_shape, len(self.rods)),)
                else:
                    result = total_method(column_px, row_px, **keyword)
                    supplied_component = np.asarray(result.density_A2_per_px2)
                    accepted_caustic_shapes = (coordinate_shape,)

            self._validate_component_provenance(result, column_px, row_px)
            supplied_caustic = np.asarray(result.caustic)
            if supplied_caustic.dtype.kind != "b":
                raise RuntimeError("a scan component returned non-boolean caustic flags")
            component_caustic = np.asarray(supplied_caustic, dtype=np.bool_)
            if component_caustic.shape not in accepted_caustic_shapes:
                raise RuntimeError("a scan component returned malformed caustic flags")
            if np.any(component_caustic):
                raise FloatingPointError(
                    "incidence-angle quadrature encountered an exact caustic "
                    f"at node {node_index} ({float(angle):.17g} rad)"
                )
            if np.iscomplexobj(supplied_component) and np.any(supplied_component.imag != 0.0):
                raise RuntimeError("a scan component returned complex detector density")
            component = np.asarray(supplied_component.real, dtype=np.float64)
            if (
                component.shape != component_shape
                or not np.all(np.isfinite(component))
                or np.any(component < 0.0)
            ):
                raise RuntimeError("a non-caustic scan component returned invalid density")
            valid_count = _validated_component_source_count(
                result,
                coordinate_shape,
                self.source_state_count,
            )
            zero_valid = valid_count == 0
            if np.any(component[zero_valid] != 0.0):
                raise RuntimeError("a source-invalid component returned nonzero density")
            aggregate_increment = float(mass) * component - aggregate_compensation
            aggregate_updated = aggregate + aggregate_increment
            aggregate_compensation[...] = (aggregate_updated - aggregate) - aggregate_increment
            aggregate[...] = aggregate_updated
            valid_increment = (
                float(mass) * valid_count / self.source_state_count - valid_compensation
            )
            valid_updated = valid_fraction + valid_increment
            valid_compensation[...] = (valid_updated - valid_fraction) - valid_increment
            valid_fraction[...] = valid_updated
            execution_identity = _consistent_component_execution(
                execution_identity,
                result,
                str(keyword["execution_backend"]),
            )

        assert execution_identity is not None
        return aggregate, valid_fraction, execution_identity[0], execution_identity[1]

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> IncidenceAngleAveragedDetectorCoordinateIntensity:
        """Stream fixed-angle evaluations into one normalized detector density."""

        column, row, keyword, aggregate_backend = _validated_evaluation_request(
            column_px,
            row_px,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )

        per_rod, valid_fraction, child_backend, child_device = self._evaluate_weighted_components(
            column,
            row,
            keyword,
            preserve_rods=True,
        )
        return IncidenceAngleAveragedDetectorCoordinateIntensity(
            column_px=column,
            row_px=row,
            rods=self.rods,
            rod_catalog_revision=self.rod_catalog_revision,
            per_rod_density_A2_per_px2=per_rod,
            density_A2_per_px2=np.sum(per_rod, axis=-1, dtype=np.float64),
            caustic=np.zeros(per_rod.shape, dtype=np.bool_),
            valid_incident_state_fraction=valid_fraction,
            source_state_count=self.source_state_count,
            source_revision=self.source_revision,
            quadrature_revision=self.quadrature.quadrature_revision,
            scan_revision=self.scan_revision,
            detector_panel_revision=self.detector_panel_revision,
            component_sample_geometry_revision=self.component_sample_geometry_revision,
            angle_evaluation_count=len(self.detectors),
            _builder_token=_RESULT_BUILDER_TOKEN,
            detector_visible_m0_q_gap_Ainv=self.detector_visible_m0_q_gap_Ainv,
            execution_backend=aggregate_backend,
            component_execution_backend=child_backend,
            execution_device=child_device,
        )

    def evaluate_detector_density_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> IncidenceAngleAveragedDetectorCoordinateDensity:
        """Stream rod-reduced child kernels for detector-native display sampling."""

        column, row, keyword, aggregate_backend = _validated_evaluation_request(
            column_px,
            row_px,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )

        density, valid_fraction, child_backend, child_device = self._evaluate_weighted_components(
            column,
            row,
            keyword,
            preserve_rods=False,
        )
        return IncidenceAngleAveragedDetectorCoordinateDensity(
            column_px=column,
            row_px=row,
            rods=self.rods,
            rod_catalog_revision=self.rod_catalog_revision,
            density_A2_per_px2=density,
            caustic=np.zeros(column.shape, dtype=np.bool_),
            valid_incident_state_fraction=valid_fraction,
            source_state_count=self.source_state_count,
            source_revision=self.source_revision,
            quadrature_revision=self.quadrature.quadrature_revision,
            scan_revision=self.scan_revision,
            detector_panel_revision=self.detector_panel_revision,
            component_sample_geometry_revision=self.component_sample_geometry_revision,
            angle_evaluation_count=len(self.detectors),
            _builder_token=_RESULT_BUILDER_TOKEN,
            detector_visible_m0_q_gap_Ainv=self.detector_visible_m0_q_gap_Ainv,
            execution_backend=aggregate_backend,
            component_execution_backend=child_backend,
            execution_device=child_device,
        )


def build_incidence_angle_averaged_detector(
    quadrature: IncidenceAngleQuadrature,
    detector_at_incidence_angle_rad: Callable[[float], _AllRootDetectorNode],
) -> IncidenceAngleAveragedDetector:
    """Build every fixed-angle component once and validate the scan contract."""

    if not isinstance(quadrature, IncidenceAngleQuadrature):
        raise TypeError("quadrature must be IncidenceAngleQuadrature")
    if not callable(detector_at_incidence_angle_rad):
        raise TypeError("detector_at_incidence_angle_rad must be callable")
    detectors = tuple(
        detector_at_incidence_angle_rad(float(angle)) for angle in quadrature.incidence_angle_rad
    )
    return IncidenceAngleAveragedDetector(quadrature, detectors)


__all__ = [
    "IncidenceAngleAveragedDetector",
    "IncidenceAngleAveragedDetectorCoordinateDensity",
    "IncidenceAngleAveragedDetectorCoordinateIntensity",
    "build_incidence_angle_averaged_detector",
]
