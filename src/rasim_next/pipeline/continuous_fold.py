"""Exact continuous-coordinate regularization of inverse-map x folds."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.measurement.continuous_regions import (
    ContinuousDetectorAreaChart,
    ContinuousRegionQuadrature,
)
from rasim_next.pipeline.source_averaged_detector import (
    SourceAveragedDetectorEwaldMeasure,
    _flatten_evaluators,
    _inverse_fold_squared_geometry,
    source_averaged_detector_geometry_revision,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    result = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result.setflags(write=False)
    return result


def _integer_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> IntArray:
    supplied = np.asarray(value)
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.shape != shape or not np.array_equal(result, supplied):
        raise ValueError(f"{name} must have shape {shape} and contain integers")
    result.setflags(write=False)
    return result


def _composite_unit_legendre_rule(order: int, subdivisions: int) -> tuple[FloatArray, FloatArray]:
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or int(order) < 2
        or isinstance(subdivisions, bool)
        or not isinstance(subdivisions, (int, np.integer))
        or int(subdivisions) < 1
    ):
        raise ValueError("continuous fold quadrature settings are invalid")
    node, weight = np.polynomial.legendre.leggauss(int(order))
    subcell = np.arange(int(subdivisions), dtype=np.float64)
    unit_node = ((subcell[:, None] + 0.5 + 0.5 * node[None, :]) / int(subdivisions)).reshape(-1)
    unit_weight = np.tile(0.5 * weight / int(subdivisions), int(subdivisions))
    return np.ascontiguousarray(unit_node), np.ascontiguousarray(unit_weight)


@dataclass(frozen=True, slots=True)
class ContinuousFoldBand:
    """One continuous radial band and its exact baseline quadrature."""

    chart: ContinuousDetectorAreaChart
    radial_interval_Ainv: tuple[float, float]
    axial_bounds: ArrayLike
    observation_row: ArrayLike
    observation_count: int
    rod_h_k: ArrayLike
    base_quadrature: ContinuousRegionQuadrature
    gauss_order: int
    radial_subdivisions: int
    axial_subdivisions: int
    revision: str = field(init=False)

    def __post_init__(self) -> None:
        chart_revision = getattr(self.chart, "revision", None)
        if not isinstance(chart_revision, str) or not chart_revision:
            raise ValueError("continuous fold chart must declare a revision")
        lower, upper = (float(value) for value in self.radial_interval_Ainv)
        if not isfinite(lower) or not isfinite(upper) or not 0.0 <= lower < upper:
            raise ValueError("continuous fold radial interval must be finite and increasing")
        supplied_axial = np.asarray(self.axial_bounds)
        axial = _float_array(self.axial_bounds, supplied_axial.shape, "axial_bounds")
        if axial.ndim != 2 or axial.shape[1:] != (2,) or np.any(axial[:, 0] >= axial[:, 1]):
            raise ValueError("axial_bounds must contain increasing rectangle intervals")
        observation = _integer_array(
            self.observation_row,
            (axial.shape[0],),
            "observation_row",
        )
        count = int(self.observation_count)
        if count < 1 or np.any((observation < 0) | (observation >= count)):
            raise ValueError("continuous fold observations lie outside their row count")
        supplied_rods = np.asarray(self.rod_h_k)
        rods = _integer_array(
            self.rod_h_k,
            supplied_rods.shape,
            "rod_h_k",
        )
        if rods.ndim != 2 or rods.shape[1:] != (2,) or not rods.size:
            raise ValueError("continuous fold rod_h_k must have shape (rod, 2)")
        if np.unique(rods, axis=0).shape[0] != rods.shape[0]:
            raise ValueError("continuous fold rod_h_k values must be unique")
        if (
            not isinstance(self.base_quadrature, ContinuousRegionQuadrature)
            or self.base_quadrature.observation_count != count
            or not np.all(np.isin(self.base_quadrature.observation_row, observation))
        ):
            raise ValueError("continuous fold baseline does not match the declared observations")
        order = int(self.gauss_order)
        radial_subdivisions = int(self.radial_subdivisions)
        axial_subdivisions = int(self.axial_subdivisions)
        _composite_unit_legendre_rule(order, radial_subdivisions)
        _composite_unit_legendre_rule(order, axial_subdivisions)
        digest = hashlib.sha256()
        for text in (
            "continuous_fold_band.v1",
            chart_revision,
            self.base_quadrature.quadrature_revision,
            str((lower, upper)),
            str(count),
            str(order),
            str(radial_subdivisions),
            str(axial_subdivisions),
        ):
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        for value in (axial, observation, rods):
            digest.update(np.ascontiguousarray(value).tobytes())
        object.__setattr__(self, "radial_interval_Ainv", (lower, upper))
        object.__setattr__(self, "axial_bounds", axial)
        object.__setattr__(self, "observation_row", observation)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "rod_h_k", rods)
        object.__setattr__(self, "gauss_order", order)
        object.__setattr__(self, "radial_subdivisions", radial_subdivisions)
        object.__setattr__(self, "axial_subdivisions", axial_subdivisions)
        object.__setattr__(self, "revision", digest.hexdigest())


@dataclass(frozen=True, slots=True)
class ContinuousFoldCorrectionPlan:
    """Geometry-frozen signed nodes replacing selected inverse-fold groups."""

    column_px: ArrayLike
    row_px: ArrayLike
    signed_detector_area_weight_px2: ArrayLike
    observation_row: ArrayLike
    evaluator_index_by_coordinate: ArrayLike
    rod_group_index_by_coordinate: ArrayLike
    group_master_rod_mask: ArrayLike
    transformed_node: ArrayLike
    observation_count: int
    detector_geometry_revision: str
    plan_sha256: str
    crossing_entry_count: int
    method_id: str = "inverse_support_fold_continuous_chart_bands.v1"

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if supplied_column.ndim != 1 or not supplied_column.size:
            raise ValueError("continuous fold plan must contain one-dimensional nodes")
        shape = supplied_column.shape
        column = _float_array(self.column_px, shape, "column_px")
        row = _float_array(self.row_px, shape, "row_px")
        weight = _float_array(
            self.signed_detector_area_weight_px2,
            shape,
            "signed_detector_area_weight_px2",
        )
        observation = _integer_array(self.observation_row, shape, "observation_row")
        evaluator = _integer_array(
            self.evaluator_index_by_coordinate,
            shape,
            "evaluator_index_by_coordinate",
        )
        group = _integer_array(
            self.rod_group_index_by_coordinate,
            shape,
            "rod_group_index_by_coordinate",
        )
        transformed = np.array(self.transformed_node, dtype=np.bool_, copy=True, order="C")
        mask = np.array(self.group_master_rod_mask, dtype=np.bool_, copy=True, order="C")
        count = int(self.observation_count)
        entries = int(self.crossing_entry_count)
        if (
            transformed.shape != shape
            or mask.ndim != 2
            or not mask.shape[0]
            or not mask.shape[1]
            or np.any(~np.any(mask, axis=1))
            or np.any(weight == 0.0)
            or np.any(weight[transformed] <= 0.0)
            or np.any(weight[~transformed] >= 0.0)
            or count < 1
            or np.any((observation < 0) | (observation >= count))
            or np.any(evaluator < 0)
            or np.any((group < 0) | (group >= mask.shape[0]))
            or entries < 1
        ):
            raise ValueError("continuous fold correction plan is inconsistent")
        for name in ("detector_geometry_revision", "plan_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty")
        if (
            len(self.plan_sha256) != 64
            or self.method_id != "inverse_support_fold_continuous_chart_bands.v1"
        ):
            raise ValueError("continuous fold plan identity is invalid")
        for value in (observation, evaluator, group, transformed, mask):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "signed_detector_area_weight_px2", weight)
        object.__setattr__(self, "observation_row", observation)
        object.__setattr__(self, "evaluator_index_by_coordinate", evaluator)
        object.__setattr__(self, "rod_group_index_by_coordinate", group)
        object.__setattr__(self, "group_master_rod_mask", mask)
        object.__setattr__(self, "transformed_node", transformed)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(self, "crossing_entry_count", entries)


@dataclass(frozen=True, slots=True)
class ContinuousFoldCorrection:
    """Observation-reduced exact-x replacement for a continuous base integral."""

    correction_A2: ArrayLike
    baseline_replaced_A2: ArrayLike
    transformed_A2: ArrayLike
    plan_sha256: str
    execution_backend: str
    execution_device: str | None
    crossing_entry_count: int
    baseline_node_count: int
    transformed_node_count: int

    def __post_init__(self) -> None:
        supplied = np.asarray(self.correction_A2)
        if supplied.ndim != 1:
            raise ValueError("continuous fold correction must be one-dimensional")
        correction = _float_array(self.correction_A2, supplied.shape, "correction_A2")
        baseline = _float_array(
            self.baseline_replaced_A2,
            supplied.shape,
            "baseline_replaced_A2",
        )
        transformed = _float_array(self.transformed_A2, supplied.shape, "transformed_A2")
        tolerance = (
            128.0
            * np.finfo(np.float64).eps
            * np.maximum.reduce((baseline, transformed, np.ones(supplied.shape, dtype=np.float64)))
        )
        if (
            np.any(baseline < 0.0)
            or np.any(transformed < 0.0)
            or np.any(np.abs(correction - (transformed - baseline)) > tolerance)
        ):
            raise ValueError("continuous fold replacement accounting is inconsistent")
        if not isinstance(self.plan_sha256, str) or len(self.plan_sha256) != 64:
            raise ValueError("continuous fold correction requires its plan identity")
        for name in (
            "crossing_entry_count",
            "baseline_node_count",
            "transformed_node_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError("continuous fold work counts must be nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "correction_A2", correction)
        object.__setattr__(self, "baseline_replaced_A2", baseline)
        object.__setattr__(self, "transformed_A2", transformed)


def _mapped_fold_geometry(
    indexed: Any,
    chart: ContinuousDetectorAreaChart,
    radial_Ainv: FloatArray,
    axial: FloatArray,
    fold_radius_Ainv: float,
    parallel_norm_Ainv: float,
) -> tuple[Any, FloatArray, FloatArray]:
    mapped = chart.map_detector_area(radial_Ainv, axial)
    if not np.all(mapped.valid):
        raise FloatingPointError("continuous fold chart left its valid detector domain")
    x_squared, w_squared, _ = _inverse_fold_squared_geometry(
        indexed.evaluator,
        np.asarray(mapped.column_px).reshape(-1),
        np.asarray(mapped.row_px).reshape(-1),
        fold_radius_Ainv,
        parallel_norm_Ainv,
    )
    return mapped, x_squared.reshape(radial_Ainv.shape), w_squared.reshape(radial_Ainv.shape)


def compile_continuous_fold_correction_plan(
    detector: SourceAveragedDetectorEwaldMeasure,
    bands: tuple[ContinuousFoldBand, ...],
) -> ContinuousFoldCorrectionPlan | None:
    """Compile source-specific exact-x nodes for continuous radial bands."""

    if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("detector must be SourceAveragedDetectorEwaldMeasure")
    indexed_evaluators = _flatten_evaluators(detector._evaluator_blocks)
    master_rod_count = len(detector.rods)
    columns: list[FloatArray] = []
    rows: list[FloatArray] = []
    weights: list[FloatArray] = []
    observations: list[IntArray] = []
    evaluators: list[IntArray] = []
    groups: list[IntArray] = []
    transformed_flags: list[BoolArray] = []
    group_masks: list[BoolArray] = []
    group_index_by_rods: dict[tuple[int, ...], int] = {}
    entry_count = 0
    machine_tolerance = 8192.0 * np.finfo(np.float64).eps

    for band in tuple(bands):
        if not isinstance(band, ContinuousFoldBand):
            raise TypeError("bands must contain ContinuousFoldBand values")
        radial_lower, radial_upper = band.radial_interval_Ainv
        master_by_h_k = {(int(rod.h), int(rod.k)): index for index, rod in enumerate(detector.rods)}
        requested_h_k = {(int(value[0]), int(value[1])) for value in np.asarray(band.rod_h_k)}
        if not requested_h_k.issubset(master_by_h_k):
            raise ValueError("continuous fold band names a rod outside the detector catalog")
        selected_master = {master_by_h_k[value] for value in requested_h_k}
        radial_unit, radial_unit_weight = _composite_unit_legendre_rule(
            band.gauss_order,
            band.radial_subdivisions,
        )
        axial_unit, axial_unit_weight = _composite_unit_legendre_rule(
            band.gauss_order,
            band.axial_subdivisions,
        )
        axial_bounds = np.asarray(band.axial_bounds)
        axial_width = np.diff(axial_bounds, axis=1)[:, 0]
        axial_node = axial_bounds[:, 0, None] + axial_width[:, None] * axial_unit[None, :]
        axial_weight = axial_width[:, None] * axial_unit_weight[None, :]

        for evaluator_index, indexed in enumerate(indexed_evaluators):
            state = indexed.evaluator.state
            inverse_groups: dict[tuple[int, int], list[int]] = {}
            for local_rod_index, master_rod_index in enumerate(indexed.master_rod_index):
                if int(master_rod_index) not in selected_master:
                    continue
                radius = np.float64(state.rod_inverse_constants[local_rod_index, 0])
                if not radial_lower < float(radius) < radial_upper:
                    continue
                parallel_norm = np.float64(state.rod_inverse_constants[local_rod_index, 1])
                key = (
                    int(radius.view(np.uint64)),
                    int(parallel_norm.view(np.uint64)),
                )
                inverse_groups.setdefault(key, []).append(local_rod_index)
            for (radius_bits, parallel_bits), local_rod_indices in sorted(inverse_groups.items()):
                fold_radius = np.asarray(radius_bits, dtype=np.uint64).view(np.float64).item()
                parallel_norm = np.asarray(parallel_bits, dtype=np.uint64).view(np.float64).item()
                radial_low = np.full(axial_node.shape, radial_lower, dtype=np.float64)
                radial_high = np.full(axial_node.shape, radial_upper, dtype=np.float64)
                _, low_g, low_w = _mapped_fold_geometry(
                    indexed,
                    band.chart,
                    radial_low,
                    axial_node,
                    fold_radius,
                    parallel_norm,
                )
                _, high_g, high_w = _mapped_fold_geometry(
                    indexed,
                    band.chart,
                    radial_high,
                    axial_node,
                    fold_radius,
                    parallel_norm,
                )
                inverse_scale = max(fold_radius**2, parallel_norm**2, 1.0)
                tolerance = machine_tolerance * inverse_scale
                low_supported = (low_g > tolerance) & (low_w > tolerance)
                high_supported = (high_g > tolerance) & (high_w > tolerance)
                crossing = ~low_supported & high_supported
                if not np.any(crossing):
                    continue
                if not np.all(crossing):
                    raise FloatingPointError(
                        "continuous fold band must bracket one outward inverse-support edge "
                        "at every axial node"
                    )
                radial_middle = np.full(
                    axial_node.shape,
                    0.5 * (radial_lower + radial_upper),
                    dtype=np.float64,
                )
                _, middle_g, middle_w = _mapped_fold_geometry(
                    indexed,
                    band.chart,
                    radial_middle,
                    axial_node,
                    fold_radius,
                    parallel_norm,
                )
                if (
                    np.any(middle_g < np.minimum(low_g, high_g) - tolerance)
                    or np.any(middle_g > np.maximum(low_g, high_g) + tolerance)
                    or np.any(middle_w < np.minimum(low_w, high_w) - tolerance)
                    or np.any(middle_w > np.maximum(low_w, high_w) + tolerance)
                ):
                    raise FloatingPointError(
                        "continuous fold coordinate is not monotone in its band"
                    )

                lower_q = np.full(axial_node.shape, radial_lower, dtype=np.float64)
                upper_q = np.full(axial_node.shape, radial_upper, dtype=np.float64)
                for _ in range(56):
                    middle_q = 0.5 * (lower_q + upper_q)
                    _, trial_g, trial_w = _mapped_fold_geometry(
                        indexed,
                        band.chart,
                        middle_q,
                        axial_node,
                        fold_radius,
                        parallel_norm,
                    )
                    trial_supported = (trial_g > 0.0) & (trial_w > 0.0)
                    lower_q = np.where(trial_supported, lower_q, middle_q)
                    upper_q = np.where(trial_supported, middle_q, upper_q)
                fold_q = upper_q
                x_extent = np.sqrt(radial_upper - fold_q)
                x = x_extent[..., None] * radial_unit[None, None, :]
                transformed_radial = fold_q[..., None] + x**2
                transformed_axial = np.broadcast_to(
                    axial_node[..., None],
                    transformed_radial.shape,
                )
                mapped, transformed_g, transformed_w = _mapped_fold_geometry(
                    indexed,
                    band.chart,
                    transformed_radial,
                    transformed_axial,
                    fold_radius,
                    parallel_norm,
                )
                if np.any(transformed_g <= 0.0) or np.any(transformed_w <= 0.0):
                    raise FloatingPointError(
                        "continuous endpoint-squared nodes are outside inverse support"
                    )
                transformed_weight = (
                    2.0
                    * x
                    * x_extent[..., None]
                    * radial_unit_weight[None, None, :]
                    * axial_weight[..., None]
                    * np.asarray(mapped.detector_area_jacobian_px2_per_chart2)
                )
                transformed_observation = np.broadcast_to(
                    np.asarray(band.observation_row)[:, None, None],
                    transformed_radial.shape,
                ).reshape(-1)

                master_group = tuple(
                    sorted(int(indexed.master_rod_index[value]) for value in local_rod_indices)
                )
                group_index = group_index_by_rods.get(master_group)
                if group_index is None:
                    group_index = len(group_masks)
                    group_index_by_rods[master_group] = group_index
                    mask = np.zeros(master_rod_count, dtype=np.bool_)
                    mask[np.asarray(master_group, dtype=np.int64)] = True
                    group_masks.append(mask)

                base = band.base_quadrature
                base_g, base_w, _ = _inverse_fold_squared_geometry(
                    indexed.evaluator,
                    np.asarray(base.column_px),
                    np.asarray(base.row_px),
                    fold_radius,
                    parallel_norm,
                )
                if np.any(np.abs(base_g) <= tolerance) or np.any(np.abs(base_w) <= tolerance):
                    raise FloatingPointError("continuous baseline contains an inverse caustic node")
                base_count = base.column_px.size
                transformed_count = transformed_radial.size
                columns.extend(
                    (
                        np.asarray(base.column_px),
                        np.asarray(mapped.column_px).reshape(-1),
                    )
                )
                rows.extend((np.asarray(base.row_px), np.asarray(mapped.row_px).reshape(-1)))
                weights.extend(
                    (
                        -np.asarray(base.detector_area_weight_px2),
                        transformed_weight.reshape(-1),
                    )
                )
                observations.extend((np.asarray(base.observation_row), transformed_observation))
                evaluators.extend(
                    (
                        np.full(base_count, evaluator_index, dtype=np.int64),
                        np.full(transformed_count, evaluator_index, dtype=np.int64),
                    )
                )
                groups.extend(
                    (
                        np.full(base_count, group_index, dtype=np.int64),
                        np.full(transformed_count, group_index, dtype=np.int64),
                    )
                )
                transformed_flags.extend(
                    (
                        np.zeros(base_count, dtype=np.bool_),
                        np.ones(transformed_count, dtype=np.bool_),
                    )
                )
                entry_count += 1

    if not columns:
        return None
    column = np.concatenate(columns)
    row = np.concatenate(rows)
    weight = np.concatenate(weights)
    observation = np.concatenate(observations)
    evaluator = np.concatenate(evaluators)
    group = np.concatenate(groups)
    transformed = np.concatenate(transformed_flags)
    group_mask = np.stack(group_masks)
    evaluator_order = np.argsort(evaluator, kind="stable")
    column = column[evaluator_order]
    row = row[evaluator_order]
    weight = weight[evaluator_order]
    observation = observation[evaluator_order]
    evaluator = evaluator[evaluator_order]
    group = group[evaluator_order]
    transformed = transformed[evaluator_order]
    digest = hashlib.sha256()
    detector_geometry_revision = source_averaged_detector_geometry_revision(detector)
    for text in (
        "inverse_support_fold_continuous_chart_bands.v1",
        detector_geometry_revision,
        str(entry_count),
        *(band.revision for band in bands),
    ):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    for value in (column, row, weight, observation, evaluator, group, group_mask, transformed):
        digest.update(np.ascontiguousarray(value).tobytes())
    return ContinuousFoldCorrectionPlan(
        column_px=column,
        row_px=row,
        signed_detector_area_weight_px2=weight,
        observation_row=observation,
        evaluator_index_by_coordinate=evaluator,
        rod_group_index_by_coordinate=group,
        group_master_rod_mask=group_mask,
        transformed_node=transformed,
        observation_count=bands[0].observation_count,
        detector_geometry_revision=detector_geometry_revision,
        plan_sha256=digest.hexdigest(),
        crossing_entry_count=entry_count,
    )


def apply_continuous_fold_correction_plan(
    detector: SourceAveragedDetectorEwaldMeasure,
    plan: ContinuousFoldCorrectionPlan,
    *,
    execution_backend: str = "cpu",
    cuda_coordinate_chunk_size: int | None = None,
) -> ContinuousFoldCorrection:
    """Evaluate and reduce one precompiled continuous fold replacement."""

    if not isinstance(plan, ContinuousFoldCorrectionPlan):
        raise TypeError("plan must be ContinuousFoldCorrectionPlan")
    if source_averaged_detector_geometry_revision(
        detector
    ) != plan.detector_geometry_revision or np.asarray(plan.group_master_rod_mask).shape[1] != len(
        detector.rods
    ):
        raise ValueError("continuous fold plan does not match the detector geometry")
    density, caustic, valid, backend_id, device = (
        detector.evaluate_selected_source_rod_groups_all_roots(
            plan.column_px,
            plan.row_px,
            plan.evaluator_index_by_coordinate,
            plan.rod_group_index_by_coordinate,
            plan.group_master_rod_mask,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )
    )
    if np.any(~valid) or np.any(caustic) or np.any(~np.isfinite(density)):
        raise FloatingPointError("continuous fold correction encountered an invalid model node")
    transformed_node = np.asarray(plan.transformed_node)
    signed_weight = np.asarray(plan.signed_detector_area_weight_px2)
    observation = np.asarray(plan.observation_row)
    transformed = np.bincount(
        observation[transformed_node],
        weights=density[transformed_node] * signed_weight[transformed_node],
        minlength=plan.observation_count,
    )
    baseline = np.bincount(
        observation[~transformed_node],
        weights=-density[~transformed_node] * signed_weight[~transformed_node],
        minlength=plan.observation_count,
    )
    return ContinuousFoldCorrection(
        correction_A2=transformed - baseline,
        baseline_replaced_A2=baseline,
        transformed_A2=transformed,
        plan_sha256=plan.plan_sha256,
        execution_backend=backend_id,
        execution_device=device,
        crossing_entry_count=plan.crossing_entry_count,
        baseline_node_count=int(np.count_nonzero(~transformed_node)),
        transformed_node_count=int(np.count_nonzero(transformed_node)),
    )


__all__ = [
    "ContinuousFoldBand",
    "ContinuousFoldCorrection",
    "ContinuousFoldCorrectionPlan",
    "apply_continuous_fold_correction_plan",
    "compile_continuous_fold_correction_plan",
]
