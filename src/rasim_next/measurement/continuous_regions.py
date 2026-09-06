"""Deterministic quadrature of continuous detector regions without a model raster."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from rasim_next.core.contracts import canonical_revision_sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
COUNT_VARIANCE_FLOOR_COUNT = 1.0


def _float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ContinuousDetectorChartAreaMeasure:
    """Detector coordinates and area Jacobian for one continuous chart."""

    column_px: ArrayLike
    row_px: ArrayLike
    detector_area_jacobian_px2_per_chart2: ArrayLike
    valid: ArrayLike

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0):
            raise ValueError("column_px must be real")
        column = np.array(supplied_column.real, dtype=np.float64, copy=True, order="C")
        shape = column.shape
        row = _float_array(self.row_px, shape, "row_px")
        jacobian = _float_array(
            self.detector_area_jacobian_px2_per_chart2,
            shape,
            "detector_area_jacobian_px2_per_chart2",
        )
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        if valid.shape != shape:
            raise ValueError("valid must align with detector chart coordinates")
        if np.any(jacobian < 0.0):
            raise ValueError("detector-area Jacobian must be nonnegative")
        for value in (column, valid):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "detector_area_jacobian_px2_per_chart2", jacobian)
        object.__setattr__(self, "valid", valid)


class ContinuousDetectorAreaChart(Protocol):
    """Map one two-coordinate chart to continuous detector area."""

    revision: str

    def map_detector_area(
        self,
        first_coordinate: FloatArray,
        second_coordinate: FloatArray,
    ) -> ContinuousDetectorChartAreaMeasure: ...


@dataclass(frozen=True, slots=True)
class ContinuousRegionQuadrature:
    """Continuous detector nodes reduced directly into observation rows."""

    column_px: ArrayLike
    row_px: ArrayLike
    detector_area_weight_px2: ArrayLike
    observation_row: ArrayLike
    background_coordinate: ArrayLike
    observation_count: int
    chart_revision: str
    quadrature_revision: str = field(init=False)

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if supplied_column.ndim != 1:
            raise ValueError("continuous region coordinates must be one-dimensional")
        column = _float_array(self.column_px, supplied_column.shape, "column_px")
        row = _float_array(self.row_px, column.shape, "row_px")
        weight = _float_array(
            self.detector_area_weight_px2,
            column.shape,
            "detector_area_weight_px2",
        )
        coordinate = _float_array(
            self.background_coordinate,
            column.shape,
            "background_coordinate",
        )
        observation = np.array(self.observation_row, dtype=np.int64, copy=True, order="C")
        count = self.observation_count
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 1:
            raise ValueError("observation_count must be a positive integer")
        count = int(count)
        if (
            not column.size
            or observation.shape != column.shape
            or np.any(weight <= 0.0)
            or np.any((observation < 0) | (observation >= count))
        ):
            raise ValueError("continuous region nodes and observation rows are invalid")
        if not isinstance(self.chart_revision, str) or not self.chart_revision:
            raise ValueError("chart_revision must be nonempty")
        observation.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "detector_area_weight_px2", weight)
        object.__setattr__(self, "observation_row", observation)
        object.__setattr__(self, "background_coordinate", coordinate)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(
            self,
            "quadrature_revision",
            canonical_revision_sha256(
                ("definition_id", "continuous_detector_region_quadrature.v1"),
                ("chart_revision", self.chart_revision),
                ("column_px", column),
                ("row_px", row),
                ("detector_area_weight_px2", weight),
                ("observation_row", observation),
                ("background_coordinate", coordinate),
                ("observation_count", count),
            ),
        )

    @property
    def observation_measure_px2(self) -> FloatArray:
        """Return the continuous detector area owned by each row."""

        result = np.bincount(
            self.observation_row,
            weights=self.detector_area_weight_px2,
            minlength=self.observation_count,
        ).astype(np.float64, copy=False)
        result.setflags(write=False)
        return result

    @property
    def observation_background_coordinate(self) -> FloatArray:
        """Return the detector-area-weighted chart coordinate of each row."""

        measure = self.observation_measure_px2
        moment = np.bincount(
            self.observation_row,
            weights=self.detector_area_weight_px2 * self.background_coordinate,
            minlength=self.observation_count,
        )
        result = np.zeros(self.observation_count, dtype=np.float64)
        np.divide(moment, measure, out=result, where=measure > 0.0)
        result.setflags(write=False)
        return result

    @property
    def observation_covered(self) -> BoolArray:
        """Return whether each declared observation owns positive detector area."""

        result = self.observation_measure_px2 > 0.0
        result.setflags(write=False)
        return result

    def integrate_density(self, density_A2_per_px2: ArrayLike) -> FloatArray:
        """Integrate one detector-coordinate density directly into observation rows."""

        density = _float_array(density_A2_per_px2, self.column_px.shape, "density_A2_per_px2")
        if np.any(density < 0.0):
            raise ValueError("density_A2_per_px2 must be nonnegative")
        result = np.bincount(
            self.observation_row,
            weights=self.detector_area_weight_px2 * density,
            minlength=self.observation_count,
        ).astype(np.float64, copy=False)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class NativePixelRegionProjection:
    """Sparse continuous-region projection of a native pixel count field."""

    detector_shape_rc: tuple[int, int]
    flat_pixel_index: ArrayLike
    observation_row: ArrayLike
    pixel_column_index: ArrayLike
    detector_area_weight_px2: ArrayLike
    observation_count: int
    quadrature_revision: str
    projection_revision: str = field(init=False)

    def __post_init__(self) -> None:
        shape_rc = tuple(self.detector_shape_rc)
        if len(shape_rc) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1
            for value in shape_rc
        ):
            raise ValueError("detector_shape_rc must contain positive integers")
        rows, columns = (int(value) for value in shape_rc)
        flat_pixel = np.array(self.flat_pixel_index, dtype=np.int64, copy=True, order="C")
        observation = np.array(self.observation_row, dtype=np.int64, copy=True, order="C")
        pixel_column = np.array(self.pixel_column_index, dtype=np.int64, copy=True, order="C")
        weight = np.array(self.detector_area_weight_px2, dtype=np.float64, copy=True, order="C")
        shape = observation.shape
        if isinstance(self.observation_count, bool) or not isinstance(
            self.observation_count, (int, np.integer)
        ):
            raise ValueError("observation_count must be a positive integer")
        count = int(self.observation_count)
        if (
            flat_pixel.ndim != 1
            or not flat_pixel.size
            or not np.array_equal(flat_pixel, np.unique(flat_pixel))
            or np.any((flat_pixel < 0) | (flat_pixel >= rows * columns))
            or pixel_column.shape != shape
            or weight.shape != shape
            or np.any((observation < 0) | (observation >= count))
            or np.any((pixel_column < 0) | (pixel_column >= flat_pixel.size))
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or np.unique(observation * flat_pixel.size + pixel_column).size != observation.size
            or count < 1
            or not isinstance(self.quadrature_revision, str)
            or not self.quadrature_revision
        ):
            raise ValueError("native-pixel region projection is invalid")
        for value in (flat_pixel, observation, pixel_column, weight):
            value.setflags(write=False)
        object.__setattr__(self, "detector_shape_rc", (int(rows), int(columns)))
        object.__setattr__(self, "flat_pixel_index", flat_pixel)
        object.__setattr__(self, "observation_row", observation)
        object.__setattr__(self, "pixel_column_index", pixel_column)
        object.__setattr__(self, "detector_area_weight_px2", weight)
        object.__setattr__(self, "observation_count", count)
        object.__setattr__(
            self,
            "projection_revision",
            canonical_revision_sha256(
                ("definition_id", "piecewise_constant_native_pixel_region_projection.v2"),
                ("detector_shape_rc", np.asarray((rows, columns), dtype=np.int64)),
                ("flat_pixel_index", flat_pixel),
                ("observation_row", observation),
                ("pixel_column_index", pixel_column),
                ("detector_area_weight_px2", weight),
                ("observation_count", count),
                ("quadrature_revision", self.quadrature_revision),
                ("count_variance_floor_count", COUNT_VARIANCE_FLOOR_COUNT),
            ),
        )

    @property
    def observation_measure_px2(self) -> FloatArray:
        result = np.bincount(
            self.observation_row,
            weights=self.detector_area_weight_px2,
            minlength=self.observation_count,
        ).astype(np.float64, copy=False)
        result.setflags(write=False)
        return result

    def integrate_counts(self, detector_native_counts: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Return projected count mass and regularized plug-in count covariance.

        Native counts are the piecewise-constant measured field. This data
        projection neither smooths the image nor rasterizes the diffraction
        model. The declared one-count variance floor keeps empty measured
        pixels finite in the downstream whitening operator.
        """

        supplied = np.asarray(detector_native_counts)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError("detector_native_counts must be real")
        counts = np.asarray(supplied.real, dtype=np.float64)
        if (
            counts.shape != self.detector_shape_rc
            or np.any(~np.isfinite(counts))
            or np.any(counts < 0.0)
        ):
            raise ValueError("detector_native_counts must be a finite nonnegative native image")
        return self.integrate_field(counts, np.maximum(counts, COUNT_VARIANCE_FLOOR_COUNT))

    def integrate_field(
        self,
        detector_native_value: ArrayLike,
        detector_native_variance: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Project a signed native field with an explicit independent-pixel variance."""

        value = _float_array(
            detector_native_value,
            self.detector_shape_rc,
            "detector_native_value",
        )
        variance = _float_array(
            detector_native_variance,
            self.detector_shape_rc,
            "detector_native_variance",
        )
        if np.any(variance < 0.0):
            raise ValueError("detector_native_variance must be nonnegative")
        selected_value = value.reshape(-1)[self.flat_pixel_index]
        selected_variance = variance.reshape(-1)[self.flat_pixel_index]
        projector = csr_matrix(
            (
                self.detector_area_weight_px2,
                (self.observation_row, self.pixel_column_index),
            ),
            shape=(self.observation_count, self.flat_pixel_index.size),
        )
        return _project_native_pixel_field(projector, selected_value, selected_variance)


def _project_native_pixel_field(
    projector: csr_matrix,
    value: FloatArray,
    variance: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply one linear pixel projection to its mean and independent-pixel variance."""

    mass = np.asarray(projector @ value, dtype=np.float64)
    weighted = projector.multiply(np.sqrt(variance)[None, :])
    covariance = np.asarray((weighted @ weighted.T).toarray(), dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    mass.setflags(write=False)
    covariance.setflags(write=False)
    return mass, covariance


def integrate_shared_native_pixel_field(
    projections: Sequence[NativePixelRegionProjection],
    detector_native_value: ArrayLike,
    detector_native_variance: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Project one shared native field and retain cross-projection covariance."""

    selected = tuple(projections)
    if not selected or any(
        not isinstance(value, NativePixelRegionProjection) for value in selected
    ):
        raise ValueError("projections must contain at least one native-pixel projection")
    detector_shape = selected[0].detector_shape_rc
    if any(value.detector_shape_rc != detector_shape for value in selected[1:]):
        raise ValueError("shared-field projections must use one detector shape")
    value = _float_array(detector_native_value, detector_shape, "detector_native_value")
    variance = _float_array(
        detector_native_variance,
        detector_shape,
        "detector_native_variance",
    )
    if np.any(variance < 0.0):
        raise ValueError("detector_native_variance must be nonnegative")

    shared_flat_pixel = np.unique(
        np.concatenate([np.asarray(projection.flat_pixel_index) for projection in selected])
    )
    observation_offset = 0
    observation_rows: list[IntArray] = []
    pixel_columns: list[IntArray] = []
    weights: list[FloatArray] = []
    for projection in selected:
        local_to_shared = np.searchsorted(shared_flat_pixel, projection.flat_pixel_index)
        observation_rows.append(np.asarray(projection.observation_row) + observation_offset)
        pixel_columns.append(local_to_shared[np.asarray(projection.pixel_column_index)])
        weights.append(np.asarray(projection.detector_area_weight_px2))
        observation_offset += projection.observation_count
    projector = csr_matrix(
        (
            np.concatenate(weights),
            (np.concatenate(observation_rows), np.concatenate(pixel_columns)),
        ),
        shape=(observation_offset, shared_flat_pixel.size),
    )
    flat_value = value.reshape(-1)[shared_flat_pixel]
    flat_variance = variance.reshape(-1)[shared_flat_pixel]
    return _project_native_pixel_field(projector, flat_value, flat_variance)


def compile_native_pixel_region_projection(
    quadrature: ContinuousRegionQuadrature,
    detector_shape_rc: tuple[int, int],
) -> NativePixelRegionProjection:
    """Reduce continuous nodes into unique observation/native-pixel overlaps."""

    if not isinstance(quadrature, ContinuousRegionQuadrature):
        raise TypeError("quadrature must be ContinuousRegionQuadrature")
    if len(detector_shape_rc) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1
        for value in detector_shape_rc
    ):
        raise ValueError("detector_shape_rc must contain two positive integers")
    rows, columns = (int(value) for value in detector_shape_rc)
    pixel_column = np.floor(np.asarray(quadrature.column_px) + 0.5).astype(np.int64)
    pixel_row = np.floor(np.asarray(quadrature.row_px) + 0.5).astype(np.int64)
    if np.any((pixel_column < 0) | (pixel_column >= columns)) or np.any(
        (pixel_row < 0) | (pixel_row >= rows)
    ):
        raise ValueError("continuous quadrature contains nodes outside the native detector")
    flat_pixel = pixel_row * columns + pixel_column
    observation = np.asarray(quadrature.observation_row, dtype=np.int64)
    pair_key = observation * (rows * columns) + flat_pixel
    order = np.argsort(pair_key, kind="stable")
    sorted_key = pair_key[order]
    boundary = np.concatenate((np.asarray((0,)), np.flatnonzero(np.diff(sorted_key)) + 1))
    unique_key = sorted_key[boundary]
    pair_weight = np.add.reduceat(
        np.asarray(quadrature.detector_area_weight_px2)[order],
        boundary,
    )
    pair_observation, pair_flat_pixel = np.divmod(unique_key, rows * columns)
    used_flat_pixel, compact_pixel = np.unique(pair_flat_pixel, return_inverse=True)
    return NativePixelRegionProjection(
        detector_shape_rc=(rows, columns),
        flat_pixel_index=used_flat_pixel,
        observation_row=pair_observation,
        pixel_column_index=compact_pixel,
        detector_area_weight_px2=pair_weight,
        observation_count=quadrature.observation_count,
        quadrature_revision=quadrature.quadrature_revision,
    )


def compile_continuous_rectangle_quadrature(
    *,
    chart: ContinuousDetectorAreaChart,
    first_coordinate_bounds: ArrayLike,
    second_coordinate_bounds: ArrayLike,
    observation_row: ArrayLike,
    observation_count: int,
    gauss_order: int,
    background_coordinate_axis: int,
    subdivision_count: int | tuple[int, int] = 1,
    first_coordinate_squared_fold_center: ArrayLike | None = None,
) -> ContinuousRegionQuadrature:
    """Compile tensor Gauss nodes for arbitrary continuous chart rectangles.

    ``first_coordinate_squared_fold_center`` optionally changes variables on
    each first-coordinate interval around an interior radial fold ``b`` using
    ``x**2 = abs(q**2 - b**2)``.  This is an exact substitution of the
    continuous chart-area integral; it does not introduce detector pixels.
    """

    first = np.asarray(first_coordinate_bounds, dtype=np.float64)
    second = np.asarray(second_coordinate_bounds, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape:
        raise ValueError("continuous rectangle bounds must align with shape (region, 2)")
    if (
        not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
        or np.any(first[:, 0] >= first[:, 1])
        or np.any(second[:, 0] >= second[:, 1])
    ):
        raise ValueError("continuous rectangle bounds must be finite and increasing")
    region_count = first.shape[0]
    row = np.asarray(observation_row, dtype=np.int64)
    if row.shape != (region_count,):
        raise ValueError("observation_row must contain one row per rectangle")
    if (
        isinstance(gauss_order, bool)
        or not isinstance(gauss_order, (int, np.integer))
        or int(gauss_order) < 2
    ):
        raise ValueError("gauss_order must be an integer of at least two")
    if background_coordinate_axis not in {0, 1}:
        raise ValueError("background_coordinate_axis must be zero or one")
    if isinstance(subdivision_count, bool):
        raise ValueError("subdivision_count must contain positive integers")
    if isinstance(subdivision_count, (int, np.integer)):
        subdivision_by_axis = (int(subdivision_count), int(subdivision_count))
    else:
        subdivision_by_axis = tuple(subdivision_count)
        if len(subdivision_by_axis) != 2:
            raise ValueError("subdivision_count must have one value per chart axis")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in subdivision_by_axis
        ):
            raise ValueError("subdivision_count must contain positive integers")
        subdivision_by_axis = tuple(int(value) for value in subdivision_by_axis)
    if any(value < 1 for value in subdivision_by_axis):
        raise ValueError("subdivision_count must contain positive integers")
    fold_center = None
    if first_coordinate_squared_fold_center is not None:
        fold_center = np.asarray(first_coordinate_squared_fold_center, dtype=np.float64)
        if (
            fold_center.shape != (region_count,)
            or not np.all(np.isfinite(fold_center))
            or np.any(first[:, 0] < 0.0)
            or np.any(fold_center <= first[:, 0])
            or np.any(fold_center >= first[:, 1])
        ):
            raise ValueError(
                "each squared-radial fold center must be finite and strictly inside a "
                "nonnegative first-coordinate interval"
            )
    revision = getattr(chart, "revision", None)
    if not isinstance(revision, str) or not revision:
        raise ValueError("chart must declare a nonempty revision")

    node, node_weight = np.polynomial.legendre.leggauss(int(gauss_order))
    composite_rules = []
    for subdivision in subdivision_by_axis:
        subcell = np.arange(subdivision, dtype=np.float64)
        composite_rules.append(
            (
                ((subcell[:, None] + 0.5 + 0.5 * node[None, :]) / subdivision).reshape(-1),
                np.tile(0.5 * node_weight / subdivision, subdivision),
            )
        )
    (
        (first_composite_node, first_composite_weight),
        (
            second_composite_node,
            second_composite_weight,
        ),
    ) = composite_rules
    first_width = np.diff(first, axis=1)[:, 0]
    second_width = np.diff(second, axis=1)[:, 0]
    if fold_center is None:
        first_node_2d = first[:, 0, None] + first_width[:, None] * first_composite_node[None, :]
        first_weight_2d = first_width[:, None] * first_composite_weight[None, :]
    else:
        lower_extent = np.sqrt((fold_center - first[:, 0]) * (fold_center + first[:, 0]))
        upper_extent = np.sqrt((first[:, 1] - fold_center) * (first[:, 1] + fold_center))
        lower_x = lower_extent[:, None] * first_composite_node[None, :]
        upper_x = upper_extent[:, None] * first_composite_node[None, :]
        lower_node = np.sqrt(fold_center[:, None] ** 2 - lower_x**2)
        upper_node = np.sqrt(fold_center[:, None] ** 2 + upper_x**2)
        lower_weight = (
            lower_extent[:, None] * first_composite_weight[None, :] * lower_x / lower_node
        )
        upper_weight = (
            upper_extent[:, None] * first_composite_weight[None, :] * upper_x / upper_node
        )
        first_node_2d = np.concatenate((lower_node, upper_node), axis=1)
        first_weight_2d = np.concatenate((lower_weight, upper_weight), axis=1)
    first_node, second_node = np.broadcast_arrays(
        first_node_2d[:, :, None],
        second[:, 0, None, None]
        + second_width[:, None, None] * second_composite_node[None, None, :],
    )
    chart_weight = (
        first_weight_2d[:, :, None]
        * second_width[:, None, None]
        * second_composite_weight[None, None, :]
    )
    mapped = chart.map_detector_area(first_node, second_node)
    if np.asarray(mapped.column_px).shape != first_node.shape:
        raise ValueError("chart mapping must preserve the supplied coordinate shape")
    detector_weight = chart_weight * mapped.detector_area_jacobian_px2_per_chart2
    selected = np.asarray(mapped.valid) & (detector_weight > 0.0)
    node_row = np.broadcast_to(row[:, None, None], first_node.shape)
    background_coordinate = first_node if background_coordinate_axis == 0 else second_node
    return ContinuousRegionQuadrature(
        column_px=np.asarray(mapped.column_px)[selected],
        row_px=np.asarray(mapped.row_px)[selected],
        detector_area_weight_px2=detector_weight[selected],
        observation_row=node_row[selected],
        background_coordinate=background_coordinate[selected],
        observation_count=observation_count,
        chart_revision=revision,
    )


__all__ = [
    "COUNT_VARIANCE_FLOOR_COUNT",
    "ContinuousDetectorAreaChart",
    "ContinuousDetectorChartAreaMeasure",
    "ContinuousRegionQuadrature",
    "NativePixelRegionProjection",
    "compile_continuous_rectangle_quadrature",
    "compile_native_pixel_region_projection",
]
