"""Incoherent incident-state average of the continuous detector field."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import BraggSpaceConfig, MosaicParameters, Rod
from rasim_next.core.contracts import MaterialOptics
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.pipeline._continuous_detector_kernel import (
    CompiledDetectorEvaluator,
    pack_bi2se3_two_h_structure,
)
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.pipeline.continuous_detector import (
    DetectorPixelMass,
    DetectorQuadrature,
    PixelIntegrationMethod,
    _compile_detector_state,
    _positive_integer,
    _subdivided_legendre_rule,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SourceAveragedDetectorCoordinateIntensity:
    """Detector-coordinate density summed over independent source states.

    A coordinate selects a different outgoing ray for every sampled source
    position. There is therefore no representative ``kf`` or ``Q`` attached
    to this reduced result; those state-specific values are evaluated before
    the incoherent intensity sum.
    """

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    branch: int
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.int64]
    source_state_count: int
    source_revision: str
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0):
            raise ValueError("column_px must be real")
        column = np.array(
            supplied_column.real,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        shape = column.shape
        if not np.all(np.isfinite(column)):
            raise ValueError("column_px must be finite")
        row = _float_array(self.row_px, shape, "row_px")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        if self.branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        supplied_per_rod = np.asarray(self.per_rod_density_A2_per_px2)
        if np.iscomplexobj(supplied_per_rod) and np.any(supplied_per_rod.imag != 0.0):
            raise ValueError("per-rod detector density must be real")
        per_rod = np.array(
            supplied_per_rod.real,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if per_rod.shape != (*shape, len(rods)) or np.any(np.isnan(per_rod)):
            raise ValueError("per-rod detector density has the wrong shape or contains NaN")
        if np.any(per_rod < 0.0):
            raise ValueError("per-rod detector density must be nonnegative")
        supplied_total = np.asarray(self.density_A2_per_px2)
        if np.iscomplexobj(supplied_total) and np.any(supplied_total.imag != 0.0):
            raise ValueError("detector density must be real")
        total = np.array(supplied_total.real, dtype=np.float64, copy=True, order="C")
        if total.shape != shape or np.any(np.isnan(total)) or np.any(total < 0.0):
            raise ValueError("detector density must be nonnegative and contain no NaN")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        finite = np.isfinite(expected)
        scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        if not np.all(
            np.abs(total[finite] - expected[finite]) <= 1024.0 * np.finfo(np.float64).eps * scale
        ) or not np.array_equal(np.isinf(total), np.isinf(expected)):
            raise ValueError("detector density must equal the physical rod sum")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != (*shape, len(rods)):
            raise ValueError("caustic must have one flag per detector coordinate and rod")
        state_count = _positive_integer(self.source_state_count, "source_state_count")
        valid_count = np.array(self.valid_source_count, dtype=np.int64, copy=True, order="C")
        if valid_count.shape != shape or np.any((valid_count < 0) | (valid_count > state_count)):
            raise ValueError("valid_source_count must lie within the source batch")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate measure")
        for value in (column, row, per_rod, total, caustic, valid_count):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_density_A2_per_px2", per_rod)
        object.__setattr__(self, "density_A2_per_px2", total)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "valid_source_count", valid_count)
        object.__setattr__(self, "source_state_count", state_count)


def _sum_compiled_evaluator_block(
    evaluators: tuple[CompiledDetectorEvaluator, ...],
    column_px: FloatArray,
    row_px: FloatArray,
    branch: int,
) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
    """Sum one fixed source-state block without materializing a state axis."""

    per_rod: FloatArray | None = None
    caustic: BoolArray | None = None
    valid_source_count = np.zeros(column_px.size, dtype=np.int64)
    for evaluator in evaluators:
        density, _, state_caustic, state_valid = evaluator.evaluate(
            column_px,
            row_px,
            branch=branch,
        )
        if per_rod is None:
            per_rod = density
            caustic = state_caustic
        else:
            per_rod += density
            caustic |= state_caustic
        valid_source_count += state_valid
    if per_rod is None or caustic is None:
        raise ValueError("a source-state block must contain at least one evaluator")
    return per_rod, caustic, valid_source_count


class SourceAveragedDetectorEwaldMeasure:
    """Continuous detector density after an incoherent incident-state sum.

    Every valid state retains its source position, wavelength, refracted
    ``ki``, outgoing detector ray, exit refraction, attenuation, and
    Ewald-constrained ``Q``. The detector-coordinate densities are summed with
    their empirical source masses before pixel quadrature. This is the exact
    common-domain reduction when sampled source positions differ; no
    representative ``kf`` or per-state detector image is formed.
    """

    __slots__ = (
        "_evaluator_blocks",
        "_incident",
        "_instrument",
        "_rods",
        "_valid_state_count",
        "_worker_count",
    )

    _MAX_STATE_BLOCK_COUNT = 32
    _COORDINATE_CHUNK_SIZE = 16_384

    def __init__(
        self,
        *,
        reciprocal_basis_Ainv: ArrayLike,
        crystal_to_sample: ArrayLike,
        rods: tuple[Rod, ...],
        mosaic: MosaicParameters,
        strength_model: Bi2Se3TwoHStrength,
        incident: IncidentTransportResult,
        material: MaterialOptics,
        instrument: CompiledInstrument,
        phase_population_weight: float = 1.0,
        polarization_weight: float = 1.0,
        worker_count: int = 1,
    ) -> None:
        if not isinstance(incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        if not isinstance(mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(strength_model, Bi2Se3TwoHStrength):
            raise TypeError("strength_model must be Bi2Se3TwoHStrength")
        if mosaic.zero_tilt_probability_mass != 0.0:
            raise ValueError("source-averaged integration does not support zero-tilt atoms")
        selected = tuple(rods)
        if not selected or not all(isinstance(rod, Rod) for rod in selected):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in selected}) != len(selected):
            raise ValueError("rods must not repeat a physical rod")
        if any(rod.family_m == 0 for rod in selected):
            raise ValueError("m=0 intensity is excluded without physical direct-beam support")
        phase_weight = float(phase_population_weight)
        polarization = float(polarization_weight)
        if not isfinite(phase_weight) or phase_weight < 0.0:
            raise ValueError("phase_population_weight must be finite and nonnegative")
        if not isfinite(polarization) or polarization < 0.0:
            raise ValueError("polarization_weight must be finite and nonnegative")
        workers = _positive_integer(worker_count, "worker_count")
        states = incident.states
        if states.material_revision != material.material_revision:
            raise ValueError("incident states and material must have the same revision")
        if states.sample_geometry_revision != instrument.sample_geometry_revision:
            raise ValueError("incident states and instrument must have the same sample geometry")
        supplied_rotation = np.asarray(crystal_to_sample)
        if np.iscomplexobj(supplied_rotation) and np.any(supplied_rotation.imag != 0.0):
            raise ValueError("crystal_to_sample must be real")
        supplied_crystal_to_sample = np.asarray(supplied_rotation.real, dtype=np.float64)
        if supplied_crystal_to_sample.shape != (3, 3) or not np.all(
            np.isfinite(supplied_crystal_to_sample)
        ):
            raise ValueError("crystal_to_sample must be finite with shape (3, 3)")
        reference_config = BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal_basis_Ainv,
            crystal_to_sample=supplied_crystal_to_sample,
            rods=selected,
            mosaic=mosaic,
            k_norm_Ainv=2.0 * np.pi / float(states.wavelength_A[0]),
        )
        rotation_tolerance = 512.0 * np.finfo(np.float64).eps
        if not np.allclose(
            reference_config.crystal_to_sample,
            instrument.sample_from_crystal.rotation,
            rtol=0.0,
            atol=rotation_tolerance,
        ):
            raise ValueError("crystal_to_sample must be the canonical instrument rotation")
        basis_scale = max(float(np.linalg.norm(reference_config.reciprocal_basis_Ainv)), 1.0)
        if not np.allclose(
            strength_model.reciprocal_basis_Ainv,
            reference_config.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * basis_scale,
        ):
            raise ValueError("strength-model and Bragg-space reciprocal bases do not match")

        evaluators: list[CompiledDetectorEvaluator] = []
        for state_index in np.flatnonzero(states.valid):
            wavelength_A = float(states.wavelength_A[state_index])
            bragg_config = BraggSpaceConfig(
                reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
                crystal_to_sample=reference_config.crystal_to_sample,
                rods=selected,
                mosaic=mosaic,
                k_norm_Ainv=2.0 * np.pi / wavelength_A,
            )
            source_phase_weight = float(
                states.source_weight[state_index]
                * states.footprint_acceptance[state_index]
                * phase_weight
                * polarization
            )
            packed = pack_bi2se3_two_h_structure(
                strength_model,
                wavelength_A=wavelength_A,
            )
            evaluators.append(
                CompiledDetectorEvaluator(
                    _compile_detector_state(
                        bragg_config=bragg_config,
                        strength_model=strength_model,
                        ki_sample_Ainv=states.k_film_phase_sample_Ainv[state_index],
                        incident=incident,
                        material=material,
                        instrument=instrument,
                        rods=selected,
                        incident_state_index=int(state_index),
                        source_phase_weight=source_phase_weight,
                        packed_structure=packed,
                    ),
                    instrument.detector_shape_rc,
                )
            )
        block_size = max(
            1,
            (len(evaluators) + self._MAX_STATE_BLOCK_COUNT - 1) // self._MAX_STATE_BLOCK_COUNT,
        )
        blocks = tuple(
            tuple(evaluators[start : start + block_size])
            for start in range(0, len(evaluators), block_size)
        )
        object.__setattr__(self, "_evaluator_blocks", blocks)
        object.__setattr__(self, "_incident", incident)
        object.__setattr__(self, "_instrument", instrument)
        object.__setattr__(self, "_rods", selected)
        object.__setattr__(self, "_valid_state_count", len(evaluators))
        object.__setattr__(self, "_worker_count", workers)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SourceAveragedDetectorEwaldMeasure is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SourceAveragedDetectorEwaldMeasure is immutable")

    @property
    def incident(self) -> IncidentTransportResult:
        return self._incident

    @property
    def instrument(self) -> CompiledInstrument:
        return self._instrument

    @property
    def rods(self) -> tuple[Rod, ...]:
        return self._rods

    @property
    def source_state_count(self) -> int:
        return int(self._incident.states.incident_state_id.size)

    @property
    def valid_source_state_count(self) -> int:
        return self._valid_state_count

    def _thread_pool(self) -> ThreadPoolExecutor | None:
        if self._worker_count <= 1 or len(self._evaluator_blocks) <= 1:
            return None
        return ThreadPoolExecutor(max_workers=min(self._worker_count, len(self._evaluator_blocks)))

    def _evaluate_flat_coordinates(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        *,
        branch: int,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
        per_rod = np.zeros((column_px.size, len(self._rods)), dtype=np.float64)
        caustic = np.zeros(per_rod.shape, dtype=np.bool_)
        valid_source_count = np.zeros(column_px.size, dtype=np.int64)
        for start in range(0, column_px.size, self._COORDINATE_CHUNK_SIZE):
            stop = min(start + self._COORDINATE_CHUNK_SIZE, column_px.size)
            chunk_column = column_px[start:stop]
            chunk_row = row_px[start:stop]
            if executor is None:
                block_results = (
                    _sum_compiled_evaluator_block(
                        block,
                        chunk_column,
                        chunk_row,
                        branch,
                    )
                    for block in self._evaluator_blocks
                )
            else:
                futures = tuple(
                    executor.submit(
                        _sum_compiled_evaluator_block,
                        block,
                        chunk_column,
                        chunk_row,
                        branch,
                    )
                    for block in self._evaluator_blocks
                )
                block_results = (future.result() for future in futures)
            for block_density, block_caustic, block_valid_count in block_results:
                per_rod[start:stop] += block_density
                caustic[start:stop] |= block_caustic
                valid_source_count[start:stop] += block_valid_count
        return per_rod, caustic, valid_source_count

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        branch: int = 2,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate the source-averaged density at arbitrary detector coordinates."""

        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
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
        shape = column.shape
        flat_column = np.ascontiguousarray(column.reshape(-1))
        flat_row = np.ascontiguousarray(row.reshape(-1))
        if self._evaluator_blocks and flat_column.size:
            self._evaluator_blocks[0][0].evaluate(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                branch=branch,
            )
        executor = self._thread_pool()
        try:
            per_rod, caustic, valid_source_count = self._evaluate_flat_coordinates(
                flat_column,
                flat_row,
                branch=branch,
                executor=executor,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        reshaped_per_rod = per_rod.reshape((*shape, len(self._rods)))
        return SourceAveragedDetectorCoordinateIntensity(
            column_px=column,
            row_px=row,
            rods=self._rods,
            branch=branch,
            per_rod_density_A2_per_px2=reshaped_per_rod,
            density_A2_per_px2=np.sum(reshaped_per_rod, axis=-1, dtype=np.float64),
            caustic=caustic.reshape((*shape, len(self._rods))),
            valid_source_count=valid_source_count.reshape(shape),
            source_state_count=self.source_state_count,
            source_revision=self._incident.states.source_revision,
        )

    def integrate_native_pixels(
        self,
        *,
        branch: int,
        quadrature: DetectorQuadrature,
    ) -> DetectorPixelMass:
        """Apply one tensor quadrature to the already-summed continuous field."""

        if not isinstance(quadrature, DetectorQuadrature):
            raise TypeError("quadrature must be DetectorQuadrature")
        if quadrature.method is not PixelIntegrationMethod.FIXED_NUMPY:
            raise ValueError("source-averaged pixel integration requires fixed_numpy")
        if (
            quadrature.fold_gauss_order != quadrature.pixel_gauss_order
            or quadrature.fold_subdivision_count != 1
        ):
            raise ValueError("source-averaged integration does not apply a second fold rule")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        rows, columns = self._instrument.detector_shape_rc
        offset, one_dimensional_weight = _subdivided_legendre_rule(
            quadrature.pixel_gauss_order,
            1,
        )
        node_weight = one_dimensional_weight[:, None] * one_dimensional_weight[None, :]
        image = np.zeros((rows, columns), dtype=np.float64)
        per_rod_mass = np.zeros(len(self._rods), dtype=np.float64)
        column_center = np.arange(columns, dtype=np.float64)
        if self._evaluator_blocks:
            self._evaluator_blocks[0][0].evaluate(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                branch=branch,
            )
        executor = self._thread_pool()
        try:
            for row_start in range(0, rows, quadrature.row_chunk_size):
                row_stop = min(row_start + quadrature.row_chunk_size, rows)
                row_center = np.arange(row_start, row_stop, dtype=np.float64)
                node_column, node_row = np.broadcast_arrays(
                    column_center[None, :, None, None] + offset[None, None, None, :],
                    row_center[:, None, None, None] + offset[None, None, :, None],
                )
                node_shape = node_column.shape
                flat_density, flat_caustic, _ = self._evaluate_flat_coordinates(
                    np.ascontiguousarray(node_column.reshape(-1)),
                    np.ascontiguousarray(node_row.reshape(-1)),
                    branch=branch,
                    executor=executor,
                )
                if np.any(flat_caustic):
                    raise FloatingPointError(
                        "a pixel quadrature node lies on a caustic; choose another even order"
                    )
                tile_per_rod = np.sum(
                    flat_density.reshape((*node_shape, len(self._rods)))
                    * node_weight[None, None, :, :, None],
                    axis=(2, 3),
                    dtype=np.float64,
                )
                image[row_start:row_stop] = np.sum(
                    tile_per_rod,
                    axis=-1,
                    dtype=np.float64,
                )
                per_rod_mass += np.sum(tile_per_rod, axis=(0, 1), dtype=np.float64)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        return DetectorPixelMass(
            image_A2=image,
            rods=self._rods,
            branch=branch,
            per_rod_detector_mass_A2=per_rod_mass,
            total_detector_mass_A2=fsum(per_rod_mass),
            quadrature=quadrature,
            coordinate_evaluation_count=rows * columns * quadrature.pixel_gauss_order**2,
            execution_backend="numba_source_averaged.v1",
        )


__all__ = [
    "SourceAveragedDetectorCoordinateIntensity",
    "SourceAveragedDetectorEwaldMeasure",
]
