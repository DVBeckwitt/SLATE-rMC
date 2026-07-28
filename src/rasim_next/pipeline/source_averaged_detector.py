"""Incoherent incident-state average of the continuous detector field."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import BraggSpaceConfig, MosaicParameters, Rod
from painted_ewald.rotations import mosaic_axes
from painted_ewald.validation import positive_integer
from rasim_next.core.contracts import MaterialOptics
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.optics import mode_decay_constant
from rasim_next.pipeline._continuous_detector_kernel import (
    CompiledDetectorEvaluator,
    pack_bi2se3_two_h_structures,
)
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength
from rasim_next.pipeline.continuous_detector import (
    DetectorPixelMass,
    DetectorQuadrature,
    PixelIntegrationMethod,
    _compile_detector_state,
    _float_array,
    _subdivided_legendre_rule,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _validated_cuda_coordinate_chunk_size(
    execution_backend: str,
    cuda_coordinate_chunk_size: int | None,
) -> int | None:
    if cuda_coordinate_chunk_size is None:
        return None
    chunk_size = positive_integer(cuda_coordinate_chunk_size, "cuda_coordinate_chunk_size")
    if execution_backend != "cuda":
        raise ValueError("cuda_coordinate_chunk_size requires the CUDA execution backend")
    return chunk_size


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
    rod_catalog_revision: str
    branch: int | None
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.int64]
    source_state_count: int
    source_revision: str
    root_policy: str = "single_nonzero_root.v1"
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

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
        if not isinstance(self.rod_catalog_revision, str) or not self.rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        if self.branch is None:
            if self.root_policy != "all_retained_roots.v1":
                raise ValueError("all-root results require all_retained_roots.v1")
        elif self.branch not in {1, 2} or self.root_policy != "single_nonzero_root.v1":
            raise ValueError("a single-root result requires branch 1 or 2")
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
        state_count = positive_integer(self.source_state_count, "source_state_count")
        valid_count = np.array(self.valid_source_count, dtype=np.int64, copy=True, order="C")
        if valid_count.shape != shape or np.any((valid_count < 0) | (valid_count > state_count)):
            raise ValueError("valid_source_count must lie within the source batch")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        }:
            raise ValueError("unsupported detector-coordinate execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        has_m0 = any(rod.family_m == 0 for rod in rods)
        if has_m0:
            if self.branch is not None:
                raise ValueError("m=0 is available only in an all-root result")
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) <= 0.0:
                raise ValueError("detector-visible m=0 requires a positive reciprocal support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("an m=0 support gap requires an m=0 rod")
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
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)


@dataclass(frozen=True, slots=True)
class SourceAveragedDetectorCoordinateDensity:
    """All-source, all-rod, all-root density on continuous detector coordinates.

    Physical rods remain explicit provenance, but no rod-valued detector array crosses this
    boundary. Every state-specific outgoing ray, inverse root, and physical-rod intensity is
    reduced before a pixel or macrobin integrator receives the result.
    """

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.int64]
    source_state_count: int
    source_revision: str
    root_policy: str = "all_retained_roots.v1"
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0):
            raise ValueError("column_px must be real")
        column = np.array(supplied_column.real, dtype=np.float64, copy=True, order="C")
        shape = column.shape
        if not np.all(np.isfinite(column)):
            raise ValueError("column_px must be finite")
        row = _float_array(self.row_px, shape, "row_px")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        supplied_density = np.asarray(self.density_A2_per_px2)
        if np.iscomplexobj(supplied_density) and np.any(supplied_density.imag != 0.0):
            raise ValueError("detector density must be real")
        density = np.array(supplied_density.real, dtype=np.float64, copy=True, order="C")
        if density.shape != shape or np.any(np.isnan(density)) or np.any(density < 0.0):
            raise ValueError("detector density must be nonnegative and contain no NaN")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != shape:
            raise ValueError("caustic must have one flag per detector coordinate")
        if np.any(np.isinf(density) & ~caustic):
            raise ValueError("infinite detector density requires a caustic")
        state_count = positive_integer(self.source_state_count, "source_state_count")
        valid_count = np.array(self.valid_source_count, dtype=np.int64, copy=True, order="C")
        if valid_count.shape != shape or np.any((valid_count < 0) | (valid_count > state_count)):
            raise ValueError("valid_source_count must lie within the source batch")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("total detector density requires all_retained_roots.v1")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
        }:
            raise ValueError("unsupported detector-coordinate execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        if (self.execution_backend == "numba_cuda_source_averaged.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        has_m0 = any(rod.family_m == 0 for rod in rods)
        if has_m0:
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) <= 0.0:
                raise ValueError("detector-visible m=0 requires a positive reciprocal support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("an m=0 support gap requires an m=0 rod")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate measure")
        for value in (column, row, density, caustic, valid_count):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "valid_source_count", valid_count)
        object.__setattr__(self, "source_state_count", state_count)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)


@dataclass(frozen=True, slots=True)
class _IndexedCompiledEvaluator:
    evaluator: CompiledDetectorEvaluator
    master_rod_index: NDArray[np.int64]
    incident_state_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, CompiledDetectorEvaluator):
            raise TypeError("evaluator must be CompiledDetectorEvaluator")
        indices = np.array(self.master_rod_index, dtype=np.int64, copy=True, order="C")
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0):
            raise ValueError("master_rod_index must contain nonnegative indices")
        if np.unique(indices).size != indices.size:
            raise ValueError("master_rod_index must not repeat an index")
        state_index = self.incident_state_index
        if isinstance(state_index, bool) or not isinstance(state_index, (int, np.integer)):
            raise TypeError("incident_state_index must be an integer")
        if state_index < 0:
            raise ValueError("incident_state_index must be nonnegative")
        indices.setflags(write=False)
        object.__setattr__(self, "master_rod_index", indices)
        object.__setattr__(self, "incident_state_index", int(state_index))

    def with_evaluator(self, evaluator: CompiledDetectorEvaluator) -> _IndexedCompiledEvaluator:
        """Reuse the already validated immutable rod-index map."""

        if not isinstance(evaluator, CompiledDetectorEvaluator):
            raise TypeError("evaluator must be CompiledDetectorEvaluator")
        rebound = object.__new__(type(self))
        object.__setattr__(rebound, "evaluator", evaluator)
        object.__setattr__(rebound, "master_rod_index", self.master_rod_index)
        object.__setattr__(rebound, "incident_state_index", self.incident_state_index)
        return rebound


def _sum_compiled_evaluator_block(
    evaluators: tuple[_IndexedCompiledEvaluator, ...],
    column_px: FloatArray,
    row_px: FloatArray,
    branch: int | None,
    master_rod_count: int,
) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
    """Sum one fixed source-state block without materializing a state axis."""

    per_rod = np.zeros((column_px.size, master_rod_count), dtype=np.float64)
    caustic = np.zeros(per_rod.shape, dtype=np.bool_)
    valid_source_count = np.zeros(column_px.size, dtype=np.int64)
    for indexed in evaluators:
        if branch is None:
            density, _, state_caustic, state_valid = indexed.evaluator.evaluate_all_roots(
                column_px,
                row_px,
            )
        else:
            density, _, state_caustic, state_valid = indexed.evaluator.evaluate(
                column_px,
                row_px,
                branch=branch,
            )
        per_rod[:, indexed.master_rod_index] += density
        caustic[:, indexed.master_rod_index] |= state_caustic
        valid_source_count += state_valid
    if not evaluators:
        raise ValueError("a source-state block must contain at least one evaluator")
    return per_rod, caustic, valid_source_count


def _reachable_master_rod_indices(
    rods: tuple[Rod, ...],
    reciprocal_basis_Ainv: FloatArray,
    k_norm_Ainv: float,
) -> NDArray[np.int64]:
    """Return stable master indices whose rod lines enter one elastic ball."""

    mean_axis, _ = mosaic_axes(reciprocal_basis_Ainv)
    q_parallel = np.asarray(
        [rod.h * reciprocal_basis_Ainv[:, 0] + rod.k * reciprocal_basis_Ainv[:, 1] for rod in rods],
        dtype=np.float64,
    )
    perpendicular = q_parallel - (q_parallel @ mean_axis)[:, None] * mean_axis
    distance = np.linalg.norm(perpendicular, axis=1)
    maximum_distance = 2.0 * float(k_norm_Ainv)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(maximum_distance, 1.0)
    result = np.flatnonzero(distance <= maximum_distance + tolerance).astype(np.int64)
    result.setflags(write=False)
    return result


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
        "_detector_visible_m0_q_gap_Ainv",
        "_evaluator_blocks",
        "_incident",
        "_instrument",
        "_material",
        "_mosaic",
        "_phase_polarization_weight",
        "_reachable_rod_count_per_source_state",
        "_rod_catalog_revision",
        "_rods",
        "_strength_model",
        "_valid_state_count",
        "_worker_count",
    )

    _MAX_STATE_BLOCK_COUNT = 32
    _COORDINATE_CHUNK_SIZE = 16_384
    _TOTAL_DENSITY_COORDINATE_CHUNK_SIZE = 4_096

    def __init__(
        self,
        *,
        reciprocal_basis_Ainv: ArrayLike,
        crystal_to_sample: ArrayLike,
        rods: tuple[Rod, ...],
        rod_catalog_revision: str,
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
        if not isinstance(rod_catalog_revision, str) or not rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        phase_weight = float(phase_population_weight)
        polarization = float(polarization_weight)
        if not isfinite(phase_weight) or phase_weight < 0.0:
            raise ValueError("phase_population_weight must be finite and nonnegative")
        if not isfinite(polarization) or polarization < 0.0:
            raise ValueError("polarization_weight must be finite and nonnegative")
        workers = positive_integer(worker_count, "worker_count")
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
        valid_state_index = np.flatnonzero(states.valid)
        if not valid_state_index.size:
            raise ValueError("source-averaged detector requires at least one valid incident state")
        maximum_air_k_Ainv = 2.0 * np.pi / float(np.min(states.wavelength_A[valid_state_index]))
        reference_config = BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal_basis_Ainv,
            crystal_to_sample=supplied_crystal_to_sample,
            rods=selected,
            mosaic=mosaic,
            k_norm_Ainv=maximum_air_k_Ainv,
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

        m0_gap: float | None = None
        if any(rod.family_m == 0 for rod in selected):
            if not valid_state_index.size:
                raise ValueError("detector-visible m=0 requires at least one valid incident state")
            incident_normal = states.k_film_phase_sample_Ainv[valid_state_index, 2]
            if np.any(incident_normal >= 0.0):
                raise ValueError(
                    "detector-visible m=0 requires every valid incident state to enter "
                    "through the negative sample-normal half-space"
                )
            m0_gap = float(np.min(-incident_normal))
            if not isfinite(m0_gap) or m0_gap <= 0.0:
                raise ValueError("detector-visible m=0 reciprocal support gap must be positive")

        reachable_count = np.zeros(states.incident_state_id.size, dtype=np.int64)
        evaluators: list[_IndexedCompiledEvaluator] = []
        (
            atom_offsets,
            atom_properties,
            f0_parameters,
            anomalous_factors,
            layers,
            normalization_divisor,
            u_radial_A2,
            u_normal_A2,
        ) = pack_bi2se3_two_h_structures(
            strength_model,
            wavelength_A=states.wavelength_A[valid_state_index],
        )
        for valid_position, state_index in enumerate(valid_state_index):
            wavelength_A = float(states.wavelength_A[state_index])
            active_index = _reachable_master_rod_indices(
                selected,
                reference_config.reciprocal_basis_Ainv,
                2.0 * np.pi / wavelength_A,
            )
            if not active_index.size:
                continue
            reachable_count[state_index] = active_index.size
            active_rods = tuple(selected[int(position)] for position in active_index)
            bragg_config = BraggSpaceConfig(
                reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
                crystal_to_sample=reference_config.crystal_to_sample,
                rods=active_rods,
                mosaic=mosaic,
                k_norm_Ainv=2.0 * np.pi / wavelength_A,
            )
            source_phase_weight = float(
                states.source_weight[state_index]
                * states.footprint_acceptance[state_index]
                * phase_weight
                * polarization
            )
            packed = (
                atom_offsets,
                atom_properties,
                f0_parameters,
                anomalous_factors[valid_position],
                layers,
                normalization_divisor,
                u_radial_A2,
                u_normal_A2,
            )
            evaluators.append(
                _IndexedCompiledEvaluator(
                    evaluator=CompiledDetectorEvaluator(
                        _compile_detector_state(
                            bragg_config=bragg_config,
                            strength_model=strength_model,
                            ki_sample_Ainv=states.k_film_phase_sample_Ainv[state_index],
                            incident=incident,
                            material=material,
                            instrument=instrument,
                            rods=active_rods,
                            incident_state_index=int(state_index),
                            source_phase_weight=source_phase_weight,
                            packed_structure=packed,
                        ),
                        instrument.detector_shape_rc,
                    ),
                    master_rod_index=active_index,
                    incident_state_index=int(state_index),
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
        reachable_count.setflags(write=False)
        object.__setattr__(self, "_evaluator_blocks", blocks)
        object.__setattr__(self, "_detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(self, "_incident", incident)
        object.__setattr__(self, "_instrument", instrument)
        object.__setattr__(self, "_material", material)
        object.__setattr__(self, "_mosaic", mosaic)
        object.__setattr__(self, "_phase_polarization_weight", phase_weight * polarization)
        object.__setattr__(self, "_reachable_rod_count_per_source_state", reachable_count)
        object.__setattr__(self, "_rod_catalog_revision", rod_catalog_revision)
        object.__setattr__(self, "_rods", selected)
        object.__setattr__(self, "_strength_model", strength_model)
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
    def material(self) -> MaterialOptics:
        return self._material

    @property
    def mosaic(self) -> MosaicParameters:
        return self._mosaic

    @property
    def strength_model(self) -> Bi2Se3TwoHStrength:
        return self._strength_model

    @property
    def rods(self) -> tuple[Rod, ...]:
        return self._rods

    @property
    def rod_catalog_revision(self) -> str:
        return self._rod_catalog_revision

    @property
    def source_state_count(self) -> int:
        return int(self._incident.states.incident_state_id.size)

    @property
    def valid_source_state_count(self) -> int:
        return self._valid_state_count

    @property
    def reachable_rod_count_per_source_state(self) -> NDArray[np.int64]:
        """Stable master-catalog reach count for each aligned incident state."""

        return self._reachable_rod_count_per_source_state

    @property
    def detector_visible_m0_q_gap_Ainv(self) -> float | None:
        """Physical lower bound on ``|Q|`` for included top-exit m=0 rays."""

        return self._detector_visible_m0_q_gap_Ainv

    def restrict_rods(
        self,
        rods: tuple[Rod, ...],
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Compile an exact source-averaged view over a physical rod subset."""

        requested = tuple(rods)
        if not requested or any(not isinstance(rod, Rod) for rod in requested):
            raise ValueError("rods must contain at least one Rod")
        configured_by_hk = {(rod.h, rod.k): rod for rod in self._rods}
        requested_hk = tuple((rod.h, rod.k) for rod in requested)
        if len(set(requested_hk)) != len(requested_hk):
            raise ValueError("rods must not repeat a physical rod")
        try:
            selected = tuple(configured_by_hk[rod_hk] for rod_hk in requested_hk)
        except KeyError as error:
            raise ValueError(f"rod subset contains unconfigured rod {error.args[0]}") from error
        if selected == self._rods:
            return self
        return type(self)(
            reciprocal_basis_Ainv=self._strength_model.reciprocal_basis_Ainv,
            crystal_to_sample=self._instrument.sample_from_crystal.rotation,
            rods=selected,
            rod_catalog_revision=self._rod_catalog_revision,
            mosaic=self._mosaic,
            strength_model=self._strength_model,
            incident=self._incident,
            material=self._material,
            instrument=self._instrument,
            phase_population_weight=self._phase_polarization_weight,
            polarization_weight=1.0,
            worker_count=self._worker_count,
        )

    def rebind_physics(
        self,
        *,
        mosaic: MosaicParameters | None = None,
        strength_model: Bi2Se3TwoHStrength | None = None,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Replace mosaic/structure arrays while retaining the combined source geometry."""

        rebound_mosaic = self._mosaic if mosaic is None else mosaic
        rebound_strength = self._strength_model if strength_model is None else strength_model
        if not isinstance(rebound_mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(rebound_strength, Bi2Se3TwoHStrength):
            raise TypeError("strength_model must be Bi2Se3TwoHStrength")
        reference = self._strength_model
        if (
            rebound_strength.crystal is not reference.crystal
            or rebound_strength.layers != reference.layers
            or rebound_strength.normalization != reference.normalization
            or rebound_strength.shared_disorder_epsilon != reference.shared_disorder_epsilon
        ):
            raise ValueError(
                "physics rebinding requires unchanged crystal, stacking, and normalization"
            )
        valid_state_index = np.flatnonzero(self._incident.states.valid)
        (
            atom_offsets,
            atom_properties,
            f0_parameters,
            anomalous_factors,
            layers,
            normalization_divisor,
            u_radial_A2,
            u_normal_A2,
        ) = pack_bi2se3_two_h_structures(
            rebound_strength,
            wavelength_A=self._incident.states.wavelength_A[valid_state_index],
        )
        valid_position_by_state = {
            int(state_index): position for position, state_index in enumerate(valid_state_index)
        }
        rebound_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        for block in self._evaluator_blocks:
            rebound_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                state = indexed.evaluator.state.rebind_physics(
                    mosaic=rebound_mosaic,
                    atom_fractional_offset=atom_offsets,
                    atom_occupancy_element=atom_properties,
                    f0_parameters=f0_parameters,
                    anomalous_factor_e=anomalous_factors[
                        valid_position_by_state[indexed.incident_state_index]
                    ],
                    layers=layers,
                    normalization_divisor=normalization_divisor,
                    u_radial_A2=u_radial_A2,
                    u_normal_A2=u_normal_A2,
                    shared_disorder_epsilon=rebound_strength.shared_disorder_epsilon,
                )
                rebound_block.append(
                    indexed.with_evaluator(
                        CompiledDetectorEvaluator(
                            state,
                            self._instrument.detector_shape_rc,
                        )
                    )
                )
            rebound_blocks.append(tuple(rebound_block))

        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", tuple(rebound_blocks))
        object.__setattr__(rebound, "_mosaic", rebound_mosaic)
        object.__setattr__(rebound, "_strength_model", rebound_strength)
        return rebound

    def with_maximum_state_block_count(
        self,
        maximum_state_block_count: int,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Return an immutable execution view with at most the requested state blocks."""

        maximum_blocks = positive_integer(
            maximum_state_block_count,
            "maximum_state_block_count",
        )
        evaluators = tuple(item for block in self._evaluator_blocks for item in block)
        block_size = max(1, (len(evaluators) + maximum_blocks - 1) // maximum_blocks)
        blocks = tuple(
            tuple(evaluators[start : start + block_size])
            for start in range(0, len(evaluators), block_size)
        )
        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", blocks)
        return rebound

    def rebind_geometry(
        self,
        *,
        incident: IncidentTransportResult,
        instrument: CompiledInstrument,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Bind new rigid geometry while reusing immutable rod, mosaic, and SF state.

        Source identities, wavelengths, material, detector calibration, crystal mounting, and
        the valid-state topology are frozen. Only detector/sample rigid geometry and the incident
        quantities causally derived from that geometry are replaced.
        """

        if not isinstance(incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        old_states = self._incident.states
        new_states = incident.states
        if new_states.sample_geometry_revision != instrument.sample_geometry_revision:
            raise ValueError(
                "geometry rebind requires incident transport and instrument to share one pose"
            )
        invariant_arrays = (
            ("incident_state_id", old_states.incident_state_id, new_states.incident_state_id),
            ("incident_sample_id", old_states.incident_sample_id, new_states.incident_sample_id),
            ("source_weight", old_states.source_weight, new_states.source_weight),
            ("wavelength_A", old_states.wavelength_A, new_states.wavelength_A),
            ("valid", old_states.valid, new_states.valid),
        )
        for name, old, new in invariant_arrays:
            if not np.array_equal(old, new):
                raise ValueError(f"geometry rebind requires unchanged {name}")
        if (
            old_states.source_revision != new_states.source_revision
            or old_states.material_revision != new_states.material_revision
            or old_states.polarization_state_id != new_states.polarization_state_id
            or old_states.incident_model_id != new_states.incident_model_id
        ):
            raise ValueError(
                "geometry rebind requires unchanged source, material, and transport identity"
            )
        old_instrument = self._instrument
        invariant_instrument = (
            instrument.detector_shape_rc == old_instrument.detector_shape_rc
            and instrument.detector_row_pitch_m == old_instrument.detector_row_pitch_m
            and instrument.detector_column_pitch_m == old_instrument.detector_column_pitch_m
            and instrument.detector_reference_coordinate_px
            == old_instrument.detector_reference_coordinate_px
            and instrument.sample_support_model_id == old_instrument.sample_support_model_id
            and instrument.sample_width_m == old_instrument.sample_width_m
            and instrument.sample_length_m == old_instrument.sample_length_m
            and instrument.film_thickness_A == old_instrument.film_thickness_A
            and np.array_equal(
                instrument.sample_from_crystal.rotation,
                old_instrument.sample_from_crystal.rotation,
            )
            and np.array_equal(
                instrument.sample_from_crystal.translation_m,
                old_instrument.sample_from_crystal.translation_m,
            )
        )
        if not invariant_instrument:
            raise ValueError(
                "geometry rebind requires unchanged detector calibration, sample support, "
                "film, and crystal mounting"
            )

        detector_rotation = instrument.lab_from_detector.rotation
        column_step_lab = detector_rotation[:, 0] * instrument.detector_column_pitch_m
        row_step_lab = detector_rotation[:, 1] * instrument.detector_row_pitch_m
        reference_column, reference_row = instrument.detector_reference_coordinate_px
        detector_zero_lab = (
            instrument.lab_from_detector.translation_m
            - reference_column * column_step_lab
            - reference_row * row_step_lab
        )
        detector_area_vector = np.cross(column_step_lab, row_step_lab)
        rebound_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        for block in self._evaluator_blocks:
            rebound_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                state_index = indexed.incident_state_index
                incident_direction = -1 if new_states.direction_sample[state_index, 2] < 0.0 else 1
                incident_decay = float(
                    mode_decay_constant(
                        new_states.kz_film_Ainv[state_index],
                        incident_direction,
                    )
                )
                source_phase_weight = float(
                    new_states.source_weight[state_index]
                    * new_states.footprint_acceptance[state_index]
                    * self._phase_polarization_weight
                )
                rebound_state = indexed.evaluator.state.rebind_geometry(
                    detector_zero_lab_m=np.ascontiguousarray(detector_zero_lab),
                    detector_column_step_lab_m=np.ascontiguousarray(column_step_lab),
                    detector_row_step_lab_m=np.ascontiguousarray(row_step_lab),
                    detector_pixel_area_vector_lab_m2=np.ascontiguousarray(detector_area_vector),
                    ray_origin_lab_m=np.ascontiguousarray(
                        new_states.sample_intersection_lab_m[state_index]
                    ),
                    sample_from_lab=np.ascontiguousarray(instrument.sample_from_lab.rotation),
                    ki_film_sample_Ainv=np.ascontiguousarray(
                        new_states.k_film_phase_sample_Ainv[state_index]
                    ),
                    internal_k_Ainv=float(
                        np.linalg.norm(new_states.k_film_phase_sample_Ainv[state_index])
                    ),
                    entrance_amplitude=complex(new_states.entrance_amplitude[state_index]),
                    incident_decay_Ainv=incident_decay,
                    source_phase_weight=source_phase_weight,
                )
                rebound_block.append(
                    indexed.with_evaluator(
                        CompiledDetectorEvaluator(
                            rebound_state,
                            instrument.detector_shape_rc,
                        )
                    )
                )
            rebound_blocks.append(tuple(rebound_block))

        m0_gap: float | None = None
        if any(rod.family_m == 0 for rod in self._rods):
            incident_normal = new_states.k_film_phase_sample_Ainv[new_states.valid, 2]
            if not incident_normal.size or np.any(incident_normal >= 0.0):
                raise ValueError("detector-visible m=0 requires negative incident sample-normal k")
            m0_gap = float(np.min(-incident_normal))
        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", tuple(rebound_blocks))
        object.__setattr__(rebound, "_detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(rebound, "_incident", incident)
        object.__setattr__(rebound, "_instrument", instrument)
        return rebound

    def _thread_pool(self) -> ThreadPoolExecutor | None:
        if self._worker_count <= 1 or len(self._evaluator_blocks) <= 1:
            return None
        return ThreadPoolExecutor(max_workers=min(self._worker_count, len(self._evaluator_blocks)))

    def _evaluate_flat_coordinates(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        *,
        branch: int | None,
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
                        len(self._rods),
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
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
                block_results = (future.result() for future in futures)
            for block_density, block_caustic, block_valid_count in block_results:
                per_rod[start:stop] += block_density
                caustic[start:stop] |= block_caustic
                valid_source_count[start:stop] += block_valid_count
        return per_rod, caustic, valid_source_count

    def _evaluate_flat_density_all_roots(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        *,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
        density = np.zeros(column_px.size, dtype=np.float64)
        caustic = np.zeros(column_px.size, dtype=np.bool_)
        valid_source_count = np.zeros(column_px.size, dtype=np.int64)
        chunk_size = self._TOTAL_DENSITY_COORDINATE_CHUNK_SIZE
        for start in range(0, column_px.size, chunk_size):
            stop = min(start + chunk_size, column_px.size)
            chunk_column = column_px[start:stop]
            chunk_row = row_px[start:stop]
            chunk_per_rod = np.zeros((stop - start, len(self._rods)), dtype=np.float64)
            chunk_per_rod_caustic = np.zeros(chunk_per_rod.shape, dtype=np.bool_)
            if executor is None:
                block_results = (
                    _sum_compiled_evaluator_block(
                        block,
                        chunk_column,
                        chunk_row,
                        None,
                        len(self._rods),
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
                        None,
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
                block_results = (future.result() for future in futures)
            for block_density, block_caustic, block_valid_count in block_results:
                chunk_per_rod += block_density
                chunk_per_rod_caustic |= block_caustic
                valid_source_count[start:stop] += block_valid_count
            density[start:stop] = np.sum(chunk_per_rod, axis=1, dtype=np.float64)
            caustic[start:stop] = np.any(chunk_per_rod_caustic, axis=1)
        return density, caustic, valid_source_count

    def _evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        branch: int | None,
        execution_backend: str,
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
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
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be 'cpu' or 'cuda'")
        cuda_chunk_size = _validated_cuda_coordinate_chunk_size(
            execution_backend,
            cuda_coordinate_chunk_size,
        )
        if execution_backend == "cuda":
            if branch is not None:
                raise ValueError("the CUDA backend currently supports all retained roots only")
            from rasim_next.pipeline._continuous_detector_cuda import (
                evaluate_source_averaged_all_roots_cuda,
            )

            per_rod, caustic, valid_source_count, execution_device = (
                evaluate_source_averaged_all_roots_cuda(
                    self._evaluator_blocks,
                    flat_column,
                    flat_row,
                    detector_shape_rc=self._instrument.detector_shape_rc,
                    master_rod_count=len(self._rods),
                    **(
                        {}
                        if cuda_chunk_size is None
                        else {"coordinate_chunk_size": cuda_chunk_size}
                    ),
                )
            )
            backend_id = "numba_cuda_source_averaged.v1"
        else:
            if self._evaluator_blocks and flat_column.size:
                first = self._evaluator_blocks[0][0].evaluator
                if branch is None:
                    first.evaluate_all_roots(
                        np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64),
                    )
                else:
                    first.evaluate(
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
            execution_device = None
            backend_id = "numba_cpu_source_averaged.v1"
        reshaped_per_rod = per_rod.reshape((*shape, len(self._rods)))
        return SourceAveragedDetectorCoordinateIntensity(
            column_px=column,
            row_px=row,
            rods=self._rods,
            rod_catalog_revision=self._rod_catalog_revision,
            branch=branch,
            per_rod_density_A2_per_px2=reshaped_per_rod,
            density_A2_per_px2=np.sum(reshaped_per_rod, axis=-1, dtype=np.float64),
            caustic=caustic.reshape((*shape, len(self._rods))),
            valid_source_count=valid_source_count.reshape(shape),
            source_state_count=self.source_state_count,
            source_revision=self._incident.states.source_revision,
            execution_backend=backend_id,
            execution_device=execution_device,
            root_policy=("all_retained_roots.v1" if branch is None else "single_nonzero_root.v1"),
            detector_visible_m0_q_gap_Ainv=(
                self._detector_visible_m0_q_gap_Ainv if branch is None else None
            ),
        )

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        branch: int = 2,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate one nonzero-rod root at arbitrary detector coordinates."""

        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        if any(rod.family_m == 0 for rod in self._rods):
            raise ValueError("branch-specific evaluation cannot include m=0")
        return self._evaluate_detector_coordinates(
            column_px,
            row_px,
            branch=branch,
            execution_backend="cpu",
        )

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate every retained physical root, including supported kinematic m=0."""

        return self._evaluate_detector_coordinates(
            column_px,
            row_px,
            branch=None,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )

    def evaluate_detector_density_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateDensity:
        """Reduce every source, physical rod, and retained root at each coordinate."""

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
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be 'cpu' or 'cuda'")
        cuda_chunk_size = _validated_cuda_coordinate_chunk_size(
            execution_backend,
            cuda_coordinate_chunk_size,
        )
        shape = column.shape
        flat_column = np.ascontiguousarray(column.reshape(-1))
        flat_row = np.ascontiguousarray(row.reshape(-1))
        if execution_backend == "cuda":
            from rasim_next.pipeline._continuous_detector_cuda import (
                evaluate_source_averaged_density_all_roots_cuda,
            )

            density, caustic, valid_source_count, execution_device = (
                evaluate_source_averaged_density_all_roots_cuda(
                    self._evaluator_blocks,
                    flat_column,
                    flat_row,
                    detector_shape_rc=self._instrument.detector_shape_rc,
                    master_rod_count=len(self._rods),
                    **(
                        {}
                        if cuda_chunk_size is None
                        else {"coordinate_chunk_size": cuda_chunk_size}
                    ),
                )
            )
            backend_id = "numba_cuda_source_averaged.v1"
        else:
            if self._evaluator_blocks and flat_column.size:
                self._evaluator_blocks[0][0].evaluator.evaluate_all_roots(
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            executor = self._thread_pool()
            try:
                density, caustic, valid_source_count = self._evaluate_flat_density_all_roots(
                    flat_column,
                    flat_row,
                    executor=executor,
                )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
            execution_device = None
            backend_id = "numba_cpu_source_averaged.v1"
        return SourceAveragedDetectorCoordinateDensity(
            column_px=column,
            row_px=row,
            rods=self._rods,
            density_A2_per_px2=density.reshape(shape),
            caustic=caustic.reshape(shape),
            valid_source_count=valid_source_count.reshape(shape),
            source_state_count=self.source_state_count,
            source_revision=self._incident.states.source_revision,
            detector_visible_m0_q_gap_Ainv=self._detector_visible_m0_q_gap_Ainv,
            execution_backend=backend_id,
            execution_device=execution_device,
        )

    def integrate_native_pixels(
        self,
        *,
        branch: int,
        quadrature: DetectorQuadrature,
        include_per_rod_evidence: bool = False,
    ) -> DetectorPixelMass:
        """Integrate detailed per-rod pixel evidence after explicit opt-in."""

        if include_per_rod_evidence is not True:
            raise ValueError(
                "per-rod pixel evidence is disabled by default; pass "
                "include_per_rod_evidence=True explicitly"
            )

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
        if any(rod.family_m == 0 for rod in self._rods):
            raise ValueError("branch-specific pixel integration cannot include m=0")
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
            self._evaluator_blocks[0][0].evaluator.evaluate(
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
    "SourceAveragedDetectorCoordinateDensity",
    "SourceAveragedDetectorCoordinateIntensity",
    "SourceAveragedDetectorEwaldMeasure",
]
