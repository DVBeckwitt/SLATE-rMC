"""One continuous conditional-position detector for native fitting and rendering."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from functools import partial
from itertools import pairwise

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from painted_ewald import MosaicParameters, Rod
from painted_ewald.normal_density import SphericalMosaicDensity, _ConeAverageGeometry
from painted_ewald.validation import proper_rotation, reciprocal_basis, reject_complex
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
from rasim_next.optics.attenuation import uniform_depth_attenuation
from rasim_next.ordered.amplitudes import AtomicQueryCache
from rasim_next.pipeline._continuous_detector_kernel import (
    _empirical_parratt_strength_A2,
    local_m0_phase_q_Ainv,
)
from rasim_next.pipeline.bragg_space import (
    Bi2X3FiniteStackStrength,
    CifFiniteStackStrength,
    IncoherentStructureMixture,
    Pbi2FiniteSurfaceStrength,
    RevisionedStructureStrengthModel,
)
from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope
from rasim_next.pipeline.detector_revisions import (
    _detector_native_chart_revision,
    _incidence_angle_static_physics_revision,
    _instrument_revision,
)
from rasim_next.pipeline.fiber_detector import (
    ConditionalFiberBatch,
    FiberIntegrationRule,
    FiberScatteringCache,
    iter_conditional_fiber_transfers,
)
from rasim_next.pipeline.pixel_error import iter_pixel_error_batches
from rasim_next.pipeline.source_spatial import (
    NativeSpatialRegionProjection,
    validate_conditional_spatial_support,
)
from rasim_next.pipeline.spatial_execution import NativeSpatialExecutor
from rasim_next.pipeline.strength_gauss import (
    finite_stack_resolution,
    identically_zero_strength,
    prepare_positive_axial_rule,
    response_panel_edges,
)
from rasim_next.reflectivity.specular import (
    LOCAL_LAMELLA_INTERFACE,
    ParrattStitchStack,
    compile_parratt_stitch,
)
from rasim_next.sampling.source import ConditionalSourceSamples

FloatArray = NDArray[np.float64]


def native_projection_bounds_px(
    projection: NativePixelRegionProjection, detector_shape_rc: tuple[int, int]
) -> FloatArray:
    """Return finite native rectangle bounds owned by an observation projection."""
    if projection.detector_shape_rc != detector_shape_rc:
        raise ValueError("native projection uses a different detector")
    owner = projection.observation_row
    row, column = np.divmod(
        projection.flat_pixel_index[projection.pixel_column_index], detector_shape_rc[1]
    )
    low = np.full((projection.observation_count, 2), np.inf)
    high = -low.copy()
    np.minimum.at(low, owner, np.column_stack((column, row)) - 0.5)
    np.maximum.at(high, owner, np.column_stack((column, row)) + 0.5)
    bounds = np.column_stack((low[:, 0], high[:, 0], low[:, 1], high[:, 1]))
    return bounds[np.all(np.isfinite(bounds), axis=1)]


def _frozen(value: ArrayLike, *, integer: bool = False) -> np.ndarray:
    reject_complex(value, "fiber response array")
    supplied = np.asarray(value)
    if integer and supplied.dtype.kind not in "iu":
        raise TypeError("fiber indices must be integer arrays")
    result = np.array(value, dtype=np.int64 if integer else np.float64, copy=True)
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError("fiber response arrays must be finite vectors")
    result.setflags(write=False)
    return result


@numba.njit(nogil=True, fastmath=False, cache=False)
def _phase_grid(external, k0, film):
    return np.array([local_m0_phase_q_Ainv(q, k0, film) for q in external])


@numba.njit(nogil=True, fastmath=False, cache=False)
def _stitch_grid(
    strength, external, k0, film, substrate, thickness, top, bottom, qc, zero, scale, lo, hi
):
    result = np.empty_like(strength)
    for sign in range(2):
        for i in range(len(external)):
            result[sign, i] = _empirical_parratt_strength_A2(
                strength[sign, i],
                external[i],
                k0,
                film,
                substrate,
                thickness,
                top,
                bottom,
                qc,
                zero,
                scale,
                lo,
                hi,
            )
    return result


@dataclass(frozen=True, slots=True)
class FiberStrengthGrid:
    """One radius/wavelength grid retaining each physical rod and both SF signs."""

    rods: tuple[Rod, ...]
    wavelength_A: float
    positive_axial_Ainv: FloatArray
    external_q_Ainv: FloatArray | None

    def __post_init__(self) -> None:
        rods = tuple(self.rods)
        axial = _frozen(self.positive_axial_Ainv)
        if not rods or any(not isinstance(rod, Rod) for rod in rods) or np.any(axial < 0):
            raise ValueError("a strength grid requires physical rods and nonnegative axial nodes")
        if not np.isfinite(self.wavelength_A) or self.wavelength_A <= 0:
            raise ValueError("strength-grid wavelength must be positive")
        external = None if self.external_q_Ainv is None else _frozen(self.external_q_Ainv)
        if external is not None and (
            external.shape != axial.shape
            or np.any(external < 0)
            or len(rods) != 1
            or (rods[0].h, rods[0].k) != (0, 0)
        ):
            raise ValueError("the local grid must be the zero rod with aligned external Q")
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "positive_axial_Ainv", axial)
        object.__setattr__(self, "external_q_Ainv", external)


@dataclass(frozen=True, slots=True)
class FiberResponseNodes:
    """Physical quadrature factors after spatial reduction, excluding source mass."""

    grid_index: int
    source_state_index: int
    axial_index: NDArray[np.int64]
    polar_angle_rad: FloatArray
    cone_angle_rad: FloatArray
    integrated_coefficient: FloatArray
    phase_q_radial_squared_Ainv2: FloatArray
    phase_q_normal_squared_Ainv2: FloatArray
    attenuation_decay_sum_Ainv: FloatArray | None
    reference_thickness_A: float

    def __post_init__(self) -> None:
        index = _frozen(self.axial_index, integer=True)
        if any(
            type(i) is not int or i < 0 for i in (self.grid_index, self.source_state_index)
        ) or np.any(index < 0):
            raise ValueError("response nodes require valid grid/source/axial indices")
        object.__setattr__(self, "axial_index", index)
        for name in (
            "polar_angle_rad",
            "cone_angle_rad",
            "integrated_coefficient",
            "phase_q_radial_squared_Ainv2",
            "phase_q_normal_squared_Ainv2",
            "attenuation_decay_sum_Ainv",
        ):
            supplied = getattr(self, name)
            if supplied is None and name == "attenuation_decay_sum_Ainv":
                continue
            value = _frozen(supplied)
            if value.shape != index.shape or np.any(value < 0):
                raise ValueError(
                    "physical response coefficients must be nonnegative aligned vectors"
                )
            if name.endswith("angle_rad") and np.any(value > np.pi):
                raise ValueError("normal angles must lie in [0,pi]")
            object.__setattr__(self, name, value)
        if not np.isfinite(self.reference_thickness_A) or self.reference_thickness_A < 0:
            raise ValueError("reference thickness must be nonnegative")


def _response_nodes(
    batch: ConditionalFiberBatch, grid_index: int, retained: NDArray[np.int64] | None = None
) -> FiberResponseNodes:
    transfer = batch.transfer
    selected = slice(None) if retained is None else retained
    decay = transfer.attenuation_decay_sum_Ainv
    return FiberResponseNodes(
        grid_index,
        batch.source_state_index,
        batch.axial_index[selected],
        transfer.polar_angle_rad[selected],
        transfer.cone_angle_rad[selected],
        batch.integrated_coefficient[selected],
        transfer.phase_q_radial_squared_Ainv2[selected],
        transfer.phase_q_normal_squared_Ainv2[selected],
        None if decay is None else decay[selected],
        transfer.reference_thickness_A,
    )


@numba.njit(nogil=True, fastmath=False, cache=False)
def _contract_event_mass(
    strength,
    cone,
    axial_index,
    integrated_coefficient,
    radial_q2,
    normal_q2,
    u_radial,
    u_normal,
    weight,
    attenuation_ratio,
    probability,
    result,
):
    """One event-weight equation, optionally accumulated directly into CSR regions."""
    for component in range(len(cone)):
        for i in range(len(axial_index)):
            index = axial_index[i]
            mass = (
                strength[0, index] * cone[component, 0, i]
                + strength[1, index] * cone[component, 1, i]
            )
            coefficient = (
                integrated_coefficient[i]
                * weight
                * np.exp(-u_radial * radial_q2[i] - u_normal * normal_q2[i])
            )
            if attenuation_ratio is not None:
                coefficient *= attenuation_ratio[i]
            mass *= coefficient
            if probability is None:
                result[component, i] = mass
            else:
                data, indices, indptr = probability
                for j in range(indptr[i], indptr[i + 1]):
                    result[component, indices[j]] += data[j] * mass


def _event_mass(
    nodes: FiberResponseNodes,
    strength: FloatArray,
    density: SphericalMosaicDensity,
    thickness_A: float,
    envelope: SampleQIntensityEnvelope,
    weight: float,
    cone_quadrature_order: int = 16,
    cone_density: FloatArray | None = None,
    *,
    probability: csr_matrix | None = None,
) -> FloatArray:
    """Shared exact SF/cone contraction, before native-region or pixel deposition."""
    index = nodes.axial_index
    if (
        strength.ndim != 2
        or strength.shape[0] != 2
        or np.max(index, initial=-1) >= strength.shape[1]
    ):
        raise ValueError("signed strength must have two rows covering every axial index")
    if probability is not None and probability.shape[0] != len(index):
        raise ValueError("sparse probability rows must align with event nodes")
    if cone_density is None:
        cone_density = np.array(
            [
                density.cone_average_sr_inv(
                    nodes.polar_angle_rad, opening, quadrature_order=cone_quadrature_order
                )
                for opening in (nodes.cone_angle_rad, np.pi - nodes.cone_angle_rad)
            ]
        )
    if cone_density.ndim < 2 or cone_density.shape[-2:] != (2, len(index)):
        raise ValueError("cone density must align with both signs and event nodes")
    decay = nodes.attenuation_decay_sum_Ainv
    attenuation_ratio = (
        None
        if decay is None
        else uniform_depth_attenuation(decay, 0.0, thickness_A)
        / uniform_depth_attenuation(decay, 0.0, nodes.reference_thickness_A)
    )
    leading_shape = cone_density.shape[:-2]
    component_count = int(np.prod(leading_shape, dtype=np.int64))
    count = len(index) if probability is None else probability.shape[1]
    result = np.zeros((component_count, count))
    _contract_event_mass(
        strength,
        cone_density.reshape(component_count, 2, len(index)),
        index,
        nodes.integrated_coefficient,
        nodes.phase_q_radial_squared_Ainv2,
        nodes.phase_q_normal_squared_Ainv2,
        envelope.u_radial_A2,
        envelope.u_normal_A2,
        weight,
        attenuation_ratio,
        None
        if probability is None
        else (probability.data, probability.indices, probability.indptr),
        result,
    )
    return result.reshape((*leading_shape, count))


def _prepare_cone_block(block, densities, node, weight):
    nodes, _ = block
    values = [np.empty((2, len(nodes.axial_index))) for _ in densities]
    for sign, opening in enumerate((nodes.cone_angle_rad, np.pi - nodes.cone_angle_rad)):
        geometry = _ConeAverageGeometry.from_angles(nodes.polar_angle_rad, opening)
        for density, value in zip(densities, values, strict=True):
            value[sign] = density._cone_average_prepared(geometry, node, weight)
        del geometry
    return values


@dataclass(frozen=True, slots=True)
class NativeFixedResponse:
    """Immutable nominal geometry probabilities; candidate factors stay separate."""

    revision: str
    observation_count: int
    grids: tuple[FiberStrengthGrid, ...]
    blocks: tuple[tuple[FiberResponseNodes, csr_matrix], ...]
    retained_bytes: int

    @property
    def cone_density_bytes(self) -> int:
        """Exact float64 storage for both signed cones on every retained node."""
        return (
            2
            * np.dtype(np.float64).itemsize
            * sum(len(nodes.axial_index) for nodes, _ in self.blocks)
        )

    def compile_cone_density(
        self, mosaic: MosaicParameters, *, quadrature_order: int, maximum_bytes: int
    ) -> NativeConeDensity:
        """Prepare explicit candidate-mosaic values for this immutable response."""
        return self.compile_cone_components(
            (mosaic,), quadrature_order=quadrature_order, maximum_bytes=maximum_bytes
        )[0]

    def compile_cone_components(
        self, mosaics: tuple[MosaicParameters, ...], *, quadrature_order: int, maximum_bytes: int
    ) -> tuple[NativeConeDensity, ...]:
        """Share angle geometry and quadrature across independent mosaic laws.

        At most four blocks are prepared concurrently. Each block retains the
        original 2048-event Gaussian batches. Narrow reductions are unchanged;
        broad panel reductions use deterministic serial float64 arithmetic.
        Geometry is temporary per block, not a retained full-response cache.
        """
        if type(maximum_bytes) is not int or maximum_bytes < 0:
            raise ValueError("cone density memory budget must be a nonnegative integer")
        if type(quadrature_order) is not int or quadrature_order < 4:
            raise ValueError("cone quadrature order must be an integer of at least four")
        mosaics = tuple(mosaics)
        if self.cone_density_bytes * len(mosaics) > maximum_bytes:
            raise MemoryError("cone density memory budget exhausted before allocation")
        densities = tuple(SphericalMosaicDensity(mosaic) for mosaic in mosaics)
        if not densities:
            return ()
        node, weight = (
            np.polynomial.legendre.leggauss(quadrature_order)
            if any(law.lorentzian_probability < 1 for law in mosaics) and self.blocks
            else (np.empty(0), np.empty(0))
        )

        values = []
        if self.blocks:
            # Warm the first block's active kernels before dispatching workers.
            values.append(_prepare_cone_block(self.blocks[0], densities, node, weight))
            workers = min(4, numba.get_num_threads(), len(self.blocks))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for start in range(1, len(self.blocks), workers):
                        futures = [
                            executor.submit(_prepare_cone_block, block, densities, node, weight)
                            for block in self.blocks[start : start + workers]
                        ]
                        values.extend(future.result() for future in futures)
            else:
                values.extend(
                    _prepare_cone_block(block, densities, node, weight) for block in self.blocks[1:]
                )
        return tuple(
            NativeConeDensity(self, density, quadrature_order, tuple(block[i] for block in values))
            for i, density in enumerate(densities)
        )

    def evaluate(
        self,
        detector: ConditionalStructureDetector,
        projection: NativePixelRegionProjection,
        *,
        cone_density: NativeConeDensity | None = None,
        cone_components: tuple[NativeConeDensity | None, NativeConeDensity | None] | None = None,
        strength_tables: NativeStrengthTables | None = None,
        query_cache: AtomicQueryCache | None = None,
    ) -> FloatArray:
        if self.revision != detector.native_response_revision(projection):
            raise ValueError("fixed response does not match current geometry, optics or projection")
        if cone_density is not None and (
            not isinstance(cone_density, NativeConeDensity)
            or cone_density.response is not self
            or cone_density.density.parameters != detector.mosaic
            or cone_density.quadrature_order != detector.integration_rule.cone_quadrature_order
        ):
            raise ValueError("cone density does not match response, mosaic or quadrature")
        self._validate_cone_components(detector, cone_components)
        if cone_density is not None and cone_components is not None:
            raise ValueError("supply mixed density or separate components, not both")
        if strength_tables is None:
            strength_tables = self.compile_strength_tables(detector, query_cache=query_cache)
        if (
            strength_tables.response is not self
            or strength_tables.revision != self.strength_revision(detector)
        ):
            raise ValueError("strength tables do not match current response, structure or stitch")
        tables = strength_tables.values
        density = (
            SphericalMosaicDensity(detector.mosaic)
            if cone_density is None
            else cone_density.density
        )
        result = np.zeros(self.observation_count)
        for block_index, (nodes, probability) in enumerate(self.blocks):
            result += _event_mass(
                nodes,
                tables[nodes.grid_index],
                density,
                detector.instrument.film_thickness_A,
                detector.intensity_envelope,
                detector.phase_population_weight
                * detector.polarization_weight
                * detector.incident.states.source_weight[nodes.source_state_index],
                detector.integration_rule.cone_quadrature_order,
                self._cone_values(detector, block_index, cone_density, cone_components),
                probability=probability,
            )
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("invalid complete fixed native response")
        return result

    def strength_revision(self, detector):
        stack = detector.specular_stitch_stack
        return canonical_revision_sha256(
            ("structure", detector.strength_model.structure_model_revision),
            ("basis", detector.reciprocal_basis_Ainv),
            ("wavelength", detector.material.wavelength_A),
            ("optics", detector.material.n_complex),
            ("stitch_grid_size", detector.integration_rule.stitch_grid_size),
            ("thickness", detector.instrument.film_thickness_A),
            ("stack", "none" if stack is None else repr(stack)),
        )

    def factor_revision(self, detector):
        """All non-strength factors; geometry/probability belong to this response."""
        return canonical_revision_sha256(
            ("mosaic", json.dumps(asdict(detector.mosaic), sort_keys=True)),
            ("cone_order", detector.integration_rule.cone_quadrature_order),
            ("thickness", detector.instrument.film_thickness_A),
            (
                "envelope",
                np.array(
                    [
                        detector.intensity_envelope.u_radial_A2,
                        detector.intensity_envelope.u_normal_A2,
                    ]
                ),
            ),
            ("source_weights", detector.incident.states.source_weight),
            ("weights", np.array([detector.phase_population_weight, detector.polarization_weight])),
        )

    def compile_strength_response(self, detector, projection, *, cone_components, maximum_bytes):
        """Aggregate repeated axial events, with explicit memory/range admission."""
        if type(maximum_bytes) is not int or maximum_bytes < 0:
            raise ValueError("strength response budget must be a nonnegative integer")
        if self.revision != detector.native_response_revision(projection):
            raise ValueError("strength response geometry differs")
        self._validate_cone_components(detector, cone_components)
        blocks, reasons, retained = [], [], 0
        density = SphericalMosaicDensity(detector.mosaic)
        for block_index in range(len(self.blocks)):
            block, reason, size = self._compile_strength_block(
                detector, block_index, density, cone_components, maximum_bytes - retained
            )
            blocks.append(block)
            reasons.append(reason)
            retained += size
        return NativeStrengthResponse(
            self, self.factor_revision(detector), tuple(blocks), tuple(reasons), retained
        )

    def _compile_strength_block(
        self, detector, block_index, density, cone_components, maximum_bytes
    ):
        # Scratch dies with this call before the next block receives its budget.
        nodes, probability = self.blocks[block_index]
        count = len(self.grids[nodes.grid_index].positive_axial_Ainv)
        if 128 * len(nodes.axial_index) + 16 * count + 24 * self.observation_count > maximum_bytes:
            return None, "memory_limit", 0
        cone = self._cone_values(detector, block_index, None, cone_components)
        if cone is None:
            return None, "unprepared_cone", 0
        basis = np.zeros((2, count))
        weights = []
        for sign in range(2):
            basis[:] = 0
            basis[sign] = 1
            weights.append(
                _event_mass(
                    nodes,
                    basis,
                    density,
                    detector.instrument.film_thickness_A,
                    detector.intensity_envelope,
                    detector.phase_population_weight
                    * detector.polarization_weight
                    * detector.incident.states.source_weight[nodes.source_state_index],
                    detector.integration_rule.cone_quadrature_order,
                    cone,
                )
            )
        weights = np.asarray(weights)
        axial, inverse = np.unique(nodes.axial_index, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        event_ptr = np.searchsorted(inverse[order], np.arange(len(axial) + 1))
        ptr, _, _, minimum, maximum, unsafe = _aggregate_axial_events(
            weights,
            order,
            event_ptr,
            probability.data,
            probability.indices,
            probability.indptr,
            self.observation_count,
            None,
        )
        # Aggregation must reduce sparse work and fit retained + construction storage.
        size = int(ptr[-1]) * 16 + ptr.nbytes + axial.nbytes
        scratch = (
            128 * len(nodes.axial_index) + 16 * count + 24 * self.observation_count + ptr.nbytes
        )
        if unsafe or np.any(
            (weights == 0) & (cone > 0) & (nodes.integrated_coefficient[None, :] > 0)
        ):
            return None, "dynamic_range", 0
        if int(ptr[-1]) >= probability.nnz:
            return None, "no_sparse_reduction", 0
        if 2 * size + scratch > maximum_bytes:
            return None, "memory_limit", 0
        ptr, indices, data, _, _, _ = _aggregate_axial_events(
            weights,
            order,
            event_ptr,
            probability.data,
            probability.indices,
            probability.indptr,
            self.observation_count,
            int(ptr[-1]),
        )
        positives = cone[cone > 0]
        if len(positives):
            minimum = min(minimum, float(positives.min()))
            maximum = max(maximum, float(positives.max()))
        positives = weights[weights > 0]
        if len(positives):
            minimum = min(minimum, float(positives.min()))
            maximum = max(maximum, float(positives.max()))
        matrix = csr_matrix(
            (data, indices, ptr), shape=(2 * len(axial), self.observation_count), copy=False
        )
        if np.any(~np.isfinite(data)):
            return None, "dynamic_range", 0
        # Keep both multiplication orders and all positive sums away from
        # float64 overflow/underflow. Other candidate ranges use direct events.
        lower = -1000.0 - np.log2(minimum) if np.isfinite(minimum) else -np.inf
        upper = (
            1000.0
            - np.log2(max(maximum, np.finfo(float).tiny))
            - np.log2(max(2, 2 * len(nodes.axial_index)))
        )
        for array in (axial, matrix.data, matrix.indices, matrix.indptr):
            array.setflags(write=False)
        size = axial.nbytes + sum(v.nbytes for v in (matrix.data, matrix.indices, matrix.indptr))
        return (axial, matrix, lower, upper), "aggregated", size

    def compile_strength_tables(self, detector, *, query_cache=None):
        return NativeStrengthTables(
            self,
            self.strength_revision(detector),
            tuple(
                detector._strength_table(
                    grid,
                    detector.strength_model,
                    detector.instrument.film_thickness_A,
                    query_cache=query_cache,
                )
                for grid in self.grids
            ),
        )

    def _validate_cone_components(self, detector, components):
        if components is None:
            return
        if len(components) != 2:
            raise ValueError("two cone components are required")
        mosaic = detector.mosaic
        for packet, weight, law in zip(
            components,
            (1 - mosaic.lorentzian_probability, mosaic.lorentzian_probability),
            (
                MosaicParameters(mosaic.gaussian_sigma_rad, 0.0, 0.0),
                MosaicParameters(0.0, mosaic.lorentzian_half_width_rad, 1.0),
            ),
            strict=True,
        ):
            if weight == 0 and packet is None:
                continue
            if (
                not isinstance(packet, NativeConeDensity)
                or packet.response is not self
                or packet.density.parameters != law
                or packet.quadrature_order != detector.integration_rule.cone_quadrature_order
            ):
                raise ValueError("cone component does not match response, width or quadrature")

    def _cone_values(self, detector, index, density, components):
        if components is None:
            return None if density is None else density.values[index]
        eta = detector.mosaic.lorentzian_probability
        if eta == 0:
            return components[0].values[index]
        if eta == 1:
            return components[1].values[index]
        return (1 - eta) * components[0].values[index] + eta * components[1].values[index]

    def evaluate_many(self, detectors, projection, *, cone_components, strength_tables):
        """Bounded candidate-axis event masses with one ordered CSR traversal."""
        if (
            not 1 <= len(detectors) <= 8
            or len(cone_components) != len(detectors)
            or len(strength_tables) != len(detectors)
        ):
            raise ValueError("a native response batch requires one to eight aligned candidates")
        for detector, components, tables in zip(
            detectors, cone_components, strength_tables, strict=True
        ):
            if self.revision != detector.native_response_revision(projection):
                raise ValueError("batched response geometry differs")
            self._validate_cone_components(detector, components)
            if tables.response is not self or tables.revision != self.strength_revision(detector):
                raise ValueError("batched strength tables differ")
        result = np.zeros((len(detectors), self.observation_count))
        densities = [SphericalMosaicDensity(detector.mosaic) for detector in detectors]
        for block_index, (nodes, probability) in enumerate(self.blocks):
            masses = np.empty((len(detectors), len(nodes.axial_index)))
            for i, (detector, components, tables) in enumerate(
                zip(detectors, cone_components, strength_tables, strict=True)
            ):
                masses[i] = _event_mass(
                    nodes,
                    tables.values[nodes.grid_index],
                    densities[i],
                    detector.instrument.film_thickness_A,
                    detector.intensity_envelope,
                    detector.phase_population_weight
                    * detector.polarization_weight
                    * detector.incident.states.source_weight[nodes.source_state_index],
                    detector.integration_rule.cone_quadrature_order,
                    self._cone_values(detector, block_index, None, components),
                )
            block = np.zeros_like(result)
            _contract_candidate_masses(
                masses, probability.data, probability.indices, probability.indptr, block
            )
            result += block
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("invalid batched native response")
        return result


@numba.njit(nogil=True, fastmath=False, cache=False)
def _contract_candidate_masses(masses, data, indices, indptr, result):
    for event in range(masses.shape[1]):
        for j in range(indptr[event], indptr[event + 1]):
            observation, probability = indices[j], data[j]
            for candidate in range(len(masses)):
                result[candidate, observation] += probability * masses[candidate, event]


@dataclass(frozen=True, slots=True)
class NativeStrengthTables:
    """Exact signed strength/stitch tables for one response and physical revision."""

    response: NativeFixedResponse
    revision: str
    values: tuple[FloatArray, ...]
    retained_bytes: int = field(init=False)

    def __post_init__(self):
        if len(self.values) != len(self.response.grids):
            raise ValueError("strength tables must align with response grids")
        for value in self.values:
            reject_complex(value, "signed strength table")
        values = tuple(np.array(value, dtype=np.float64, copy=True) for value in self.values)
        for value, grid in zip(values, self.response.grids, strict=True):
            if (
                value.shape != (2, len(grid.positive_axial_Ainv))
                or np.any(~np.isfinite(value))
                or np.any(value < 0)
            ):
                raise ValueError("signed strength tables must be finite and nonnegative")
            value.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "retained_bytes", sum(value.nbytes for value in values))


@numba.njit(nogil=True, fastmath=False, cache=False)
def _aggregate_axial_events(weights, event_order, event_ptr, data, indices, indptr, nobs, count):
    """Two passes bound sparse allocation; each row retains original event order."""
    rows = 2 * (len(event_ptr) - 1)
    out_ptr = np.zeros(rows + 1, np.int64)
    out_index = np.empty(0 if count is None else count, np.int64)
    out_data = np.empty(0 if count is None else count)
    seen = np.full(nobs, -1, np.int64)
    touched = np.empty(nobs, np.int64)
    sums = np.zeros(nobs)
    cursor, minimum, maximum, unsafe = 0, np.inf, 0.0, False
    for sign in range(2):
        for axial in range(len(event_ptr) - 1):
            row = sign * (len(event_ptr) - 1) + axial
            used = 0
            for slot in range(event_ptr[axial], event_ptr[axial + 1]):
                event = event_order[slot]
                coefficient = weights[sign, event]
                for j in range(indptr[event], indptr[event + 1]):
                    value = coefficient * data[j]
                    if (
                        not np.isfinite(value)
                        or (0 < value < 2.0**-1000)
                        or (value == 0 and coefficient > 0 and data[j] > 0)
                    ):
                        unsafe = True
                    if value <= 0:
                        continue
                    minimum, maximum = min(minimum, value), max(maximum, value)
                    observation = indices[j]
                    if seen[observation] != row:
                        seen[observation] = row
                        touched[used] = observation
                        used += 1
                        sums[observation] = 0.0
                    sums[observation] += value
            if count is not None:
                for k in range(used):
                    observation = touched[k]
                    out_index[cursor + k] = observation
                    out_data[cursor + k] = sums[observation]
            cursor += used
            out_ptr[row + 1] = cursor
    return out_ptr, out_index, out_data, minimum, maximum, unsafe


@dataclass(frozen=True, slots=True)
class NativeStrengthResponse:
    """Bounded axial aggregation; declined blocks keep canonical event arithmetic."""

    response: NativeFixedResponse
    factor_revision: str
    blocks: tuple
    admission: tuple[str, ...]
    retained_bytes: int

    def admission_for(self, tables: NativeStrengthTables) -> tuple[str, ...]:
        """Explain each candidate block's aggregated or canonical arithmetic."""
        if tables.response is not self.response:
            raise ValueError("strength admission requires this exact response")
        reasons = list(self.admission)
        for index, ((nodes, _), aggregate) in enumerate(
            zip(self.response.blocks, self.blocks, strict=True)
        ):
            if aggregate is None:
                continue
            axial, _, lower, upper = aggregate
            selected = tables.values[nodes.grid_index][:, axial]
            positive = selected[selected > 0]
            if len(positive) and (
                np.log2(positive.min()) < lower or np.log2(positive.max()) > upper
            ):
                reasons[index] = "strength_dynamic_range"
        return tuple(reasons)

    def evaluate_many(self, detectors, projection, *, cone_components, strength_tables):
        if not 1 <= len(detectors) <= 8 or len(strength_tables) != len(detectors):
            raise ValueError("strength response batch must contain one to eight aligned candidates")
        for detector, tables in zip(detectors, strength_tables, strict=True):
            if (
                self.response.revision != detector.native_response_revision(projection)
                or self.factor_revision != self.response.factor_revision(detector)
                or tables.response is not self.response
                or tables.revision != self.response.strength_revision(detector)
            ):
                raise ValueError("aggregated response dependencies differ")
            self.response._validate_cone_components(detector, cone_components)
        result = np.zeros((len(detectors), self.response.observation_count))
        densities = [SphericalMosaicDensity(detector.mosaic) for detector in detectors]
        admissions = [self.admission_for(tables) for tables in strength_tables]
        for index, ((nodes, probability), aggregate) in enumerate(
            zip(self.response.blocks, self.blocks, strict=True)
        ):
            for i, (detector, tables) in enumerate(zip(detectors, strength_tables, strict=True)):
                strength = tables.values[nodes.grid_index]
                if admissions[i][index] == "aggregated":
                    axial, matrix, _, _ = aggregate
                    selected = strength[:, axial].reshape(-1)
                    result[i] += matrix.T @ selected
                    continue
                result[i] += _event_mass(
                    nodes,
                    strength,
                    densities[i],
                    detector.instrument.film_thickness_A,
                    detector.intensity_envelope,
                    detector.phase_population_weight
                    * detector.polarization_weight
                    * detector.incident.states.source_weight[nodes.source_state_index],
                    detector.integration_rule.cone_quadrature_order,
                    self.response._cone_values(detector, index, None, cone_components),
                    probability=probability,
                )
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("invalid aggregated native response")
        return result


@dataclass(frozen=True, slots=True)
class NativeConeDensity:
    """Owned two-sign mosaic densities, aligned with one exact response instance.

    Geometry remains immutable and shared. The density owns its actual mosaic
    and spherical law; quadrature order is separate from response admission.
    Retained bytes exclude the shared response and temporary preparation arrays.
    """

    response: NativeFixedResponse
    density: SphericalMosaicDensity
    quadrature_order: int
    values: tuple[FloatArray, ...]
    retained_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.response, NativeFixedResponse) or not isinstance(
            self.density, SphericalMosaicDensity
        ):
            raise TypeError("cone density requires an exact native response and spherical law")
        if type(self.quadrature_order) is not int or self.quadrature_order < 4:
            raise ValueError("cone quadrature order must be an integer of at least four")
        if len(self.values) != len(self.response.blocks):
            raise ValueError("cone density blocks must align with response blocks")
        values = []
        for supplied, (nodes, _) in zip(self.values, self.response.blocks, strict=True):
            reject_complex(supplied, "cone density")
            value = np.array(supplied, dtype=np.float64, copy=True, order="C")
            if (
                value.shape != (2, len(nodes.axial_index))
                or np.any(~np.isfinite(value))
                or np.any(value < 0)
            ):
                raise ValueError(
                    "cone densities must be finite, nonnegative and aligned with both signs"
                )
            value.setflags(write=False)
            values.append(value)
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "retained_bytes", sum(value.nbytes for value in values))


@dataclass(frozen=True, slots=True)
class ConditionalStructureDetector:
    """Shared source/SF/optical continuum for native fits and full-panel images."""

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    source: ConditionalSourceSamples
    incident: IncidentTransportResult
    material: MaterialOptics
    instrument: CompiledInstrument
    strength_model: RevisionedStructureStrengthModel
    mosaic: MosaicParameters
    integration_rule: FiberIntegrationRule = field(default_factory=FiberIntegrationRule)
    specular_stitch_stack: ParrattStitchStack | None = None
    intensity_envelope: SampleQIntensityEnvelope = field(default_factory=SampleQIntensityEnvelope)
    phase_population_weight: float = 1.0
    polarization_weight: float = 1.0
    spatial_quadrature_order: int = 16
    gaussian_tail_radius: float = 8.0
    incidence_axis_angle_rad: float | None = None
    scan_calibration_binding_revision: str | None = None
    proposal_mosaic: MosaicParameters | None = None
    spatial_execution: str = "auto"
    spatial_executor: NativeSpatialExecutor = field(
        default_factory=NativeSpatialExecutor, repr=False, compare=False
    )
    fixed_physics_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if self.spatial_execution not in {"auto", "cpu", "cuda"}:
            raise ValueError("native spatial execution must be auto, cpu or cuda")
        if not isinstance(self.spatial_executor, NativeSpatialExecutor):
            raise TypeError("spatial_executor must be a NativeSpatialExecutor")
        basis, rotation = (
            reciprocal_basis(self.reciprocal_basis_Ainv),
            proper_rotation(self.crystal_to_sample),
        )
        rods = tuple(self.rods)
        if self.proposal_mosaic is not None and not isinstance(
            self.proposal_mosaic, MosaicParameters
        ):
            raise TypeError("proposal_mosaic must be an explicit MosaicParameters value")
        if (
            not rods
            or any(not isinstance(r, Rod) for r in rods)
            or len({(r.h, r.k) for r in rods}) != len(rods)
        ):
            raise ValueError("physical rods must be nonempty and individually unique")
        if not self.rod_catalog_revision:
            raise ValueError("rod catalog revision is required")
        if self.incidence_axis_angle_rad is not None and not np.isfinite(
            self.incidence_axis_angle_rad
        ):
            raise ValueError("incidence axis angle must be finite")
        if (
            self.scan_calibration_binding_revision is not None
            and not self.scan_calibration_binding_revision
        ):
            raise ValueError("scan calibration binding must be nonempty")
        validate_conditional_spatial_support(self.source, self.instrument)
        if (
            self.incident.states.source_revision != self.source.mean_rays.source_revision
            or self.incident.states.material_revision != self.material.material_revision
            or self.incident.states.sample_geometry_revision
            != self.instrument.sample_geometry_revision
            or not np.allclose(
                rotation, self.instrument.sample_from_crystal.rotation, rtol=0, atol=1e-12
            )
        ):
            raise ValueError("conditional source, material and instrument states must agree")
        if not isinstance(self.mosaic, MosaicParameters) or not isinstance(
            self.integration_rule, FiberIntegrationRule
        ):
            raise TypeError("mosaic and integration rule must be explicit typed values")
        if not isinstance(self.intensity_envelope, SampleQIntensityEnvelope):
            raise TypeError("invalid intensity envelope")
        if (
            self.specular_stitch_stack is not None
            and self.specular_stitch_stack.interface_assumption != LOCAL_LAMELLA_INTERFACE
        ):
            raise ValueError("the conditional local chart requires the local-lamella composite")
        if self.specular_stitch_stack is not None and not np.allclose(
            basis[:, 2] / np.linalg.norm(basis[:, 2]), [0.0, 0.0, 1.0], rtol=0, atol=1e-12
        ):
            raise ValueError("the local-lamella chart requires reciprocal b3 along crystal z")
        for name in ("phase_population_weight", "polarization_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError("population/polarization factors must be nonnegative")
            object.__setattr__(self, name, value)
        if type(self.spatial_quadrature_order) is not int or self.spatial_quadrature_order < 4:
            raise ValueError("spatial quadrature order must be an integer of at least four")
        if not np.isfinite(self.gaussian_tail_radius) or self.gaussian_tail_radius <= 0:
            raise ValueError("Gaussian tail radius must be finite and positive")
        basis.setflags(write=False)
        rotation.setflags(write=False)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", rotation)
        object.__setattr__(self, "rods", rods)
        self._validate_strength(self.strength_model)
        rule = self.integration_rule
        finite_stack_resolution(self.strength_model)
        stack = self.specular_stitch_stack
        object.__setattr__(
            self,
            "fixed_physics_revision",
            canonical_revision_sha256(
                ("definition_id", "independent_azimuth_conditional_detector.v3"),
                ("source", self.source.revision),
                ("incident_model", self.incident.states.incident_model_id),
                ("material", self.material.material_revision),
                ("instrument", _instrument_revision(self.instrument)),
                ("basis", basis),
                ("rods", np.array([(r.h, r.k, r.population) for r in rods])),
                ("rod_catalog", self.rod_catalog_revision),
                ("engine", rule.regular_rule),
                ("strength_measure_revision", self.strength_model.structure_model_revision),
                ("integration_rule", json.dumps(asdict(rule), sort_keys=True, allow_nan=False)),
                ("overlap_measure", "none" if stack is None else stack.overlap_measure),
                (
                    "actual_mosaic",
                    np.array(
                        [
                            self.mosaic.gaussian_sigma_rad,
                            self.mosaic.lorentzian_half_width_rad,
                            self.mosaic.lorentzian_probability,
                        ]
                    ),
                ),
                (
                    "reference_mosaic",
                    np.array(
                        [
                            (self.proposal_mosaic or self.mosaic).gaussian_sigma_rad,
                            (self.proposal_mosaic or self.mosaic).lorentzian_half_width_rad,
                            (self.proposal_mosaic or self.mosaic).lorentzian_probability,
                        ]
                    ),
                ),
                (
                    "spatial_rule",
                    np.array([self.spatial_quadrature_order, self.gaussian_tail_radius]),
                ),
                ("weights", np.array([self.phase_population_weight, self.polarization_weight])),
                (
                    "envelope",
                    np.array(
                        [self.intensity_envelope.u_radial_A2, self.intensity_envelope.u_normal_A2]
                    ),
                ),
                ("local_composite", stack is not None),
                ("spatial_execution", self.spatial_execution),
                (
                    "stack",
                    np.array([])
                    if stack is None
                    else np.array(
                        [
                            stack.substrate_refractive_index,
                            stack.top_roughness_A,
                            stack.bottom_roughness_A,
                        ]
                    ),
                ),
            ),
        )

    def _validate_strength(self, model: RevisionedStructureStrengthModel) -> None:
        revision = getattr(model, "structure_model_revision", None)
        if not isinstance(revision, str) or not revision:
            raise ValueError("a structure model revision is required")
        if not np.allclose(
            model.reciprocal_basis_Ainv, self.reciprocal_basis_Ainv, rtol=0, atol=1e-12
        ):
            raise ValueError("strength and detector reciprocal bases disagree")

    @property
    def source_revision(self) -> str:
        return self.source.revision

    @property
    def incidence_angle_static_physics_revision(self) -> str:
        return canonical_revision_sha256(
            ("source", self.source.revision),
            (
                "static_physics",
                _incidence_angle_static_physics_revision(
                    reciprocal_basis_Ainv=self.reciprocal_basis_Ainv,
                    strength_model_revision=self.strength_model.structure_model_revision,
                    mosaic=self.mosaic,
                    intensity_envelope=self.intensity_envelope,
                    material_revision=self.material.material_revision,
                    incident_model_id=self.incident.states.incident_model_id,
                    instrument=self.instrument,
                    phase_polarization_weight=self.phase_population_weight
                    * self.polarization_weight,
                    specular_stitch_stack=self.specular_stitch_stack,
                ),
            ),
        )

    @property
    def detector_shape_rc(self) -> tuple[int, int]:
        return self.instrument.detector_shape_rc

    @property
    def detector_visible_m0_q_gap_Ainv(self) -> float | None:
        if self.specular_stitch_stack is not None or not any(r.h == r.k == 0 for r in self.rods):
            return None
        normal = self.incident.states.k_film_phase_sample_Ainv[self.incident.states.valid, 2]
        return float(np.min(-normal)) if len(normal) else None

    @property
    def detector_panel_revision(self) -> str:
        return _detector_native_chart_revision(self.instrument)

    def restrict_rods(self, rods: tuple[Rod, ...]) -> ConditionalStructureDetector:
        requested = tuple(rods)
        configured = {(r.h, r.k): r for r in self.rods}
        if any((r.h, r.k) not in configured or configured[r.h, r.k] != r for r in requested):
            raise ValueError("a rod subset must retain the configured physical populations")
        return replace(self, rods=requested)

    def _batches(
        self,
        bounds: FloatArray,
        cancel_requested: Callable[[], bool] | None = None,
        *,
        scattering_cache: FiberScatteringCache | None = None,
        include_source_mass: bool = True,
        source_state_indices: tuple[int, ...] | None = None,
        include_local_m0: bool = True,
    ) -> Iterator[ConditionalFiberBatch]:
        return iter_conditional_fiber_transfers(
            rods=self.rods,
            reciprocal_basis_Ainv=self.reciprocal_basis_Ainv,
            crystal_to_sample=self.crystal_to_sample,
            source=self.source,
            incident=self.incident,
            material=self.material,
            instrument=self.instrument,
            native_bounds_px=bounds,
            reference_mosaic=self.proposal_mosaic or self.mosaic,
            rule=self.integration_rule,
            local_stitched_m0=self.specular_stitch_stack is not None,
            cancel_requested=cancel_requested,
            scattering_cache=scattering_cache,
            include_source_mass=include_source_mass,
            source_state_indices=source_state_indices,
            regular_integrator=self._regular_product_batches,
            include_local_m0=include_local_m0,
        )

    def native_response_revision(self, projection):
        """Dependency identity for explicit strength-independent geometry reuse."""
        if self.integration_rule.regular_rule != "fixed_importance.v1":
            raise ValueError("strength-dependent rules cannot compile a fixed response")
        if self.spatial_execution == "cuda":
            raise ValueError("fixed native region compilation is CPU-only; use auto or cpu")
        states = self.incident.states
        return canonical_revision_sha256(
            ("definition", "fixed_importance_native_response.v1"),
            ("projection", projection.projection_revision),
            ("source", self.source.revision),
            ("instrument", _instrument_revision(replace(self.instrument, film_thickness_A=0.0))),
            ("basis", self.reciprocal_basis_Ainv),
            ("rods", np.array([(r.h, r.k, r.population) for r in self.rods])),
            ("wavelength", self.material.wavelength_A),
            ("optics", self.material.n_complex),
            ("incident_model", states.incident_model_id),
            ("direction", states.direction_sample),
            ("intersection", states.sample_intersection_lab_m),
            ("polarization", states.polarization_state_id),
            ("phase_wavevector", states.k_film_phase_sample_Ainv),
            ("entrance", states.entrance_amplitude),
            ("normal_wavevector", states.kz_film_Ainv),
            ("valid", states.valid),
            ("footprint", states.footprint_acceptance),
            ("rule", json.dumps(asdict(self.integration_rule), sort_keys=True, allow_nan=False)),
            ("proposal", json.dumps(asdict(self.proposal_mosaic or self.mosaic), sort_keys=True)),
            ("spatial_rule", np.array([self.spatial_quadrature_order, self.gaussian_tail_radius])),
            ("local_chart", self.specular_stitch_stack is not None),
        )

    def compile_native_response(self, projection, *, maximum_bytes=1024**3, cancel_requested=None):
        """Compile region probabilities with an explicit retained-array byte cap.

        Both Bragg cells and profile memberships use the same spatial owner. No
        strength-selected nodes or detector survivor masks are reused across keys.
        The cap includes grid/node/CSR arrays; temporary transfer/projection storage
        and Python metadata are excluded and require a caller process-memory limit.
        """
        revision = self.native_response_revision(projection)
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("response memory budget must be a positive integer")
        spatial = NativeSpatialRegionProjection(projection)
        grids, blocks, grid_ids = [], [], {}
        retained_bytes = 0
        for batch in self._batches(
            native_projection_bounds_px(projection, self.detector_shape_rc),
            cancel_requested,
            include_source_mass=False,
        ):
            key = (
                batch.radial_Ainv,
                float(self.source.mean_rays.wavelength_A[batch.source_state_index]),
            )
            if key not in grid_ids:
                grid_ids[key] = len(grids)
                grids.append(self._grid(batch))
                grid = grids[-1]
                retained_bytes += grid.positive_axial_Ainv.nbytes
                if grid.external_q_Ainv is not None:
                    retained_bytes += grid.external_q_Ainv.nbytes
            else:
                grid = grids[grid_ids[key]]
                expected = (
                    grid.external_q_Ainv if batch.local_m0 is not None else grid.positive_axial_Ainv
                )
                if grid.rods != batch.rods or not np.array_equal(
                    expected, batch.positive_axial_Ainv
                ):
                    raise ValueError("fixed response group changed its shared axial grid")
            probability = spatial.probabilities(
                batch.transfer.spatial,
                quadrature_order=self.spatial_quadrature_order,
                gaussian_tail_radius=self.gaussian_tail_radius,
                executor=self.spatial_executor,
            )
            retained = np.flatnonzero(np.diff(probability.indptr))
            # Removing empty rows changes only row pointers, not nonzero order.
            probability = csr_matrix(
                (
                    probability.data,
                    probability.indices,
                    np.concatenate((probability.indptr[retained], probability.indptr[-1:])),
                ),
                shape=(len(retained), probability.shape[1]),
                copy=False,
            )
            nodes = _response_nodes(batch, grid_ids[key], retained)
            retained_bytes += sum(
                value.nbytes
                for field in fields(nodes)
                if isinstance(value := getattr(nodes, field.name), np.ndarray)
            ) + sum(a.nbytes for a in (probability.data, probability.indices, probability.indptr))
            if retained_bytes > maximum_bytes:
                raise MemoryError("fixed native response memory budget exhausted")
            for array in (probability.data, probability.indices, probability.indptr):
                array.setflags(write=False)
            blocks.append((nodes, probability))
        return NativeFixedResponse(
            revision, projection.observation_count, tuple(grids), tuple(blocks), retained_bytes
        )

    def _regular_strength_at(self, axial, *, rods, wavelength):
        return self._strength_table(
            FiberStrengthGrid(rods, wavelength, axial, None),
            self.strength_model,
            self.instrument.film_thickness_A,
        )

    def _batch_pixel_patch(self, batch, *, maximum_bytes, density, mass=None):
        """Canonical signed contraction before the existing spatial16 deposition."""
        if mass is None:
            mass = self._batch_event_mass(batch, density=density)
        kernels = batch.transfer.spatial
        bounds = kernels.native_pixel_bounds(
            self.detector_shape_rc,
            integrated_mass=mass,
            gaussian_tail_radius=self.gaussian_tail_radius,
        )
        if bounds is None:
            return None
        r0, r1, c0, c1 = bounds
        if (r1 - r0) * (c1 - c0) * 8 > maximum_bytes:
            raise MemoryError("pixel-error patch budget exhausted before deposition")
        image = kernels.integrate_native_pixels(
            (r1 - r0, c1 - c0),
            integrated_mass=mass,
            quadrature_order=self.spatial_quadrature_order,
            gaussian_tail_radius=self.gaussian_tail_radius,
            row_offset=r0,
            column_offset=c0,
            execution=self.spatial_execution,
            executor=self.spatial_executor,
        )
        return bounds, image

    def _batch_pixel_patches(self, batch, *, panel_count, maximum_bytes, density):
        """Joint cone/event contraction; split only native deposition windows."""
        if batch is None:
            return [(None, None)] * panel_count
        mass = self._batch_event_mass(batch, density=density)
        transfer = batch.transfer
        owner = transfer.quadrature_index // 8
        patches, resident = [], 0
        for panel in range(panel_count):
            take = owner == panel
            if not np.any(take):
                patches.append((None, None))
                continue
            spatial = replace(
                transfer.spatial,
                mean_px=transfer.spatial.mean_px[take],
                factor_px=transfer.spatial.factor_px[take],
                backward_probability_bound=transfer.spatial.backward_probability_bound[take],
            )
            subset = replace(
                transfer,
                spatial=spatial,
                quadrature_index=transfer.quadrature_index[take],
                polar_angle_rad=transfer.polar_angle_rad[take],
                cone_angle_rad=transfer.cone_angle_rad[take],
                coefficient_per_L_rad=transfer.coefficient_per_L_rad[take],
                phase_q_radial_squared_Ainv2=transfer.phase_q_radial_squared_Ainv2[take],
                phase_q_normal_squared_Ainv2=transfer.phase_q_normal_squared_Ainv2[take],
                attenuation_decay_sum_Ainv=(
                    None
                    if transfer.attenuation_decay_sum_Ainv is None
                    else transfer.attenuation_decay_sum_Ainv[take]
                ),
            )
            part = replace(
                batch,
                transfer=subset,
                axial_index=batch.axial_index[take],
                integrated_coefficient=batch.integrated_coefficient[take],
            )
            patch = self._batch_pixel_patch(
                part, maximum_bytes=maximum_bytes - resident, density=density, mass=mass[take]
            )
            resident += 0 if patch is None else patch[1].nbytes
            patches.append((part, patch))
        return patches

    def _regular_product_batches(
        self, *, coordinate_parameters, batch_parameters, cancel_requested, channel_count
    ):
        """Per-source preparation; model/geometry changes always rebuild W and nodes."""
        if identically_zero_strength(self.strength_model):
            return
        c, b = coordinate_parameters, batch_parameters
        lower, upper = c["axial_bounds_Ainv"]
        edges = response_panel_edges(
            lower,
            upper,
            self.reciprocal_basis_Ainv,
            b["rods"],
            b["radial_Ainv"],
            c["ki_sample_Ainv"],
            c["normal_sample"],
            c["source_region_bounds"],
        )
        wavelength = float(self.source.mean_rays.wavelength_A[b["source_state_index"]])
        strength_at = partial(self._regular_strength_at, rods=b["rods"], wavelength=wavelength)
        rule = self.integration_rule
        pixel_patch = partial(
            self._batch_pixel_patches, density=SphericalMosaicDensity(self.mosaic)
        )
        # Relative indicators add over panels. Allocate absolute tolerance over
        # all source/group/panel slots, including those with no retained events.
        absolute_budget = rule.pixel_error_atol / (channel_count * (len(edges) - 1))
        for left, right in pairwise(edges):
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            axial = prepare_positive_axial_rule(
                left,
                right,
                basis=self.reciprocal_basis_Ainv,
                rods=b["rods"],
                model=self.strength_model,
                strength_at=strength_at,
                order=rule.strength_gauss_order,
                scalar_order=rule.strength_scalar_order,
                scalar_phase_step_rad=rule.strength_scalar_phase_step_rad,
            )
            yield from iter_pixel_error_batches(
                axial,
                coordinate_parameters=c,
                batch_parameters=b,
                pixel_patch=pixel_patch,
                absolute_budget=absolute_budget,
                cancel_requested=cancel_requested,
            )

    def _grid(self, batch: ConditionalFiberBatch) -> FiberStrengthGrid:
        wavelength = float(self.source.mean_rays.wavelength_A[batch.source_state_index])
        external = batch.positive_axial_Ainv if batch.local_m0 is not None else None
        axial = batch.positive_axial_Ainv
        if external is not None:
            film = self.material.n_complex[
                np.flatnonzero(self.material.wavelength_A == wavelength)[0]
            ]
            axial = _phase_grid(external, 2 * np.pi / wavelength, film)
        return FiberStrengthGrid(batch.rods, wavelength, axial, external)

    def _strength_table(
        self,
        grid: FiberStrengthGrid,
        model: RevisionedStructureStrengthModel,
        thickness: float,
        *,
        query_cache: AtomicQueryCache | None = None,
    ) -> FloatArray:
        basis = self.reciprocal_basis_Ainv
        b3 = np.linalg.norm(basis[:, 2])
        normal = basis[:, 2] / b3
        offset = np.array([(r.h * basis[:, 0] + r.k * basis[:, 1]) @ normal for r in grid.rods])
        ell = (
            np.array([1.0, -1.0])[None, :, None] * grid.positive_axial_Ainv - offset[:, None, None]
        ) / b3
        models = model.components if isinstance(model, IncoherentStructureMixture) else (model,)
        probabilities = (
            model.probabilities if isinstance(model, IncoherentStructureMixture) else (1.0,)
        )
        result = np.zeros((2, len(grid.positive_axial_Ainv)))
        k0 = 2 * np.pi / grid.wavelength_A
        for component, probability in zip(models, probabilities, strict=True):
            preparation = (
                {"query_cache": query_cache}
                if type(component)
                in (Bi2X3FiniteStackStrength, CifFiniteStackStrength, Pbi2FiniteSurfaceStrength)
                else {}
            )
            value = component.evaluate_hkl(
                h=np.array([r.h for r in grid.rods])[:, None, None],
                k=np.array([r.k for r in grid.rods])[:, None, None],
                L=ell,
                k_norm_Ainv=k0,
                **preparation,
            )
            reject_complex(value, "structure strength")
            value = np.asarray(value, dtype=np.float64)
            if value.shape != ell.shape or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError("structure strength must be finite, nonnegative and grid-aligned")
            if grid.external_q_Ainv is not None:
                film = self.material.n_complex[
                    np.flatnonzero(self.material.wavelength_A == grid.wavelength_A)[0]
                ]
                stitch = compile_parratt_stitch(
                    self.specular_stitch_stack,
                    lambda L, component=component, preparation=preparation: component.evaluate_hkl(
                        h=0, k=0, L=L, k_norm_Ainv=k0, **preparation
                    ),
                    wavelength_A=grid.wavelength_A,
                    film_refractive_index=film,
                    film_thickness_A=thickness,
                    c_A=2 * np.pi / b3,
                    grid_size=self.integration_rule.stitch_grid_size,
                )
                value = _stitch_grid(
                    value[0],
                    grid.external_q_Ainv,
                    k0,
                    film,
                    stitch.substrate_refractive_index,
                    thickness,
                    stitch.top_roughness_A,
                    stitch.bottom_roughness_A,
                    stitch.qc_Ainv,
                    stitch.zero_strength_A2,
                    stitch.dimensionless_scale_factor,
                    *stitch.blend_bounds_q_over_qc,
                )[None, :, :]
            result += probability * np.einsum(
                "r,rst->st", np.array([r.population for r in grid.rods]), value
            )
        return result

    def _batch_event_mass(self, batch, *, density, tables=None):
        strength = batch.signed_strength_fractions
        if strength is None:
            wavelength = float(self.source.mean_rays.wavelength_A[batch.source_state_index])
            key = (batch.radial_Ainv, wavelength)
            if key not in tables:
                tables[key] = self._strength_table(
                    self._grid(batch), self.strength_model, self.instrument.film_thickness_A
                )
            strength = tables[key]
        return _event_mass(
            _response_nodes(batch, 0),
            strength,
            density,
            self.instrument.film_thickness_A,
            self.intensity_envelope,
            self.phase_population_weight * self.polarization_weight,
            self.integration_rule.cone_quadrature_order,
        )

    def _local_pixel_patch(self, batch, tables, density):
        mass = self._batch_event_mass(batch, density=density, tables=tables)
        kernels = batch.transfer.spatial
        bounds = kernels.native_pixel_bounds(
            self.detector_shape_rc,
            integrated_mass=mass,
            gaussian_tail_radius=self.gaussian_tail_radius,
        )
        if bounds is None:
            return None
        r0, r1, c0, c1 = bounds
        value = kernels.integrate_native_pixels(
            (r1 - r0, c1 - c0),
            integrated_mass=mass,
            quadrature_order=self.spatial_quadrature_order,
            gaussian_tail_radius=self.gaussian_tail_radius,
            row_offset=r0,
            column_offset=c0,
            execution=self.spatial_execution,
            executor=self.spatial_executor,
        )
        return bounds, value

    def _pixel_patches(
        self, bounds, *, cancel_requested=None, source_state_indices=None, scattering_cache=None
    ):
        """Compact windows; both signs contract before one spatial projection."""
        tables = {}
        density = SphericalMosaicDensity(self.mosaic)
        for batch in self._batches(
            bounds,
            cancel_requested,
            source_state_indices=source_state_indices,
            scattering_cache=scattering_cache,
        ):
            patch = (
                batch.native_pixel_patch
                if batch.signed_strength_fractions is not None
                else self._local_pixel_patch(batch, tables, density)
            )
            if patch is not None:
                yield patch
            batch = patch = None

    def integrate_native_regions(
        self,
        projection: NativePixelRegionProjection,
        *,
        scattering_cache=None,
        source_state_indices=None,
    ):
        """Reprepare current strength/mosaic/geometry, then project native pixel mass.

        No strength-dependent axial or angular response survives this call.
        Fractional observation coverage multiplies pixel masses exactly once.
        Source subsets retain original probabilities; callers sum disjoint subsets.
        """
        bounds = native_projection_bounds_px(projection, self.detector_shape_rc)
        result = np.zeros(projection.observation_count)
        for patch_bounds, image in self._pixel_patches(
            bounds, scattering_cache=scattering_cache, source_state_indices=source_state_indices
        ):
            projection.accumulate_native_patch(result, patch_bounds, image)
            image = None
        return result

    def density_at(self, column_px: ArrayLike, row_px: ArrayLike) -> FloatArray:
        """Evaluate continuous A2/px2 density through the sole prepared event stream.

        The native-pixel angular indicator does not qualify this distinct pointwise
        observable; callers must assess continuous-region quadrature separately.
        """
        reject_complex(column_px, "column_px")
        reject_complex(row_px, "row_px")
        column, row = np.broadcast_arrays(
            np.asarray(column_px, dtype=np.float64), np.asarray(row_px, dtype=np.float64)
        )
        if np.any(~np.isfinite(column)) or np.any(~np.isfinite(row)):
            raise ValueError("detector coordinates must be finite")
        result = np.zeros(column.shape)
        if not column.size:
            return result
        bounds = np.array(
            [[column.min() - 0.5, column.max() + 0.5, row.min() - 0.5, row.max() + 0.5]]
        )
        tables, density = {}, SphericalMosaicDensity(self.mosaic)
        for batch in self._batches(bounds):
            mass = self._batch_event_mass(batch, density=density, tables=tables)
            result += batch.transfer.spatial.density_at(column, row, integrated_mass=mass)
            batch = mass = None
        return result

    def iter_native_pixel_patches(self, *, batch_offset=0, bin_size_px=1):
        """Yield compact (cell bounds, mass) windows in deterministic checkpoint order.

        Bounds index the output cells (native pixels when bin_size_px is one).
        Bins sum native pixel masses; callers own the accumulated image. Resuming
        reprepares the same candidate stream and skips already accumulated windows.
        """
        rows, columns = self.detector_shape_rc
        if type(batch_offset) is not int or batch_offset < 0:
            raise ValueError("batch_offset must be a nonnegative integer")
        if (
            type(bin_size_px) is not int
            or bin_size_px < 1
            or rows % bin_size_px
            or columns % bin_size_px
        ):
            raise ValueError("bin size must exactly divide the native detector")
        bounds = np.array([[-0.5, columns - 0.5, -0.5, rows - 0.5]])
        count = 0
        for patch in self._pixel_patches(bounds):
            count += 1
            if count <= batch_offset:
                patch = None
                continue
            if bin_size_px == 1:
                yield patch
            else:
                (r0, r1, c0, c1), value = patch
                a, b, c, d = (
                    r0 // bin_size_px,
                    (r1 + bin_size_px - 1) // bin_size_px,
                    c0 // bin_size_px,
                    (c1 + bin_size_px - 1) // bin_size_px,
                )
                padded = np.zeros(((b - a) * bin_size_px, (d - c) * bin_size_px))
                padded[
                    r0 - a * bin_size_px : r1 - a * bin_size_px,
                    c0 - c * bin_size_px : c1 - c * bin_size_px,
                ] = value
                yield (
                    (a, b, c, d),
                    padded.reshape(b - a, bin_size_px, d - c, bin_size_px).sum(axis=(1, 3)),
                )
                value = padded = None
            patch = None
        if batch_offset > count:
            raise ValueError("batch_offset exceeds completed native windows")

    def integrate_native_pixels(self, *, row_bounds=None):
        """Accumulate compact accepted windows in one caller-owned native image."""
        rows, columns = self.detector_shape_rc
        first, stop = (0, rows) if row_bounds is None else row_bounds
        if type(first) is not int or type(stop) is not int or not 0 <= first < stop <= rows:
            raise ValueError("row_bounds must identify ordered native detector rows")
        image = np.zeros((stop - first, columns))
        for (r0, r1, c0, c1), value in self.iter_native_pixel_patches():
            a, b = max(first, r0), min(stop, r1)
            if a < b:
                image[a - first : b - first, c0:c1] += value[a - r0 : b - r0]
            value = None
        return image
