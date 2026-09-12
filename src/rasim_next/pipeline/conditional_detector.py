"""One continuous conditional-position detector for native fitting and rendering."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from painted_ewald import MosaicParameters, Rod
from painted_ewald.normal_density import SphericalMosaicDensity
from painted_ewald.validation import proper_rotation, reciprocal_basis, reject_complex
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
from rasim_next.optics.attenuation import uniform_depth_attenuation
from rasim_next.pipeline._continuous_detector_kernel import (
    _empirical_parratt_strength_A2,
    local_m0_phase_q_Ainv,
)
from rasim_next.pipeline.bragg_space import (
    IncoherentStructureMixture,
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
    iter_conditional_fiber_transfers,
)
from rasim_next.pipeline.source_spatial import (
    NativeSpatialRegionProjection,
    validate_conditional_spatial_support,
)
from rasim_next.reflectivity.specular import (
    LOCAL_LAMELLA_INTERFACE,
    ParrattStitchStack,
    compile_parratt_stitch,
)
from rasim_next.sampling.source import ConditionalSourceSamples

FloatArray = NDArray[np.float64]


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
    """Physical quadrature factors after the spatial kernels have been reduced."""

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


def _project_batches(batches, projector, order, tail_radius, worker_count):
    """At most worker_count spatial jobs in flight; yield in source iterator order."""
    if worker_count == 1:
        for batch in batches:
            yield (
                batch,
                projector.probabilities(
                    batch.transfer.spatial, quadrature_order=order, gaussian_tail_radius=tail_radius
                ),
            )
        return
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = deque()
        for batch in batches:
            future = executor.submit(
                projector.probabilities,
                batch.transfer.spatial,
                quadrature_order=order,
                gaussian_tail_radius=tail_radius,
            )
            pending.append((batch, future))
            if len(pending) >= worker_count:
                first, result = pending.popleft()
                yield first, result.result()
        for first, result in pending:
            yield first, result.result()


def _event_mass(
    nodes: FiberResponseNodes,
    strength: FloatArray,
    density: SphericalMosaicDensity,
    thickness_A: float,
    envelope: SampleQIntensityEnvelope,
    weight: float,
    cone_quadrature_order: int = 16,
    cone_density: FloatArray | None = None,
) -> FloatArray:
    """Shared exact SF/cone contraction, before native-region or pixel deposition."""
    index = nodes.axial_index
    if cone_density is None:
        cone_density = np.array(
            [
                density.cone_average_sr_inv(
                    nodes.polar_angle_rad, opening, quadrature_order=cone_quadrature_order
                )
                for opening in (nodes.cone_angle_rad, np.pi - nodes.cone_angle_rad)
            ]
        )
    mass = strength[0, index] * cone_density[0] + strength[1, index] * cone_density[1]
    coefficient = (
        nodes.integrated_coefficient
        * weight
        * np.exp(
            -envelope.u_radial_A2 * nodes.phase_q_radial_squared_Ainv2
            - envelope.u_normal_A2 * nodes.phase_q_normal_squared_Ainv2
        )
    )
    decay = nodes.attenuation_decay_sum_Ainv
    if decay is not None:
        coefficient *= uniform_depth_attenuation(
            decay, 0.0, thickness_A
        ) / uniform_depth_attenuation(decay, 0.0, nodes.reference_thickness_A)
    return mass * coefficient


@dataclass(frozen=True, slots=True)
class NativeMosaicCache:
    """Explicit bounded execution state for one immutable response's cone averages.

    Retain one width/order entry per normalized component. Eta mixes the two
    probability laws; thickness, structure and attenuation do not alter them.
    """

    response: NativeFiberResponse
    _gaussian: tuple | None = field(default=None, init=False, repr=False)
    _lorentzian: tuple | None = field(default=None, init=False, repr=False)

    def components(self, density: SphericalMosaicDensity, order: int) -> tuple:
        parameters = density.parameters
        components = []
        for name, width, active, pure in (
            (
                "_gaussian",
                parameters.gaussian_sigma_rad,
                parameters.lorentzian_probability < 1,
                MosaicParameters(parameters.gaussian_sigma_rad, 0.0, 0.0),
            ),
            (
                "_lorentzian",
                parameters.lorentzian_half_width_rad,
                parameters.lorentzian_probability > 0,
                MosaicParameters(0.0, parameters.lorentzian_half_width_rad, 1.0),
            ),
        ):
            if not active:
                components.append(None)
                continue
            saved = getattr(self, name)
            if saved is None or saved[:2] != (width, order):
                law = SphericalMosaicDensity(pure)
                values = []
                for node in self.response.nodes:
                    value = np.array(
                        [
                            law.cone_average_sr_inv(
                                node.polar_angle_rad, opening, quadrature_order=order
                            )
                            for opening in (node.cone_angle_rad, np.pi - node.cone_angle_rad)
                        ]
                    )
                    value.setflags(write=False)
                    values.append(value)
                saved = (width, order, tuple(values))
                object.__setattr__(self, name, saved)
            components.append(saved[2])
        return tuple(components)


@dataclass(frozen=True, slots=True)
class NativeFiberResponse:
    """Immutable native observable; contains no retained Gaussian image kernels."""

    detector: ConditionalStructureDetector
    grids: tuple[FiberStrengthGrid, ...]
    nodes: tuple[FiberResponseNodes, ...]
    region_probability: tuple[csr_matrix, ...]
    observation_measure_px2: FloatArray
    projection_revision: str
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        grids, nodes = tuple(self.grids), tuple(self.nodes)
        area = _frozen(self.observation_measure_px2)
        if (
            np.any(area < 0)
            or not self.projection_revision
            or len(nodes) != len(self.region_probability)
        ):
            raise ValueError("native response requires aligned probabilities and declared support")
        probabilities = []
        for node, supplied in zip(nodes, self.region_probability, strict=True):
            if not isinstance(supplied, csr_matrix) or not 0 <= node.grid_index < len(grids):
                raise ValueError("native response grid and sparse probabilities must align")
            probability = supplied.copy()
            probability.sum_duplicates()
            probability.sort_indices()
            if (
                probability.shape != (len(node.axial_index), len(area))
                or np.any(~np.isfinite(probability.data))
                or np.any(probability.data < 0)
                or np.any(node.axial_index >= len(grids[node.grid_index].positive_axial_Ainv))
            ):
                raise ValueError("native probabilities or axial assignments are invalid")
            for array in (probability.data, probability.indices, probability.indptr):
                array.setflags(write=False)
            probabilities.append(probability)
        object.__setattr__(self, "grids", grids)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "region_probability", tuple(probabilities))
        object.__setattr__(self, "observation_measure_px2", area)
        object.__setattr__(
            self,
            "response_revision",
            canonical_revision_sha256(
                ("definition_id", "continuous_conditional_native_response.v1"),
                ("detector", self.detector.fixed_physics_revision),
                ("reference_structure", self.detector.strength_model.structure_model_revision),
                ("projection", self.projection_revision),
            ),
        )

    def evaluate(
        self,
        strength_model: RevisionedStructureStrengthModel | None = None,
        *,
        mosaic: MosaicParameters | None = None,
        thickness_A: float | None = None,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
        specular_stitch_stack: ParrattStitchStack | None = None,
        mosaic_cache: NativeMosaicCache | None = None,
        cone_quadrature_order: int | None = None,
    ) -> FloatArray:
        """Return raw integrated A² per native observation, with one shared scale owner."""
        detector = self.detector
        if specular_stitch_stack is not None:
            if detector.specular_stitch_stack is None:
                raise ValueError("response reuse cannot add a local specular channel")
            detector = replace(detector, specular_stitch_stack=specular_stitch_stack)
        model = detector.strength_model if strength_model is None else strength_model
        detector._validate_strength(model)
        thickness = (
            detector.instrument.film_thickness_A if thickness_A is None else float(thickness_A)
        )
        if not np.isfinite(thickness) or thickness < 0:
            raise ValueError("thickness must be finite and nonnegative")
        density = SphericalMosaicDensity(detector.mosaic if mosaic is None else mosaic)
        order = (
            detector.integration_rule.cone_quadrature_order
            if cone_quadrature_order is None
            else cone_quadrature_order
        )
        if type(order) is not int or order < 4:
            raise ValueError("cone quadrature order must be an integer of at least four")
        components = None
        if mosaic_cache is not None:
            if mosaic_cache.response is not self:
                raise ValueError("mosaic cache belongs to another native response")
            components = mosaic_cache.components(density, order)
        envelope = detector.intensity_envelope if intensity_envelope is None else intensity_envelope
        tables = [detector._strength_table(grid, model, thickness) for grid in self.grids]
        result = np.zeros(len(self.observation_measure_px2))
        for i, (node, probability) in enumerate(
            zip(self.nodes, self.region_probability, strict=True)
        ):
            cone_density = None
            if components is not None:
                gaussian, lorentzian = components
                eta = density.parameters.lorentzian_probability
                cone_density = np.zeros((2, len(node.axial_index)))
                if gaussian is not None:
                    cone_density += (1 - eta) * gaussian[i]
                if lorentzian is not None:
                    cone_density += eta * lorentzian[i]
            mass = _event_mass(
                node,
                tables[node.grid_index],
                density,
                thickness,
                envelope,
                detector.phase_population_weight * detector.polarization_weight,
                order,
                cone_density,
            )
            result += mass @ probability
        return result


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
    fixed_physics_revision: str = field(init=False)

    def __post_init__(self) -> None:
        basis, rotation = (
            reciprocal_basis(self.reciprocal_basis_Ainv),
            proper_rotation(self.crystal_to_sample),
        )
        rods = tuple(self.rods)
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
        stack = self.specular_stitch_stack
        object.__setattr__(
            self,
            "fixed_physics_revision",
            canonical_revision_sha256(
                ("definition_id", "independent_azimuth_conditional_detector.v1"),
                ("source", self.source.revision),
                ("incident_model", self.incident.states.incident_model_id),
                ("material", self.material.material_revision),
                ("instrument", _instrument_revision(self.instrument)),
                ("basis", basis),
                ("rods", np.array([(r.h, r.k, r.population) for r in rods])),
                ("rod_catalog", self.rod_catalog_revision),
                (
                    "proposal",
                    np.array(
                        [
                            rule.axial_power,
                            rule.angular_power,
                            rule.axial_peak_spacing_L,
                            rule.axial_peak_half_width_L,
                            rule.source_latent_radius,
                            rule.maximum_backward_probability,
                            rule.cone_quadrature_order,
                            rule.stitch_grid_size,
                        ]
                    ),
                ),
                ("proposal_seed", rule.seed),
                (
                    "reference_mosaic",
                    np.array(
                        [
                            self.mosaic.gaussian_sigma_rad,
                            self.mosaic.lorentzian_half_width_rad,
                            self.mosaic.lorentzian_probability,
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
        self, bounds: FloatArray, cancel_requested: Callable[[], bool] | None = None
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
            reference_mosaic=self.mosaic,
            rule=self.integration_rule,
            local_stitched_m0=self.specular_stitch_stack is not None,
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
        self, grid: FiberStrengthGrid, model: RevisionedStructureStrengthModel, thickness: float
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
            value = component.evaluate_hkl(
                h=np.array([r.h for r in grid.rods])[:, None, None],
                k=np.array([r.k for r in grid.rods])[:, None, None],
                L=ell,
                k_norm_Ainv=k0,
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
                    lambda L, component=component: component.evaluate_hkl(
                        h=0, k=0, L=L, k_norm_Ainv=k0
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

    def compile_native_response(
        self,
        projection: NativePixelRegionProjection,
        *,
        worker_count: int = 1,
        spatial_projection: NativeSpatialRegionProjection | None = None,
    ) -> NativeFiberResponse:
        """Compile probabilities with bounded parallel work and fixed reduction order."""
        if type(worker_count) is not int or worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        if projection.detector_shape_rc != self.detector_shape_rc:
            raise ValueError("native projection uses a different detector")
        owner = projection.observation_row
        row, column = np.divmod(
            projection.flat_pixel_index[projection.pixel_column_index], self.detector_shape_rc[1]
        )
        low = np.full((projection.observation_count, 2), np.inf)
        high = -low.copy()
        np.minimum.at(low, owner, np.column_stack((column, row)) - 0.5)
        np.maximum.at(high, owner, np.column_stack((column, row)) + 0.5)
        bounds = np.column_stack((low[:, 0], high[:, 0], low[:, 1], high[:, 1]))
        bounds = bounds[np.all(np.isfinite(bounds), axis=1)]
        if spatial_projection is not None and (
            not isinstance(spatial_projection, NativeSpatialRegionProjection)
            or spatial_projection.projection is not projection
        ):
            raise ValueError("spatial projection belongs to another native observation operator")
        projector = (
            spatial_projection
            if spatial_projection is not None
            else NativeSpatialRegionProjection(projection)
        )
        grids, nodes, probabilities, indices = [], [], [], {}
        if len(bounds):
            for batch, probability in _project_batches(
                self._batches(bounds),
                projector,
                self.spatial_quadrature_order,
                self.gaussian_tail_radius,
                worker_count,
            ):
                wavelength = float(self.source.mean_rays.wavelength_A[batch.source_state_index])
                key = (batch.radial_Ainv, wavelength)
                if key not in indices:
                    indices[key] = len(grids)
                    grids.append(self._grid(batch))
                if not probability.nnz:
                    continue
                retained = np.flatnonzero(np.diff(probability.indptr))
                nodes.append(_response_nodes(batch, indices[key], retained))
                probabilities.append(probability[retained])
        return NativeFiberResponse(
            self,
            tuple(grids),
            tuple(nodes),
            tuple(probabilities),
            projection.observation_measure_px2,
            projection.projection_revision,
        )

    def _intensity_batches(
        self, bounds: FloatArray, cancel_requested: Callable[[], bool] | None = None
    ):
        tables = {}
        density = SphericalMosaicDensity(self.mosaic)
        for batch in self._batches(bounds, cancel_requested):
            wavelength = float(self.source.mean_rays.wavelength_A[batch.source_state_index])
            key = (batch.radial_Ainv, wavelength)
            if key not in tables:
                tables[key] = self._strength_table(
                    self._grid(batch), self.strength_model, self.instrument.film_thickness_A
                )
            mass = _event_mass(
                _response_nodes(batch, 0),
                tables[key],
                density,
                self.instrument.film_thickness_A,
                self.intensity_envelope,
                self.phase_population_weight * self.polarization_weight,
                self.integration_rule.cone_quadrature_order,
            )
            yield batch.transfer.spatial, mass

    def density_at(self, column_px: ArrayLike, row_px: ArrayLike) -> FloatArray:
        column, row = np.broadcast_arrays(
            np.asarray(column_px, dtype=float), np.asarray(row_px, dtype=float)
        )
        if np.any(~np.isfinite(column)) or np.any(~np.isfinite(row)):
            raise ValueError("detector coordinates must be finite")
        result = np.zeros(column.shape)
        if not column.size:
            return result
        bounds = np.array(
            [[column.min() - 0.5, column.max() + 0.5, row.min() - 0.5, row.max() + 0.5]]
        )
        for kernels, mass in self._intensity_batches(bounds):
            result += kernels.density_at(column, row, integrated_mass=mass)
        return result

    def integrate_native_pixels(self) -> FloatArray:
        """Integrate the continuous detector function over every native pixel."""
        rows, columns = self.detector_shape_rc
        image = np.zeros((rows, columns))
        bounds = np.array([[-0.5, columns - 0.5, -0.5, rows - 0.5]])
        for kernels, mass in self._intensity_batches(bounds):
            image += kernels.integrate_native_pixels(
                self.detector_shape_rc,
                integrated_mass=mass,
                quadrature_order=self.spatial_quadrature_order,
                gaussian_tail_radius=self.gaussian_tail_radius,
            )
        return image

    def sample_native_pixel_mass(
        self,
        *,
        draws_per_batch: int,
        seed: int,
        draw_offset: int = 0,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> FloatArray:
        """Monte Carlo kernel selection, with continuous beam position integrated."""
        if type(draws_per_batch) is not int or draws_per_batch <= 0:
            raise ValueError("draws_per_batch must be positive")
        if type(draw_offset) is not int or draw_offset < 0 or type(seed) is not int or seed < 0:
            raise ValueError("draw offset and seed must be nonnegative integers")
        if cancel_requested is not None and cancel_requested():
            raise CancelledError
        rows, columns = self.detector_shape_rc
        image = np.zeros((rows, columns))
        bounds = np.array([[-0.5, columns - 0.5, -0.5, rows - 0.5]])
        for batch_index, (kernels, mass) in enumerate(
            self._intensity_batches(bounds, cancel_requested)
        ):
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            if mass.sum() <= 0:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, batch_index]))
            rng.random(draw_offset)
            image += kernels.sample_native_pixel_mass(
                draws_per_batch,
                self.detector_shape_rc,
                integrated_mass=mass,
                rng=rng,
                quadrature_order=self.spatial_quadrature_order,
                gaussian_tail_radius=self.gaussian_tail_radius,
            )
        if cancel_requested is not None and cancel_requested():
            raise CancelledError
        return image
