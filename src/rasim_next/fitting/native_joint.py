"""One response owner for joint Bi/Pb and acquisition refinement."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import BiJointModel
from rasim_next.fitting.native_instrument import (
    NATIVE_INSTRUMENT_PARAMETER_NAMES,
    NativeInstrumentModel,
)
from rasim_next.fitting.native_observations import NativeFitObservations
from rasim_next.fitting.native_structure import native_stitch_records
from rasim_next.fitting.pb_native import PbJointModel
from rasim_next.materials.optics import OpticalFactorCache
from rasim_next.ordered.amplitudes import AtomicQueryCache
from rasim_next.pipeline.fiber_detector import FiberScatteringCache
from rasim_next.pipeline.spatial_execution import NativeSpatialExecutor


class NativeRefinementModel(Protocol):
    """Material boundary: explicit coordinates, bound physics and known inactivity."""

    @property
    def parameter_names(self) -> tuple[str, ...]: ...

    @property
    def parameter_units(self) -> tuple[str, ...]: ...

    def bind(self, values, coherent_repeats):
        """Return physics, detector arguments, mosaic and optional specular stack."""
        ...

    def inactive_parameters(self, values) -> dict[str, str]: ...


@dataclass(frozen=True, slots=True)
class NativeJointEvaluator:
    """Bounded explicit caches; every trial rebinds atomic amplitudes and optics.

    Adaptive rules always prepare candidate-dependent responses. The explicit
    fixed importance rule reuses geometry and region probabilities while
    contracting current signed strength, mosaic, attenuation and source masses.
    Pure Gaussian/Lorentzian packets reuse only the exact response, component
    width and cone order, under ``cone_cache_maximum_bytes``. Mixtures combine
    per block without retaining a third packet. Zero disables retention; an
    oversized active pair uses the canonical direct calculation. Strength tables,
    atomic queries and admitted axial aggregates have separate explicit budgets.
    Exact completed candidate predictions reuse
    only within this immutable evaluator and observation projection.
    ``spatial_execution`` defaults to per-batch automatic CPU/CUDA deposition;
    all physical preparation and adaptive acceptance retain their shared owners.
    """

    model: NativeRefinementModel
    observations: NativeFitObservations
    proposal_mosaic: MosaicParameters
    instrument_model: NativeInstrumentModel | None = None
    worker_count: int = 1
    spatial_execution: str = "auto"
    spatial_executor: NativeSpatialExecutor = field(
        default_factory=NativeSpatialExecutor, repr=False, compare=False
    )
    compile_count: int = field(default=0, init=False)
    evaluation_count: int = field(default=0, init=False)
    contraction_count: int = field(default=0, init=False)
    cone_compile_count: int = field(default=0, init=False)
    cone_reuse_count: int = field(default=0, init=False)
    compile_seconds: float = field(default=0, init=False)
    _predictions: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _responses: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _cone_densities: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    scattering_cache: FiberScatteringCache = field(default_factory=FiberScatteringCache, repr=False)
    cone_cache_maximum_bytes: int = 256 * 1024**2
    strength_cache_maximum_bytes: int = 128 * 1024**2
    atomic_query_cache: AtomicQueryCache = field(
        default_factory=AtomicQueryCache, repr=False, compare=False
    )
    optical_factor_cache: OpticalFactorCache = field(
        default_factory=OpticalFactorCache, repr=False, compare=False, kw_only=True
    )
    _strength_tables: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    aggregation_cache_maximum_bytes: int = 128 * 1024**2
    _strength_responses: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _factor_uses: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self):
        if type(self.cone_cache_maximum_bytes) is not int or self.cone_cache_maximum_bytes < 0:
            raise ValueError("cone cache memory budget must be a nonnegative integer")
        if (
            type(self.strength_cache_maximum_bytes) is not int
            or self.strength_cache_maximum_bytes < 0
        ):
            raise ValueError("strength cache memory budget must be a nonnegative integer")
        if not isinstance(self.atomic_query_cache, AtomicQueryCache):
            raise TypeError("atomic_query_cache must be an AtomicQueryCache")
        if not isinstance(self.optical_factor_cache, OpticalFactorCache):
            raise TypeError("optical_factor_cache must be an OpticalFactorCache")
        if (
            type(self.aggregation_cache_maximum_bytes) is not int
            or self.aggregation_cache_maximum_bytes < 0
        ):
            raise ValueError("aggregation cache memory budget must be a nonnegative integer")
        if self.spatial_execution not in {"auto", "cpu", "cuda"}:
            raise ValueError("native spatial execution must be auto, cpu or cuda")
        if not isinstance(self.spatial_executor, NativeSpatialExecutor):
            raise TypeError("spatial_executor must be a NativeSpatialExecutor")
        if type(self.worker_count) is not int or self.worker_count != 1:
            raise ValueError(
                "native preparation uses one worker; use prediction_workers for parallel candidates"
            )

    @property
    def parameter_names(self):
        names = self.model.parameter_names
        return (
            *names,
            *(NATIVE_INSTRUMENT_PARAMETER_NAMES if self.instrument_model is not None else ()),
        )

    @property
    def parameter_units(self):
        """Units are fixed by the physical coordinate contract, never by a plan."""
        return (
            *self.model.parameter_units,
            *(self.instrument_model.parameter_units if self.instrument_model is not None else ()),
        )

    def bind(self, values, coherent_repeats):
        values = np.asarray(values)
        if (
            values.shape != (len(self.parameter_names),)
            or np.iscomplexobj(values)
            or np.any(~np.isfinite(values))
            or type(coherent_repeats) is not int
            or coherent_repeats < 1
        ):
            raise ValueError(
                "native candidate requires a finite aligned vector and positive integer N"
            )
        physical = values[:-18] if self.instrument_model is not None else values
        preparation = (
            {"optical_factor_cache": self.optical_factor_cache}
            if type(self.model) in (BiJointModel, PbJointModel)
            else {}
        )
        physics, arguments, mosaic, stack = self.model.bind(
            physical, coherent_repeats, **preparation
        )
        if self.instrument_model is not None:
            preparation = (
                {"optical_factor_cache": self.optical_factor_cache}
                if type(self.instrument_model) is NativeInstrumentModel
                else {}
            )
            physics = self.instrument_model.bind(physics, values[-18:], **preparation)
        return physics, arguments, mosaic, stack

    def predict(self, values, coherent_repeats, *, resolve_mosaic_components=False):
        """Complete raw native masses with the same preparation as batched calls."""
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("a native candidate must be a vector")
        return self.predict_many(
            values[None, :], coherent_repeats, resolve_mosaic_components=resolve_mosaic_components
        )[0]

    def predict_many(self, values, coherent_repeats, *, resolve_mosaic_components=False):
        """Reuse exact dependencies across bounded groups; preserve input ordering."""
        values = np.asarray(values)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.parameter_names)
            or np.iscomplexobj(values)
            or np.any(~np.isfinite(values))
            or type(coherent_repeats) is not int
            or coherent_repeats < 1
        ):
            raise ValueError("native candidates require finite aligned rows and positive integer N")
        if type(resolve_mosaic_components) is not bool:
            raise TypeError("resolve_mosaic_components must be boolean")
        values = np.asarray(values, dtype=np.float64)
        shape = (
            (len(values), 2, len(self.observations.net_count))
            if resolve_mosaic_components
            else (len(values), len(self.observations.net_count))
        )
        result = np.empty(shape)
        for start in range(0, len(values), 8):
            rows = values[start : start + 8]
            keys = [(coherent_repeats, row.tobytes(), resolve_mosaic_components) for row in rows]
            missing = {}
            for key, row in zip(keys, rows, strict=True):
                if key not in self._predictions:
                    missing.setdefault(key, row)
            predictions = {key: np.zeros(shape[1:]) for key in missing}
            contributions = {key: [] for key in missing}
            groups = {}
            for key, row in missing.items():
                physics, arguments, mosaic, stack = self.bind(row, coherent_repeats)
                if resolve_mosaic_components:
                    if mosaic.gaussian_sigma_rad <= 0 or mosaic.lorentzian_half_width_rad <= 0:
                        raise ValueError(
                            "component predictions require both positive mosaic widths"
                        )
                    laws = (
                        MosaicParameters(mosaic.gaussian_sigma_rad, 0.0, 0.0),
                        MosaicParameters(0.0, mosaic.lorentzian_half_width_rad, 1.0),
                    )
                else:
                    laws = (mosaic,)
                for part in physics.integration_parts():
                    for component, law in enumerate(laws):
                        slot = len(contributions[key])
                        contributions[key].append(None)
                        detector = replace(
                            part.detector(mosaic=law, **arguments),
                            proposal_mosaic=self.proposal_mosaic,
                            specular_stitch_stack=stack,
                            spatial_execution=self.spatial_execution,
                            spatial_executor=self.spatial_executor,
                        )
                        if detector.integration_rule.regular_rule == "fixed_importance.v1":
                            revision = detector.native_response_revision(
                                self.observations.projection
                            )
                            groups.setdefault(revision, []).append((key, component, detector, slot))
                        else:
                            begun = perf_counter()
                            raw = detector.integrate_native_regions(
                                self.observations.projection, scattering_cache=self.scattering_cache
                            )
                            contributions[key][slot] = component, raw
                            object.__setattr__(
                                self,
                                "compile_seconds",
                                self.compile_seconds + perf_counter() - begun,
                            )
                            object.__setattr__(self, "compile_count", self.compile_count + 1)
                            object.__setattr__(
                                self, "contraction_count", self.contraction_count + 1
                            )
            for revision, entries in groups.items():
                response = self._response(entries[0][2], revision)
                # Each batch binds a single actual mosaic. Width probes therefore
                # do not retain a packet per candidate outside the cone byte cap.
                compatible = {}
                for entry in entries:
                    compatible.setdefault(entry[2].mosaic, []).append(entry)
                for subset in compatible.values():
                    for offset in range(0, len(subset), 8):
                        chunk = subset[offset : offset + 8]
                        detectors = [entry[2] for entry in chunk]
                        cones = self._cones(response, detectors[0], revision)
                        tables = [
                            self._tables(response, detector, revision) for detector in detectors
                        ]
                        factors = {response.factor_revision(detector) for detector in detectors}
                        aggregate = (
                            self._aggregate(response, detectors[0], revision, cones, len(chunk))
                            if len(factors) == 1
                            else None
                        )
                        if aggregate is not None:
                            raw = aggregate.evaluate_many(
                                detectors,
                                self.observations.projection,
                                cone_components=cones,
                                strength_tables=tables,
                            )
                        elif len(chunk) == 1:
                            raw = response.evaluate(
                                detectors[0],
                                self.observations.projection,
                                cone_components=cones,
                                strength_tables=tables[0],
                            )[None, :]
                        else:
                            raw = response.evaluate_many(
                                detectors,
                                self.observations.projection,
                                cone_components=[cones] * len(chunk),
                                strength_tables=tables,
                            )
                        for (key, component, _, slot), row in zip(chunk, raw, strict=True):
                            contributions[key][slot] = component, row
                        object.__setattr__(
                            self, "contraction_count", self.contraction_count + len(chunk)
                        )
            for key, prediction in predictions.items():
                # Response grouping must not reorder the physical part reduction.
                for component, raw in contributions[key]:
                    if resolve_mosaic_components:
                        prediction[component] += raw
                    else:
                        prediction += raw
                prediction.setflags(write=False)
                self._predictions[key] = prediction
                object.__setattr__(self, "evaluation_count", self.evaluation_count + 1)
            for i, key in enumerate(keys):
                self._predictions.move_to_end(key)
                result[start + i] = self._predictions[key]
            while len(self._predictions) > 64:
                self._predictions.popitem(last=False)
        result.setflags(write=False)
        return result

    def _response(self, detector, revision):
        if revision not in self._responses:
            while len(self._responses) >= 2:
                evicted = next(iter(self._responses))
                del self._responses[evicted]
                for cache in (
                    self._cone_densities,
                    self._strength_tables,
                    self._strength_responses,
                    self._factor_uses,
                ):
                    for key in tuple(cache):
                        if key[0] == evicted:
                            del cache[key]
            begun = perf_counter()
            self._responses[revision] = detector.compile_native_response(
                self.observations.projection
            )
            object.__setattr__(self, "compile_count", self.compile_count + 1)
            object.__setattr__(
                self, "compile_seconds", self.compile_seconds + perf_counter() - begun
            )
        self._responses.move_to_end(revision)
        return self._responses[revision]

    def _cones(self, response, detector, revision):
        mosaic = detector.mosaic
        eta = mosaic.lorentzian_probability
        if (
            response.cone_density_bytes * (int(eta < 1) + int(eta > 0))
            > self.cone_cache_maximum_bytes
            or not self.cone_cache_maximum_bytes
        ):
            return None
        order = detector.integration_rule.cone_quadrature_order
        laws = (
            MosaicParameters(mosaic.gaussian_sigma_rad, 0.0, 0.0),
            MosaicParameters(0.0, mosaic.lorentzian_half_width_rad, 1.0),
        )
        active = [
            (revision, law, order)
            for law, weight in zip(laws, (1 - eta, eta), strict=True)
            if weight > 0
        ]
        # Reserve the whole active pair before compiling; inactive old widths may evict.
        required = sum(
            response.cone_density_bytes for key in active if key not in self._cone_densities
        )
        for key in tuple(self._cone_densities):
            if self.cone_cache_retained_bytes + required <= self.cone_cache_maximum_bytes:
                break
            if key not in active:
                del self._cone_densities[key]
        components = []
        missing = [key for key in active if key not in self._cone_densities]
        prepared = response.compile_cone_components(
            tuple(key[1] for key in missing),
            quadrature_order=order,
            maximum_bytes=self.cone_cache_maximum_bytes,
        )
        for key, packet in zip(missing, prepared, strict=True):
            self._cone_densities[key] = packet
            object.__setattr__(self, "cone_compile_count", self.cone_compile_count + 1)
        for law, weight in zip(laws, (1 - eta, eta), strict=True):
            if weight == 0:
                components.append(None)
                continue
            key = revision, law, order
            packet = self._cone_densities[key]
            if key not in missing:
                if packet.response is not response:
                    raise ValueError("cone cache response ownership differs")
                self._cone_densities.move_to_end(key)
                object.__setattr__(self, "cone_reuse_count", self.cone_reuse_count + 1)
            components.append(packet)
        return tuple(components)

    def _tables(self, response, detector, revision):
        key = revision, response.strength_revision(detector)
        tables = self._strength_tables.get(key)
        if tables is not None:
            self._strength_tables.move_to_end(key)
            return tables
        tables = response.compile_strength_tables(detector, query_cache=self.atomic_query_cache)
        if (
            self.strength_cache_maximum_bytes
            and tables.retained_bytes <= self.strength_cache_maximum_bytes
        ):
            while (
                self._strength_tables
                and sum(t.retained_bytes for t in self._strength_tables.values())
                + tables.retained_bytes
                > self.strength_cache_maximum_bytes
            ):
                self._strength_tables.popitem(last=False)
            self._strength_tables[key] = tables
        return tables

    def _aggregate(self, response, detector, revision, cones, candidate_count):
        if not self.aggregation_cache_maximum_bytes or cones is None:
            return None
        key = revision, response.factor_revision(detector)
        uses = self._factor_uses.get(key, 0) + candidate_count
        self._factor_uses[key] = uses
        self._factor_uses.move_to_end(key)
        while len(self._factor_uses) > 16:
            self._factor_uses.popitem(last=False)
        if uses < 2:
            return None
        aggregate = self._strength_responses.get(key)
        if aggregate is None:
            # Evict before construction; its cap includes temporary aggregation work.
            self._strength_responses.clear()
            aggregate = response.compile_strength_response(
                detector,
                self.observations.projection,
                cone_components=cones,
                maximum_bytes=self.aggregation_cache_maximum_bytes,
            )
            self._strength_responses[key] = aggregate
        return aggregate if aggregate.retained_bytes else None

    @property
    def cone_cache_retained_bytes(self):
        """Additional owned float64 storage; response geometry is counted separately."""
        return sum(value.retained_bytes for value in self._cone_densities.values())

    def clear_responses(self):
        self.scattering_cache.entries.clear()
        self.scattering_cache.retained_bytes = 0
        self._predictions.clear()
        self._responses.clear()
        self._cone_densities.clear()
        self._strength_tables.clear()
        self._strength_responses.clear()
        self._factor_uses.clear()
        self.atomic_query_cache.clear()
        self.optical_factor_cache.clear()

    def inactive_parameters(self, values, coherent_repeats):
        """Exact physical inactivity at mixture boundaries; coordinates stay free.

        These statements concern the declared integral, not finite-quadrature
        derivatives or a claim that every remaining parameter is identifiable.
        """
        self.bind(values, coherent_repeats)
        parameters = dict(zip(self.parameter_names, values, strict=True))
        inactive = {}
        eta = parameters.get("lorentzian_probability")
        if eta == 0:
            inactive["lorentzian_half_width_rad"] = "zero Lorentzian population"
        if eta == 1:
            inactive["gaussian_sigma_rad"] = "zero Gaussian population"
        if parameters.get("surface_fraction_0") == 1:
            inactive["surface_1_share_of_remainder"] = "zero surface remainder"
        inactive.update(
            self.model.inactive_parameters(values[:-18] if self.instrument_model else values)
        )
        if self.instrument_model is not None:
            probability = parameters["source_line_0_probability"]
            if probability in (0, 1):
                absent = 0 if probability == 0 else 1
                inactive[f"source_line_{absent}_wavelength_A"] = "zero spectral line probability"
        return inactive

    def stitch_state(self, values, coherent_repeats):
        """Surface/wavelength handoff selections and normalization at this candidate."""
        physics, arguments, _, stack = self.bind(values, coherent_repeats)
        return [
            record
            for part in physics.integration_parts()
            if any(r.h == r.k == 0 for r in part.rods)
            for record in native_stitch_records(part, arguments, stack)
        ]
