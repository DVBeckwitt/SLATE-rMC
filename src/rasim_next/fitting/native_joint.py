"""One response owner for joint Bi/Pb and acquisition refinement."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Protocol

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.fitting.native_instrument import (
    NATIVE_INSTRUMENT_PARAMETER_NAMES,
    NativeInstrumentModel,
)
from rasim_next.fitting.native_observations import NativeFitObservations
from rasim_next.fitting.native_structure import native_stitch_records
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
    Candidate mosaic densities reuse only for the same response, full mosaic
    and cone order. One packet per response is retained under the explicit total
    ``cone_cache_maximum_bytes`` cap; zero disables retention. Oversized packets
    use the same direct contraction without retaining density arrays.
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

    def __post_init__(self):
        if type(self.cone_cache_maximum_bytes) is not int or self.cone_cache_maximum_bytes < 0:
            raise ValueError("cone cache memory budget must be a nonnegative integer")
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
        physics, arguments, mosaic, stack = self.model.bind(physical, coherent_repeats)
        if self.instrument_model is not None:
            physics = self.instrument_model.bind(physics, values[-18:])
        return physics, arguments, mosaic, stack

    def predict(self, values, coherent_repeats, *, resolve_mosaic_components: bool = False):
        """Raw native masses, or complete (Gaussian/Lorentzian, observation) columns.

        Each pure component prepares its own angular panels, including inactive
        eta-boundary components. Exact component predictions have their own cache
        identity. Their recombination remains subject to numerical qualification.
        """
        if type(resolve_mosaic_components) is not bool:
            raise TypeError("resolve_mosaic_components must be boolean")
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
        values = np.asarray(values, dtype=float)
        candidate_key = coherent_repeats, values.tobytes(), resolve_mosaic_components
        if candidate_key in self._predictions:
            self._predictions.move_to_end(candidate_key)
            return self._predictions[candidate_key]
        physics, arguments, mosaic, stack = self.bind(values, coherent_repeats)
        prediction = sum(
            (
                self._predict_part(
                    part,
                    arguments,
                    mosaic,
                    stack,
                    resolve_mosaic_components=resolve_mosaic_components,
                )
                for part in physics.integration_parts()
            ),
            np.zeros(
                (2, len(self.observations.net_count))
                if resolve_mosaic_components
                else len(self.observations.net_count)
            ),
        )
        prediction.setflags(write=False)
        self._predictions[candidate_key] = prediction
        while len(self._predictions) > 64:
            self._predictions.popitem(last=False)
        object.__setattr__(self, "evaluation_count", self.evaluation_count + 1)
        return prediction

    def _predict_part(self, physics, arguments, mosaic, stack, *, resolve_mosaic_components=False):
        """Reprepare adaptive candidates; reuse only explicitly strength-free responses."""
        if resolve_mosaic_components:
            if mosaic.gaussian_sigma_rad <= 0 or mosaic.lorentzian_half_width_rad <= 0:
                raise ValueError("component predictions require both positive mosaic widths")
            laws = (
                MosaicParameters(mosaic.gaussian_sigma_rad, 0.0, 0.0),
                MosaicParameters(0.0, mosaic.lorentzian_half_width_rad, 1.0),
            )
        else:
            laws = (mosaic,)
        predictions = []
        for law in laws:
            start = perf_counter()
            detector = physics.detector(mosaic=law, **arguments)
            detector = replace(
                detector,
                proposal_mosaic=self.proposal_mosaic,
                specular_stitch_stack=stack,
                spatial_execution=self.spatial_execution,
                spatial_executor=self.spatial_executor,
            )
            projection = self.observations.projection
            if detector.integration_rule.regular_rule == "fixed_importance.v1":
                key = detector.native_response_revision(projection)
                if key not in self._responses:
                    while len(self._responses) >= 2:
                        evicted = next(iter(self._responses))
                        del self._responses[evicted]
                        self._cone_densities.pop(evicted, None)
                    response = detector.compile_native_response(projection)
                    self._responses[key] = response
                    object.__setattr__(self, "compile_count", self.compile_count + 1)
                    object.__setattr__(
                        self, "compile_seconds", self.compile_seconds + perf_counter() - start
                    )
                self._responses.move_to_end(key)
                response = self._responses[key]
                cone = self._cone_densities.get(key)
                order = detector.integration_rule.cone_quadrature_order
                if cone is not None and (
                    cone.response is not response
                    or cone.density.parameters != law
                    or cone.quadrature_order != order
                ):
                    del self._cone_densities[key]
                    cone = None
                if cone is not None:
                    self._cone_densities.move_to_end(key)
                    object.__setattr__(self, "cone_reuse_count", self.cone_reuse_count + 1)
                elif (
                    self.cone_cache_maximum_bytes
                    and response.cone_density_bytes <= self.cone_cache_maximum_bytes
                ):
                    while self._cone_densities and (
                        self.cone_cache_retained_bytes + response.cone_density_bytes
                        > self.cone_cache_maximum_bytes
                    ):
                        self._cone_densities.popitem(last=False)
                    cone = response.compile_cone_density(
                        law, quadrature_order=order, maximum_bytes=self.cone_cache_maximum_bytes
                    )
                    self._cone_densities[key] = cone
                    object.__setattr__(self, "cone_compile_count", self.cone_compile_count + 1)
                predictions.append(response.evaluate(detector, projection, cone_density=cone))
            else:
                predictions.append(
                    detector.integrate_native_regions(
                        projection, scattering_cache=self.scattering_cache
                    )
                )
                object.__setattr__(
                    self, "compile_seconds", self.compile_seconds + perf_counter() - start
                )
                object.__setattr__(self, "compile_count", self.compile_count + 1)
            object.__setattr__(self, "contraction_count", self.contraction_count + 1)
        return np.stack(predictions) if resolve_mosaic_components else predictions[0]

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
