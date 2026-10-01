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
from rasim_next.pipeline.conditional_detector import NativeMosaicCache
from rasim_next.pipeline.detector_revisions import _instrument_revision
from rasim_next.pipeline.fiber_detector import FiberScatteringCache
from rasim_next.pipeline.source_spatial import NativeSpatialRegionProjection


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

    The reference projection, numerical proposal and instrument owner are fixed.
    Source, optics, basis or rigid geometry changes invalidate compiled responses.
    Film thickness, ADPs and stacking reuse geometry and re-evaluate intensities.
    """

    model: NativeRefinementModel
    observations: NativeFitObservations
    proposal_mosaic: MosaicParameters
    instrument_model: NativeInstrumentModel | None = None
    worker_count: int = 1
    compile_count: int = field(default=0, init=False)
    evaluation_count: int = field(default=0, init=False)
    contraction_count: int = field(default=0, init=False)
    compile_seconds: float = field(default=0, init=False)
    _responses: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _predictions: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    scattering_cache: FiberScatteringCache = field(default_factory=FiberScatteringCache, repr=False)
    _spatial_projection: NativeSpatialRegionProjection | None = field(
        default=None, init=False, repr=False
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

        Component requests share response and strength work. Their cache identity
        is distinct from mixed predictions; contraction_count includes each
        component in every disjoint source/rod partition. No eta is extrapolated.
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

    def predict_axial_panels(self, values, coherent_repeats):
        """Return raw (panel, observation) contributions in declared mesh order.

        This decomposition uses the ordinary response and physical contraction;
        it is not an alternative detector model or a normalized partial signal.
        """
        physics, arguments, mosaic, stack = self.bind(values, coherent_repeats)
        meshes = physics.integration_rule.axial_meshes
        if not meshes:
            raise ValueError("panel predictions require explicit axial meshes")
        return sum(
            (
                self._predict_part(part, arguments, mosaic, stack, resolve_axial_panels=True)
                for part in physics.integration_parts()
            ),
            np.zeros(
                (sum(len(m.edges_Ainv) - 1 for m in meshes), len(self.observations.net_count))
            ),
        )

    def _predict_part(
        self,
        physics,
        arguments,
        mosaic,
        stack,
        *,
        resolve_axial_panels=False,
        resolve_mosaic_components=False,
    ):
        """Evaluate one disjoint rod partition with its normalized source rule."""
        detector = physics.detector(mosaic=self.proposal_mosaic, **arguments)
        # Thickness changes attenuation weights, never the accepted spatial ray map.
        # This backend rejects finite footprint/external absorption at construction.
        key = (
            physics.rods,
            physics.integration_rule,
            physics.material.material_revision,
            physics.reciprocal_basis_Ainv.tobytes(),
            physics.source.mean_rays.origin_lab_m.tobytes(),
            physics.source.mean_rays.direction_lab.tobytes(),
            physics.source.mean_rays.wavelength_A.tobytes(),
            physics.source.mean_rays.polarization_state_id,
            physics.source.conditional_origin_factor_lab_m.tobytes(),
            _instrument_revision(replace(physics.instrument, film_thickness_A=1.0)),
        )
        if key in self._responses:
            cache = self._responses.pop(key)
        else:
            start = perf_counter()
            if self._spatial_projection is None:
                object.__setattr__(
                    self,
                    "_spatial_projection",
                    NativeSpatialRegionProjection(self.observations.projection),
                )
            cache = NativeMosaicCache(
                detector.compile_native_response(
                    self.observations.projection,
                    worker_count=self.worker_count,
                    spatial_projection=self._spatial_projection,
                    scattering_cache=self.scattering_cache,
                )
            )
            object.__setattr__(
                self, "compile_seconds", self.compile_seconds + perf_counter() - start
            )
            object.__setattr__(self, "compile_count", self.compile_count + 1)
        self._responses[key] = cache
        matching = [k for k in self._responses if k[0] == physics.rods]
        for old_key in matching[:-2]:
            del self._responses[old_key]
        prediction = cache.response.evaluate(
            detector.strength_model,
            mosaic=mosaic,
            thickness_A=arguments["film_thickness_A"],
            specular_stitch_stack=stack,
            mosaic_cache=cache,
            source_weights=physics.source.mean_rays.source_weight,
            resolve_axial_panels=resolve_axial_panels,
            resolve_mosaic_components=resolve_mosaic_components,
        )
        object.__setattr__(
            self,
            "contraction_count",
            self.contraction_count + (2 if resolve_mosaic_components else 1),
        )
        return prediction

    def clear_responses(self):
        self._responses.clear()
        self.scattering_cache.entries.clear()
        self.scattering_cache.retained_bytes = 0
        self._predictions.clear()
        object.__setattr__(self, "_spatial_projection", None)

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
