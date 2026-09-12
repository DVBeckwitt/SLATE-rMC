"""One response owner for joint Bi/Pb and acquisition refinement."""

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from painted_ewald import MosaicParameters
from rasim_next.fitting.bi_joint import BI_JOINT_PARAMETER_NAMES, BiJointCandidate
from rasim_next.fitting.bi_native import BiCellSiteParameters, BiNativeStructureModel
from rasim_next.fitting.native_instrument import (
    NATIVE_INSTRUMENT_PARAMETER_NAMES,
    NativeInstrumentModel,
)
from rasim_next.fitting.native_observations import NativeFitObservations
from rasim_next.fitting.native_structure import native_stitch_records
from rasim_next.fitting.pb_native import PbJointModel, simplex_fractions
from rasim_next.pipeline.conditional_detector import NativeMosaicCache
from rasim_next.pipeline.detector_revisions import _instrument_revision
from rasim_next.pipeline.source_spatial import NativeSpatialRegionProjection


@dataclass(frozen=True, slots=True)
class NativeJointEvaluator:
    """Bounded explicit caches; every trial rebinds atomic amplitudes and optics.

    The reference projection, numerical proposal and instrument owner are fixed.
    Source, optics, basis or rigid geometry changes invalidate compiled responses.
    Film thickness, ADPs and stacking reuse geometry and re-evaluate intensities.
    """

    model: BiNativeStructureModel | PbJointModel
    observations: NativeFitObservations
    proposal_mosaic: MosaicParameters
    instrument_model: NativeInstrumentModel | None = None
    worker_count: int = 1
    compile_count: int = field(default=0, init=False)
    evaluation_count: int = field(default=0, init=False)
    compile_seconds: float = field(default=0, init=False)
    _responses: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _predictions: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _spatial_projection: NativeSpatialRegionProjection | None = field(
        default=None, init=False, repr=False
    )

    @property
    def parameter_names(self):
        names = (
            BI_JOINT_PARAMETER_NAMES
            if isinstance(self.model, BiNativeStructureModel)
            else self.model.parameter_names
        )
        return (
            *names,
            *(NATIVE_INSTRUMENT_PARAMETER_NAMES if self.instrument_model is not None else ()),
        )

    @property
    def parameter_units(self):
        """Units are fixed by the physical coordinate contract, never by a plan."""
        suffixes = (
            ("_A2", "angstrom^2"),
            ("_A", "angstrom"),
            ("_rad", "radian"),
            ("_m", "metre"),
            ("_px", "pixel"),
        )
        return tuple(
            next((unit for suffix, unit in suffixes if name.endswith(suffix)), "1")
            for name in self.parameter_names
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
        if isinstance(self.model, BiNativeStructureModel):
            candidate = BiJointCandidate(physical, coherent_repeats)
            physics = self.model.bind(BiCellSiteParameters.from_array(physical[:13]))
            stack = physics.specular_stitch_stack
            if stack is None:
                raise ValueError("Bi joint refinement requires its declared local composite")
            stack = replace(
                stack, top_roughness_A=float(physical[19]), bottom_roughness_A=float(physical[20])
            )
            arguments = dict(
                coherent_repeats=coherent_repeats,
                film_thickness_A=candidate.film_thickness_A,
                surface_fractions=candidate.surface_fractions,
                phase_fractions=(1.0,),
                fault_parameters={},
            )
            mosaic = MosaicParameters(*physical[13:16])
        else:
            physics, arguments = self.model.bind(physical, coherent_repeats)
            mosaic = MosaicParameters(*physical[9:12])
            stack = None
        if self.instrument_model is not None:
            physics = self.instrument_model.bind(physics, values[-18:])
        return physics, arguments, mosaic, stack

    def predict(self, values, coherent_repeats):
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
        candidate_key = coherent_repeats, values.tobytes()
        if candidate_key in self._predictions:
            self._predictions.move_to_end(candidate_key)
            return self._predictions[candidate_key]
        physics, arguments, mosaic, stack = self.bind(values, coherent_repeats)
        detector = physics.detector(mosaic=self.proposal_mosaic, **arguments)
        # Thickness changes attenuation weights, never the accepted spatial ray map.
        # This backend rejects finite footprint/external absorption at construction.
        key = (
            physics.material.material_revision,
            physics.reciprocal_basis_Ainv.tobytes(),
            physics.source.revision,
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
                )
            )
            object.__setattr__(
                self, "compile_seconds", self.compile_seconds + perf_counter() - start
            )
            object.__setattr__(self, "compile_count", self.compile_count + 1)
        self._responses[key] = cache
        while len(self._responses) > 2:
            self._responses.popitem(last=False)
        prediction = cache.response.evaluate(
            detector.strength_model,
            mosaic=mosaic,
            thickness_A=arguments["film_thickness_A"],
            specular_stitch_stack=stack,
            mosaic_cache=cache,
        )
        prediction.setflags(write=False)
        self._predictions[candidate_key] = prediction
        while len(self._predictions) > 64:
            self._predictions.popitem(last=False)
        object.__setattr__(self, "evaluation_count", self.evaluation_count + 1)
        return prediction

    def clear_responses(self):
        self._responses.clear()
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
        eta = parameters["lorentzian_probability"]
        if eta == 0:
            inactive["lorentzian_half_width_rad"] = "zero Lorentzian population"
        if eta == 1:
            inactive["gaussian_sigma_rad"] = "zero Gaussian population"
        if parameters["surface_fraction_0"] == 1:
            inactive["surface_1_share_of_remainder"] = "zero surface remainder"
        if isinstance(self.model, PbJointModel):
            phases = self.model.atomic.reference.structure.stacking_phases
            share_names = [f"phase_{i}_share_of_remainder" for i in range(len(phases) - 1)]
            fractions = simplex_fractions([parameters[name] for name in share_names])
            exhausted = False
            for name in share_names:
                if exhausted:
                    inactive[name] = "zero phase remainder"
                exhausted = exhausted or parameters[name] == 1
            for i, (phase, fraction) in enumerate(zip(phases, fractions, strict=True)):
                if fraction == 0:
                    inactive[f"phase_{i}_epsilon"] = "zero phase population"
                exhausted = False
                for j in range(len(phase.parents) - 1):
                    name = f"phase_{i}_parent_{j}_share_of_remainder"
                    if fraction == 0 or exhausted:
                        inactive[name] = "zero phase population or parent remainder"
                    exhausted = exhausted or parameters[name] == 1
        if self.instrument_model is not None:
            probability = parameters["source_line_0_probability"]
            if probability in (0, 1):
                absent = 0 if probability == 0 else 1
                inactive[f"source_line_{absent}_wavelength_A"] = "zero spectral line probability"
        return inactive

    def stitch_state(self, values, coherent_repeats):
        """Surface/wavelength handoff selections and normalization at this candidate."""
        physics, arguments, _, stack = self.bind(values, coherent_repeats)
        return native_stitch_records(physics, arguments, stack)
