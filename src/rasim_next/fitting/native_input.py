"""Explicit physical inputs for native fits; no executable checkpoints or legacy state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from painted_ewald import MosaicParameters, Rod
from rasim_next.core.contracts import EventIntensityNormalization, MaterialOptics
from rasim_next.core.transforms import RigidTransform
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import build_incident_states
from rasim_next.materials.crystal import CrystalSite, CrystalStructure
from rasim_next.ordered.amplitudes import _validated_site_displacement_tensors
from rasim_next.ordered.motifs import SiteDisplacementProfile, TransverseIsotropicSiteDisplacement
from rasim_next.pipeline.bragg_space import (
    CifFiniteStackStrength,
    IncoherentStructureMixture,
    Pbi2FiniteSurfaceStrength,
    RevisionedStructureStrengthModel,
)
from rasim_next.pipeline.conditional_detector import ConditionalStructureDetector
from rasim_next.pipeline.fiber_detector import FiberIntegrationRule
from rasim_next.reflectivity.specular import ParrattStitchStack
from rasim_next.sampling.source import (
    ConditionalSourceSamples,
    quadrature_conditional_gaussian_source,
    sample_conditional_gaussian_source,
)
from rasim_next.stacking import (
    InitialPopulation,
    Parent,
    RichEpsilonModel,
    StackingPopulation,
    TransitionLaw,
)


@dataclass(frozen=True, slots=True)
class StackingPhaseRecipe:
    """A parent-rich phase, distinguishing transition-law and intensity averages."""

    fault_parameter: str
    parents: tuple[Parent, ...]
    weights: tuple[float, ...]
    average_transition_laws: bool

    def __post_init__(self) -> None:
        parents = tuple(Parent(p) for p in self.parents)
        weights = tuple(float(w) for w in self.weights)
        if (
            not self.fault_parameter
            or not parents
            or len(parents) != len(weights)
            or np.any(~np.isfinite(weights))
            or np.any(np.asarray(weights) < 0)
            or not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12)
            or type(self.average_transition_laws) is not bool
        ):
            raise ValueError("stacking phases require explicit normalized parent weights")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "weights", weights)


@dataclass(frozen=True, slots=True)
class FiniteStructureRecipe:
    """Expanded finite motifs, or Pb stacking phases with explicit initial orientation."""

    crystals: tuple[CrystalStructure, ...]
    normalization: EventIntensityNormalization
    unknown_u_iso_A2: float | None
    stacking_phases: tuple[StackingPhaseRecipe, ...] = ()
    initial_population: InitialPopulation | None = None
    site_displacement_tensors_A2: tuple[np.ndarray, ...] | None = None
    site_displacement_profile: SiteDisplacementProfile | None = None

    def __post_init__(self) -> None:
        crystals, phases = tuple(self.crystals), tuple(self.stacking_phases)
        if not crystals or any(not isinstance(c, CrystalStructure) for c in crystals):
            raise ValueError("finite structure requires explicit expanded crystals")
        if any(not isinstance(p, StackingPhaseRecipe) for p in phases):
            raise TypeError("stacking phases must be StackingPhaseRecipe values")
        if phases:
            if len(crystals) != 1 or not isinstance(self.initial_population, InitialPopulation):
                raise ValueError("Pb stacking requires one crystal and its initial orientation")
        elif self.initial_population is not None:
            raise ValueError("an initial orientation requires stacking phases")
        for crystal in crystals[1:]:
            if not np.array_equal(crystal.direct_basis_A, crystals[0].direct_basis_A):
                raise ValueError("surface motifs must share one direct lattice")
        object.__setattr__(self, "crystals", crystals)
        object.__setattr__(self, "stacking_phases", phases)
        object.__setattr__(self, "normalization", EventIntensityNormalization(self.normalization))
        if self.site_displacement_profile is not None and (
            not phases
            or not isinstance(self.site_displacement_profile, SiteDisplacementProfile)
            or self.site_displacement_tensors_A2 is not None
            or self.unknown_u_iso_A2 is not None
        ):
            raise ValueError(
                "the Pb site profile requires stacking and exclusive displacement state"
            )
        if self.site_displacement_tensors_A2 is not None:
            if phases or len(self.site_displacement_tensors_A2) != len(crystals):
                raise ValueError("site tensors require one tensor array per ordered CIF motif")
            tensors = tuple(
                _validated_site_displacement_tensors(value, len(crystal.sites))
                for crystal, value in zip(crystals, self.site_displacement_tensors_A2, strict=True)
            )
            if self.unknown_u_iso_A2 is not None:
                raise ValueError("site tensors and unknown isotropic displacement are exclusive")
            object.__setattr__(self, "site_displacement_tensors_A2", tensors)

    def strength(
        self,
        *,
        coherent_repeats: int,
        surface_fractions: tuple[float, ...],
        phase_fractions: tuple[float, ...],
        fault_parameters: dict[str, float],
    ) -> RevisionedStructureStrengthModel:
        if not self.stacking_phases:
            if phase_fractions != (1.0,) or fault_parameters:
                raise ValueError("finite CIF motifs have one ordered phase and no fault parameters")
            models = tuple(
                CifFiniteStackStrength(
                    crystal,
                    coherent_repeats,
                    self.normalization,
                    self.unknown_u_iso_A2,
                    None
                    if self.site_displacement_tensors_A2 is None
                    else self.site_displacement_tensors_A2[i],
                )
                for i, crystal in enumerate(self.crystals)
            )
            return IncoherentStructureMixture(models, surface_fractions)
        phases = self.stacking_phases
        expected = {phase.fault_parameter for phase in phases}
        fractions = np.asarray(phase_fractions, dtype=float)
        if (
            set(fault_parameters) != expected
            or fractions.shape != (len(phases),)
            or np.any(~np.isfinite(fractions))
            or np.any(fractions < 0)
            or not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "stacking fractions and fault parameters must match the declared phases"
            )
        populations, weights = [], []
        for phase, fraction in zip(phases, fractions, strict=True):
            laws = tuple(
                RichEpsilonModel(parent, fault_parameters[phase.fault_parameter]).transition_law()
                for parent in phase.parents
            )
            if phase.average_transition_laws:
                laws = (
                    TransitionLaw.from_array(
                        np.asarray(phase.weights) @ np.array([law.as_array() for law in laws])
                    ),
                )
                term_weights = (1.0,)
            else:
                term_weights = phase.weights
            for law, weight in zip(laws, term_weights, strict=True):
                populations.append(
                    StackingPopulation(str(len(populations)), law, self.initial_population)
                )
                weights.append(float(fraction * weight))
        return Pbi2FiniteSurfaceStrength(
            self.crystals[0],
            coherent_repeats,
            tuple(populations),
            tuple(weights),
            surface_fractions,
            self.normalization,
            self.unknown_u_iso_A2,
            self.site_displacement_profile,
        )


@dataclass(frozen=True, slots=True)
class NativeSourceDefinition:
    """Physical source inputs and an explicit integration rule retained at I/O."""

    mean_origin_lab_m: np.ndarray
    mean_direction_lab: np.ndarray
    transverse_axes_lab: np.ndarray
    spatial_sigma_m: np.ndarray
    divergence_sigma_rad: np.ndarray
    line_wavelength_A: np.ndarray
    line_probability: np.ndarray
    common_wavelength_sigma_A: float
    polarization_state_id: str
    kind: str
    position_divergence_correlation: tuple[float, float] = (0.0, 0.0)
    sample_count: int = 32
    seed: int = 0
    divergence_order: int = 6
    wavelength_order: int = 6
    local_m0_divergence_order: int | None = None

    def __post_init__(self):
        for name in (
            "mean_origin_lab_m",
            "mean_direction_lab",
            "transverse_axes_lab",
            "spatial_sigma_m",
            "divergence_sigma_rad",
            "line_wavelength_A",
            "line_probability",
            "position_divergence_correlation",
        ):
            raw = np.asarray(getattr(self, name))
            if np.iscomplexobj(raw) or np.any(~np.isfinite(raw)):
                raise ValueError("source parameters must be finite real arrays")
            value = np.array(raw, dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.kind not in ("gauss_hermite", "latin_hypercube"):
            raise ValueError("unrecognized conditional source integration rule")
        if self.local_m0_divergence_order is not None and (
            self.kind != "gauss_hermite"
            or type(self.local_m0_divergence_order) is not int
            or self.local_m0_divergence_order < 1
        ):
            raise ValueError(
                "local_m0_divergence_order requires a positive integer and Gauss-Hermite"
            )

    def sample(self):
        """Rebuild common-latent rows through the authoritative source sampler."""
        parameters = dict(
            mean_origin_lab_m=self.mean_origin_lab_m,
            mean_direction_lab=self.mean_direction_lab,
            transverse_axes_lab=self.transverse_axes_lab,
            spatial_sigma_m=self.spatial_sigma_m,
            divergence_sigma_rad=self.divergence_sigma_rad,
            line_wavelength_A=self.line_wavelength_A,
            line_probability=self.line_probability,
            common_wavelength_sigma_A=self.common_wavelength_sigma_A,
            polarization_state_id=self.polarization_state_id,
            position_divergence_correlation=self.position_divergence_correlation,
        )
        if self.kind == "latin_hypercube":
            return sample_conditional_gaussian_source(
                **parameters, sample_count=self.sample_count, seed=self.seed
            )
        return quadrature_conditional_gaussian_source(
            **parameters,
            divergence_order=self.divergence_order,
            wavelength_order=self.wavelength_order,
        )


@dataclass(frozen=True, slots=True)
class NativeFitPhysics:
    """Complete fixed geometry and source; every candidate supplies its fitted parameters."""

    sample_id: str
    instrument: CompiledInstrument
    material: MaterialOptics
    source: ConditionalSourceSamples
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    reciprocal_basis_Ainv: np.ndarray
    crystal_to_sample: np.ndarray
    structure: FiniteStructureRecipe
    integration_rule: FiberIntegrationRule
    spatial_quadrature_order: int
    phase_population_weight: float
    polarization_weight: float
    specular_stitch_stack: ParrattStitchStack | None
    input_revision: str
    source_definition: NativeSourceDefinition | None = None

    def __post_init__(self) -> None:
        for name in ("reciprocal_basis_Ainv", "crystal_to_sample"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "rods", tuple(self.rods))
        if self.structure.stacking_phases and self.specular_stitch_stack is not None:
            raise ValueError("this Pb finite-surface recipe has no reflectivity channel")

    def integration_parts(self) -> tuple[NativeFitPhysics, ...]:
        """Partition independent rods only when stitched m0 needs a different source rule.

        Every part retains the same physical source distribution and uses its own
        normalized numerical samples. Sum raw predictions before fitting a scale.
        """
        definition = self.source_definition
        if (
            definition is None
            or definition.local_m0_divergence_order in (None, definition.divergence_order)
            or self.specular_stitch_stack is None
            or not any(r.h == r.k == 0 for r in self.rods)
        ):
            return (self,)
        from rasim_next.materials.optics import material_optics

        parts = []
        for local in (False, True):
            rods = tuple(r for r in self.rods if (r.h == r.k == 0) == local)
            if not rods:
                continue
            rule = replace(
                definition,
                divergence_order=(
                    definition.local_m0_divergence_order if local else definition.divergence_order
                ),
                local_m0_divergence_order=None,
            )
            source = rule.sample() if local else self.source
            parts.append(
                replace(
                    self,
                    rods=rods,
                    source_definition=rule,
                    source=source,
                    material=(
                        material_optics(self.structure.crystals[0], source.mean_rays.wavelength_A)
                        if local
                        else self.material
                    ),
                )
            )
        return tuple(parts)

    def detector(
        self,
        *,
        mosaic: MosaicParameters,
        coherent_repeats: int,
        film_thickness_A: float,
        surface_fractions: tuple[float, ...],
        phase_fractions: tuple[float, ...],
        fault_parameters: dict[str, float],
    ) -> ConditionalStructureDetector:
        definition = self.source_definition
        if (
            definition is not None
            and definition.local_m0_divergence_order not in (None, definition.divergence_order)
            and self.specular_stitch_stack is not None
            and any(r.h == r.k == 0 for r in self.rods)
        ):
            raise ValueError("resolve integration_parts before constructing a detector")
        strength = self.structure.strength(
            coherent_repeats=coherent_repeats,
            surface_fractions=surface_fractions,
            phase_fractions=phase_fractions,
            fault_parameters=fault_parameters,
        )
        # Coherent stack extent and physical attenuation thickness remain distinct.
        repeat_A = 2 * np.pi / np.linalg.norm(self.reciprocal_basis_Ainv[:, 2])
        if coherent_repeats * repeat_A > film_thickness_A * (1 + 1e-12):
            raise ValueError("coherent stack extent exceeds the physical film thickness")
        instrument = replace(self.instrument, film_thickness_A=film_thickness_A)
        return ConditionalStructureDetector(
            reciprocal_basis_Ainv=self.reciprocal_basis_Ainv,
            crystal_to_sample=self.crystal_to_sample,
            rods=self.rods,
            rod_catalog_revision=self.rod_catalog_revision,
            source=self.source,
            incident=build_incident_states(self.source.mean_rays, self.material, instrument),
            material=self.material,
            instrument=instrument,
            strength_model=strength,
            mosaic=mosaic,
            integration_rule=self.integration_rule,
            specular_stitch_stack=self.specular_stitch_stack,
            spatial_quadrature_order=self.spatial_quadrature_order,
            phase_population_weight=self.phase_population_weight,
            polarization_weight=self.polarization_weight,
        )


def load_native_fit_physics(path: Path, *, source_bytes: bytes | None = None) -> NativeFitPhysics:
    """Load versioned JSON values into the same typed owners used by fitting and images.

    Expanded crystal sites are retained exactly; CIF paths are provenance, never
    reloaded as a substitute for fitted sites. Complex values use explicit real/
    imaginary pairs. Source nodes are rebuilt from the declared divergence rule;
    both rules integrate the conditional beam-position distribution continuously.
    """
    payload = Path(path).read_bytes() if source_bytes is None else source_bytes
    if not isinstance(payload, bytes):
        raise TypeError("native physical source must be immutable bytes")
    record = json.loads(payload)
    if record.get("schema") != "rasim-native-fit-physics-v2":
        raise ValueError(
            "native physics requires schema v2; use scripts/migrate_native_physics.py for v1 inputs"
        )
    expected = {
        "schema",
        "sample_id",
        "instrument",
        "material",
        "source",
        "source_rule",
        "rods",
        "rod_catalog_revision",
        "reciprocal_basis_Ainv",
        "crystal_to_sample",
        "structure",
        "integration_rule",
        "spatial_quadrature_order",
        "phase_population_weight",
        "polarization_weight",
        "specular_stitch_stack",
    }
    if set(record) != expected:
        raise ValueError("native fit physics contains missing or unrecognized fields")
    instrument = dict(record["instrument"])
    for name in ("lab_from_sample", "sample_from_crystal", "lab_from_detector"):
        instrument[name] = RigidTransform(**instrument[name])
    material = dict(record["material"])
    real = np.asarray(material.pop("n_real"), dtype=float)
    imaginary = np.asarray(material.pop("n_imag"), dtype=float)
    wavelength = np.asarray(material["wavelength_A"], dtype=float)
    if real.ndim != 1 or real.shape != wavelength.shape or imaginary.shape != real.shape:
        raise ValueError("real and imaginary refractive indices must match the wavelength vector")
    material["n_complex"] = real + 1j * imaginary
    structure = dict(record["structure"])
    crystals = []
    for item in structure.pop("crystals"):
        crystal = dict(item)
        crystal["sites"] = tuple(CrystalSite(**site) for site in crystal["sites"])
        crystals.append(CrystalStructure(**crystal))
    phases = tuple(StackingPhaseRecipe(**p) for p in structure.pop("stacking_phases"))
    initial = structure.pop("initial_population")
    if structure.get("site_displacement_profile") is not None:
        profile = dict(structure["site_displacement_profile"])
        profile["sites"] = tuple(
            TransverseIsotropicSiteDisplacement(**site) for site in profile["sites"]
        )
        structure["site_displacement_profile"] = SiteDisplacementProfile(**profile)
    recipe = FiniteStructureRecipe(
        crystals=tuple(crystals),
        stacking_phases=phases,
        initial_population=None if initial is None else InitialPopulation(**initial),
        **structure,
    )
    stack = record["specular_stitch_stack"]
    if stack is not None:
        stack = dict(stack)
        stack["substrate_refractive_index"] = complex(*stack["substrate_refractive_index"])
        stack = ParrattStitchStack(**stack)
    definition = NativeSourceDefinition(**record["source"], **record["source_rule"])
    source = definition.sample()
    return NativeFitPhysics(
        sample_id=record["sample_id"],
        instrument=CompiledInstrument(**instrument),
        material=MaterialOptics(**material),
        source=source,
        rods=tuple(Rod(**rod) for rod in record["rods"]),
        rod_catalog_revision=record["rod_catalog_revision"],
        reciprocal_basis_Ainv=record["reciprocal_basis_Ainv"],
        crystal_to_sample=record["crystal_to_sample"],
        structure=recipe,
        integration_rule=FiberIntegrationRule(**record["integration_rule"]),
        spatial_quadrature_order=record["spatial_quadrature_order"],
        phase_population_weight=record["phase_population_weight"],
        polarization_weight=record["polarization_weight"],
        specular_stitch_stack=stack,
        input_revision=hashlib.sha256(payload).hexdigest(),
        source_definition=definition,
    )
