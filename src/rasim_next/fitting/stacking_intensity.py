"""Fixed-parent PbI2 structure strength and parameter binding."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import pi
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import Rod
from painted_ewald.validation import reject_complex
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    EventIntensityResult,
    LayerNormalQBatch,
    RodQueryBatch,
    canonical_revision_sha256,
)
from rasim_next.fitting.pbi2_geometry import PBI2_IDEAL_PARENTS
from rasim_next.materials.crystal import CrystalStructure, crystal_structure_revision
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking.finite_intensity import finite_population_event_intensity
from rasim_next.stacking.parent_models import RichEpsilonModel, StackingPopulation
from rasim_next.stacking.transition import InitialPopulation, RegistryPhaseModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

STACKING_COMPONENT_IDS = ("2H", "4H+", "4H-", "6H+", "6H-")
STACKING_PHASE_IDS = ("2H", "4H", "6H")
_PBI2_EPSILON = 0.001


def _readonly_float(value: ArrayLike, shape: tuple[int | None, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    result = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _readonly_int(value: ArrayLike, shape: tuple[int | None, ...], name: str) -> IntArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=True)
    ):
        raise ValueError(f"{name} has the wrong shape")
    result.setflags(write=False)
    return result


def _positive_integer(value: int, name: str) -> int:
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sha256_revision(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 revision")
    return value


def _canonical_allowed_component_ids(
    supplied: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    allowed = tuple(supplied)
    if (
        not allowed
        or len(set(allowed)) != len(allowed)
        or any(value not in STACKING_COMPONENT_IDS for value in allowed)
        or allowed != tuple(value for value in STACKING_COMPONENT_IDS if value in allowed)
    ):
        raise ValueError("allowed_component_ids must be a nonempty canonical-order subset")
    component_indices = tuple(STACKING_COMPONENT_IDS.index(value) for value in allowed)
    phase_indices = tuple(
        sorted(
            {
                STACKING_PHASE_IDS.index(STACKING_COMPONENT_IDS[index][:2])
                for index in component_indices
            }
        )
    )
    return allowed, component_indices, phase_indices


def _parent_populations() -> tuple[StackingPopulation, ...]:
    return tuple(
        StackingPopulation(
            population_id=parent.value,
            model=RichEpsilonModel(parent, _PBI2_EPSILON).transition_law(),
            initial=InitialPopulation.plus_only(),
        )
        for parent in PBI2_IDEAL_PARENTS
    )


def _pbi2_fixed_parent_model_revision(
    crystal: CrystalStructure,
    source_cif_sha256: str,
    layers: int,
) -> str:
    """Bind the five fixed parents to resolved motif and finite-stack physics."""

    source_revision = _sha256_revision(source_cif_sha256, "source_cif_sha256")
    if crystal.source_sha256 != source_revision:
        raise ValueError("source_cif_sha256 does not identify the supplied PbI2 crystal")
    layer_count = _positive_integer(layers, "layers")
    if len(extract_pbi2_motifs(crystal)) != 1:
        raise ValueError("five-parent PbI2 strength requires one trilayer motif per unit cell")
    return canonical_revision_sha256(
        ("model", "pbi2-five-parent-finite-profile.v2"),
        ("source_cif_sha256", source_revision),
        ("resolved_crystal_revision", crystal_structure_revision(crystal)),
        ("layers", layer_count),
        ("component_ids", STACKING_COMPONENT_IDS),
        ("epsilon", _PBI2_EPSILON),
        ("initial_population", "plus_only"),
        ("normalization", EventIntensityNormalization.FINITE_PER_LAYER.value),
        ("unknown_u_iso_A2", 0.0),
        ("registry_phase_model", RegistryPhaseModel.FORWARD_H_PLUS_2K.value),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Pbi2ParentMixtureStrength:
    """Incoherent mixture of the existing five finite PbI2 parent providers.

    This is a basis-bound strength provider for the shared sparse detector. It
    does not turn the five fixed near-parent templates into a general stacking
    transition law.
    """

    crystal: CrystalStructure
    source_cif_sha256: str
    layers: int
    domain_fraction: FloatArray
    reciprocal_basis_Ainv: FloatArray = field(init=False, repr=False)
    fixed_parent_model_revision: str = field(init=False)
    structure_model_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.crystal, CrystalStructure):
            raise TypeError("crystal must be CrystalStructure")
        layer_count = _positive_integer(self.layers, "layers")
        fraction = _readonly_float(
            self.domain_fraction,
            (len(STACKING_COMPONENT_IDS),),
            "domain_fraction",
        )
        if np.any(fraction < 0.0) or not np.isclose(
            np.sum(fraction),
            1.0,
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("domain_fraction must be a nonnegative unit simplex")
        fixed_revision = _pbi2_fixed_parent_model_revision(
            self.crystal,
            self.source_cif_sha256,
            layer_count,
        )
        basis = np.array(
            ReciprocalLattice.from_crystal(self.crystal).basis_Ainv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        basis.setflags(write=False)
        revision = canonical_revision_sha256(
            ("definition_id", "pbi2_parent_mixture_strength.v1"),
            ("fixed_parent_model_revision", fixed_revision),
            ("domain_fraction", fraction),
        )
        object.__setattr__(
            self,
            "source_cif_sha256",
            _sha256_revision(
                self.source_cif_sha256,
                "source_cif_sha256",
            ),
        )
        object.__setattr__(self, "layers", layer_count)
        object.__setattr__(self, "domain_fraction", fraction)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "fixed_parent_model_revision", fixed_revision)
        object.__setattr__(self, "structure_model_revision", revision)

    def evaluate_hkl(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: ArrayLike,
    ) -> FloatArray:
        """Evaluate mixed signed rods, wavelengths, and exact layer coordinates."""

        reject_complex(h, "h")
        reject_complex(k, "k")
        reject_complex(L, "L")
        reject_complex(k_norm_Ainv, "k_norm_Ainv")
        h_value, k_value, ell, k_norm = np.broadcast_arrays(
            np.asarray(h),
            np.asarray(k),
            np.asarray(L, dtype=np.float64),
            np.asarray(k_norm_Ainv, dtype=np.float64),
        )
        if (
            np.any(~np.isfinite(h_value))
            or np.any(~np.isfinite(k_value))
            or np.any(~np.isfinite(ell))
            or np.any(~np.isfinite(k_norm))
            or np.any(k_norm <= 0.0)
            or np.any(h_value != np.rint(h_value))
            or np.any(k_value != np.rint(k_value))
        ):
            raise ValueError("h, k, L, and k_norm_Ainv must be finite valid rod queries")
        bounds = np.iinfo(np.int32)
        if np.any((h_value < bounds.min) | (h_value > bounds.max)) or np.any(
            (k_value < bounds.min) | (k_value > bounds.max)
        ):
            raise ValueError("h and k must fit signed 32-bit integers")
        if ell.size == 0:
            return np.empty(ell.shape, dtype=np.float64)
        parents = _parent_populations()
        active_indices = tuple(
            int(component_index) for component_index in np.flatnonzero(self.domain_fraction > 0.0)
        )
        components, _, _, _, layer_count = _compile_pbi2_population_profile_components(
            self.crystal,
            signed_hk=np.column_stack(
                (
                    h_value.ravel().astype(np.int32),
                    k_value.ravel().astype(np.int32),
                )
            ),
            l_coordinate=ell.ravel(),
            wavelength_A=(2.0 * pi / k_norm).ravel(),
            layers=self.layers,
            populations=tuple(parents[component_index] for component_index in active_indices),
        )
        if (
            _pbi2_fixed_parent_model_revision(
                self.crystal,
                self.source_cif_sha256,
                layer_count,
            )
            != self.fixed_parent_model_revision
        ):
            raise RuntimeError("PbI2 parent physics changed while evaluating a bound provider")
        component_ids = tuple(component.model_component_id for component in components)
        expected_ids = tuple(STACKING_COMPONENT_IDS[index] for index in active_indices)
        if component_ids != expected_ids:
            raise RuntimeError(
                "finite stacking components did not preserve the active parent order"
            )
        active_fraction = self.domain_fraction[np.asarray(active_indices)]
        component_response_A2 = np.column_stack(
            tuple(component.scattering_strength_A2 for component in components)
        )
        return np.asarray(
            (component_response_A2 @ active_fraction).reshape(ell.shape),
            dtype=np.float64,
        )

    def evaluate_profile(
        self,
        *,
        rod: Rod,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Evaluate one signed physical rod over continuous layer coordinate."""

        if not isinstance(rod, Rod):
            raise TypeError("rod must be Rod")
        ell = np.asarray(L)
        return self.evaluate_hkl(
            h=np.full(ell.shape, rod.h, dtype=np.int32),
            k=np.full(ell.shape, rod.k, dtype=np.int32),
            L=ell,
            k_norm_Ainv=np.full(ell.shape, k_norm_Ainv, dtype=np.float64),
        )

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        """Evaluate one exact Ewald intersection."""

        return float(self.evaluate_profile(rod=rod, L=L, k_norm_Ainv=k_norm_Ainv))

    def rebind_domain_fraction(self, domain_fraction: ArrayLike) -> Pbi2ParentMixtureStrength:
        """Return the same fixed parent physics with a new incoherent simplex."""

        return replace(self, domain_fraction=domain_fraction)


@dataclass(frozen=True, slots=True, kw_only=True)
class Pbi2ParentLogRatioParameterization:
    """Gauge-free log-ratio coordinates for a declared PbI2 parent roster."""

    reference_strength: Pbi2ParentMixtureStrength
    active_component_ids: tuple[str, ...]
    reference_component_id: str = "2H"
    parameter_names: tuple[str, ...] = field(init=False)
    parameter_units: tuple[str, ...] = field(init=False)
    reference_parameters: FloatArray = field(init=False)
    parameterization_revision: str = field(init=False)
    _active_indices: tuple[int, ...] = field(init=False, repr=False)
    _reference_active_index: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_strength, Pbi2ParentMixtureStrength):
            raise TypeError("reference_strength must be Pbi2ParentMixtureStrength")
        active, component_indices, _ = _canonical_allowed_component_ids(self.active_component_ids)
        if len(active) < 2 or self.reference_component_id not in active:
            raise ValueError("active PbI2 roster requires at least two parents including reference")
        fraction = self.reference_strength.domain_fraction
        inactive_indices = tuple(
            index for index in range(len(STACKING_COMPONENT_IDS)) if index not in component_indices
        )
        if np.any(fraction[np.asarray(component_indices)] <= 0.0) or (
            inactive_indices and np.any(fraction[np.asarray(inactive_indices)] != 0.0)
        ):
            raise ValueError("reference fractions must be positive exactly on the active roster")
        reference_active_index = active.index(self.reference_component_id)
        reference_fraction = fraction[component_indices[reference_active_index]]
        parameter_components = tuple(
            component for component in active if component != self.reference_component_id
        )
        reference = np.asarray(
            [
                np.log(fraction[STACKING_COMPONENT_IDS.index(component)] / reference_fraction)
                for component in parameter_components
            ],
            dtype=np.float64,
        )
        reference.setflags(write=False)
        names = tuple(
            f"log_fraction_ratio:{component}:{self.reference_component_id}"
            for component in parameter_components
        )
        revision = canonical_revision_sha256(
            ("definition_id", "pbi2_parent_log_ratio_parameterization.v1"),
            (
                "reference_structure_model_revision",
                self.reference_strength.structure_model_revision,
            ),
            ("active_component_ids", active),
            ("reference_component_id", self.reference_component_id),
        )
        object.__setattr__(self, "active_component_ids", active)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", ("1",) * len(names))
        object.__setattr__(self, "reference_parameters", reference)
        object.__setattr__(self, "parameterization_revision", revision)
        object.__setattr__(self, "_active_indices", component_indices)
        object.__setattr__(self, "_reference_active_index", reference_active_index)

    def bind_strength(self, parameters: ArrayLike) -> Pbi2ParentMixtureStrength:
        """Map additive log ratios to one exact nonnegative unit simplex."""

        values = _readonly_float(
            parameters,
            (len(self.parameter_names),),
            "parameters",
        )
        logits = np.empty(len(self.active_component_ids), dtype=np.float64)
        logits[self._reference_active_index] = 0.0
        logits[np.arange(logits.size) != self._reference_active_index] = values
        logits -= np.max(logits)
        active_fraction = np.exp(logits)
        active_fraction /= np.sum(active_fraction)
        fraction = np.zeros(len(STACKING_COMPONENT_IDS), dtype=np.float64)
        fraction[np.asarray(self._active_indices)] = active_fraction
        return self.reference_strength.rebind_domain_fraction(fraction)


def _compile_pbi2_population_profile_components(
    crystal: CrystalStructure,
    *,
    signed_hk: ArrayLike,
    l_coordinate: ArrayLike,
    wavelength_A: ArrayLike,
    layers: int,
    populations: tuple[StackingPopulation, ...],
) -> tuple[tuple[EventIntensityResult, ...], IntArray, FloatArray, FloatArray, int]:
    """Evaluate selected fixed PbI2 parents through the shared finite-stack physics."""

    if not isinstance(crystal, CrystalStructure):
        raise TypeError("crystal must be CrystalStructure")
    layer_count = _positive_integer(layers, "layers")
    hk = _readonly_int(signed_hk, (None, 2), "signed_hk")
    if hk.shape[0] == 0:
        raise ValueError("signed_hk must be nonempty")
    int32 = np.iinfo(np.int32)
    if np.any((hk < int32.min) | (hk > int32.max)):
        raise ValueError("signed_hk values must fit in int32")
    ell = _readonly_float(l_coordinate, (hk.shape[0],), "l_coordinate")
    supplied_wavelength = np.asarray(wavelength_A)
    try:
        wavelength = _readonly_float(
            np.broadcast_to(supplied_wavelength, (hk.shape[0],)),
            (hk.shape[0],),
            "wavelength_A",
        )
    except ValueError as error:
        raise ValueError("wavelength_A must be scalar or align with signed_hk") from error
    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be positive")
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    hkl = np.column_stack((hk, ell))
    layer_normal = np.cross(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1])
    layer_normal /= np.linalg.norm(layer_normal)
    layer_q = reciprocal.q_cartesian_Ainv(hkl) @ layer_normal
    _, rod_id = np.unique(hk, axis=0, return_inverse=True)
    event_id = np.arange(hk.shape[0], dtype=np.int64)
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=rod_id,
        phase_id=(crystal.phase_id,) * hk.shape[0],
        h=hk[:, 0].astype(np.int32),
        k=hk[:, 1].astype(np.int32),
        q_sample_normal_Ainv=layer_q,
        l_coordinate=ell,
        wavelength_A=wavelength,
    )
    amplitudes = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)
    components = finite_population_event_intensity(
        query,
        amplitudes,
        populations,
        layer_normal_q=LayerNormalQBatch(
            event_id=event_id,
            rod_id=query.rod_id,
            phase_id=query.phase_id,
            layer_normal_q_Ainv=layer_q,
            gauge_id=amplitudes.gauge_id,
        ),
        layers=layer_count,
        population_group_id="pbi2-stacking-parents",
        normalization=EventIntensityNormalization.FINITE_PER_LAYER,
        phase_model=RegistryPhaseModel.FORWARD_H_PLUS_2K,
    )
    return components, hk, ell, wavelength, layer_count


__all__ = [
    "STACKING_COMPONENT_IDS",
    "STACKING_PHASE_IDS",
    "Pbi2ParentLogRatioParameterization",
    "Pbi2ParentMixtureStrength",
]
