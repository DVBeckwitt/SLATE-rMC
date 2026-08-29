"""Direct PbI2 stacking-profile compilation and population fitting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import isfinite, pi, sqrt
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from painted_ewald import Rod
from painted_ewald.validation import reject_complex
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    EventIntensityResult,
    LayerNormalQBatch,
    RodQueryBatch,
    canonical_revision_sha256,
)
from rasim_next.fitting.geometry import LayerLMarkerDefinition, LayerLMarkerObservations
from rasim_next.fitting.mosaic import MosaicReflectionGroupKey
from rasim_next.fitting.pbi2_geometry import PBI2_IDEAL_PARENTS, Pbi2PolytypeLandmarkCatalogue
from rasim_next.materials.crystal import CrystalStructure, crystal_structure_revision
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking.finite_intensity import finite_population_event_intensity
from rasim_next.stacking.parent_models import RichEpsilonModel, StackingPopulation
from rasim_next.stacking.transition import InitialPopulation, RegistryPhaseModel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

STACKING_COMPONENT_IDS = ("2H", "4H+", "4H-", "6H+", "6H-")
STACKING_PHASE_IDS = ("2H", "4H", "6H")
_LAYER_L_STACKING_MEASURE_ID = "pointwise-intrinsic-summed-signed-rods-layer-L-strength-A2.v1"
_PBI2_EPSILON = 0.001
_PHASE_AGGREGATION = np.asarray(
    ((1.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0, 1.0))
)
_PHASE_AGGREGATION.setflags(write=False)
_LAYER_L_COMPONENT_ROSTERS = (
    ("2H",),
    ("2H", "6H+", "6H-"),
    STACKING_COMPONENT_IDS,
)


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


def _canonical_layer_l_component_ids(supplied: tuple[str, ...]) -> tuple[str, ...]:
    allowed, _, _ = _canonical_allowed_component_ids(supplied)
    if allowed not in _LAYER_L_COMPONENT_ROSTERS:
        raise ValueError(
            "layer-L stacking fits require 2H, 2H+6H+/- or the full five-parent roster"
        )
    return allowed


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledStackingResponse:
    """Five pointwise finite-parent strengths at explicit signed-rod/L queries."""

    component_response_A2: FloatArray
    signed_hk: IntArray
    l_coordinate: FloatArray
    wavelength_A: FloatArray
    fixed_model_revision: str
    component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_PER_LAYER
    sampling_revision: str = field(init=False)
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if tuple(self.component_ids) != STACKING_COMPONENT_IDS:
            raise ValueError("component_ids must use the canonical five-parent order")
        response = _readonly_float(
            self.component_response_A2, (None, len(STACKING_COMPONENT_IDS)), "component_response_A2"
        )
        if response.shape[0] == 0 or np.any(response < 0.0):
            raise ValueError("component_response_A2 must be nonempty and nonnegative")
        signed_hk = _readonly_int(self.signed_hk, (response.shape[0], 2), "signed_hk")
        ell = _readonly_float(self.l_coordinate, (response.shape[0],), "l_coordinate")
        wavelength = _readonly_float(self.wavelength_A, (response.shape[0],), "wavelength_A")
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_A must be positive")
        if not isinstance(self.fixed_model_revision, str) or not self.fixed_model_revision:
            raise ValueError("fixed_model_revision must be nonempty")
        normalization = EventIntensityNormalization(self.normalization)
        if normalization is not EventIntensityNormalization.FINITE_PER_LAYER:
            raise ValueError("stacking profiles require FINITE_PER_LAYER normalization")
        sampling_revision = canonical_revision_sha256(
            ("measure", "pointwise-intrinsic-signed-rod-L-strength-A2.v1"),
            ("signed_hk", signed_hk),
            ("l_coordinate", ell),
            ("wavelength_A", wavelength),
        )
        response_revision = canonical_revision_sha256(
            ("component_ids", STACKING_COMPONENT_IDS),
            ("component_response_A2", response),
            ("sampling_revision", sampling_revision),
            ("fixed_model_revision", self.fixed_model_revision),
            ("normalization", normalization.value),
        )
        object.__setattr__(self, "component_response_A2", response)
        object.__setattr__(self, "signed_hk", signed_hk)
        object.__setattr__(self, "l_coordinate", ell)
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(self, "component_ids", STACKING_COMPONENT_IDS)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "sampling_revision", sampling_revision)
        object.__setattr__(self, "response_revision", response_revision)


def _validate_layer_l_group_keys(groups: tuple[MosaicReflectionGroupKey, ...]) -> None:
    if not groups or any(not isinstance(group, MosaicReflectionGroupKey) for group in groups):
        raise ValueError("group_keys must contain at least one MosaicReflectionGroupKey")
    if len(set(groups)) != len(groups):
        raise ValueError("layer-L stacking reflection-group identities must be unique")
    if any(
        group.branch_mode != "EXPLICIT_NONZERO"
        or group.layered_family_m is None
        or group.layered_family_m <= 0
        or group.layered_integer_L is not None
        or group.layered_layer_order is None
        or group.layered_reciprocal_basis_revision is None
        for group in groups
    ):
        raise ValueError("stacking rows require exact nonzero layer-L reflection groups")
    if len({group.layered_reciprocal_basis_revision for group in groups}) != 1:
        raise ValueError("layer-L stacking groups must use one reciprocal-basis revision")
    if len({group.rod_catalog_revision for group in groups}) != 1:
        raise ValueError("layer-L stacking groups must use one rod-catalog revision")
    structural_keys = tuple(
        (
            group.layered_family_m,
            group.layered_layer_order,
            group.layered_reciprocal_basis_revision,
        )
        for group in groups
    )
    if len(set(structural_keys)) != len(structural_keys):
        raise ValueError("each exact structural layer-L identity may occur only once")
    if any(
        h * h + h * k + k * k != group.layered_family_m
        for group in groups
        for h, k in group.member_rod_hk
    ):
        raise ValueError("reflection-group rods must match the declared layered family")


def _validate_source_marker_definitions(
    groups: tuple[MosaicReflectionGroupKey, ...],
    source_groups: tuple[tuple[LayerLMarkerDefinition, ...], ...],
) -> None:
    if len(source_groups) != len(groups):
        raise ValueError("source marker definitions must align with reflection groups")
    for group, supplied in zip(groups, source_groups, strict=True):
        definitions = tuple(supplied)
        if (
            not definitions
            or any(not isinstance(item, LayerLMarkerDefinition) for item in definitions)
            or len(set(definitions)) != len(definitions)
        ):
            raise ValueError("each reflection group requires unique source marker definitions")
        if any(
            definition.key.family_m != group.layered_family_m
            or definition.key.layer_order != group.layered_layer_order
            or definition.key.reciprocal_basis_revision != group.layered_reciprocal_basis_revision
            or definition.contributing_rod_hk != group.member_rod_hk
            for definition in definitions
        ):
            raise ValueError("source marker definitions disagree with the reflection group")
        tag_branches = tuple(definition.key.tag_branch for definition in definitions)
        if len(set(tag_branches)) != len(tag_branches):
            raise ValueError("source marker definitions must have unique detector tag sides")
        if set(tag_branches) == {1, 2} and len({item.key.branch for item in definitions}) != 1:
            raise ValueError("paired source marker definitions must share one analytic branch")


def _layer_l_group_revision_fields(
    groups: tuple[MosaicReflectionGroupKey, ...],
) -> tuple[tuple[str, object], ...]:
    rod_rows = np.asarray(
        [
            (group_index, h, k)
            for group_index, group in enumerate(groups)
            for h, k in group.member_rod_hk
        ],
        dtype=np.int64,
    )
    return (
        ("group_id", tuple(group.group_id for group in groups)),
        ("rod_catalog_revision", tuple(group.rod_catalog_revision for group in groups)),
        ("branch_mode", tuple(group.branch_mode for group in groups)),
        (
            "family_m",
            np.asarray([group.layered_family_m for group in groups], dtype=np.int64),
        ),
        (
            "layer_order_numerator",
            np.asarray([group.layered_layer_order.numerator for group in groups], dtype=np.int64),
        ),
        (
            "layer_order_denominator",
            np.asarray([group.layered_layer_order.denominator for group in groups], dtype=np.int64),
        ),
        (
            "reciprocal_basis_revision",
            tuple(group.layered_reciprocal_basis_revision for group in groups),
        ),
        ("contributing_rod_rows", rod_rows),
    )


def _source_marker_revision_rows(
    source_groups: tuple[tuple[LayerLMarkerDefinition, ...], ...],
) -> IntArray:
    return np.asarray(
        [
            (
                group_index,
                definition.key.family_m,
                definition.key.layer_order.numerator,
                definition.key.layer_order.denominator,
                definition.key.branch,
                definition.key.root_sign,
            )
            for group_index, definitions in enumerate(source_groups)
            for definition in definitions
        ],
        dtype=np.int64,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledLayerLStackingResponse:
    """Five fixed-parent strengths summed once per exact structural landmark."""

    group_keys: tuple[MosaicReflectionGroupKey, ...]
    source_marker_definitions: tuple[tuple[LayerLMarkerDefinition, ...], ...]
    component_response_A2: FloatArray
    wavelength_A: FloatArray
    specimen_id: str
    fixed_state_revision: str
    catalogue_revision: str
    fixed_model_revision: str
    component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_PER_LAYER
    measure_id: str = _LAYER_L_STACKING_MEASURE_ID
    sampling_revision: str = field(init=False)
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        groups = tuple(self.group_keys)
        source_groups = tuple(tuple(group) for group in self.source_marker_definitions)
        _validate_layer_l_group_keys(groups)
        _validate_source_marker_definitions(groups, source_groups)
        if tuple(self.component_ids) != STACKING_COMPONENT_IDS:
            raise ValueError("component_ids must use the canonical five-parent order")
        response = _readonly_float(
            self.component_response_A2,
            (len(groups), len(STACKING_COMPONENT_IDS)),
            "component_response_A2",
        )
        if np.any(response < 0.0):
            raise ValueError("component_response_A2 must be nonnegative")
        wavelength = _readonly_float(
            self.wavelength_A,
            (len(groups),),
            "wavelength_A",
        )
        if np.any(wavelength <= 0.0):
            raise ValueError("wavelength_A must be positive")
        if not isinstance(self.specimen_id, str) or not self.specimen_id:
            raise ValueError("specimen_id must be nonempty")
        for name in ("fixed_state_revision", "catalogue_revision"):
            _sha256_revision(getattr(self, name), name)
        if {group.rod_catalog_revision for group in groups} != {self.catalogue_revision}:
            raise ValueError("reflection groups disagree with catalogue_revision")
        if not isinstance(self.fixed_model_revision, str) or not self.fixed_model_revision:
            raise ValueError("fixed_model_revision must be nonempty")
        normalization = EventIntensityNormalization(self.normalization)
        if normalization is not EventIntensityNormalization.FINITE_PER_LAYER:
            raise ValueError("stacking landmarks require FINITE_PER_LAYER normalization")
        if self.measure_id != _LAYER_L_STACKING_MEASURE_ID:
            raise ValueError("unsupported layer-L stacking measure")
        sampling_revision = canonical_revision_sha256(
            ("measure", self.measure_id),
            ("specimen_id", self.specimen_id),
            ("fixed_state_revision", self.fixed_state_revision),
            ("catalogue_revision", self.catalogue_revision),
            *_layer_l_group_revision_fields(groups),
            ("source_marker_rows", _source_marker_revision_rows(source_groups)),
            ("wavelength_A", wavelength),
        )
        response_revision = canonical_revision_sha256(
            ("component_ids", STACKING_COMPONENT_IDS),
            ("component_response_A2", response),
            ("sampling_revision", sampling_revision),
            ("fixed_model_revision", self.fixed_model_revision),
            ("normalization", normalization.value),
        )
        object.__setattr__(self, "group_keys", groups)
        object.__setattr__(self, "source_marker_definitions", source_groups)
        object.__setattr__(self, "component_response_A2", response)
        object.__setattr__(self, "wavelength_A", wavelength)
        object.__setattr__(self, "component_ids", STACKING_COMPONENT_IDS)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "sampling_revision", sampling_revision)
        object.__setattr__(self, "response_revision", response_revision)


@dataclass(frozen=True, slots=True, kw_only=True)
class LayerLStackingObservations:
    """Keyed intrinsic landmark strengths for one separately fitted specimen."""

    specimen_id: str
    group_keys: tuple[MosaicReflectionGroupKey, ...]
    observed_strength_A2: FloatArray
    variance_A4: FloatArray
    sampling_revision: str
    measure_id: str = _LAYER_L_STACKING_MEASURE_ID
    observation_revision: str = field(init=False)

    def __post_init__(self) -> None:
        groups = tuple(self.group_keys)
        _validate_layer_l_group_keys(groups)
        if not isinstance(self.specimen_id, str) or not self.specimen_id:
            raise ValueError("specimen_id must be nonempty")
        observed = _readonly_float(
            self.observed_strength_A2,
            (len(groups),),
            "observed_strength_A2",
        )
        variance = _readonly_float(self.variance_A4, (len(groups),), "variance_A4")
        if np.any(variance <= 0.0):
            raise ValueError("variance_A4 must be positive")
        _sha256_revision(self.sampling_revision, "sampling_revision")
        if self.measure_id != _LAYER_L_STACKING_MEASURE_ID:
            raise ValueError("unsupported layer-L stacking observation measure")
        observation_revision = canonical_revision_sha256(
            ("measure", self.measure_id),
            ("specimen_id", self.specimen_id),
            ("sampling_revision", self.sampling_revision),
            *_layer_l_group_revision_fields(groups),
            ("observed_strength_A2", observed),
            ("variance_A4", variance),
        )
        object.__setattr__(self, "group_keys", groups)
        object.__setattr__(self, "observed_strength_A2", observed)
        object.__setattr__(self, "variance_A4", variance)
        object.__setattr__(self, "observation_revision", observation_revision)


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


def compile_pbi2_stacking_profile_response(
    crystal: CrystalStructure,
    crystal_revision: str,
    *,
    signed_hk: ArrayLike,
    l_coordinate: ArrayLike,
    wavelength_A: ArrayLike,
    layers: int,
) -> CompiledStackingResponse:
    """Evaluate the five fixed PbI2 parents directly at signed-rod/L points."""

    if (
        not isinstance(crystal_revision, str)
        or len(crystal_revision) != 64
        or any(character not in "0123456789abcdef" for character in crystal_revision)
    ):
        raise ValueError("crystal_revision must be a lowercase SHA-256 digest")
    components, hk, ell, wavelength, layer_count = _compile_pbi2_population_profile_components(
        crystal,
        signed_hk=signed_hk,
        l_coordinate=l_coordinate,
        wavelength_A=wavelength_A,
        layers=layers,
        populations=_parent_populations(),
    )
    if tuple(value.model_component_id for value in components) != STACKING_COMPONENT_IDS:
        raise RuntimeError("finite stacking components did not preserve canonical order")
    fixed_model_revision = _pbi2_fixed_parent_model_revision(
        crystal,
        crystal_revision,
        layer_count,
    )
    return CompiledStackingResponse(
        component_response_A2=np.column_stack(
            tuple(value.scattering_strength_A2 for value in components)
        ),
        signed_hk=hk,
        l_coordinate=ell,
        wavelength_A=wavelength,
        fixed_model_revision=fixed_model_revision,
    )


def compile_pbi2_layer_l_stacking_response(
    crystal: CrystalStructure,
    crystal_revision: str,
    *,
    catalogue: Pbi2PolytypeLandmarkCatalogue,
    observations: LayerLMarkerObservations,
    specimen_id: str,
    fixed_state_revision: str,
    layers: int,
) -> CompiledLayerLStackingResponse:
    """Collapse detector-root duplicates and sum unique rods at exact rational landmarks."""

    if not isinstance(catalogue, Pbi2PolytypeLandmarkCatalogue):
        raise TypeError("catalogue must be Pbi2PolytypeLandmarkCatalogue")
    if not isinstance(observations, LayerLMarkerObservations):
        raise TypeError("observations must be LayerLMarkerObservations")
    if set(catalogue.selected_parents) != set(PBI2_IDEAL_PARENTS):
        raise ValueError("stacking compilation requires a catalogue with all five ideal parents")
    if crystal_revision != catalogue.source_cif_sha256:
        raise ValueError("crystal revision disagrees with the landmark catalogue")
    reciprocal = ReciprocalLattice.from_crystal(crystal)
    reciprocal_basis_revision = canonical_revision_sha256(
        ("definition_id", "commensurate_layer_coordinate_basis.v1"),
        ("phase_id", crystal.phase_id),
        ("cif_sha256", crystal_revision),
        ("reciprocal_basis_Ainv", reciprocal.basis_Ainv),
    )
    if reciprocal_basis_revision != catalogue.reciprocal_basis_revision:
        raise ValueError("crystal reciprocal basis disagrees with the landmark catalogue")
    if any(
        key.reciprocal_basis_revision != catalogue.reciprocal_basis_revision
        for key in observations.keys
    ):
        raise ValueError("observations changed the catalogue reciprocal-basis revision")
    catalogue_definitions = {definition.key: definition for definition in catalogue.definitions}
    for definition in observations.definitions:
        if catalogue_definitions.get(definition.key) != definition:
            raise ValueError("an observation does not match its exact catalogue definition")

    grouped: dict[tuple[int, int, int, str], list[LayerLMarkerDefinition]] = {}
    for definition in observations.definitions:
        marker_key = definition.key
        structural_key = (
            marker_key.family_m,
            marker_key.layer_order.numerator,
            marker_key.layer_order.denominator,
            marker_key.reciprocal_basis_revision,
        )
        grouped.setdefault(structural_key, []).append(definition)

    grouped_definitions = sorted(
        grouped.values(),
        key=lambda definitions: (
            definitions[0].key.family_m,
            definitions[0].key.layer_order,
            definitions[0].key.reciprocal_basis_revision,
        ),
    )
    source_groups = tuple(
        tuple(
            sorted(
                definitions,
                key=lambda item: (item.key.tag_branch, item.key.branch, item.key.root_sign),
            )
        )
        for definitions in grouped_definitions
    )
    group_keys = tuple(
        MosaicReflectionGroupKey(
            group_id=(
                f"layer-L:m={definitions[0].key.family_m}:"
                f"L={definitions[0].key.layer_order.numerator}/"
                f"{definitions[0].key.layer_order.denominator}:"
                f"basis={definitions[0].key.reciprocal_basis_revision}"
            ),
            rod_catalog_revision=catalogue.catalogue_revision,
            member_rod_hk=definitions[0].contributing_rod_hk,
            branch_mode="EXPLICIT_NONZERO",
            layered_family_m=definitions[0].key.family_m,
            layered_layer_order=definitions[0].key.layer_order,
            layered_reciprocal_basis_revision=definitions[0].key.reciprocal_basis_revision,
        )
        for definitions in source_groups
    )
    rod_counts = np.asarray(
        [len(group.member_rod_hk) for group in group_keys],
        dtype=np.int64,
    )
    flat_hk = np.asarray(
        [rod for group in group_keys for rod in group.member_rod_hk],
        dtype=np.int64,
    )
    flat_l = np.concatenate(
        tuple(
            np.full(count, group.layered_layer_order.as_float(), dtype=np.float64)
            for group, count in zip(group_keys, rod_counts, strict=True)
        )
    )
    pointwise = compile_pbi2_stacking_profile_response(
        crystal,
        crystal_revision,
        signed_hk=flat_hk,
        l_coordinate=flat_l,
        wavelength_A=observations.reference_wavelength_A,
        layers=layers,
    )
    row_starts = np.concatenate((np.asarray((0,), dtype=np.int64), np.cumsum(rod_counts)[:-1]))
    grouped_response = np.add.reduceat(pointwise.component_response_A2, row_starts, axis=0)
    return CompiledLayerLStackingResponse(
        group_keys=group_keys,
        source_marker_definitions=source_groups,
        component_response_A2=grouped_response,
        wavelength_A=np.full(len(group_keys), observations.reference_wavelength_A),
        specimen_id=specimen_id,
        fixed_state_revision=fixed_state_revision,
        catalogue_revision=catalogue.catalogue_revision,
        fixed_model_revision=pointwise.fixed_model_revision,
    )


class StackingPopulationIdentifiabilityError(ValueError):
    """The sampled profiles cannot distinguish all three phase totals."""

    def __init__(
        self,
        message: str,
        *,
        phase_contrast_rank: int,
        phase_contrast_singular_values: FloatArray,
        phase_contrast_condition: float,
    ) -> None:
        self.phase_contrast_rank = int(phase_contrast_rank)
        singular = np.array(phase_contrast_singular_values, dtype=np.float64, copy=True)
        singular.setflags(write=False)
        self.phase_contrast_singular_values = singular
        self.phase_contrast_condition = float(phase_contrast_condition)
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class StackingPopulationFitResult:
    """Fit and diagnostics; rank-deficient domain fractions are one NNLS representative."""

    domain_amount: FloatArray
    domain_fraction: FloatArray
    phase_fraction: FloatArray
    global_scale: float
    predicted_strength_A2: FloatArray
    weighted_residual: FloatArray
    chi_square: float
    active_component_ids: tuple[str, ...]
    domain_response_full_rank: bool
    response_singular_values: FloatArray
    response_rank: int
    response_condition: float
    phase_contrast_singular_values: FloatArray
    phase_contrast_rank: int
    phase_contrast_condition: float
    phase_profile_bounds: FloatArray
    phase_estimate_on_boundary: BoolArray
    profile_delta_chi_square: float
    response_revision: str
    allowed_component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    observation_revision: str | None = None
    component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS
    phase_ids: tuple[str, ...] = STACKING_PHASE_IDS

    def __post_init__(self) -> None:
        if (
            tuple(self.component_ids) != STACKING_COMPONENT_IDS
            or tuple(self.phase_ids) != STACKING_PHASE_IDS
        ):
            raise ValueError("component_ids and phase_ids must use canonical order")
        arrays = {
            "domain_amount": _readonly_float(self.domain_amount, (5,), "domain_amount"),
            "domain_fraction": _readonly_float(self.domain_fraction, (5,), "domain_fraction"),
            "phase_fraction": _readonly_float(self.phase_fraction, (3,), "phase_fraction"),
            "predicted_strength_A2": _readonly_float(
                self.predicted_strength_A2, (None,), "predicted_strength_A2"
            ),
            "weighted_residual": _readonly_float(
                self.weighted_residual, (None,), "weighted_residual"
            ),
            "response_singular_values": _readonly_float(
                self.response_singular_values, (None,), "response_singular_values"
            ),
            "phase_contrast_singular_values": _readonly_float(
                self.phase_contrast_singular_values, (2,), "phase_contrast_singular_values"
            ),
            "phase_profile_bounds": _readonly_float(
                self.phase_profile_bounds, (3, 2), "phase_profile_bounds"
            ),
        }
        boundary = np.array(self.phase_estimate_on_boundary, dtype=np.bool_, copy=True)
        if boundary.shape != (3,):
            raise ValueError("phase_estimate_on_boundary must have shape (3,)")
        boundary.setflags(write=False)
        amount = arrays["domain_amount"]
        domain = arrays["domain_fraction"]
        phase = arrays["phase_fraction"]
        scale = float(self.global_scale)
        if np.any(amount < 0.0) or not isfinite(scale) or scale <= 0.0:
            raise ValueError("domain amounts and global_scale must be physically positive")
        if np.any(domain < 0.0) or not np.isclose(np.sum(domain), 1.0, rtol=0.0, atol=2.0e-12):
            raise ValueError("domain_fraction must lie on the probability simplex")
        if np.any(phase < 0.0) or not np.isclose(np.sum(phase), 1.0, rtol=0.0, atol=2.0e-12):
            raise ValueError("phase_fraction must lie on the probability simplex")
        if not np.allclose(amount, scale * domain, rtol=2.0e-12, atol=0.0):
            raise ValueError("global_scale must normalize domain_amount")
        if not np.allclose(phase, _PHASE_AGGREGATION @ domain, rtol=0.0, atol=2.0e-12):
            raise ValueError("phase fractions must aggregate domain fractions")
        bounds = arrays["phase_profile_bounds"]
        if np.any((bounds < 0.0) | (bounds > 1.0)) or np.any(
            (bounds[:, 0] > phase) | (bounds[:, 1] < phase)
        ):
            raise ValueError("phase_profile_bounds must contain each phase estimate")
        if arrays["predicted_strength_A2"].shape != arrays["weighted_residual"].shape:
            raise ValueError("prediction and residual arrays must align")
        if (
            not isfinite(self.chi_square)
            or self.chi_square < 0.0
            or not np.isclose(
                self.chi_square,
                arrays["weighted_residual"] @ arrays["weighted_residual"],
                rtol=2.0e-12,
                atol=1.0e-14,
            )
        ):
            raise ValueError("chi_square must be the squared weighted-residual norm")
        allowed, allowed_indices, _ = _canonical_allowed_component_ids(self.allowed_component_ids)
        active = tuple(self.active_component_ids)
        if (
            len(set(active)) != len(active)
            or active != tuple(value for value in STACKING_COMPONENT_IDS if value in active)
            or set(active) - set(allowed)
        ):
            raise ValueError(
                "active_component_ids must be a canonical-order subset of allowed components"
            )
        excluded_indices = tuple(
            index for index in range(len(STACKING_COMPONENT_IDS)) if index not in allowed_indices
        )
        if excluded_indices and (
            np.any(amount[list(excluded_indices)] != 0.0)
            or np.any(domain[list(excluded_indices)] != 0.0)
        ):
            raise ValueError("excluded stacking components must have exactly zero amount")
        if not isfinite(self.profile_delta_chi_square) or self.profile_delta_chi_square <= 0.0:
            raise ValueError("profile_delta_chi_square must be positive")
        if not isinstance(self.response_revision, str) or not self.response_revision:
            raise ValueError("response_revision must be nonempty")
        if self.observation_revision is not None:
            _sha256_revision(self.observation_revision, "observation_revision")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "global_scale", scale)
        object.__setattr__(self, "phase_estimate_on_boundary", boundary)
        object.__setattr__(self, "active_component_ids", active)
        object.__setattr__(self, "allowed_component_ids", allowed)
        object.__setattr__(self, "component_ids", STACKING_COMPONENT_IDS)
        object.__setattr__(self, "phase_ids", STACKING_PHASE_IDS)


@dataclass(frozen=True, slots=True)
class _LeastSquaresSolution:
    amount: FloatArray
    objective: float
    support: tuple[int, ...]


def _rank(singular_values: FloatArray, reference_scale: float) -> int:
    if singular_values.size == 0 or reference_scale == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > sqrt(np.finfo(np.float64).eps) * reference_scale))


def _scale_safe_column_norm(matrix: FloatArray) -> FloatArray:
    maximum = np.max(np.abs(matrix), axis=0)
    nonzero = maximum > 0.0
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    scaled_norm = np.linalg.norm(matrix[:, nonzero] / maximum[nonzero], axis=0)
    with np.errstate(over="ignore"):
        result[nonzero] = maximum[nonzero] * scaled_norm
    overflow = ~np.isfinite(result)
    result[overflow] = maximum[overflow]
    return result


def _condition(singular_values: FloatArray, rank: int, required_rank: int) -> float:
    return (
        float(singular_values[0] / singular_values[required_rank - 1])
        if rank == required_rank and singular_values[required_rank - 1] > 0.0
        else float("inf")
    )


def _response_diagnostics(
    response: FloatArray,
    domain_fraction: FloatArray,
    allowed_component_indices: tuple[int, ...],
    allowed_phase_indices: tuple[int, ...],
) -> tuple[FloatArray, int, float, FloatArray, int, float]:
    allowed_response = response[:, allowed_component_indices]
    singular = np.linalg.svd(allowed_response, compute_uv=False)
    reference_scale = float(singular[0]) if singular.size else 0.0
    rank = _rank(singular, reference_scale)
    response_condition = _condition(singular, rank, len(allowed_component_indices))
    phase_profiles: dict[int, FloatArray] = {}
    hand_nuisance: list[FloatArray] = []
    for phase_index in allowed_phase_indices:
        phase_components = tuple(
            component
            for component in allowed_component_indices
            if STACKING_PHASE_IDS.index(STACKING_COMPONENT_IDS[component][:2]) == phase_index
        )
        phase_profiles[phase_index] = np.mean(response[:, phase_components], axis=1)
        if len(phase_components) == 2:
            hand_nuisance.append(
                0.5 * (response[:, phase_components[0]] - response[:, phase_components[1]])
            )
    reference_phase = allowed_phase_indices[0]
    phase_contrast = (
        np.column_stack(
            tuple(
                phase_profiles[phase_index] - phase_profiles[reference_phase]
                for phase_index in allowed_phase_indices[1:]
            )
        )
        if len(allowed_phase_indices) > 1
        else np.zeros((response.shape[0], 0))
    )
    nuisance = np.column_stack((response @ domain_fraction, *hand_nuisance))
    nuisance_norm = _scale_safe_column_norm(nuisance)
    retained_nuisance = nuisance[:, nuisance_norm > 0.0] / nuisance_norm[nuisance_norm > 0.0]
    nuisance_left, nuisance_singular, _ = np.linalg.svd(retained_nuisance, full_matrices=False)
    nuisance_reference = float(nuisance_singular[0]) if nuisance_singular.size else 0.0
    nuisance_rank = _rank(nuisance_singular, nuisance_reference)
    if nuisance_rank and phase_contrast.shape[1]:
        nuisance_basis = nuisance_left[:, :nuisance_rank]
        phase_contrast -= nuisance_basis @ (nuisance_basis.T @ phase_contrast)
    raw_phase_singular = np.linalg.svd(phase_contrast, compute_uv=False)
    phase_singular = np.zeros(2, dtype=np.float64)
    phase_singular[: raw_phase_singular.size] = raw_phase_singular
    phase_rank = _rank(phase_singular, reference_scale)
    required_phase_rank = len(allowed_phase_indices) - 1
    phase_condition = (
        1.0
        if required_phase_rank == 0
        else _condition(phase_singular, phase_rank, required_phase_rank)
    )
    return (
        singular,
        rank,
        response_condition,
        phase_singular,
        phase_rank,
        phase_condition,
    )


def _nonnegative_least_squares(
    response: FloatArray,
    signal: FloatArray,
    equality: FloatArray | None = None,
    allowed_component_indices: tuple[int, ...] = tuple(range(5)),
) -> _LeastSquaresSolution:
    signal_norm_squared = float(signal @ signal)
    best = _LeastSquaresSolution(np.zeros(5), signal_norm_squared, ())
    rcond = sqrt(np.finfo(np.float64).eps)
    for mask in range(1, 1 << len(allowed_component_indices)):
        support = tuple(
            component
            for bit, component in enumerate(allowed_component_indices)
            if mask & (1 << bit)
        )
        selected = response[:, support]
        scale = _scale_safe_column_norm(selected)
        safe_scale = np.where(scale > 0.0, scale, 1.0)
        scaled = selected / safe_scale
        if equality is None:
            scaled_amount = np.linalg.lstsq(scaled, signal, rcond=rcond)[0]
        else:
            constraint = equality[list(support)] / safe_scale
            _, constraint_singular, constraint_right = np.linalg.svd(
                constraint[None, :], full_matrices=True
            )
            constraint_rank = _rank(constraint_singular, float(np.linalg.norm(constraint)))
            null_basis = constraint_right[constraint_rank:].T
            if null_basis.shape[1] == 0:
                continue
            scaled_amount = (
                null_basis @ np.linalg.lstsq(scaled @ null_basis, signal, rcond=rcond)[0]
            )
        feasibility_scale = max(
            float(np.linalg.norm(scaled_amount, ord=np.inf)),
            float(np.linalg.norm(signal)),
            np.finfo(np.float64).tiny,
        )
        positivity_tolerance = 8192.0 * np.finfo(np.float64).eps * feasibility_scale
        if np.any(scaled_amount < -positivity_tolerance):
            continue
        amount = np.zeros(5, dtype=np.float64)
        amount[list(support)] = np.maximum(scaled_amount, 0.0) / safe_scale
        if equality is not None:
            equality_tolerance = (
                8192.0
                * np.finfo(np.float64).eps
                * max(
                    float(np.linalg.norm(equality) * np.linalg.norm(amount)),
                    np.finfo(np.float64).tiny,
                )
            )
            if abs(float(equality @ amount)) > equality_tolerance:
                continue
        residual = response @ amount - signal
        objective = float(residual @ residual)
        effective_support = tuple(np.flatnonzero(amount > 0.0).tolist())
        tolerance = (
            64.0
            * np.finfo(np.float64).eps
            * max(objective, best.objective, np.finfo(np.float64).tiny)
        )
        if objective < best.objective - tolerance or (
            abs(objective - best.objective) <= tolerance
            and (len(effective_support), effective_support) < (len(best.support), best.support)
        ):
            best = _LeastSquaresSolution(amount, objective, effective_support)
    return best


def _profile_delta_function(
    response: FloatArray,
    signal: FloatArray,
    best_objective: float,
    phase_index: int,
    delta_chi_square: float,
    allowed_component_indices: tuple[int, ...],
) -> Callable[[float], float]:
    cache: dict[float, float] = {}

    def profile_delta(value: float) -> float:
        key = float(value)
        if key not in cache:
            equality = _PHASE_AGGREGATION[phase_index] - key
            constrained = _nonnegative_least_squares(
                response,
                signal,
                equality,
                allowed_component_indices,
            )
            cache[key] = constrained.objective - best_objective - delta_chi_square
        return cache[key]

    return profile_delta


def _profile_bounds(
    response: FloatArray,
    signal: FloatArray,
    best: _LeastSquaresSolution,
    phase_fraction: FloatArray,
    delta_chi_square: float,
    allowed_component_indices: tuple[int, ...],
    allowed_phase_indices: tuple[int, ...],
) -> tuple[FloatArray, BoolArray]:
    bounds = np.zeros((3, 2), dtype=np.float64)
    for phase_index, optimum in enumerate(phase_fraction):
        if phase_index not in allowed_phase_indices:
            continue
        if len(allowed_phase_indices) == 1:
            bounds[phase_index] = (1.0, 1.0)
            continue
        profile_delta = _profile_delta_function(
            response,
            signal,
            best.objective,
            phase_index,
            delta_chi_square,
            allowed_component_indices,
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(best.objective, 1.0)
        if profile_delta(float(optimum)) > tolerance:
            raise FloatingPointError("best phase fraction is outside its own profile bounds")
        bounds[phase_index, 0] = (
            0.0
            if profile_delta(0.0) <= tolerance
            else brentq(profile_delta, 0.0, float(optimum), xtol=2.0e-13)
        )
        bounds[phase_index, 1] = (
            1.0
            if profile_delta(1.0) <= tolerance
            else brentq(profile_delta, float(optimum), 1.0, xtol=2.0e-13)
        )
    boundary = (phase_fraction <= 4096.0 * np.finfo(np.float64).eps) | (
        phase_fraction >= 1.0 - 4096.0 * np.finfo(np.float64).eps
    )
    return bounds, boundary


def _broadcast_row(value: ArrayLike, size: int, name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(np.broadcast_to(supplied.real, (size,)), dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"{name} must be scalar or have shape ({size},)") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _fit_stacking_phase_totals_matrix(
    component_response_A2: FloatArray,
    response_revision: str,
    observed_strength_A2: ArrayLike,
    variance_A4: ArrayLike,
    *,
    allowed_component_ids: tuple[str, ...],
    profile_delta_chi_square: float = 1.0,
    observation_revision: str | None = None,
) -> StackingPopulationFitResult:
    allowed, allowed_indices, allowed_phase_indices = _canonical_allowed_component_ids(
        allowed_component_ids
    )
    row_count = component_response_A2.shape[0]
    observed = _broadcast_row(observed_strength_A2, row_count, "observed_strength_A2")
    variance = _broadcast_row(variance_A4, row_count, "variance_A4")
    if np.any(variance <= 0.0):
        raise ValueError("variance must be positive")
    profile_delta = float(profile_delta_chi_square)
    if not isfinite(profile_delta) or profile_delta <= 0.0:
        raise ValueError("profile_delta_chi_square must be finite and positive")
    inverse_sigma = 1.0 / np.sqrt(variance)
    weighted_response = component_response_A2 * inverse_sigma[:, None]
    weighted_signal = observed * inverse_sigma
    solution = _nonnegative_least_squares(
        weighted_response,
        weighted_signal,
        allowed_component_indices=allowed_indices,
    )
    global_scale = float(np.sum(solution.amount))
    if not solution.support or global_scale <= 0.0:
        raise ValueError("fitted stacking scale is zero; phase fractions are undefined")
    domain_fraction = solution.amount / global_scale
    phase_fraction = _PHASE_AGGREGATION @ domain_fraction
    (
        response_singular,
        response_rank,
        response_condition,
        phase_singular,
        phase_rank,
        phase_condition,
    ) = _response_diagnostics(
        weighted_response,
        domain_fraction,
        allowed_indices,
        allowed_phase_indices,
    )
    required_phase_rank = len(allowed_phase_indices) - 1
    if phase_rank < required_phase_rank:
        raise StackingPopulationIdentifiabilityError(
            "sampled profiles cannot separate the allowed stacking phase totals",
            phase_contrast_rank=phase_rank,
            phase_contrast_singular_values=phase_singular,
            phase_contrast_condition=phase_condition,
        )
    predicted = component_response_A2 @ solution.amount
    weighted_residual = (observed - predicted) * inverse_sigma
    profile_bounds, boundary = _profile_bounds(
        weighted_response,
        weighted_signal,
        solution,
        phase_fraction,
        profile_delta,
        allowed_indices,
        allowed_phase_indices,
    )
    return StackingPopulationFitResult(
        domain_amount=solution.amount,
        domain_fraction=domain_fraction,
        phase_fraction=phase_fraction,
        global_scale=global_scale,
        predicted_strength_A2=predicted,
        weighted_residual=weighted_residual,
        chi_square=float(weighted_residual @ weighted_residual),
        active_component_ids=tuple(STACKING_COMPONENT_IDS[value] for value in solution.support),
        domain_response_full_rank=response_rank == len(allowed_indices),
        response_singular_values=response_singular,
        response_rank=response_rank,
        response_condition=response_condition,
        phase_contrast_singular_values=phase_singular,
        phase_contrast_rank=phase_rank,
        phase_contrast_condition=phase_condition,
        phase_profile_bounds=profile_bounds,
        phase_estimate_on_boundary=boundary,
        profile_delta_chi_square=profile_delta,
        response_revision=response_revision,
        allowed_component_ids=allowed,
        observation_revision=observation_revision,
    )


def fit_stacking_phase_totals(
    response: CompiledStackingResponse,
    observed_strength_A2: ArrayLike,
    variance_A4: ArrayLike,
    *,
    profile_delta_chi_square: float = 1.0,
) -> StackingPopulationFitResult:
    """Fit one global nonnegative amount vector, then normalize to phase fractions."""

    if not isinstance(response, CompiledStackingResponse):
        raise TypeError("response must be CompiledStackingResponse")
    return _fit_stacking_phase_totals_matrix(
        response.component_response_A2,
        response.response_revision,
        observed_strength_A2,
        variance_A4,
        allowed_component_ids=STACKING_COMPONENT_IDS,
        profile_delta_chi_square=profile_delta_chi_square,
    )


def fit_layer_l_stacking_phase_totals(
    response: CompiledLayerLStackingResponse,
    observations: LayerLStackingObservations,
    *,
    allowed_component_ids: tuple[str, ...] = STACKING_COMPONENT_IDS,
    profile_delta_chi_square: float = 1.0,
) -> StackingPopulationFitResult:
    """Fit one specimen after joining intrinsic strengths by exact structural identity."""

    if not isinstance(response, CompiledLayerLStackingResponse):
        raise TypeError("response must be CompiledLayerLStackingResponse")
    if not isinstance(observations, LayerLStackingObservations):
        raise TypeError("observations must be LayerLStackingObservations")
    allowed_component_ids = _canonical_layer_l_component_ids(allowed_component_ids)
    if observations.specimen_id != response.specimen_id:
        raise ValueError("observation specimen_id disagrees with the compiled response")
    if observations.sampling_revision != response.sampling_revision:
        raise ValueError("observations changed the response sampling revision")
    observation_by_key = {key: index for index, key in enumerate(observations.group_keys)}
    if set(observation_by_key) != set(response.group_keys):
        raise ValueError("observed and compiled stacking reflection-group identities differ")
    ordered_indices: list[int] = []
    for group in response.group_keys:
        ordered_indices.append(observation_by_key[group])
    indices = np.asarray(ordered_indices, dtype=np.int64)
    return _fit_stacking_phase_totals_matrix(
        response.component_response_A2,
        response.response_revision,
        observations.observed_strength_A2[indices],
        observations.variance_A4[indices],
        allowed_component_ids=allowed_component_ids,
        profile_delta_chi_square=profile_delta_chi_square,
        observation_revision=observations.observation_revision,
    )


__all__ = [
    "STACKING_COMPONENT_IDS",
    "STACKING_PHASE_IDS",
    "CompiledLayerLStackingResponse",
    "CompiledStackingResponse",
    "LayerLStackingObservations",
    "Pbi2ParentLogRatioParameterization",
    "Pbi2ParentMixtureStrength",
    "StackingPopulationFitResult",
    "StackingPopulationIdentifiabilityError",
    "compile_pbi2_layer_l_stacking_response",
    "compile_pbi2_stacking_profile_response",
    "fit_layer_l_stacking_phase_totals",
    "fit_stacking_phase_totals",
]
