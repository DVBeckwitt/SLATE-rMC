"""Material-to-mosaic integration for continuous pre-Ewald Bragg space."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from operator import index
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import BasisBoundStrengthModel, Rod
from painted_ewald.validation import finite_scalar, reject_complex
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerNormalQBatch,
    RodQueryBatch,
    canonical_revision_sha256,
)
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.materials import (
    AffineCifSiteBasis,
    CrystalStructure,
    crystal_structure_revision,
)
from rasim_next.ordered import (
    Bi2X3QuintupleLayerParameters,
    SiteDisplacementProfile,
    bi2x3_quintuple_layer_amplitudes,
    finite_periodic_repeat_amplitude_factor,
    quintuple_layer_site_labels,
    uniform_finite_stack,
    unit_cell_amplitude,
)
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking import (
    InitialPopulation,
    Parent,
    RichEpsilonModel,
    finite_event_intensity,
    registry_phase,
)
from rasim_next.stacking.finite_intensity import finite_intensity_reduced

FloatArray = NDArray[np.float64]


def finite_stack_integer_l_display_nodes(
    lower_l: float,
    upper_l: float,
    *,
    layer_count: int,
    background_count: int,
) -> FloatArray:
    """Return coarse coverage plus integer-L and +/-0.5/N display landmarks."""

    bounds = np.asarray((lower_l, upper_l), dtype=np.float64)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError("finite-stack display bounds must be finite and ordered")
    if (
        isinstance(layer_count, (bool, np.bool_))
        or isinstance(background_count, (bool, np.bool_))
        or index(layer_count) < 1
        or index(background_count) < 2
    ):
        raise ValueError("layer_count must be positive and background_count at least two")
    layer_count = index(layer_count)
    background_count = index(background_count)
    integer_l = np.arange(math.ceil(bounds[0]), math.floor(bounds[1]) + 1, dtype=np.float64)
    if integer_l.size:
        shoulder_l = 0.5 / layer_count
        resolved_l = (integer_l[:, None] + (-shoulder_l, 0.0, shoulder_l)).reshape(-1)
        resolved_l = resolved_l[(resolved_l >= bounds[0]) & (resolved_l <= bounds[1])]
    else:
        resolved_l = np.empty(0, dtype=np.float64)
    background_l = np.linspace(bounds[0], bounds[1], background_count)
    result = np.array(
        np.unique(np.concatenate((background_l, resolved_l))),
        dtype=np.float64,
        copy=True,
        order="C",
    )
    result.setflags(write=False)
    return result


class RevisionedStructureStrengthModel(BasisBoundStrengthModel, Protocol):
    """Basis-bound strength provider with immutable structure lineage."""

    @property
    def structure_model_revision(self) -> str: ...

    def evaluate_hkl(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: ArrayLike,
    ) -> FloatArray: ...


class StructureStrengthParameterization(Protocol):
    """Bind one explicit parameter vector to a basis-bound strength model."""

    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    reference_parameters: FloatArray
    parameterization_revision: str
    reference_strength: RevisionedStructureStrengthModel

    def bind_strength(self, parameters: ArrayLike) -> RevisionedStructureStrengthModel: ...


@dataclass(frozen=True, slots=True)
class CifFiniteStackStrength:
    """Coherent finite repeat of one complete periodic CIF unit cell.

    The CIF's first two direct-lattice vectors define the surface lattice and
    its third reciprocal vector defines the continuous rod coordinate ``L``.
    Repeating the complete unit cell is an ordered-film default; it does not
    infer polytypes, faults, terminations, or substrate structure.
    """

    crystal: CrystalStructure
    repeats: int
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_TOTAL
    unknown_u_iso_A2: float | None = None
    _lattice: ReciprocalLattice = field(init=False, repr=False, compare=False)
    structure_model_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.crystal, CrystalStructure):
            raise TypeError("crystal must be a CrystalStructure")
        try:
            repeats = index(self.repeats)
        except TypeError as error:
            raise ValueError("repeats must be a positive integer") from error
        if isinstance(self.repeats, bool) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        normalization = EventIntensityNormalization(self.normalization)
        if normalization is EventIntensityNormalization.UNIT_CELL:
            raise ValueError("finite CIF strength requires FINITE_TOTAL or FINITE_PER_LAYER")
        unknown_u = self.unknown_u_iso_A2
        if unknown_u is not None:
            unknown_u = finite_scalar(unknown_u, "unknown_u_iso_A2")
            if unknown_u < 0.0:
                raise ValueError("unknown_u_iso_A2 must be nonnegative")
        if unknown_u is None and any(site.u_iso_A2 is None for site in self.crystal.sites):
            raise ValueError("unknown CIF displacement requires explicit unknown_u_iso_A2")

        lattice = ReciprocalLattice.from_crystal(self.crystal)
        effective_u_iso_A2 = np.asarray(
            [unknown_u if site.u_iso_A2 is None else site.u_iso_A2 for site in self.crystal.sites],
            dtype=np.float64,
        )
        revision = canonical_revision_sha256(
            ("definition_id", "cif_finite_periodic_stack_strength.v1"),
            ("phase_id", self.crystal.phase_id),
            ("spacegroup_hm", self.crystal.spacegroup_hm),
            ("direct_basis_A", self.crystal.direct_basis_A),
            ("site_source_label", tuple(site.source_label for site in self.crystal.sites)),
            ("site_species", tuple(site.species for site in self.crystal.sites)),
            ("site_element", tuple(site.element for site in self.crystal.sites)),
            (
                "site_charge",
                np.asarray([site.charge for site in self.crystal.sites], dtype=np.int64),
            ),
            (
                "site_occupancy",
                np.asarray([site.occupancy for site in self.crystal.sites], dtype=np.float64),
            ),
            (
                "site_fractional",
                np.asarray([site.fractional for site in self.crystal.sites], dtype=np.float64),
            ),
            ("site_effective_u_iso_A2", effective_u_iso_A2),
            ("repeats", repeats),
            ("normalization", normalization.value),
        )
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "unknown_u_iso_A2", unknown_u)
        object.__setattr__(self, "_lattice", lattice)
        object.__setattr__(self, "structure_model_revision", revision)

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray:
        """CIF-derived reciprocal basis used by every rod query."""

        return self._lattice.basis_Ainv

    def evaluate_profile(
        self,
        *,
        rod: Rod,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Evaluate one signed physical rod at arbitrary continuous ``L``."""

        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        reject_complex(L, "L")
        ell = np.asarray(L, dtype=np.float64)
        if not np.all(np.isfinite(ell)):
            raise ValueError("L must be finite")
        k_norm = finite_scalar(k_norm_Ainv, "k_norm_Ainv")
        if k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be positive")
        return self.evaluate_hkl(
            h=np.full(ell.shape, rod.h, dtype=np.int32),
            k=np.full(ell.shape, rod.k, dtype=np.int32),
            L=ell,
            k_norm_Ainv=np.full(ell.shape, k_norm, dtype=np.float64),
        )

    def evaluate_hkl(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: ArrayLike,
    ) -> FloatArray:
        """Vectorize the CIF strength over mixed rods, wavelengths, and exact ``L``."""

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
        integer_bounds = np.iinfo(np.int32)
        if np.any((h_value < integer_bounds.min) | (h_value > integer_bounds.max)) or np.any(
            (k_value < integer_bounds.min) | (k_value > integer_bounds.max)
        ):
            raise ValueError("h and k must fit signed 32-bit integers")
        hkl = np.stack((h_value, k_value, ell), axis=-1)
        amplitude = unit_cell_amplitude(
            self.crystal,
            hkl,
            2.0 * np.pi / k_norm,
            unknown_u_iso_A2=self.unknown_u_iso_A2,
        ).amplitude_e
        repeat = finite_periodic_repeat_amplitude_factor(ell, self.repeats)
        strength = electron_squared_to_scattering_strength_A2(np.abs(amplitude * repeat) ** 2)
        if self.normalization is EventIntensityNormalization.FINITE_PER_LAYER:
            strength = strength / float(self.repeats)
        return np.asarray(strength, dtype=np.float64)

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        """Implement the scalar reciprocal-basis-bound strength protocol."""

        return float(self.evaluate_profile(rod=rod, L=L, k_norm_Ainv=k_norm_Ainv))

    def rebind_crystal(self, crystal: CrystalStructure) -> CifFiniteStackStrength:
        """Return the same declared finite-repeat model for a candidate crystal."""

        if not isinstance(crystal, CrystalStructure):
            raise TypeError("crystal must be a CrystalStructure")
        return replace(self, crystal=crystal)


@dataclass(frozen=True, slots=True)
class AffineCifFiniteStackParameterization:
    """Bind an affine expanded-CIF basis to the generic finite-repeat provider."""

    reference_strength: CifFiniteStackStrength
    site_basis: AffineCifSiteBasis
    parameter_names: tuple[str, ...] = field(init=False)
    parameter_units: tuple[str, ...] = field(init=False)
    reference_parameters: FloatArray = field(init=False)
    parameterization_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_strength, CifFiniteStackStrength):
            raise TypeError("reference_strength must be CifFiniteStackStrength")
        if not isinstance(self.site_basis, AffineCifSiteBasis):
            raise TypeError("site_basis must be AffineCifSiteBasis")
        if self.site_basis.reference_crystal_revision != crystal_structure_revision(
            self.reference_strength.crystal
        ):
            raise ValueError("affine CIF basis and reference strength identify different crystals")
        if self.site_basis.unknown_u_iso_A2 != self.reference_strength.unknown_u_iso_A2:
            raise ValueError("affine CIF basis and strength require the same unknown-U policy")
        reference = np.array(self.site_basis.reference_parameters, copy=True)
        reference.setflags(write=False)
        revision = canonical_revision_sha256(
            ("definition_id", "affine_cif_finite_stack_parameterization.v1"),
            (
                "reference_structure_model_revision",
                self.reference_strength.structure_model_revision,
            ),
            ("site_basis_revision", self.site_basis.basis_revision),
        )
        object.__setattr__(self, "parameter_names", self.site_basis.parameter_names)
        object.__setattr__(self, "parameter_units", self.site_basis.parameter_units)
        object.__setattr__(self, "reference_parameters", reference)
        object.__setattr__(self, "parameterization_revision", revision)

    def bind_strength(self, parameters: ArrayLike) -> CifFiniteStackStrength:
        """Apply one validated vector without changing repeat or normalization state."""

        return self.reference_strength.rebind_crystal(self.site_basis.apply(parameters))


def _nearest_physical_occupancy_quadratic(
    quadratic_A2: FloatArray,
    *,
    plus_basis_e: NDArray[np.complex128],
    minus_basis_e: NDArray[np.complex128],
    layers: int,
    normalization: EventIntensityNormalization,
) -> FloatArray:
    """Remove only roundoff-sized negative modes from a physical intensity quadratic."""

    flat = np.asarray(quadratic_A2, dtype=np.float64).reshape(-1, 6)
    matrix = np.empty((flat.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = flat[:, 0]
    matrix[:, 1, 1] = flat[:, 1]
    matrix[:, 2, 2] = flat[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = 0.5 * flat[:, 3]
    matrix[:, 0, 2] = matrix[:, 2, 0] = 0.5 * flat[:, 4]
    matrix[:, 1, 2] = matrix[:, 2, 1] = 0.5 * flat[:, 5]
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    negative = eigenvalues[:, 0] < 0.0
    if not np.any(negative):
        return np.asarray(quadratic_A2, dtype=np.float64)

    state_norm_squared = np.maximum(
        np.sum(np.abs(plus_basis_e) ** 2, axis=1),
        np.sum(np.abs(minus_basis_e) ** 2, axis=1),
    )
    normalization_count = (
        float(layers)
        if normalization is EventIntensityNormalization.FINITE_PER_LAYER
        else float(layers * layers)
    )
    coherent_envelope_A2 = electron_squared_to_scattering_strength_A2(
        normalization_count * state_norm_squared
    )
    work = 256.0 * float(layers)
    rho = work * np.finfo(np.float64).eps
    if rho >= 1.0:
        raise FloatingPointError("finite-stack roundoff bound is undefined")
    gamma = rho / (1.0 - rho)
    local_scale = np.max(np.abs(eigenvalues), axis=1)
    tolerance = 8.0 * gamma * local_scale + 8.0 * gamma * gamma * coherent_envelope_A2
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise FloatingPointError("occupancy intensity quadratic is meaningfully non-PSD")

    clipped = np.maximum(eigenvalues[negative], 0.0)
    vectors = eigenvectors[negative]
    projected = np.einsum(
        "nij,nj,nkj->nik",
        vectors,
        clipped,
        vectors,
        optimize=True,
    )
    result = flat.copy()
    result[negative, 0] = projected[:, 0, 0]
    result[negative, 1] = projected[:, 1, 1]
    result[negative, 2] = projected[:, 2, 2]
    result[negative, 3] = 2.0 * projected[:, 0, 1]
    result[negative, 4] = 2.0 * projected[:, 0, 2]
    result[negative, 5] = 2.0 * projected[:, 1, 2]
    return np.asarray(result.reshape(quadratic_A2.shape), dtype=np.float64)


def _physical_rod_id(rod: Rod) -> int:
    """Return a reversible nonnegative key for one signed integer pair."""

    bounds = np.iinfo(np.int32)
    if not bounds.min <= rod.h <= bounds.max or not bounds.min <= rod.k <= bounds.max:
        raise ValueError("Bi2Se3 rod indices must fit signed 32-bit integers")
    h_key = 2 * rod.h if rod.h >= 0 else -2 * rod.h - 1
    k_key = 2 * rod.k if rod.k >= 0 else -2 * rod.k - 1
    diagonal = h_key + k_key
    result = diagonal * (diagonal + 1) // 2 + k_key
    if result > np.iinfo(np.int64).max:
        raise ValueError("Bi2Se3 rod identity does not fit signed 64-bit integers")
    return result


@dataclass(frozen=True, slots=True)
class Bi2X3FiniteStackStrength:
    """Finite-stack strength from a CIF-derived Bi2-chalcogen3 layer.

    The parent is explicit: ``Parent.TWO_H`` is registry-fixed AA, while
    ``Parent.THREE_R`` is the native R-centered registry cycle.  A nonzero
    shared disorder epsilon assigns ``1-epsilon`` to the selected parent
    transition and ``epsilon/4`` to each alternative transition.
    """

    crystal: CrystalStructure
    layers: int
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_PER_LAYER
    parent: Parent = Parent.TWO_H
    shared_disorder_epsilon: float = 0.0
    structure_parameters: Bi2X3QuintupleLayerParameters | None = None
    site_displacement_profile: SiteDisplacementProfile | None = None
    _lattice: ReciprocalLattice = field(init=False, repr=False, compare=False)
    structure_model_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.crystal, CrystalStructure):
            raise TypeError("crystal must be a CrystalStructure")
        try:
            layers = index(self.layers)
        except TypeError as error:
            raise ValueError("layers must be a positive integer") from error
        if isinstance(self.layers, bool) or layers < 1:
            raise ValueError("layers must be a positive integer")
        normalization = EventIntensityNormalization(self.normalization)
        if normalization is EventIntensityNormalization.UNIT_CELL:
            raise ValueError(
                "finite stacking normalization must be FINITE_TOTAL or FINITE_PER_LAYER"
            )
        parent = Parent(self.parent)
        epsilon = RichEpsilonModel(parent, self.shared_disorder_epsilon).epsilon
        parameters = (
            Bi2X3QuintupleLayerParameters.from_crystal(self.crystal)
            if self.structure_parameters is None
            else self.structure_parameters
        )
        if not isinstance(parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("structure_parameters must be Bi2X3QuintupleLayerParameters")
        profile = self.site_displacement_profile
        if profile is not None and not isinstance(profile, SiteDisplacementProfile):
            raise TypeError("site_displacement_profile must be a SiteDisplacementProfile")
        if profile is not None:
            expected_labels = set(self.site_labels)
            actual_labels = {site.source_label for site in profile.sites}
            if actual_labels != expected_labels:
                raise ValueError("site displacement profile must cover each quintuple-layer orbit")
            if parameters.u_radial_A2 != 0.0 or parameters.u_normal_A2 != 0.0:
                raise ValueError(
                    "site-resolved and shared quintuple-layer displacements are mutually exclusive"
                )
        parameter_values = np.asarray(
            [
                parameters.bi_fractional_z,
                parameters.se2_fractional_z,
                parameters.bi_occupancy,
                parameters.se1_occupancy,
                parameters.se2_occupancy,
                parameters.u_radial_A2,
                parameters.u_normal_A2,
                parameters.outer_bi_antisite_fraction,
            ],
            dtype=np.float64,
        )
        revision_fields: list[tuple[str, object]] = [
            ("definition_id", "bi2x3_finite_quintuple_stack_strength.v1"),
            ("crystal_revision", crystal_structure_revision(self.crystal)),
            ("layers", layers),
            ("normalization", normalization.value),
            ("parent", parent.value),
            ("shared_disorder_epsilon", epsilon),
            ("structure_parameter_values", parameter_values),
        ]
        if profile is None:
            revision_fields.append(("site_displacement_profile", "shared_parameters"))
        else:
            revision_fields.extend(
                (
                    ("site_displacement_profile", "site_resolved"),
                    (
                        "site_displacement_labels",
                        tuple(site.source_label for site in profile.sites),
                    ),
                    (
                        "site_displacement_components_A2",
                        np.asarray(
                            [(site.u_radial_A2, site.u_normal_A2) for site in profile.sites],
                            dtype=np.float64,
                        ),
                    ),
                    ("site_displacement_scale", profile.scale),
                    ("site_displacement_provenance", profile.provenance),
                )
            )
        revision = canonical_revision_sha256(*revision_fields)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "parent", parent)
        object.__setattr__(self, "shared_disorder_epsilon", epsilon)
        object.__setattr__(self, "structure_parameters", parameters)
        object.__setattr__(self, "_lattice", ReciprocalLattice.from_crystal(self.crystal))
        object.__setattr__(self, "structure_model_revision", revision)

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray:
        """CIF-derived basis to which this strength model is bound."""

        return self._lattice.basis_Ainv

    @property
    def site_labels(self) -> tuple[str, str, str]:
        """CIF labels for Bi, central chalcogen, and outer chalcogen."""

        return quintuple_layer_site_labels(self.crystal)

    def evaluate_profile(
        self,
        *,
        rod: Rod,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Evaluate the continuous exact-L finite-stack profile for one physical rod."""

        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        ell = np.asarray(L, dtype=np.float64)
        return self.evaluate_hkl(
            h=np.full(ell.shape, rod.h, dtype=np.int32),
            k=np.full(ell.shape, rod.k, dtype=np.int32),
            L=ell,
            k_norm_Ainv=k_norm_Ainv,
        )

    def evaluate_hkl(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: ArrayLike,
    ) -> FloatArray:
        """Vectorize the authoritative strength over mixed physical rods and exact L."""

        shape, query, layer_normal_q, _ = self._hkl_query(
            h=h,
            k=k,
            L=L,
            k_norm_Ainv=k_norm_Ainv,
        )
        amplitudes = bi2x3_quintuple_layer_amplitudes(
            self.crystal,
            query,
            structure_parameters=self.structure_parameters,
            site_displacement_profile=self.site_displacement_profile,
        )
        law = RichEpsilonModel(
            self.parent,
            self.shared_disorder_epsilon,
        ).transition_law()
        result = finite_event_intensity(
            query,
            amplitudes,
            law,
            layer_normal_q=LayerNormalQBatch(
                event_id=query.event_id,
                rod_id=query.rod_id,
                phase_id=query.phase_id,
                layer_normal_q_Ainv=layer_normal_q,
                gauge_id=amplitudes.gauge_id,
            ),
            layers=self.layers,
            initial=InitialPopulation.plus_only(),
            model_component_id=self.parent.value,
            population_group_id=None,
            normalization=self.normalization,
        )
        return np.asarray(result.scattering_strength_A2.reshape(shape), dtype=np.float64)

    def _hkl_query(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: ArrayLike,
    ) -> tuple[tuple[int, ...], RodQueryBatch, FloatArray, FloatArray]:
        """Validate mixed indices once and build the virtual-normal query."""

        reject_complex(h, "h")
        reject_complex(k, "k")
        reject_complex(L, "L")
        h_value, k_value, ell = np.broadcast_arrays(
            np.asarray(h),
            np.asarray(k),
            np.asarray(L, dtype=np.float64),
        )
        if (
            not np.all(np.isfinite(h_value))
            or not np.all(np.isfinite(k_value))
            or not np.all(np.isfinite(ell))
        ):
            raise ValueError("h, k, and L must be finite")
        if np.any(h_value != np.rint(h_value)) or np.any(k_value != np.rint(k_value)):
            raise ValueError("h and k must contain integer rod indices")
        integer_bounds = np.iinfo(np.int32)
        if np.any((h_value < integer_bounds.min) | (h_value > integer_bounds.max)) or np.any(
            (k_value < integer_bounds.min) | (k_value > integer_bounds.max)
        ):
            raise ValueError("h and k must fit signed 32-bit integers")
        h_integer = np.asarray(h_value, dtype=np.int32)
        k_integer = np.asarray(k_value, dtype=np.int32)
        shape = ell.shape
        reject_complex(k_norm_Ainv, "k_norm_Ainv")
        k_norm_value = np.asarray(k_norm_Ainv, dtype=np.float64)
        try:
            k_norm = np.broadcast_to(k_norm_value, shape)
        except ValueError as error:
            raise ValueError("k_norm_Ainv must broadcast with h, k, and L") from error
        if np.any(~np.isfinite(k_norm)) or np.any(k_norm <= 0.0):
            raise ValueError("k_norm_Ainv must be finite and positive")
        ell_flat = ell.reshape(-1)
        h_flat = h_integer.reshape(-1)
        k_flat = k_integer.reshape(-1)
        wavelength_A = 2.0 * np.pi / k_norm.reshape(-1)
        hkl = np.column_stack(
            (
                h_flat,
                k_flat,
                ell_flat,
            )
        )
        q_crystal = self._lattice.q_cartesian_Ainv(hkl)
        layer_normal = np.cross(
            self.crystal.direct_basis_A[:, 0],
            self.crystal.direct_basis_A[:, 1],
        )
        layer_normal /= np.linalg.norm(layer_normal)
        if np.dot(layer_normal, self.crystal.direct_basis_A[:, 2]) < 0.0:
            layer_normal = -layer_normal
        layer_normal_q = q_crystal @ layer_normal
        q_radial_squared = np.maximum(
            np.einsum("ij,ij->i", q_crystal, q_crystal) - layer_normal_q**2,
            0.0,
        )
        event_id = np.arange(ell_flat.size, dtype=np.int64)
        unique_rod_hk, inverse_rod = np.unique(
            np.column_stack((h_flat, k_flat)),
            axis=0,
            return_inverse=True,
        )
        unique_rod_id = np.fromiter(
            (_physical_rod_id(Rod(int(h_row), int(k_row))) for h_row, k_row in unique_rod_hk),
            dtype=np.int64,
            count=unique_rod_hk.shape[0],
        )
        rod_id = unique_rod_id[inverse_rod]
        # Structure profiles use a virtual sample frame whose normal is the
        # crystal layer normal. This makes the sample-normal query field exact;
        # later mosaic rotations act only on the already evaluated scalar SF.
        query = RodQueryBatch(
            event_id=event_id,
            rod_id=rod_id,
            phase_id=(self.crystal.phase_id,) * ell_flat.size,
            h=h_flat,
            k=k_flat,
            q_sample_normal_Ainv=layer_normal_q,
            l_coordinate=ell_flat,
            wavelength_A=wavelength_A,
        )
        return shape, query, layer_normal_q, q_radial_squared

    def fixed_position_occupancy_quadratic(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Compile six occupancy coefficients with coordinates and stacking fixed."""

        shape, query, layer_normal_q, _ = self._hkl_query(
            h=h,
            k=k,
            L=L,
            k_norm_Ainv=k_norm_Ainv,
        )
        fixed = self.structure_parameters
        if not isinstance(fixed, Bi2X3QuintupleLayerParameters):
            raise TypeError("layered-quintuple strength requires resolved structure parameters")
        reference = replace(
            fixed,
            bi_occupancy=0.0,
            se1_occupancy=0.0,
            se2_occupancy=0.0,
            u_radial_A2=0.0,
            u_normal_A2=0.0,
        )

        amplitude_basis = tuple(
            bi2x3_quintuple_layer_amplitudes(
                self.crystal,
                query,
                structure_parameters=replace(
                    reference,
                    bi_occupancy=occupancies[0],
                    se1_occupancy=occupancies[1],
                    se2_occupancy=occupancies[2],
                ),
                site_displacement_profile=self.site_displacement_profile,
            )
            for occupancies in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        repeat_spacing_A = amplitude_basis[0].layer_repeat_A
        plus_basis = np.column_stack(tuple(item.f_plus_e for item in amplitude_basis))
        minus_basis = np.column_stack(tuple(item.f_minus_e for item in amplitude_basis))

        if self.shared_disorder_epsilon != 0.0:
            plus_mixture = np.column_stack(
                (
                    plus_basis,
                    plus_basis[:, 0] + plus_basis[:, 1],
                    plus_basis[:, 0] + plus_basis[:, 2],
                    plus_basis[:, 1] + plus_basis[:, 2],
                )
            )
            minus_mixture = np.column_stack(
                (
                    minus_basis,
                    minus_basis[:, 0] + minus_basis[:, 1],
                    minus_basis[:, 0] + minus_basis[:, 2],
                    minus_basis[:, 1] + minus_basis[:, 2],
                )
            )
            law = RichEpsilonModel(
                self.parent,
                self.shared_disorder_epsilon,
            ).transition_law()
            raw = finite_intensity_reduced(
                self.layers,
                plus_mixture,
                minus_mixture,
                np.asarray(registry_phase(query.h, query.k))[:, None],
                np.exp(1.0j * layer_normal_q * repeat_spacing_A)[:, None],
                law,
                InitialPopulation.plus_only(),
            )
            if self.normalization is EventIntensityNormalization.FINITE_PER_LAYER:
                raw = raw / float(self.layers)
            strength_mixture = electron_squared_to_scattering_strength_A2(raw)
            quadratic = np.column_stack(
                (
                    strength_mixture[:, :3],
                    strength_mixture[:, 3] - strength_mixture[:, 0] - strength_mixture[:, 1],
                    strength_mixture[:, 4] - strength_mixture[:, 0] - strength_mixture[:, 2],
                    strength_mixture[:, 5] - strength_mixture[:, 1] - strength_mixture[:, 2],
                )
            )
            result = _nearest_physical_occupancy_quadratic(
                np.asarray(quadratic.reshape((*shape, 6)), dtype=np.float64),
                plus_basis_e=plus_basis,
                minus_basis_e=minus_basis,
                layers=self.layers,
                normalization=self.normalization,
            )
            return result

        parent_law = RichEpsilonModel(
            self.parent,
            self.shared_disorder_epsilon,
        ).transition_law()
        registry = np.asarray(registry_phase(query.h, query.k))[:, None]
        vertical = np.exp(1.0j * layer_normal_q * repeat_spacing_A)[:, None]

        def ordered_strength(
            plus_amplitude_e: NDArray[np.complex128],
            minus_amplitude_e: NDArray[np.complex128],
        ) -> FloatArray:
            if self.parent is Parent.TWO_H:
                result = uniform_finite_stack(
                    query.event_id,
                    layer_normal_q,
                    plus_amplitude_e,
                    repeat_spacing_A,
                    self.layers,
                ).scattering_strength_A2
            else:
                raw = finite_intensity_reduced(
                    self.layers,
                    plus_amplitude_e[:, None],
                    minus_amplitude_e[:, None],
                    registry,
                    vertical,
                    parent_law,
                    InitialPopulation.plus_only(),
                )[:, 0]
                result = electron_squared_to_scattering_strength_A2(raw)
            if self.normalization is EventIntensityNormalization.FINITE_PER_LAYER:
                result = result / float(self.layers)
            return np.asarray(result, dtype=np.float64)

        bi_plus, se1_plus, se2_plus = (item.f_plus_e for item in amplitude_basis)
        bi_minus, se1_minus, se2_minus = (item.f_minus_e for item in amplitude_basis)
        bi_strength = ordered_strength(bi_plus, bi_minus)
        se1_strength = ordered_strength(se1_plus, se1_minus)
        se2_strength = ordered_strength(se2_plus, se2_minus)
        quadratic = np.column_stack(
            (
                bi_strength,
                se1_strength,
                se2_strength,
                ordered_strength(bi_plus + se1_plus, bi_minus + se1_minus)
                - bi_strength
                - se1_strength,
                ordered_strength(bi_plus + se2_plus, bi_minus + se2_minus)
                - bi_strength
                - se2_strength,
                ordered_strength(se1_plus + se2_plus, se1_minus + se2_minus)
                - se1_strength
                - se2_strength,
            )
        )
        result = _nearest_physical_occupancy_quadratic(
            np.asarray(quadratic.reshape((*shape, 6)), dtype=np.float64),
            plus_basis_e=plus_basis,
            minus_basis_e=minus_basis,
            layers=self.layers,
            normalization=self.normalization,
        )
        return result

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        """Implement the scalar painted-Ewald strength protocol."""

        return float(self.evaluate_profile(rod=rod, L=L, k_norm_Ainv=k_norm_Ainv))

    def rebind_structure_parameters(
        self,
        parameters: Bi2X3QuintupleLayerParameters,
    ) -> Bi2X3FiniteStackStrength:
        """Safely replace resolved QL parameters without retaining stale derived state."""

        if not isinstance(parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("parameters must be Bi2X3QuintupleLayerParameters")
        return replace(self, structure_parameters=parameters)


_BI2X3_PARAMETER_UNITS = {
    "bi_fractional_z": "fractional",
    "se2_fractional_z": "fractional",
    "bi_occupancy": "1",
    "se1_occupancy": "1",
    "se2_occupancy": "1",
    "u_radial_A2": "A2",
    "u_normal_A2": "A2",
    "outer_bi_antisite_fraction": "1",
}


@dataclass(frozen=True, slots=True)
class Bi2X3FiniteStackParameterization:
    """Explicit reusable parameter basis for the specialized QL strength provider."""

    reference_strength: Bi2X3FiniteStackStrength
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...] = field(init=False)
    reference_parameters: FloatArray = field(init=False)
    parameterization_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_strength, Bi2X3FiniteStackStrength):
            raise TypeError("reference_strength must be Bi2X3FiniteStackStrength")
        names = tuple(self.parameter_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in _BI2X3_PARAMETER_UNITS for name in names)
        ):
            raise ValueError("parameter_names must select unique supported QL parameters")
        if self.reference_strength.site_displacement_profile is not None and any(
            name in {"u_radial_A2", "u_normal_A2"} for name in names
        ):
            raise ValueError("shared displacement parameters conflict with the site profile")
        parameters = self.reference_strength.structure_parameters
        if not isinstance(parameters, Bi2X3QuintupleLayerParameters):
            raise TypeError("reference strength must contain resolved structure parameters")
        units = tuple(_BI2X3_PARAMETER_UNITS[name] for name in names)
        reference = np.asarray([getattr(parameters, name) for name in names], dtype=np.float64)
        reference.setflags(write=False)
        revision = canonical_revision_sha256(
            ("definition_id", "bi2x3_finite_stack_parameterization.v1"),
            (
                "reference_structure_model_revision",
                self.reference_strength.structure_model_revision,
            ),
            ("parameter_names", names),
            ("parameter_units", units),
            ("reference_parameters", reference),
        )
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", units)
        object.__setattr__(self, "reference_parameters", reference)
        object.__setattr__(self, "parameterization_revision", revision)

    def bind_strength(self, values: ArrayLike) -> Bi2X3FiniteStackStrength:
        """Bind one finite vector through the authoritative QL parameter dataclass."""

        supplied = np.asarray(values, dtype=np.float64)
        if supplied.shape != (len(self.parameter_names),) or np.any(~np.isfinite(supplied)):
            raise ValueError("values must be a finite vector aligned with parameter_names")
        reference = self.reference_strength.structure_parameters
        if not isinstance(reference, Bi2X3QuintupleLayerParameters):
            raise TypeError("reference strength must contain resolved structure parameters")
        candidate = replace(
            reference,
            **dict(zip(self.parameter_names, supplied.tolist(), strict=True)),
        )
        return self.reference_strength.rebind_structure_parameters(candidate)


__all__ = [
    "AffineCifFiniteStackParameterization",
    "Bi2X3FiniteStackParameterization",
    "Bi2X3FiniteStackStrength",
    "CifFiniteStackStrength",
    "RevisionedStructureStrengthModel",
    "StructureStrengthParameterization",
]
