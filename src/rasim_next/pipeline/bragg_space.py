"""Material-to-mosaic integration for continuous pre-Ewald Bragg space."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import Rod
from painted_ewald.validation import finite_scalar, reject_complex
from rasim_next.core.contracts import (
    EventIntensityNormalization,
    LayerNormalQBatch,
    RodQueryBatch,
)
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.materials import CrystalStructure
from rasim_next.ordered import (
    Bi2Se3QuintupleLayerParameters,
    bi2se3_ql_amplitudes,
    uniform_finite_stack,
)
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking import (
    InitialPopulation,
    Parent,
    RichEpsilonModel,
    TransitionLaw,
    finite_event_intensity,
    registry_phase,
)
from rasim_next.stacking.finite_intensity import finite_intensity_reduced

FloatArray = NDArray[np.float64]


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
class Bi2Se3TwoHStrength:
    """Finite parent-2H strength from a CIF-derived Bi2Se3 quintuple layer.

    ``Parent.TWO_H`` is the registry-fixed AA sequence. The source R-3m CIF
    supplies the internal quintuple-layer motif but its native registry-cycling
    3R sequence is intentionally not selected by this model. A nonzero shared
    disorder epsilon assigns ``1-epsilon`` to the 2H parent transition and
    ``epsilon/4`` to each of the four alternative transitions.
    """

    crystal: CrystalStructure
    layers: int
    normalization: EventIntensityNormalization = EventIntensityNormalization.FINITE_PER_LAYER
    shared_disorder_epsilon: float = 0.0
    structure_parameters: Bi2Se3QuintupleLayerParameters | None = None
    _lattice: ReciprocalLattice = field(init=False, repr=False, compare=False)

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
            raise ValueError("2H stacking normalization must be FINITE_TOTAL or FINITE_PER_LAYER")
        epsilon = RichEpsilonModel(
            Parent.TWO_H,
            self.shared_disorder_epsilon,
        ).epsilon
        parameters = (
            Bi2Se3QuintupleLayerParameters.from_crystal(self.crystal)
            if self.structure_parameters is None
            else self.structure_parameters
        )
        if not isinstance(parameters, Bi2Se3QuintupleLayerParameters):
            raise TypeError("structure_parameters must be Bi2Se3QuintupleLayerParameters")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "shared_disorder_epsilon", epsilon)
        object.__setattr__(self, "structure_parameters", parameters)
        object.__setattr__(self, "_lattice", ReciprocalLattice.from_crystal(self.crystal))

    @property
    def reciprocal_basis_Ainv(self) -> FloatArray:
        """CIF-derived basis to which this strength model is bound."""

        return self._lattice.basis_Ainv

    def evaluate_profile(
        self,
        *,
        rod: Rod,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Evaluate the continuous exact-L 2H profile for one physical rod."""

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
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Vectorize the authoritative strength over mixed physical rods and exact L."""

        shape, query, layer_normal_q = self._hkl_query(
            h=h,
            k=k,
            L=L,
            k_norm_Ainv=k_norm_Ainv,
        )
        amplitudes = bi2se3_ql_amplitudes(
            self.crystal,
            query,
            structure_parameters=self.structure_parameters,
        )
        law = (
            TransitionLaw.for_parent(Parent.TWO_H)
            if self.shared_disorder_epsilon == 0.0
            else RichEpsilonModel(
                Parent.TWO_H,
                self.shared_disorder_epsilon,
            ).transition_law()
        )
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
            model_component_id="2H",
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
        k_norm_Ainv: float,
    ) -> tuple[tuple[int, ...], RodQueryBatch, FloatArray]:
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
        k_norm = finite_scalar(k_norm_Ainv, "k_norm_Ainv")
        if k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be positive")
        shape = ell.shape
        ell_flat = ell.reshape(-1)
        h_flat = h_integer.reshape(-1)
        k_flat = k_integer.reshape(-1)
        wavelength_A = 2.0 * np.pi / k_norm
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
            wavelength_A=np.full(ell_flat.size, wavelength_A),
        )
        return shape, query, layer_normal_q

    def fixed_position_occupancy_quadratic(
        self,
        *,
        h: ArrayLike,
        k: ArrayLike,
        L: ArrayLike,
        k_norm_Ainv: float,
    ) -> FloatArray:
        """Compile six occupancy coefficients with coordinates and stacking fixed."""

        shape, query, layer_normal_q = self._hkl_query(
            h=h,
            k=k,
            L=L,
            k_norm_Ainv=k_norm_Ainv,
        )
        fixed = self.structure_parameters
        if not isinstance(fixed, Bi2Se3QuintupleLayerParameters):
            raise TypeError("Bi2Se3 strength requires resolved structure parameters")
        reference = replace(
            fixed,
            bi_occupancy=0.0,
            se1_occupancy=0.0,
            se2_occupancy=0.0,
            u_radial_A2=0.0,
            u_normal_A2=0.0,
        )

        amplitude_basis = tuple(
            bi2se3_ql_amplitudes(
                self.crystal,
                query,
                structure_parameters=replace(
                    reference,
                    bi_occupancy=occupancies[0],
                    se1_occupancy=occupancies[1],
                    se2_occupancy=occupancies[2],
                ),
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
                Parent.TWO_H,
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
            return _nearest_physical_occupancy_quadratic(
                np.asarray(quadratic.reshape((*shape, 6)), dtype=np.float64),
                plus_basis_e=plus_basis,
                minus_basis_e=minus_basis,
                layers=self.layers,
                normalization=self.normalization,
            )

        def ordered_strength(amplitude_e: NDArray[np.complex128]) -> FloatArray:
            result = uniform_finite_stack(
                query.event_id,
                layer_normal_q,
                amplitude_e,
                repeat_spacing_A,
                self.layers,
            ).scattering_strength_A2
            if self.normalization is EventIntensityNormalization.FINITE_PER_LAYER:
                result = result / float(self.layers)
            return np.asarray(result, dtype=np.float64)

        bi_amplitude, se1_amplitude, se2_amplitude = (item.f_plus_e for item in amplitude_basis)
        bi_strength = ordered_strength(bi_amplitude)
        se1_strength = ordered_strength(se1_amplitude)
        se2_strength = ordered_strength(se2_amplitude)
        quadratic = np.column_stack(
            (
                bi_strength,
                se1_strength,
                se2_strength,
                ordered_strength(bi_amplitude + se1_amplitude) - bi_strength - se1_strength,
                ordered_strength(bi_amplitude + se2_amplitude) - bi_strength - se2_strength,
                ordered_strength(se1_amplitude + se2_amplitude) - se1_strength - se2_strength,
            )
        )
        return _nearest_physical_occupancy_quadratic(
            np.asarray(quadratic.reshape((*shape, 6)), dtype=np.float64),
            plus_basis_e=plus_basis,
            minus_basis_e=minus_basis,
            layers=self.layers,
            normalization=self.normalization,
        )

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        """Implement the scalar painted-Ewald strength protocol."""

        return float(self.evaluate_profile(rod=rod, L=L, k_norm_Ainv=k_norm_Ainv))


__all__ = ["Bi2Se3TwoHStrength"]
