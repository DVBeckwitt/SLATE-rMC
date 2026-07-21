"""Material-to-mosaic integration for continuous pre-Ewald Bragg space."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from rasim_next.materials import CrystalStructure
from rasim_next.ordered import bi2se3_ql_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.stacking import (
    InitialPopulation,
    Parent,
    RichEpsilonModel,
    TransitionLaw,
    finite_event_intensity,
)

FloatArray = NDArray[np.float64]


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
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "shared_disorder_epsilon", epsilon)
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
        reject_complex(L, "L")
        ell = np.asarray(L, dtype=np.float64)
        if not np.all(np.isfinite(ell)):
            raise ValueError("L must be finite")
        k_norm = finite_scalar(k_norm_Ainv, "k_norm_Ainv")
        if k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be positive")
        shape = ell.shape
        ell_flat = ell.reshape(-1)
        wavelength_A = 2.0 * np.pi / k_norm
        hkl = np.column_stack(
            (
                np.full(ell_flat.size, rod.h),
                np.full(ell_flat.size, rod.k),
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
        rod_id = np.full(ell_flat.size, _physical_rod_id(rod), dtype=np.int64)
        # Structure profiles use a virtual sample frame whose normal is the
        # crystal layer normal. This makes the sample-normal query field exact;
        # later mosaic rotations act only on the already evaluated scalar SF.
        query = RodQueryBatch(
            event_id=event_id,
            rod_id=rod_id,
            phase_id=(self.crystal.phase_id,) * ell_flat.size,
            h=np.full(ell_flat.size, rod.h, dtype=np.int32),
            k=np.full(ell_flat.size, rod.k, dtype=np.int32),
            q_sample_normal_Ainv=layer_normal_q,
            l_coordinate=ell_flat,
            wavelength_A=np.full(ell_flat.size, wavelength_A),
        )
        amplitudes = bi2se3_ql_amplitudes(self.crystal, query)
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
                event_id=event_id,
                rod_id=rod_id,
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

    def evaluate(self, *, rod: Rod, L: float, k_norm_Ainv: float) -> float:
        """Implement the scalar painted-Ewald strength protocol."""

        return float(self.evaluate_profile(rod=rod, L=L, k_norm_Ainv=k_norm_Ainv))


__all__ = ["Bi2Se3TwoHStrength"]
