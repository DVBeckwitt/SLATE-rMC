"""General physical reciprocal-lattice calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.materials.crystal import CrystalStructure


@dataclass(frozen=True, slots=True)
class ReciprocalLattice:
    """Direct and physical reciprocal bases, both stored as column vectors."""

    direct_basis_A: NDArray[np.float64]
    basis_Ainv: NDArray[np.float64]
    volume_A3: float

    def __post_init__(self) -> None:
        direct = np.array(self.direct_basis_A, dtype=np.float64, copy=True, order="C")
        reciprocal = np.array(self.basis_Ainv, dtype=np.float64, copy=True, order="C")
        if direct.shape != (3, 3) or not np.all(np.isfinite(direct)):
            raise ValueError("direct basis must be a finite 3 by 3 matrix")
        if reciprocal.shape != (3, 3) or not np.all(np.isfinite(reciprocal)):
            raise ValueError("reciprocal basis must be a finite 3 by 3 matrix")
        determinant = float(np.linalg.det(direct))
        if determinant <= 0.0:
            raise ValueError("direct basis must be nonsingular and right-handed")
        if not np.isfinite(self.volume_A3) or not np.isclose(
            self.volume_A3, determinant, rtol=2e-12, atol=1e-12
        ):
            raise ValueError("volume must agree with the direct basis")
        if not np.allclose(direct.T @ reciprocal, 2.0 * np.pi * np.eye(3), rtol=2e-12, atol=1e-12):
            raise ValueError("reciprocal basis columns must be dual to the direct basis")
        direct.setflags(write=False)
        reciprocal.setflags(write=False)
        object.__setattr__(self, "direct_basis_A", direct)
        object.__setattr__(self, "basis_Ainv", reciprocal)
        object.__setattr__(self, "volume_A3", determinant)

    @classmethod
    def from_direct_basis(cls, direct_basis_A: ArrayLike) -> ReciprocalLattice:
        direct = np.array(direct_basis_A, dtype=np.float64, copy=True, order="C")
        if direct.shape != (3, 3) or not np.all(np.isfinite(direct)):
            raise ValueError("direct_basis_A must be a finite 3 by 3 matrix")
        determinant = float(np.linalg.det(direct))
        if determinant <= 0.0:
            raise ValueError("direct_basis_A must be nonsingular and right-handed")
        reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
        return cls(direct_basis_A=direct, basis_Ainv=reciprocal, volume_A3=determinant)

    @classmethod
    def from_crystal(cls, crystal: CrystalStructure) -> ReciprocalLattice:
        return cls.from_direct_basis(crystal.direct_basis_A)

    @property
    def inplane_metric_Ainv2(self) -> NDArray[np.float64]:
        rod_axis = self.basis_Ainv[:, 2]
        rod_axis = rod_axis / np.linalg.norm(rod_axis)
        radial_projector = np.eye(3) - np.outer(rod_axis, rod_axis)
        metric = self.basis_Ainv[:, :2].T @ radial_projector @ self.basis_Ainv[:, :2]
        metric.setflags(write=False)
        return metric

    def q_cartesian_Ainv(self, hkl: ArrayLike) -> NDArray[np.float64]:
        """Return crystal-frame ``B @ [h, k, L]`` vectors for scalar or batched indices."""

        indices = np.asarray(hkl, dtype=np.float64)
        if indices.ndim == 0 or indices.shape[-1] != 3 or not np.all(np.isfinite(indices)):
            raise ValueError("hkl must be finite and end with a length-3 axis")
        return np.asarray(indices @ self.basis_Ainv.T, dtype=np.float64)

    def qr_Ainv(self, hk: ArrayLike) -> NDArray[np.float64]:
        indices = np.asarray(hk, dtype=np.float64)
        if indices.ndim == 0 or indices.shape[-1] != 2 or not np.all(np.isfinite(indices)):
            raise ValueError("hk must be finite and end with a length-2 axis")
        inplane_q = indices @ self.basis_Ainv[:, :2].T
        rod_axis = self.basis_Ainv[:, 2]
        rod_axis = rod_axis / np.linalg.norm(rod_axis)
        radial_q = inplane_q - np.sum(inplane_q * rod_axis, axis=-1)[..., None] * rod_axis
        return np.asarray(np.linalg.norm(radial_q, axis=-1), dtype=np.float64)

    def validate_layer_qz(self, l_coordinate: ArrayLike, qz_Ainv: ArrayLike) -> NDArray[np.float64]:
        """Return L after checking the c-axis-normal layered-cell relation to Qz."""

        if not np.allclose(
            self.direct_basis_A[2, :2], 0.0, rtol=0.0, atol=1e-12
        ) or not np.allclose(self.direct_basis_A[:2, 2], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("layer Qz validation requires the crystallographic c axis normal")
        l_value, qz = np.broadcast_arrays(
            np.asarray(l_coordinate, dtype=np.float64), np.asarray(qz_Ainv, dtype=np.float64)
        )
        if not np.all(np.isfinite(l_value)) or not np.all(np.isfinite(qz)):
            raise ValueError("L and Qz must be finite")
        expected_qz = l_value * self.basis_Ainv[2, 2]
        if not np.allclose(qz, expected_qz, rtol=2e-12, atol=1e-12):
            raise ValueError("L and Qz are inconsistent for this layered cell")
        return np.asarray(l_value, dtype=np.float64)
