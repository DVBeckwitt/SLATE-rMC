"""RASIM-facing adapter to the authoritative standalone mosaic probability."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald.mosaic import (
    _build_mosaic_space,
    wrapped_mosaic_line_density_rad_inv,
)
from painted_ewald.types import MosaicParameters as WrappedMosaicParameters
from painted_ewald.validation import readonly_float_array, reciprocal_basis

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class MosaicOrientationBatch:
    """Integrated axisymmetric orientation masses under the existing event contract."""

    orientation_id: IntArray
    alpha_rad: FloatArray
    azimuth_rad: FloatArray
    rotation_crystal: FloatArray
    probability_mass: FloatArray
    reciprocal_basis_Ainv: FloatArray
    model_id: str

    def __post_init__(self) -> None:
        supplied_ids = np.asarray(self.orientation_id)
        if supplied_ids.dtype.kind not in "iu":
            raise ValueError("orientation_id must contain integers")
        ids = np.array(supplied_ids, dtype=np.int64, copy=True, order="C")
        if ids.ndim != 1 or np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("orientation_id must be nonnegative and unique")
        ids.setflags(write=False)
        size = ids.size
        alpha = readonly_float_array(self.alpha_rad, (size,), "alpha_rad")
        azimuth = readonly_float_array(self.azimuth_rad, (size,), "azimuth_rad")
        rotations = readonly_float_array(self.rotation_crystal, (size, 3, 3), "rotation_crystal")
        mass = readonly_float_array(self.probability_mass, (size,), "probability_mass")
        basis = reciprocal_basis(self.reciprocal_basis_Ainv)
        if np.any(mass < 0.0) or not np.isclose(mass.sum(), 1.0, rtol=0.0, atol=1.0e-10):
            raise ValueError("probability_mass must be nonnegative and sum to one")
        if np.any((alpha < 0.0) | (alpha > np.pi)):
            raise ValueError("alpha_rad must lie in [0, pi]")
        if np.any((azimuth < 0.0) | (azimuth >= 2.0 * np.pi)):
            raise ValueError("azimuth_rad must lie in [0, 2*pi)")
        if not np.allclose(
            rotations @ np.swapaxes(rotations, 1, 2),
            np.eye(3),
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("rotation_crystal must contain proper rotations")
        if self.model_id != "manuscript_axisymmetric_v1":
            raise ValueError("unsupported mosaic orientation model")
        object.__setattr__(self, "orientation_id", ids)
        object.__setattr__(self, "alpha_rad", alpha)
        object.__setattr__(self, "azimuth_rad", azimuth)
        object.__setattr__(self, "rotation_crystal", rotations)
        object.__setattr__(self, "probability_mass", mass)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)


def manuscript_axisymmetric_v1_orientation_quadrature(
    parameters: WrappedMosaicParameters,
    *,
    reciprocal_basis_Ainv: ArrayLike,
    alpha_cell_count: int,
    azimuth_cell_count: int,
    azimuth_phase_rad: float = np.pi,
) -> MosaicOrientationBatch:
    """Preserve the accepted nested-azimuth event quadrature over shared physics."""

    configured = replace(
        parameters,
        alpha_panel_count=alpha_cell_count,
        alpha_gauss_order=16,
        azimuth_count=azimuth_cell_count,
        azimuth_phase_rad=azimuth_phase_rad,
    )
    space = _build_mosaic_space(
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=np.eye(3),
        parameters=configured,
        azimuth_rule="nested",
    )
    return MosaicOrientationBatch(
        orientation_id=space.orientation_id,
        alpha_rad=space.alpha_rad,
        azimuth_rad=space.beta_rad,
        rotation_crystal=space.rotation_crystal,
        probability_mass=space.probability_mass,
        reciprocal_basis_Ainv=space.reciprocal_basis_Ainv,
        model_id="manuscript_axisymmetric_v1",
    )


__all__ = [
    "MosaicOrientationBatch",
    "WrappedMosaicParameters",
    "manuscript_axisymmetric_v1_orientation_quadrature",
    "wrapped_mosaic_line_density_rad_inv",
]
