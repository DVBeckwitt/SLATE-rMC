"""Deterministic reciprocal-rod catalog helpers."""

from __future__ import annotations

from math import ceil, sqrt

import numpy as np
from numpy.typing import ArrayLike

from painted_ewald.ewald import _stable_line_components
from painted_ewald.types import Rod
from painted_ewald.validation import finite_scalar, reciprocal_basis


def enumerate_rods_within_ewald_sphere(
    *,
    reciprocal_basis_Ainv: ArrayLike,
    k_norm_Ainv: float,
    population: float = 1.0,
) -> tuple[Rod, ...]:
    """Return every integer rod line whose origin distance is at most ``2K``.

    Rigid mosaic rotations preserve a rod line's distance from reciprocal
    origin. An elastic q-space Ewald sphere centered ``K`` from the origin and
    having radius ``K`` can therefore meet that line for some orientation only
    when its in-plane distance is no greater than ``2K``.
    """

    basis = reciprocal_basis(reciprocal_basis_Ainv)
    k_norm = finite_scalar(k_norm_Ainv, "k_norm_Ainv")
    if k_norm <= 0.0:
        raise ValueError("k_norm_Ainv must be positive")
    rod_population = Rod(0, 0, population).population

    mean_axis = basis[:, 2] / np.linalg.norm(basis[:, 2])
    projected_basis = np.column_stack(
        (
            _stable_line_components(basis[:, 0], mean_axis)[1],
            _stable_line_components(basis[:, 1], mean_axis)[1],
        )
    )
    in_plane_metric = projected_basis.T @ projected_basis
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(in_plane_metric)))
    if minimum_eigenvalue <= 0.0:
        raise ValueError("reciprocal in-plane metric must be positive definite")

    maximum_distance = 2.0 * k_norm
    tolerance = 256.0 * np.finfo(np.float64).eps * max(maximum_distance, 1.0)
    index_bound = ceil((maximum_distance + tolerance) / sqrt(minimum_eigenvalue))
    rods: list[Rod] = []
    for h in range(-index_bound, index_bound + 1):
        for k in range(-index_bound, index_bound + 1):
            line_offset = projected_basis @ np.array([h, k], dtype=np.float64)
            if np.linalg.norm(line_offset) <= maximum_distance + tolerance:
                rods.append(Rod(h, k, rod_population))
    return tuple(sorted(rods, key=lambda rod: (rod.family_m, rod.h, rod.k)))


__all__ = ["enumerate_rods_within_ewald_sphere"]
