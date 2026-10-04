"""Positive physical axial quadrature for finite periodic CIF stacks.

These rules integrate W(u) du, where W is the canonical sum of signed strengths.
They are strength dependent and cannot be reused as strength-free geometry.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.special import roots_legendre

from painted_ewald.validation import reject_complex
from rasim_next.pipeline.bragg_space import (
    Bi2X3FiniteStackStrength,
    CifFiniteStackStrength,
    IncoherentStructureMixture,
    Pbi2FiniteSurfaceStrength,
)


@dataclass(frozen=True, slots=True)
class PositiveAxialRule:
    """Physical W du weights and canonical signed fractions at interior nodes."""

    nodes_Ainv: NDArray[np.float64]
    strength_weighted_mass: NDArray[np.float64]
    signed_fractions: NDArray[np.float64]

    def __post_init__(self):
        for name in ("nodes_Ainv", "strength_weighted_mass", "signed_fractions"):
            reject_complex(getattr(self, name), name)
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError("positive axial rule must be finite and nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        n = len(self.nodes_Ainv)
        if (
            n < 1
            or self.nodes_Ainv.shape != (n,)
            or self.strength_weighted_mass.shape != (n,)
            or self.signed_fractions.shape != (2, n)
            or np.any(self.strength_weighted_mass <= 0)
            or np.any(np.diff(self.nodes_Ainv) <= 0)
            or not np.allclose(self.signed_fractions.sum(axis=0), 1, rtol=0, atol=5e-15)
        ):
            raise ValueError(
                "positive axial nodes, physical masses and signed fractions must align"
            )


def finite_stack_resolution(model):
    """Conservative actual phase-depth extent, including unwrapped CIF site lifts.

    Pb partial endpoints span N translations, while CIF stacks span N-1.
    Quintuple layers use their existing c/3 spacing and retained orbit offsets.
    Registry/disorder probabilities affect canonical strengths, not depth span.
    """
    models = model.components if isinstance(model, IncoherentStructureMixture) else (model,)
    divisions, extents = [], []
    for component in models:
        basis = component.reciprocal_basis_Ainv
        normal = basis[:, 2] / np.linalg.norm(basis[:, 2])
        d = 2 * np.pi / np.linalg.norm(basis[:, 2])
        if isinstance(component, CifFiniteStackStrength):
            height = (
                np.array(
                    [
                        component.crystal.direct_basis_A @ np.array(site.fractional)
                        for site in component.crystal.sites
                    ]
                )
                @ normal
            )
            extents.append(max(d, (component.repeats - 1) * d + np.ptp(height)))
            divisions.append(component.repeats)
        elif isinstance(component, Pbi2FiniteSurfaceStrength):
            heights = np.concatenate(
                [
                    np.array(
                        [motif.direct_basis_A @ np.array(site.fractional) for site in motif.sites]
                    )
                    @ normal
                    for orientation in component._motifs
                    for motif in orientation
                ]
            )
            extents.append(max(d, component.repeats * d + np.ptp(heights)))
            divisions.append(component.repeats + 1)
        elif isinstance(component, Bi2X3FiniteStackStrength):
            p = component.structure_parameters
            span = 2 * d * max(p.bi_fractional_z - 1 / 3, 1 / 3 - p.se2_fractional_z)
            extents.append(max(d, (component.layers - 1) * d / 3 + span))
            divisions.append(component.layers)
        else:
            raise ValueError(
                "strength Gaussian preparation needs a supported finite phase-depth model"
            )
    return tuple(sorted(set(divisions))), max(extents)


def identically_zero_strength(model):
    """Only analytic zero populations/occupancies; sampled zeros remain failures."""
    models = model.components if isinstance(model, IncoherentStructureMixture) else (model,)
    probabilities = model.probabilities if isinstance(model, IncoherentStructureMixture) else (1.0,)
    for component, probability in zip(models, probabilities, strict=True):
        if probability == 0:
            continue
        if isinstance(component, (CifFiniteStackStrength, Pbi2FiniteSurfaceStrength)):
            if all(site.occupancy == 0 for site in component.crystal.sites):
                continue
        elif isinstance(component, Bi2X3FiniteStackStrength):
            p = component.structure_parameters
            if p.bi_occupancy == p.se1_occupancy == p.se2_occupancy == 0:
                continue
        return False
    return True


def positive_gauss_rule(scaled_nodes, physical_masses, order):
    """Reorthogonalized Stieltjes/Lanczos rule with physical mass restoration.

    The input masses already contain physical du. Scaling the coordinate to
    [-1,1] adds no Jacobian. Polynomial checks qualify the discrete measure only.
    """
    reject_complex(scaled_nodes, "scaled nodes")
    reject_complex(physical_masses, "physical masses")
    x, mass = np.asarray(scaled_nodes, dtype=float), np.asarray(physical_masses, dtype=float)
    if (
        x.ndim != 1
        or mass.shape != x.shape
        or np.any(~np.isfinite(x))
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0)
        or np.any(abs(x) > 1)
        or type(order) is not int
        or order < 2
        or np.count_nonzero(mass) < order
    ):
        raise ValueError("finite positive discrete measure and adequate quadrature order required")
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("unresolved or zero strength measure; no panel may be silently removed")
    vector = np.sqrt(mass / total)
    previous = np.zeros_like(vector)
    previous_beta = 0.0
    basis, alpha, beta = [], [], []
    for degree in range(order):
        basis.append(vector.copy())
        diagonal = float(vector @ (x * vector))
        alpha.append(diagonal)
        if degree == order - 1:
            break
        residual = x * vector - diagonal * vector - previous_beta * previous
        vectors = np.column_stack(basis)
        for _ in range(2):
            residual -= vectors @ (vectors.T @ residual)
        off_diagonal = float(np.linalg.norm(residual))
        if not np.isfinite(off_diagonal) or off_diagonal <= 64 * np.finfo(float).eps:
            raise ValueError("strength recurrence lost rank; refine its scalar measure")
        beta.append(off_diagonal)
        previous, vector = vector, residual / off_diagonal
        previous_beta = off_diagonal
    nodes, eigenvectors = eigh_tridiagonal(np.asarray(alpha), np.asarray(beta))
    weights = total * eigenvectors[0] ** 2
    error = (
        np.max(abs(weights @ legvander(nodes, 2 * order - 1) - mass @ legvander(x, 2 * order - 1)))
        / total
    )
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
        or np.any(abs(nodes) >= 1)
        or error > 5e-13
        or abs(weights.sum() / total - 1) > 5e-14
    ):
        raise ValueError("positive Gaussian rule failed discrete moment or mass qualification")
    return nodes, weights


def rod_phase_knots(lower, upper, basis, rods, divisions):
    """Actual signed rod offsets; divisions=1 seeds integer full-cell Bragg phases."""
    b3 = np.linalg.norm(basis[:, 2])
    normal = basis[:, 2] / b3
    values = []
    for rod in rods:
        offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ normal
        for sign in (1, -1):
            limits = np.sort((sign * np.array([lower, upper]) - offset) / b3 * divisions)
            phase = np.arange(np.floor(limits[0]), np.ceil(limits[1]) + 1) / divisions
            values.extend(sign * (offset + b3 * phase))
    return np.asarray(values)


def response_panel_edges(lower, upper, basis, rods, radius, ki, normal, bounds):
    """Complete response panels seeded by full-cell phases, visibility and folds.

    Fold landmarks are seeds, never mosaic-tail cutoffs. Every panel is retained.
    A half-cell cap keeps response and cheap fringe meshes distinct.
    """
    reject_complex([lower, upper], "response bounds")
    if not np.all(np.isfinite([lower, upper])) or not 0 <= lower < upper:
        raise ValueError("response support must have finite positive width")
    b3 = np.linalg.norm(basis[:, 2])
    parts = [rod_phase_knots(lower, upper, basis, rods, 2)]
    endpoints = np.asarray(bounds)[:, :2].ravel()
    parts.append(np.sqrt(np.maximum(0, endpoints**2 - radius**2)))
    a = float(ki @ normal)
    p = np.linalg.norm(ki - a * normal)
    roots = []
    for sign in (1, -1):
        for side in (1, -1):
            discriminant = a * a - radius * radius + side * 2 * radius * p
            if discriminant >= 0:
                roots.extend((-sign * a + np.sqrt(discriminant), -sign * a - np.sqrt(discriminant)))
    cutoff_squared = 4 * float(ki @ ki) - radius**2
    if cutoff_squared >= 0:
        roots.append(np.sqrt(cutoff_squared))
    parts.append(np.asarray(roots))
    seeds = np.unique(np.concatenate(parts))
    tolerance = 128 * np.finfo(float).eps * max(1.0, upper)
    interior = seeds[(seeds > lower + tolerance) & (seeds < upper - tolerance)]
    if len(interior):
        interior = interior[np.r_[True, np.diff(interior) > tolerance]]
    edges = np.r_[lower, interior, upper]
    # Subdivide actual landmarks, rather than overlaying an unrelated uniform grid.
    pieces = [
        np.linspace(a, b, max(1, int(np.ceil((b - a) / (b3 / 2)))) + 1)[:-1]
        for a, b in pairwise(edges)
    ]
    edges = np.r_[np.concatenate(pieces), upper]
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("physical response panels are unresolved")
    return edges


def prepare_positive_axial_rule(
    lower, upper, *, basis, rods, model, strength_at, order, scalar_order, scalar_phase_step_rad
):
    """Resolve cheap canonical fringes separately from expensive response nodes."""
    reject_complex([lower, upper, scalar_phase_step_rad], "scalar controls")
    if (
        not np.all(np.isfinite([lower, upper, scalar_phase_step_rad]))
        or not 0 <= lower < upper
        or scalar_phase_step_rad <= 0
        or type(scalar_order) is not int
        or scalar_order < max(8, order)
    ):
        raise ValueError("finite positive scalar support, phase step and adequate order required")
    repeats, extent = finite_stack_resolution(model)
    cap = scalar_phase_step_rad / extent
    parts = [np.array([lower, upper]), np.arange(lower, upper, cap)]
    parts.extend(rod_phase_knots(lower, upper, basis, rods, n) for n in repeats)
    edges = np.unique(np.concatenate(parts))
    edges = edges[(edges >= lower) & (edges <= upper)]
    x, w = roots_legendre(scalar_order)
    half = np.diff(edges) / 2
    cheap = (edges[:-1, None] + half[:, None] * (x + 1)).ravel()
    du = (half[:, None] * w).ravel()
    strength = strength_at(cheap)
    mid, scale = (lower + upper) / 2, (upper - lower) / 2
    generated, masses = positive_gauss_rule((cheap - mid) / scale, du * strength.sum(axis=0), order)
    axial = mid + scale * generated
    signed = strength_at(axial)
    common = signed.sum(axis=0)
    if np.any(~np.isfinite(common)) or np.any(common <= 0):
        raise ValueError("zero or invalid canonical strength at generated response node")
    return PositiveAxialRule(axial, masses, signed / common)
