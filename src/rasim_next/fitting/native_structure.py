"""Common lattice, optics and physical rod-roster binding for native fits."""

from dataclasses import replace

import numpy as np

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.fixed_lattice import hexagonal_direct_basis
from rasim_next.materials.crystal import crystal_with_direct_basis
from rasim_next.materials.optics import atomic_scattering_factor_e, material_optics
from rasim_next.pipeline.bragg_space import IncoherentStructureMixture
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity.specular import compile_parratt_stitch


def native_stitch_records(physics, arguments, stack):
    """Trace the same empirical handoff used by the native finite-strength table."""
    if stack is None:
        return []
    strength = physics.structure.strength(
        **{k: v for k, v in arguments.items() if k != "film_thickness_A"}
    )
    components = (
        strength.components if isinstance(strength, IncoherentStructureMixture) else (strength,)
    )
    records = []
    for surface, component in enumerate(components):
        for wave, index in zip(
            physics.material.wavelength_A, physics.material.n_complex, strict=True
        ):
            state = compile_parratt_stitch(
                stack,
                lambda ell, component=component, wave=wave: component.evaluate_hkl(
                    h=0, k=0, L=ell, k_norm_Ainv=2 * np.pi / wave
                ),
                wavelength_A=wave,
                film_refractive_index=index,
                film_thickness_A=arguments["film_thickness_A"],
                c_A=2 * np.pi / np.linalg.norm(physics.reciprocal_basis_Ainv[:, 2]),
                grid_size=physics.integration_rule.stitch_grid_size,
            )
            records.append(
                dict(
                    surface=surface,
                    wavelength_A=float(wave),
                    selection=state.blend_selection,
                    bounds_q_over_qc=state.blend_bounds_q_over_qc,
                    scale=state.dimensionless_scale_factor,
                    zero_strength_A2=state.zero_strength_A2,
                    qc_Ainv=state.qc_Ainv,
                    grid_size=physics.integration_rule.stitch_grid_size,
                    overlap_measure=state.overlap_measure,
                )
            )
    return records


def rebind_native_structure(reference, recipe, revision, *, optical_factor_cache=None):
    """Recompute optics and reciprocal geometry from the occupied candidate cell."""
    reciprocal = ReciprocalLattice.from_crystal(recipe.crystals[0]).basis_Ainv
    material = material_optics(
        recipe.crystals[0],
        reference.source.mean_rays.wavelength_A,
        factor_cache=optical_factor_cache,
    )
    rod_revision = reference.rod_catalog_revision
    if not np.array_equal(reciprocal, reference.reciprocal_basis_Ainv):
        rod_revision = canonical_revision_sha256(
            ("definition_id", "fixed_physical_rod_roster_rebound_basis.v1"),
            ("reference_catalog", rod_revision),
            ("reciprocal_basis_Ainv", reciprocal),
        )
    return replace(
        reference,
        structure=recipe,
        material=material,
        reciprocal_basis_Ainv=reciprocal,
        rod_catalog_revision=rod_revision,
        input_revision=revision,
    )


def validate_native_rod_coverage(reference, *, a_bounds_A, c_bounds_A):
    """Bound the phase sphere over a hexagonal cell box and occupancies in [0,1].

    Positive forward factors and minimum volume bound beta. For 0<Re(n)<=1,
    Kphase squared <= k0 squared * (1+beta_max). The smallest reciprocal metric
    eigenvalue bounds an exhaustive integer search for every elastic rod.
    The source wavelengths are fixed; recheck this bound after changing them.
    """
    bounds = np.asarray([a_bounds_A, c_bounds_A], dtype=float)
    if (
        bounds.shape != (2, 2)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds <= 0)
        or np.any(bounds[:, 1] < bounds[:, 0])
    ):
        raise ValueError("cell coverage requires ordered positive a/c ranges")
    waves = np.unique(reference.source.mean_rays.wavelength_A)
    cell = reference.structure.crystals[0]
    for species, element, charge in {(s.species, s.element, s.charge) for s in cell.sites}:
        factor, _ = atomic_scattering_factor_e(
            species=species,
            element=element,
            charge=charge,
            q_magnitude_Ainv=np.zeros(len(waves)),
            wavelength_A=waves,
        )
        if np.any(factor.real <= 0) or np.any(factor.imag < 0):
            raise ValueError("rod coverage optical bound requires positive forward factors")
    basis = hexagonal_direct_basis(
        cell.direct_basis_A,
        np.log(bounds[:, 0] / np.linalg.norm(cell.direct_basis_A[:, [0, 2]], axis=0)),
    )
    dense = crystal_with_direct_basis(
        replace(cell, sites=tuple(replace(site, occupancy=1.0) for site in cell.sites)),
        basis,
        provenance="rod coverage optical bound",
    )
    optics = material_optics(dense, waves)
    if np.any(optics.n_complex.real <= 0):
        raise ValueError("cell box exceeds the passive phase-sphere coverage bound")
    beta = float(np.max(optics.n_complex.imag))
    maximum_q = float(4 * np.pi / np.min(waves) * np.sqrt(1 + beta))
    basis = hexagonal_direct_basis(
        cell.direct_basis_A,
        np.log(bounds[:, 1] / np.linalg.norm(cell.direct_basis_A[:, [0, 2]], axis=0)),
    )
    crystal = crystal_with_direct_basis(cell, basis, provenance="rod coverage cell bound")
    lattice = ReciprocalLattice.from_crystal(crystal)
    minimum_eigenvalue = float(np.linalg.eigvalsh(lattice.inplane_metric_Ainv2)[0])
    limit = int(np.ceil(maximum_q / np.sqrt(minimum_eigenvalue)))
    catalog = build_rod_catalog(crystal, h_bounds=(-limit, limit), k_bounds=(-limit, limit))
    configured = {(rod.h, rod.k) for rod in reference.rods}
    required = {
        (int(h), int(k))
        for h, k, q in zip(catalog.h, catalog.k, catalog.qr_Ainv, strict=True)
        if q <= maximum_q * (1 + 1e-12)
    }
    if missing := required - configured:
        raise ValueError(f"fixed rod roster misses potentially elastic rods: {sorted(missing)}")
    absent_q = [
        float(q)
        for h, k, q in zip(catalog.h, catalog.k, catalog.qr_Ainv, strict=True)
        if (int(h), int(k)) not in configured
    ]
    first_excluded = min([np.sqrt(minimum_eigenvalue) * (limit + 1), *absent_q])
    return dict(
        maximum_elastic_q_Ainv=maximum_q,
        first_excluded_radial_Ainv=first_excluded,
        exclusion_margin_Ainv=first_excluded - maximum_q,
        required_rod_count=len(required),
        enumerated_index_limit=limit,
    )
