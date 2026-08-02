"""Raw complex crystallographic structure amplitudes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xraydb
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import EventIntensityResult, RodCatalog, RodQueryBatch
from rasim_next.core.scattering import electron_squared_to_scattering_strength_A2
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.materials.optics import atomic_scattering_factor_e
from rasim_next.reciprocal.lattice import ReciprocalLattice


@dataclass(frozen=True, slots=True)
class StructureAmplitudeResult:
    """Unnormalized complex unit-cell amplitude in electron units."""

    amplitude_e: NDArray[np.complex128]
    provenance: str

    def __post_init__(self) -> None:
        amplitude = np.array(self.amplitude_e, dtype=np.complex128, copy=True, order="C")
        if not np.all(np.isfinite(amplitude)) or not self.provenance:
            raise ValueError("finite amplitude and provenance are required")
        amplitude.setflags(write=False)
        object.__setattr__(self, "amplitude_e", amplitude)


def unit_cell_amplitude(
    crystal: CrystalStructure,
    hkl: ArrayLike,
    wavelength_A: ArrayLike,
    *,
    unknown_u_iso_A2: float | None = None,
    shared_displacement_tensor_A2: ArrayLike | None = None,
    site_displacement_tensors_A2: ArrayLike | None = None,
) -> StructureAmplitudeResult:
    """Evaluate the positive-phase structure sum at arbitrary Miller coordinates."""

    indices = np.asarray(hkl, dtype=np.float64)
    if indices.ndim == 0 or indices.shape[-1] != 3 or not np.all(np.isfinite(indices)):
        raise ValueError("hkl must be finite and end with a length-3 axis")
    leading_shape = indices.shape[:-1]
    try:
        wavelength = np.broadcast_to(np.asarray(wavelength_A, dtype=np.float64), leading_shape)
    except ValueError as error:
        raise ValueError("wavelength_A must broadcast to the hkl batch") from error
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be finite and positive")
    if unknown_u_iso_A2 is not None and (
        not np.isfinite(unknown_u_iso_A2) or unknown_u_iso_A2 < 0.0
    ):
        raise ValueError("unknown_u_iso_A2 must be finite and nonnegative")
    if (
        sum(
            value is not None
            for value in (
                unknown_u_iso_A2,
                shared_displacement_tensor_A2,
                site_displacement_tensors_A2,
            )
        )
        > 1
    ):
        raise ValueError("unknown, shared, and site displacement overrides are mutually exclusive")
    displacement_tensor = None
    if shared_displacement_tensor_A2 is not None:
        displacement_tensor = np.asarray(shared_displacement_tensor_A2, dtype=np.float64)
        if displacement_tensor.shape != (3, 3) or not np.all(np.isfinite(displacement_tensor)):
            raise ValueError("shared_displacement_tensor_A2 must be finite with shape (3, 3)")
        scale = max(float(np.linalg.norm(displacement_tensor, ord=2)), 1.0)
        tolerance = 256.0 * np.finfo(np.float64).eps * scale
        if (
            not np.allclose(
                displacement_tensor,
                displacement_tensor.T,
                rtol=0.0,
                atol=tolerance,
            )
            or np.min(np.linalg.eigvalsh(displacement_tensor)) < -tolerance
        ):
            raise ValueError(
                "shared_displacement_tensor_A2 must be symmetric positive semidefinite"
            )
        displacement_tensor = 0.5 * (displacement_tensor + displacement_tensor.T)
    site_displacement_tensors = None
    if site_displacement_tensors_A2 is not None:
        site_displacement_tensors = np.asarray(site_displacement_tensors_A2, dtype=np.float64)
        expected_shape = (len(crystal.sites), 3, 3)
        if site_displacement_tensors.shape != expected_shape or not np.all(
            np.isfinite(site_displacement_tensors)
        ):
            raise ValueError(
                "site_displacement_tensors_A2 must be finite with shape (site_count, 3, 3)"
            )
        scale = max(
            float(np.linalg.norm(site_displacement_tensors, ord=2, axis=(1, 2)).max()),
            1.0,
        )
        tolerance = 256.0 * np.finfo(np.float64).eps * scale
        if (
            not np.allclose(
                site_displacement_tensors,
                np.swapaxes(site_displacement_tensors, 1, 2),
                rtol=0.0,
                atol=tolerance,
            )
            or np.min(np.linalg.eigvalsh(site_displacement_tensors)) < -tolerance
        ):
            raise ValueError("site_displacement_tensors_A2 must be symmetric positive semidefinite")
        site_displacement_tensors = 0.5 * (
            site_displacement_tensors + np.swapaxes(site_displacement_tensors, 1, 2)
        )
    has_unknown_u_iso = any(site.u_iso_A2 is None for site in crystal.sites)
    if (
        displacement_tensor is None
        and site_displacement_tensors is None
        and unknown_u_iso_A2 is None
        and has_unknown_u_iso
    ):
        raise ValueError("unknown isotropic displacement requires an explicit calculation value")

    lattice = ReciprocalLattice.from_crystal(crystal)
    q_vectors = lattice.q_cartesian_Ainv(indices).reshape(-1, 3)
    q_magnitude = np.linalg.norm(q_vectors, axis=1)
    wavelength_flat = wavelength.reshape(-1)
    fractional = np.asarray([site.fractional for site in crystal.sites], dtype=np.float64)
    positions_A = fractional @ crystal.direct_basis_A.T
    phase = np.exp(1.0j * (q_vectors @ positions_A.T))
    if displacement_tensor is None and site_displacement_tensors is None:
        u_iso = np.asarray(
            [
                unknown_u_iso_A2 if site.u_iso_A2 is None else site.u_iso_A2
                for site in crystal.sites
            ],
            dtype=np.float64,
        )
        damping = np.exp(-0.5 * q_magnitude[:, None] ** 2 * u_iso[None, :])
    elif displacement_tensor is not None:
        isotropic_u_A2 = float(displacement_tensor[0, 0])
        if np.array_equal(displacement_tensor, isotropic_u_A2 * np.eye(3)):
            exponent = q_magnitude**2 * isotropic_u_A2
        else:
            exponent = np.einsum(
                "ni,ij,nj->n",
                q_vectors,
                displacement_tensor,
                q_vectors,
                optimize=True,
            )
        damping = np.exp(-0.5 * np.maximum(exponent, 0.0))[:, None]
    else:
        exponent = np.einsum(
            "ni,sij,nj->ns",
            q_vectors,
            site_displacement_tensors,
            q_vectors,
            optimize=True,
        )
        damping = np.exp(-0.5 * np.maximum(exponent, 0.0))
    occupancy = np.asarray([site.occupancy for site in crystal.sites], dtype=np.float64)
    site_sum = phase * damping * occupancy[None, :]

    amplitude = np.zeros(q_vectors.shape[0], dtype=np.complex128)
    mappings: list[str] = []
    factor_groups = sorted({(site.species, site.element, site.charge) for site in crystal.sites})
    for species, element, charge in factor_groups:
        mask = np.fromiter(
            (
                site.species == species and site.element == element and site.charge == charge
                for site in crystal.sites
            ),
            dtype=np.bool_,
            count=len(crystal.sites),
        )
        factor, mapping = atomic_scattering_factor_e(
            species=species,
            element=element,
            charge=charge,
            q_magnitude_Ainv=q_magnitude,
            wavelength_A=wavelength_flat,
        )
        amplitude += factor * np.sum(site_sum[:, mask], axis=1)
        mappings.append(mapping)

    return StructureAmplitudeResult(
        amplitude_e=amplitude.reshape(leading_shape),
        provenance=(
            f"XrayDB {xraydb.__version__}; "
            f"database={xraydb.get_xraydb().get_version().split(',')[0].removeprefix('XrayDB Version: ')}; "
            "f=f0+f1+i*f2; q=|Q|/(4*pi); "
            f"species={','.join(mappings)}"
            + (
                f"; unknown_u_iso_A2={unknown_u_iso_A2:g}"
                if has_unknown_u_iso
                and displacement_tensor is None
                and site_displacement_tensors is None
                else ""
            )
            + (
                "; site_displacement_tensors_A2=declared"
                if site_displacement_tensors is not None
                else ""
            )
            + (
                "; shared_displacement_tensor_A2=declared"
                if displacement_tensor is not None
                else ""
            )
        ),
    )


def ordered_event_result(
    crystal: CrystalStructure,
    catalog: RodCatalog,
    query: RodQueryBatch,
    *,
    unknown_u_iso_A2: float | None = None,
) -> EventIntensityResult:
    """Return arbitrary-L unit-cell scattering strength after rod-identity validation."""

    lattice = ReciprocalLattice.from_crystal(crystal)
    if not np.allclose(catalog.reciprocal_basis_Ainv, lattice.basis_Ainv, rtol=2e-12, atol=1e-12):
        raise ValueError("catalog reciprocal basis does not match the crystal-frame basis")
    row_by_rod_id = {int(rod_id): index for index, rod_id in enumerate(catalog.rod_id)}
    for row, rod_id in enumerate(query.rod_id):
        catalog_row = row_by_rod_id.get(int(rod_id))
        if catalog_row is None or (
            query.phase_id[row] != crystal.phase_id
            or catalog.phase_id[catalog_row] != crystal.phase_id
            or int(query.h[row]) != int(catalog.h[catalog_row])
            or int(query.k[row]) != int(catalog.k[catalog_row])
        ):
            raise ValueError(f"rod identity mismatch at query row {row}")
    hkl = np.column_stack((query.h, query.k, query.l_coordinate))
    amplitude = unit_cell_amplitude(
        crystal,
        hkl,
        query.wavelength_A,
        unknown_u_iso_A2=unknown_u_iso_A2,
    )
    return EventIntensityResult(
        event_id=query.event_id,
        scattering_strength_A2=electron_squared_to_scattering_strength_A2(
            np.abs(amplitude.amplitude_e) ** 2
        ),
        model_id="ordered",
        model_component_id="raw_unit_cell",
        population_group_id=None,
        normalization="UNIT_CELL",
    )
