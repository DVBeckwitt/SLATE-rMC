"""Complete individual-rod catalogs with exact family metadata."""

from __future__ import annotations

import gemmi
import numpy as np

from rasim_next.core.contracts import RodCatalog
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.reciprocal.lattice import ReciprocalLattice


def _integer_bounds(bounds: tuple[int, int], name: str) -> tuple[int, int]:
    if (
        len(bounds) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in bounds
        )
        or bounds[0] > bounds[1]
    ):
        raise ValueError(f"{name} must contain two ordered integers")
    return int(bounds[0]), int(bounds[1])


def _is_hexagonal(direct_basis_A: np.ndarray) -> bool:
    a_vector, b_vector, c_vector = direct_basis_A.T
    a_length = np.linalg.norm(a_vector)
    return bool(
        np.isclose(np.linalg.norm(b_vector), a_length, rtol=1e-8, atol=1e-10)
        and np.isclose(np.dot(a_vector, b_vector), -0.5 * a_length**2, rtol=1e-8, atol=1e-10)
        and np.isclose(np.dot(a_vector, c_vector), 0.0, rtol=0.0, atol=1e-10)
        and np.isclose(np.dot(b_vector, c_vector), 0.0, rtol=0.0, atol=1e-10)
    )


def _inplane_orbit(spacegroup_hm: str, h_value: int, k_value: int) -> tuple[tuple[int, int], ...]:
    spacegroup = gemmi.find_spacegroup_by_name(spacegroup_hm)
    if spacegroup is None:
        raise ValueError(f"cannot resolve space group {spacegroup_hm!r}")
    orbit: set[tuple[int, int]] = set()
    for operation in spacegroup.operations():
        transformed_axis = operation.apply_to_hkl([0, 0, 1])
        transformed = operation.apply_to_hkl([h_value, k_value, 0])
        if transformed_axis[0] == transformed_axis[1] == 0 and transformed[2] == 0:
            orbit.add((int(transformed[0]), int(transformed[1])))
    if not orbit:
        raise ValueError("space group has no operation preserving the crystallographic rod axis")
    return tuple(sorted(orbit))


def _reciprocal_cell_key(basis_Ainv: np.ndarray) -> str:
    return ",".join(float(value).hex() for value in basis_Ainv.ravel())


def build_rod_catalog(
    crystal: CrystalStructure,
    *,
    h_bounds: tuple[int, int],
    k_bounds: tuple[int, int],
) -> RodCatalog:
    """Enumerate all integer rods in inclusive bounds without pruning or family collapse."""

    h_min, h_max = _integer_bounds(h_bounds, "h_bounds")
    k_min, k_max = _integer_bounds(k_bounds, "k_bounds")
    hk = np.asarray(
        [
            (h_value, k_value)
            for h_value in range(h_min, h_max + 1)
            for k_value in range(k_min, k_max + 1)
        ],
        dtype=np.int32,
    )
    lattice = ReciprocalLattice.from_crystal(crystal)
    hexagonal = _is_hexagonal(crystal.direct_basis_A)
    family_keys: list[str] = []
    family_ids: list[str] = []
    metadata: list[str] = []
    for h_value, k_value in hk.tolist():
        orbit = _inplane_orbit(crystal.spacegroup_hm, h_value, k_value)
        orbit_text = ";".join(f"{h},{k}" for h, k in orbit)
        if hexagonal:
            family_number = h_value * h_value + h_value * k_value + k_value * k_value
            key = f"hex:m={family_number}"
        else:
            key = f"cell:{_reciprocal_cell_key(lattice.basis_Ainv)}|orbit:{orbit_text}"
        family_keys.append(key)
        family_ids.append(f"{crystal.phase_id}:{key}")
        metadata.append(
            f"crystal-frame; Qr-metadata-only; multiplicity={len(orbit)}; orbit={orbit_text}"
        )
    size = hk.shape[0]
    return RodCatalog(
        rod_id=np.arange(size, dtype=np.int64),
        phase_id=(crystal.phase_id,) * size,
        h=hk[:, 0],
        k=hk[:, 1],
        family_id=tuple(family_ids),
        family_key=tuple(family_keys),
        qr_Ainv=lattice.qr_Ainv(hk),
        reciprocal_basis_Ainv=lattice.basis_Ainv,
        symmetry_metadata=tuple(metadata),
    )
