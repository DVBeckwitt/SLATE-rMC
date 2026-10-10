"""One XrayDB authority for atomic factors and material optics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import xraydb
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import MaterialOptics
from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
from rasim_next.materials.crystal import CrystalStructure

HC_EV_A = 12398.419843320026
AVOGADRO_PER_MOL = 6.02214076e23


@dataclass(slots=True)
class OpticalFactorCache:
    """Explicit Q=0 factor reuse; occupied counts and volume remain candidate data.

    The budget includes retained factor arrays and wavelength key bytes, with a
    separate entry bound for Python/database metadata. No optical state is cached.
    """

    maximum_bytes: int = 1024**2
    maximum_entries: int = 256
    retained_bytes: int = field(default=0, init=False)
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    _entries: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self):
        if any(
            type(value) is not int or value < 0
            for value in (self.maximum_bytes, self.maximum_entries)
        ):
            raise ValueError("optical factor cache limits must be nonnegative integers")

    def _forward(self, species, element, charge, wavelength):
        """Consume material_optics' validated, unique float64 wavelength vector."""
        database = xraydb.get_xraydb()
        key = (
            database,
            xraydb.__version__,
            database.get_version(),
            species,
            element,
            charge,
            wavelength.shape,
            wavelength.tobytes(),
        )
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
            return self._entries[key][:2]
        self.misses += 1
        factor, mapping = atomic_scattering_factor_e(
            species=species,
            element=element,
            charge=charge,
            q_magnitude_Ainv=np.zeros(wavelength.size),
            wavelength_A=wavelength,
        )
        size = factor.nbytes + wavelength.nbytes
        if self.maximum_entries and size <= self.maximum_bytes:
            while self._entries and (
                self.retained_bytes + size > self.maximum_bytes
                or len(self._entries) >= self.maximum_entries
            ):
                self.retained_bytes -= self._entries.popitem(last=False)[1][2]
            factor.setflags(write=False)
            self._entries[key] = factor, mapping, size
            self.retained_bytes += size
        return factor, mapping

    def clear(self):
        self._entries.clear()
        self.retained_bytes = 0


def _f0_species(element: str, charge: int) -> str:
    if charge == 0:
        requested = element
    else:
        sign = "+" if charge > 0 else "-"
        requested = f"{element}{abs(charge)}{sign}"
    supported = xraydb.f0_ions(element)
    return requested if requested in supported else element


def atomic_scattering_factor_e(
    *,
    species: str,
    element: str,
    charge: int,
    q_magnitude_Ainv: ArrayLike,
    wavelength_A: ArrayLike,
) -> tuple[NDArray[np.complex128], str]:
    """Return ``f0 + f' + i*f''`` using XrayDB's documented momentum convention."""

    q_magnitude, wavelength = np.broadcast_arrays(
        np.asarray(q_magnitude_Ainv, dtype=np.float64),
        np.asarray(wavelength_A, dtype=np.float64),
    )
    if (
        not np.all(np.isfinite(q_magnitude))
        or np.any(q_magnitude < 0.0)
        or not np.all(np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("Q magnitude and wavelength must be finite; wavelength must be positive")
    selected_species = _f0_species(element, charge)
    q_xraydb = q_magnitude / (4.0 * np.pi)
    energy_eV = np.array(HC_EV_A / wavelength, dtype=np.float64, copy=True)
    shape = q_magnitude.shape
    if q_magnitude.size == 0:
        return np.empty(shape, dtype=np.complex128), f"{species}->{selected_species}"
    chantler_energy_eV = np.asarray(xraydb.chantler_energies(element), dtype=np.float64)
    if np.any(energy_eV < chantler_energy_eV[0]) or np.any(energy_eV > chantler_energy_eV[-1]):
        raise ValueError(
            f"wavelength requires energy outside the tabulated Chantler energy range for {element}"
        )
    f0 = np.asarray(xraydb.f0(selected_species, q_xraydb.ravel()), dtype=np.float64).reshape(shape)
    unique_energy_eV, inverse = np.unique(energy_eV.ravel(), return_inverse=True)
    interval = np.searchsorted(chantler_energy_eV, unique_energy_eV, side="right") - 1
    f1_unique = np.empty_like(unique_energy_eV)
    f2_unique = np.empty_like(unique_energy_eV)
    for interval_index in np.unique(interval):
        selected = interval == interval_index
        selected_energy_eV = unique_energy_eV[selected]
        f1_unique[selected] = np.asarray(
            xraydb.f1_chantler(element, selected_energy_eV), dtype=np.float64
        ).reshape(-1)
        f2_unique[selected] = np.asarray(
            xraydb.f2_chantler(element, selected_energy_eV), dtype=np.float64
        ).reshape(-1)
    f1 = f1_unique[inverse].reshape(shape)
    f2 = f2_unique[inverse].reshape(shape)
    factor = np.asarray(f0 + f1 + 1.0j * f2, dtype=np.complex128)
    return factor, f"{species}->{selected_species}"


def mass_density_g_cm3(crystal: CrystalStructure) -> float:
    """Return occupied unit-cell mass divided by its physical volume."""

    molar_mass_g_mol = sum(
        site.occupancy * float(xraydb.atomic_mass(site.element)) for site in crystal.sites
    )
    density = molar_mass_g_mol / (AVOGADRO_PER_MOL * crystal.volume_A3 * 1e-24)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("occupied structure must have finite positive mass density")
    return float(density)


def material_optics(
    crystal: CrystalStructure,
    wavelength_A: ArrayLike,
    *,
    factor_cache: OpticalFactorCache | None = None,
) -> MaterialOptics:
    """Derive wavelength-resolved optical constants from the occupied expanded structure."""

    if factor_cache is not None and not isinstance(factor_cache, OpticalFactorCache):
        raise TypeError("factor_cache must be an OpticalFactorCache")
    wavelength = np.asarray(wavelength_A, dtype=np.float64)
    if wavelength.ndim != 1 or not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be a finite positive one-dimensional array")
    wavelength = np.unique(wavelength)
    forward_factor_e = np.zeros(wavelength.size, dtype=np.complex128)
    mappings: list[str] = []
    groups = sorted({(site.species, site.element, site.charge) for site in crystal.sites})
    for species, element, charge in groups:
        occupied_count = sum(
            site.occupancy
            for site in crystal.sites
            if (site.species, site.element, site.charge) == (species, element, charge)
        )
        if factor_cache is None:
            factor, mapping = atomic_scattering_factor_e(
                species=species,
                element=element,
                charge=charge,
                q_magnitude_Ainv=np.zeros(wavelength.size),
                wavelength_A=wavelength,
            )
        else:
            factor, mapping = factor_cache._forward(species, element, charge, wavelength)
        forward_factor_e += occupied_count * factor
        mappings.append(mapping)

    prefactor = CLASSICAL_ELECTRON_RADIUS_A * wavelength**2 / (2.0 * np.pi * crystal.volume_A3)
    delta = prefactor * forward_factor_e.real
    beta = prefactor * forward_factor_e.imag
    if np.any(beta < 0.0):
        raise ValueError("XrayDB forward factors produced negative absorption")
    density = mass_density_g_cm3(crystal)
    return MaterialOptics(
        material_id=crystal.phase_id,
        wavelength_A=wavelength,
        n_complex=1.0 - delta + 1.0j * beta,
        provenance=(
            f"Gemmi-expanded occupied structure; XrayDB {xraydb.__version__}; "
            f"database={xraydb.get_xraydb().get_version().split(',')[0].removeprefix('XrayDB Version: ')}; "
            f"density_g_cm3={density:.16g}; species={','.join(mappings)}"
        ),
    )
