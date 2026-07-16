"""Raw complex crystallographic structure amplitudes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xraydb
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import EventIntensityResult, RodCatalog, RodQueryBatch
from rasim_next.materials.crystal import CrystalStructure
from rasim_next.materials.optics import atomic_scattering_factor_e
from rasim_next.reciprocal.lattice import ReciprocalLattice


@dataclass(frozen=True, slots=True)
class StructureAmplitudeResult:
    """Unnormalized complex unit-cell amplitude in electron units."""

    amplitude_e: NDArray[np.complex128]
    provenance: str
    cache_identity: str
    basis_mode: str

    def __post_init__(self) -> None:
        amplitude = np.array(self.amplitude_e, dtype=np.complex128, copy=True, order="C")
        if (
            not np.all(np.isfinite(amplitude))
            or not self.provenance
            or not self.cache_identity
            or self.basis_mode not in {"exact_provider", "bi2se3_whole_cell_compat"}
        ):
            raise ValueError("finite amplitude and complete basis metadata are required")
        amplitude.setflags(write=False)
        object.__setattr__(self, "amplitude_e", amplitude)


@dataclass(frozen=True, slots=True)
class OrderedEventResult:
    """Raw event-aligned amplitude and unscaled electron-squared intensity."""

    event_id: NDArray[np.int64]
    amplitude_e: NDArray[np.complex128]
    intensity: EventIntensityResult
    provenance: str
    cache_identity: str
    basis_mode: str

    def __post_init__(self) -> None:
        event_id = np.array(self.event_id, dtype=np.int64, copy=True, order="C")
        amplitude = np.array(self.amplitude_e, dtype=np.complex128, copy=True, order="C")
        if event_id.ndim != 1 or amplitude.shape != event_id.shape:
            raise ValueError("event_id and amplitude_e must be aligned one-dimensional arrays")
        if (
            not np.array_equal(event_id, self.intensity.event_id)
            or not self.provenance
            or not self.cache_identity
            or self.basis_mode not in {"exact_provider", "bi2se3_whole_cell_compat"}
        ):
            raise ValueError("raw amplitude and intensity must preserve identical event identity")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("amplitude_e must be finite")
        event_id.setflags(write=False)
        amplitude.setflags(write=False)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "amplitude_e", amplitude)


_BI2SE3_COMPAT_CIF_SHA256 = "a25bc39732a01887faadcfc4b1286044ee98edec67ab7cc2964d7953bfa39888"
_BI2SE3_COMPAT_LEGACY_REVISION = "494accdc2655bd677fafaf070b3dad816b65fa3c"
_BI2SE3_COMPAT_RODS = frozenset({(0, 0), (-1, 0), (-2, 0), (-3, 1)})
_BI2SE3_COMPAT_FACTORS = (
    (
        "Bi",
        (33.3689, 0.704),
        (12.951, 2.9238),
        (16.5877, 8.7937),
        (6.4692, 48.0093),
        13.5782,
        -3.6853811239916183,
        -9.2942695857503,
    ),
    (
        "Se",
        (17.0006, 2.4098),
        (5.8196, 0.2726),
        (3.9731, 15.2372),
        (4.3543, 43.8163),
        2.8409,
        -0.7944157549455682,
        -1.1804401764903047,
    ),
)


def _source_sha256(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bi2se3_compat_amplitude(
    crystal: CrystalStructure,
    indices: NDArray[np.float64],
    wavelength: NDArray[np.float64],
    source_sha256: str,
) -> NDArray[np.complex128]:
    if source_sha256 != _BI2SE3_COMPAT_CIF_SHA256:
        raise ValueError("bi2se3 compatibility requires the frozen CIF SHA-256")
    cell_lengths = np.linalg.norm(crystal.direct_basis_A, axis=0)
    if not np.allclose(cell_lengths, (4.143, 4.143, 28.636), rtol=0.0, atol=1e-12):
        raise ValueError("bi2se3 compatibility requires the frozen a and c lattice constants")
    if not np.all(wavelength == 1.54):
        raise ValueError("bi2se3 compatibility requires wavelength_A=1.54")

    flat = indices.reshape(-1, 3)
    rounded_hk = np.rint(flat[:, :2])
    if not np.array_equal(flat[:, :2], rounded_hk):
        raise ValueError("bi2se3 compatibility requires integer h and k")
    rods = tuple((int(h), int(k)) for h, k in rounded_hk)
    if any(rod not in _BI2SE3_COMPAT_RODS for rod in rods):
        raise ValueError(
            "bi2se3 compatibility supported rods are (0,0), (-1,0), (-2,0), and (-3,1)"
        )
    if np.any(flat[:, 2] < 0.0) or np.any(flat[:, 2] > 10.0):
        raise ValueError("bi2se3 compatibility requires external L in [0, 10]")

    elements = tuple(site.element for site in crystal.sites)
    if elements.count("Bi") != 6 or elements.count("Se") != 9 or len(elements) != 15:
        raise ValueError("bi2se3 compatibility requires the frozen Bi6Se9 conventional cell")
    fractional = np.asarray([site.fractional for site in crystal.sites], dtype=np.float64)
    h = flat[:, 0]
    k = flat[:, 1]
    l_coordinate = flat[:, 2]
    radial_index = h * h + h * k + k * k
    q_magnitude = (
        2.0 * np.pi * np.sqrt((4.0 / 3.0) * radial_index / 4.557**2 + l_coordinate**2 / 28.636**2)
    )
    q_scaled_squared = (q_magnitude / (4.0 * np.pi)) ** 2
    phase = np.exp(
        2.0j
        * np.pi
        * (
            h[:, None] * fractional[None, :, 0]
            + k[:, None] * fractional[None, :, 1]
            + l_coordinate[:, None] * fractional[None, :, 2]
        )
    )
    amplitude = np.zeros(flat.shape[0], dtype=np.complex128)
    for element, *coefficients in _BI2SE3_COMPAT_FACTORS:
        gaussian_terms = coefficients[:4]
        factor = np.full(flat.shape[0], coefficients[4], dtype=np.float64)
        for coefficient, exponent in gaussian_terms:
            factor += coefficient * np.exp(-exponent * q_scaled_squared)
        complex_factor = factor + coefficients[5] + 1.0j * coefficients[6]
        mask = np.fromiter(
            (site_element == element for site_element in elements),
            dtype=np.bool_,
            count=len(elements),
        )
        amplitude += complex_factor * np.sum(phase[:, mask], axis=1)
    return amplitude.reshape(indices.shape[:-1])


def unit_cell_amplitude(
    crystal: CrystalStructure,
    hkl: ArrayLike,
    wavelength_A: ArrayLike,
    *,
    unknown_u_iso_A2: float | None = None,
    basis_mode: Literal["exact_provider", "bi2se3_whole_cell_compat"] = "exact_provider",
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
    has_unknown_u_iso = any(site.u_iso_A2 is None for site in crystal.sites)
    if unknown_u_iso_A2 is None and has_unknown_u_iso:
        raise ValueError("unknown isotropic displacement requires an explicit calculation value")

    if basis_mode not in {"exact_provider", "bi2se3_whole_cell_compat"}:
        raise ValueError("basis_mode must be 'exact_provider' or 'bi2se3_whole_cell_compat'")
    source_sha256 = _source_sha256(crystal.source_path)
    if basis_mode == "bi2se3_whole_cell_compat":
        identity = (
            "basis_mode=bi2se3_whole_cell_compat;"
            f"legacy_revision={_BI2SE3_COMPAT_LEGACY_REVISION};"
            f"cif_sha256={source_sha256};a_A=4.143;c_A=28.636;wavelength_A=1.54;"
            "legacy_form_factor_a_A=4.557;phase=+2*pi*(h*x+k*y+L*z);"
            "phase_z_divisor=1;occupancy=ignored;u_iso=ignored"
        )
        return StructureAmplitudeResult(
            amplitude_e=_bi2se3_compat_amplitude(crystal, indices, wavelength, source_sha256),
            provenance=identity,
            cache_identity=identity,
            basis_mode=basis_mode,
        )

    lattice = ReciprocalLattice.from_crystal(crystal)
    q_vectors = lattice.q_cartesian_Ainv(indices).reshape(-1, 3)
    q_magnitude = np.linalg.norm(q_vectors, axis=1)
    wavelength_flat = wavelength.reshape(-1)
    fractional = np.asarray([site.fractional for site in crystal.sites], dtype=np.float64)
    positions_A = fractional @ crystal.direct_basis_A.T
    phase = np.exp(1.0j * (q_vectors @ positions_A.T))
    u_iso = np.asarray(
        [unknown_u_iso_A2 if site.u_iso_A2 is None else site.u_iso_A2 for site in crystal.sites],
        dtype=np.float64,
    )
    damping = np.exp(-0.5 * q_magnitude[:, None] ** 2 * u_iso[None, :])
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

    identity = (
        f"basis_mode=exact_provider;cif_sha256={source_sha256};provider=XrayDB {xraydb.__version__};"
        "phase=positive;occupancy=applied;u_iso=applied;q=|Q|/(4*pi)"
    )
    return StructureAmplitudeResult(
        amplitude_e=amplitude.reshape(leading_shape),
        provenance=(
            f"{identity}; "
            f"database={xraydb.get_xraydb().get_version().split(',')[0].removeprefix('XrayDB Version: ')}; "
            "f=f0+f1+i*f2; q=|Q|/(4*pi); "
            f"species={','.join(mappings)}"
            + (f"; unknown_u_iso_A2={unknown_u_iso_A2:g}" if has_unknown_u_iso else "")
        ),
        cache_identity=identity,
        basis_mode=basis_mode,
    )


def ordered_event_result(
    crystal: CrystalStructure,
    catalog: RodCatalog,
    query: RodQueryBatch,
    *,
    unknown_u_iso_A2: float | None = None,
    basis_mode: Literal["exact_provider", "bi2se3_whole_cell_compat"] = "exact_provider",
) -> OrderedEventResult:
    """Evaluate arbitrary-L ordered amplitudes after exact rod-identity validation."""

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
    lattice.validate_layer_qz(query.l_coordinate, query.qz_Ainv)
    hkl = np.column_stack((query.h, query.k, query.l_coordinate))
    amplitude = unit_cell_amplitude(
        crystal,
        hkl,
        query.wavelength_A,
        unknown_u_iso_A2=unknown_u_iso_A2,
        basis_mode=basis_mode,
    )
    intensity = EventIntensityResult(
        event_id=query.event_id,
        intensity_per_sr=np.abs(amplitude.amplitude_e) ** 2,
        model_id="ordered",
        model_component_id=(
            "raw_unit_cell" if basis_mode == "exact_provider" else "bi2se3_whole_cell_compat"
        ),
        population_group_id=None,
        normalization="|F_e|^2; electron2; no external factors",
    )
    return OrderedEventResult(
        event_id=query.event_id,
        amplitude_e=amplitude.amplitude_e,
        intensity=intensity,
        provenance=amplitude.provenance,
        cache_identity=amplitude.cache_identity,
        basis_mode=amplitude.basis_mode,
    )
