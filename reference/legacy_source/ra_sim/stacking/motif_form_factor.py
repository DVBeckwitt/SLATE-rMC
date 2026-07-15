"""Exact Cartesian motif form factors for stacking-orientation pairs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ra_sim.structure_factors.options import StructureFactorOptions
from ra_sim.structure_factors.raw_f import _load_crystal
from ra_sim.structure_factors.vesta_like_atomic_factors import anomalous_terms, f0

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
DEFAULT_WAVELENGTH_ANGSTROM = 1.5406


def _coerce_vector3(value, name: str) -> Vector3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(float(component) for component in arr)


def _coerce_matrix3(value, name: str) -> Matrix3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return tuple(tuple(float(component) for component in row) for row in arr)


def _coerce_vector_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a final dimension of 3")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _coerce_int_vector3(value, name: str) -> tuple[int, int, int]:
    coerced = tuple(int(v) for v in value)
    if len(coerced) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return coerced


def _require_instance(value, expected: type, name: str) -> None:
    if not isinstance(value, expected):
        article = "an" if expected.__name__[0].lower() in "aeiou" else "a"
        raise TypeError(f"{name} must be {article} {expected.__name__}")


def _coerce_element(value) -> str:
    element = "".join(ch for ch in str(value).strip() if ch.isalpha())
    if not element:
        raise ValueError("element must contain an element symbol")
    return element[:1].upper() + element[1:].lower()


def _coerce_nonnegative_float(value, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _coerce_optional_finite_float(value, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite when provided")
    return parsed


@dataclass(frozen=True, slots=True)
class CellGeometry:
    """Direct-cell matrix with Cartesian lattice vectors stored as columns."""

    direct_matrix: Matrix3

    def __post_init__(self) -> None:
        matrix = _coerce_matrix3(self.direct_matrix, "direct_matrix")
        if abs(float(np.linalg.det(matrix))) < 1e-14:
            raise ValueError("direct_matrix must be invertible")
        object.__setattr__(self, "direct_matrix", matrix)

    @property
    def direct(self) -> np.ndarray:
        return np.asarray(self.direct_matrix, dtype=float)

    @property
    def reciprocal(self) -> np.ndarray:
        return 2.0 * np.pi * np.linalg.inv(self.direct).T


@dataclass(frozen=True, slots=True)
class ExpandedSite:
    element: str
    occupancy: float
    frac: Vector3
    cart: Vector3
    uiso: float | None
    source_label: str
    b_iso: float | None = None

    def __post_init__(self) -> None:
        element = _coerce_element(self.element)
        occupancy = _coerce_nonnegative_float(self.occupancy, "occupancy")
        uiso = _coerce_optional_finite_float(self.uiso, "uiso")
        b_iso = _coerce_optional_finite_float(self.b_iso, "b_iso")
        frac = _coerce_vector3(self.frac, "frac")
        cart = _coerce_vector3(self.cart, "cart")
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "frac", frac)
        object.__setattr__(self, "cart", cart)
        object.__setattr__(self, "uiso", uiso)
        object.__setattr__(self, "b_iso", b_iso)
        object.__setattr__(self, "source_label", str(self.source_label))


@dataclass(frozen=True, slots=True)
class ExpandedStructure:
    cell: CellGeometry
    sites: tuple[ExpandedSite, ...]

    def __post_init__(self) -> None:
        _require_instance(self.cell, CellGeometry, "cell")
        sites = tuple(self.sites)
        if not all(isinstance(site, ExpandedSite) for site in sites):
            raise TypeError("sites must contain ExpandedSite instances")
        object.__setattr__(self, "sites", sites)


@dataclass(frozen=True, slots=True)
class MotifAtom:
    element: str
    local_cartesian: Vector3
    occupancy: float = 1.0
    uiso: float | None = None
    b_iso: float | None = None
    source_index: int | None = None

    def __post_init__(self) -> None:
        element = _coerce_element(self.element)
        occupancy = _coerce_nonnegative_float(self.occupancy, "occupancy")
        uiso = _coerce_optional_finite_float(self.uiso, "uiso")
        b_iso = _coerce_optional_finite_float(self.b_iso, "b_iso")
        local_cartesian = _coerce_vector3(self.local_cartesian, "cart")
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "local_cartesian", local_cartesian)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "uiso", uiso)
        object.__setattr__(self, "b_iso", b_iso)
        if self.source_index is not None:
            object.__setattr__(self, "source_index", int(self.source_index))


@dataclass(frozen=True, slots=True)
class ExactLayerMotif:
    atoms: tuple[MotifAtom, ...]
    center_cartesian: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        atoms = tuple(self.atoms)
        if not atoms:
            raise ValueError("ExactLayerMotif requires at least one atom")
        if not all(isinstance(atom, MotifAtom) for atom in atoms):
            raise TypeError("atoms must contain MotifAtom instances")
        center_cartesian = _coerce_vector3(self.center_cartesian, "center_cartesian")
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "center_cartesian", center_cartesian)


@dataclass(frozen=True, slots=True)
class OrientationTransform:
    matrix: Matrix3
    permutation: tuple[int, ...] | None = None
    residual_translation: Vector3 = (0.0, 0.0, 0.0)
    registry_translation: Vector3 = (0.0, 0.0, 0.0)
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        matrix = _coerce_matrix3(self.matrix, "matrix")
        arr = np.asarray(matrix, dtype=float)
        tol = float(self.tolerance)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tolerance must be positive")
        if not np.allclose(arr.T @ arr, np.eye(3), rtol=0.0, atol=max(tol, 1e-12)):
            raise ValueError("matrix must be orthogonal")
        permutation = None if self.permutation is None else tuple(int(i) for i in self.permutation)
        residual_translation = _coerce_vector3(self.residual_translation, "residual_translation")
        registry_translation = _coerce_vector3(self.registry_translation, "registry_translation")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "residual_translation", residual_translation)
        object.__setattr__(self, "registry_translation", registry_translation)
        object.__setattr__(self, "tolerance", tol)


@dataclass(frozen=True, slots=True)
class MotifDiagnostics:
    orientation_status: str
    max_internal_residual: float = 0.0
    registry_translation: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class GenericOrientationFormFactorProvider:
    """Exact F+/F- provider with source and orientation diagnostics."""

    source_cif_path: str
    motif: ExactLayerMotif
    orientation_transform: OrientationTransform
    cell: CellGeometry
    normal: Vector3
    stoichiometry: tuple[tuple[str, int], ...]
    orientation_rmsd_angstrom: float = 0.0
    max_internal_residual_angstrom: float = 0.0
    orientation_status: str = "DISTINCT_ORIENTATIONS"
    removed_registry_translation: Vector3 = (0.0, 0.0, 0.0)
    wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM
    options: StructureFactorOptions = field(default_factory=StructureFactorOptions.package_default)

    def __post_init__(self) -> None:
        _require_instance(self.motif, ExactLayerMotif, "motif")
        _require_instance(self.orientation_transform, OrientationTransform, "orientation_transform")
        _require_instance(self.cell, CellGeometry, "cell")
        normal = np.asarray(_coerce_vector3(self.normal, "normal"), dtype=float)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 0.0:
            raise ValueError("normal must be non-zero")
        stoichiometry = tuple((str(element), int(count)) for element, count in self.stoichiometry)
        if not stoichiometry:
            raise ValueError("stoichiometry must not be empty")
        wavelength = float(self.wavelength_angstrom)
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            raise ValueError("wavelength_angstrom must be positive")
        registry_translation = _coerce_vector3(
            self.removed_registry_translation, "removed_registry_translation"
        )
        max_residual = float(self.max_internal_residual_angstrom)
        object.__setattr__(self, "source_cif_path", str(self.source_cif_path))
        object.__setattr__(self, "normal", _coerce_vector3(normal / normal_norm, "normal"))
        object.__setattr__(self, "stoichiometry", stoichiometry)
        object.__setattr__(self, "removed_registry_translation", registry_translation)
        object.__setattr__(self, "orientation_rmsd_angstrom", float(self.orientation_rmsd_angstrom))
        object.__setattr__(self, "max_internal_residual_angstrom", max_residual)
        object.__setattr__(self, "orientation_status", str(self.orientation_status))
        object.__setattr__(self, "wavelength_angstrom", wavelength)

    @classmethod
    def from_matched_blocks(
        cls,
        cif_path: str | Path,
        structure: ExpandedStructure,
        source_block,
        target_block,
        *,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
        options: StructureFactorOptions | None = None,
        tolerance: float = 1e-7,
    ) -> GenericOrientationFormFactorProvider:
        """Build a provider from two species-matched extracted motif blocks."""

        from ra_sim.stacking.motif_validation import match_species_preserving_rigid_operation

        _require_instance(structure, ExpandedStructure, "structure")
        match = match_species_preserving_rigid_operation(
            source_block.motif, target_block.motif, tolerance=tolerance
        )
        orientation_rmsd = max((rmsd for _, rmsd in match.species_rmsd), default=match.max_rmsd)
        transform = OrientationTransform(
            match.matrix, permutation=match.permutation,
            registry_translation=match.removed_registry_translation, tolerance=tolerance
        )
        return cls(
            source_cif_path=str(Path(cif_path).resolve()), motif=source_block.motif,
            orientation_transform=transform, cell=structure.cell,
            normal=source_block.stacking_normal, stoichiometry=source_block.stoichiometry,
            orientation_rmsd_angstrom=orientation_rmsd,
            max_internal_residual_angstrom=match.max_rmsd, orientation_status=match.status,
            removed_registry_translation=transform.registry_translation,
            wavelength_angstrom=wavelength_angstrom,
            options=StructureFactorOptions.package_default() if options is None else options,
        )

    @classmethod
    def from_pbi2_2h_cif(
        cls,
        cif_path: str | Path,
        *,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
        options: StructureFactorOptions | None = None,
    ) -> GenericOrientationFormFactorProvider:
        """Build the PbI2 production provider from one 2H trilayer CIF."""

        from ra_sim.stacking.motif_validation import extract_pbi2_blocks

        structure = expand_cif_to_p1(cif_path)
        blocks = extract_pbi2_blocks(structure)
        if len(blocks) != 1:
            raise ValueError("PbI2 2H provider requires exactly one I-Pb-I motif block")
        block = blocks[0]
        transform = OrientationTransform(np.diag((1.0, 1.0, -1.0)), tolerance=1e-7)
        minus = transform_motif(block.motif, transform)
        return cls(
            source_cif_path=str(Path(cif_path).resolve()), motif=block.motif,
            orientation_transform=transform, cell=structure.cell,
            normal=block.stacking_normal, stoichiometry=block.stoichiometry,
            orientation_status=_orientation_status(block.motif, minus, transform.tolerance),
            removed_registry_translation=transform.registry_translation,
            wavelength_angstrom=wavelength_angstrom,
            options=StructureFactorOptions.package_default() if options is None else options,
        )

    def q_cartesian_for_hkl(self, h: int, k: int, l_vals) -> np.ndarray:
        l_arr = np.asarray(l_vals, dtype=float)
        hkl = np.zeros(l_arr.shape + (3,), dtype=float)
        hkl[..., :2], hkl[..., 2] = (int(h), int(k)), l_arr
        return physical_q(self.cell, hkl)

    def evaluate(self, q_cartesian) -> tuple[np.ndarray, np.ndarray]:
        return form_factor_pair(
            self.motif, self.orientation_transform, q_cartesian,
            wavelength_angstrom=self.wavelength_angstrom, options=self.options,
        )[:2]

    def with_expanded_site_occupancies(
        self,
        occupancies,
    ) -> GenericOrientationFormFactorProvider:
        """Return this provider with canonical expanded-site occupancies."""

        values = np.asarray(occupancies, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Expanded-site occupancies must be a non-empty vector.")
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Expanded-site occupancies must be finite and within [0, 1].")
        atoms = []
        for atom in self.motif.atoms:
            if atom.source_index is None or not 0 <= atom.source_index < values.size:
                raise ValueError("Motif atom has no matching expanded-site occupancy.")
            atoms.append(replace(atom, occupancy=float(values[atom.source_index])))
        return replace(
            self,
            motif=ExactLayerMotif(
                tuple(atoms),
                center_cartesian=self.motif.center_cartesian,
            ),
        )


def physical_q(cell: CellGeometry, hkl) -> np.ndarray:
    """Map Miller coordinates to physical Cartesian Q using 2*pi*A^-T."""

    _require_instance(cell, CellGeometry, "cell")
    hkl_arr = _coerce_vector_array(hkl, "hkl")
    return np.einsum("ij,...j->...i", cell.reciprocal, hkl_arr)


def expand_cif_to_p1(cif_path: str | Path) -> ExpandedStructure:
    """Return the same P1-expanded sites used by SLATE-rMC raw-F validation."""

    xtl = _load_crystal(cif_path)
    basis_rows = np.asarray(xtl.Cell.calculateR(np.eye(3)), dtype=float)
    cell = CellGeometry(basis_rows.T)
    frac, atom_type, labels, occ, uiso, _mxmymz = xtl.Structure.get()
    cart = np.asarray(xtl.Cell.calculateR(frac), dtype=float)
    sites = []
    for idx, frac_value in enumerate(np.asarray(frac, dtype=float)):
        uiso_value = float(uiso[idx])
        sites.append(
            ExpandedSite(
                str(atom_type[idx]),
                float(occ[idx]),
                frac_value,
                cart[idx],
                uiso_value,
                str(labels[idx]),
                8.0 * np.pi**2 * uiso_value,
            )
        )
    return ExpandedStructure(cell=cell, sites=tuple(sites))


def extract_motif_by_indices(
    structure: ExpandedStructure,
    indices: Sequence[int],
    *,
    image_shifts: Sequence[Sequence[int]] | None = None,
    center: str | Sequence[float] = "centroid",
) -> ExactLayerMotif:
    _require_instance(structure, ExpandedStructure, "structure")
    selected = [int(index) for index in indices]
    if not selected:
        raise ValueError("indices must select at least one site")
    if image_shifts is None:
        shifts = [(0, 0, 0)] * len(selected)
    else:
        shifts = [_coerce_int_vector3(shift, "image_shift") for shift in image_shifts]
    if len(shifts) != len(selected):
        raise ValueError("image_shifts must match indices length")

    unwrapped_sites = []
    for site_index, shift in zip(selected, shifts, strict=True):
        site = structure.sites[site_index]
        shift_cart = structure.cell.direct @ np.asarray(shift, dtype=float)
        cart = np.asarray(site.cart, dtype=float) + shift_cart
        unwrapped_sites.append((site_index, site, cart))

    coords = np.asarray([item[2] for item in unwrapped_sites], dtype=float)
    if isinstance(center, str):
        if center == "centroid":
            center_cart = np.mean(coords, axis=0)
        elif center == "occupancy_centroid":
            weights = np.asarray([item[1].occupancy for item in unwrapped_sites], dtype=float)
            if float(np.sum(weights)) <= 0.0:
                raise ValueError("occupancy_centroid requires positive total occupancy")
            center_cart = np.average(coords, axis=0, weights=weights)
        else:
            raise ValueError(f"Unsupported center mode: {center}")
    else:
        center_cart = np.asarray(_coerce_vector3(center, "center"), dtype=float)

    atoms = tuple(
        MotifAtom(
            site.element,
            cart - center_cart,
            occupancy=site.occupancy,
            uiso=site.uiso,
            b_iso=site.b_iso,
            source_index=site_index,
        )
        for site_index, site, cart in unwrapped_sites
    )
    return ExactLayerMotif(atoms, center_cartesian=center_cart)


def transform_motif(motif: ExactLayerMotif, transform: OrientationTransform) -> ExactLayerMotif:
    _require_instance(motif, ExactLayerMotif, "motif")
    _require_instance(transform, OrientationTransform, "transform")
    permutation = transform.permutation
    if permutation is None:
        permutation = tuple(range(len(motif.atoms)))
    if sorted(permutation) != list(range(len(motif.atoms))):
        raise ValueError("permutation must contain each motif atom index exactly once")

    matrix = np.asarray(transform.matrix, dtype=float)
    translation = np.asarray(transform.residual_translation, dtype=float)
    transformed = tuple(
        MotifAtom(
            atom.element,
            matrix @ np.asarray(atom.local_cartesian, dtype=float) + translation,
            occupancy=atom.occupancy,
            uiso=atom.uiso,
            b_iso=atom.b_iso,
            source_index=atom.source_index,
        )
        for atom in (motif.atoms[atom_index] for atom_index in permutation)
    )
    return ExactLayerMotif(transformed, center_cartesian=motif.center_cartesian)


def _debye_waller_for_atoms(
    atoms: Sequence[MotifAtom],
    q_norm: np.ndarray,
    options: StructureFactorOptions,
) -> np.ndarray:
    if options.debye_waller_mode == "off":
        return np.ones((q_norm.size, len(atoms)), dtype=float)
    if options.debye_waller_mode != "cif":
        raise ValueError(f"Unsupported Debye-Waller mode: {options.debye_waller_mode}")
    values: np.ndarray = np.ones((q_norm.size, len(atoms)), dtype=float)
    for atom_idx, atom in enumerate(atoms):
        if atom.b_iso is not None:
            values[:, atom_idx] = np.exp(-float(atom.b_iso) * q_norm * q_norm / (16.0 * np.pi**2))
        elif atom.uiso is not None:
            values[:, atom_idx] = np.exp(-0.5 * float(atom.uiso) * q_norm * q_norm)
    return values


def _atomic_factor_grid(
    atoms: Sequence[MotifAtom],
    q_norm: np.ndarray,
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> np.ndarray:
    elements = np.asarray([atom.element for atom in atoms], dtype=str)
    q_values = np.asarray(q_norm, dtype=float).reshape(-1)
    if q_values.size == 0:
        return np.empty((0, len(atoms)), dtype=complex)
    if options.scattering_table == "constant":
        missing = sorted({el for el in elements if el not in options.constant_factors})
        if missing:
            raise ValueError(f"Missing constant factors for elements: {missing}")
        factors = np.asarray([complex(options.constant_factors[el]) for el in elements])
        return np.broadcast_to(factors, (q_values.size, factors.size)).astype(complex)

    s_values = q_values / (4.0 * np.pi)
    base = f0(elements, s_values, table=options.scattering_table).astype(complex)
    f_prime, f_double_prime = anomalous_terms(
        elements,
        wavelength_angstrom,
        mode=options.anomalous_mode,
    )
    return base + f_prime[None, :] + 1j * f_double_prime[None, :]


def form_factor(
    motif: ExactLayerMotif,
    q_cartesian,
    *,
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> np.ndarray:
    _require_instance(motif, ExactLayerMotif, "motif")
    q_vals = _coerce_vector_array(q_cartesian, "q_cartesian")
    out_shape = q_vals.shape[:-1]
    q_flat = q_vals.reshape(-1, 3)
    q_norm = np.linalg.norm(q_flat, axis=1)
    coords = np.asarray([atom.local_cartesian for atom in motif.atoms], dtype=float)
    if options.phase_sign not in {-1, 1}:
        raise ValueError("phase_sign must be -1 or 1.")
    if options.occupancy_mode == "unit":
        occupancies = np.ones(len(motif.atoms), dtype=float)
    elif options.occupancy_mode == "cif":
        occupancies = np.asarray([atom.occupancy for atom in motif.atoms], dtype=float)
    else:
        raise ValueError(f"Unsupported occupancy mode: {options.occupancy_mode}")
    factors = _atomic_factor_grid(motif.atoms, q_norm, float(wavelength_angstrom), options)
    debye_waller = _debye_waller_for_atoms(motif.atoms, q_norm, options)
    phase = np.exp(options.phase_sign * 1j * (q_flat @ coords.T))
    total = np.sum(factors * occupancies[None, :] * debye_waller * phase, axis=1)
    return total.reshape(out_shape)


def _orientation_status(plus: ExactLayerMotif, minus: ExactLayerMotif, tol: float) -> str:
    if len(plus.atoms) != len(minus.atoms):
        return "INDEPENDENT_COORDINATE_SETS"
    plus_coords = np.asarray([atom.local_cartesian for atom in plus.atoms], dtype=float)
    minus_coords = np.asarray([atom.local_cartesian for atom in minus.atoms], dtype=float)
    if all(
        a.element == b.element and abs(a.occupancy - b.occupancy) <= tol
        for a, b in zip(plus.atoms, minus.atoms, strict=True)
    ) and np.allclose(plus_coords, minus_coords, rtol=0.0, atol=tol):
        return "EQUIVALENT_UP_TO_TRANSLATION"
    return "DISTINCT_ORIENTATIONS"


def form_factor_pair(
    motif: ExactLayerMotif,
    transform: OrientationTransform,
    q_cartesian,
    *,
    wavelength_angstrom: float,
    options: StructureFactorOptions,
) -> tuple[np.ndarray, np.ndarray, MotifDiagnostics]:
    minus_motif = transform_motif(motif, transform)
    factor_kwargs = {"wavelength_angstrom": wavelength_angstrom, "options": options}
    f_plus = form_factor(motif, q_cartesian, **factor_kwargs)
    f_minus = form_factor(minus_motif, q_cartesian, **factor_kwargs)
    return f_plus, f_minus, MotifDiagnostics(
        _orientation_status(motif, minus_motif, transform.tolerance),
        registry_translation=transform.registry_translation,
    )
