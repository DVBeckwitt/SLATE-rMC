"""CIF parsing with one explicit symmetry-expansion boundary."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from operator import index
from pathlib import Path

import gemmi
import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import canonical_revision_sha256


@dataclass(frozen=True, slots=True)
class CrystalSite:
    """One occupied site in the expanded conventional unit cell."""

    source_label: str
    species: str
    element: str
    charge: int
    occupancy: float
    fractional: tuple[float, float, float]
    u_iso_A2: float | None
    source_multiplicity: int

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.fractional, dtype=np.float64)
        if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("fractional coordinates must contain three finite values")
        if not np.isfinite(self.occupancy) or not 0.0 <= self.occupancy <= 1.0:
            raise ValueError("site occupancy must be finite and in [0, 1]")
        if self.u_iso_A2 is not None and (not np.isfinite(self.u_iso_A2) or self.u_iso_A2 < 0.0):
            raise ValueError("isotropic displacement must be finite and nonnegative")
        try:
            charge = index(self.charge)
            multiplicity = index(self.source_multiplicity)
        except TypeError as error:
            raise ValueError("charge and source multiplicity must be integers") from error
        if isinstance(self.charge, bool) or isinstance(self.source_multiplicity, bool):
            raise ValueError("charge and source multiplicity must be integers")
        if not self.source_label or not self.species or not self.element or multiplicity < 1:
            raise ValueError("site identity and positive source multiplicity are required")
        object.__setattr__(self, "fractional", tuple(float(value) for value in coordinates))
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "occupancy", float(self.occupancy))
        object.__setattr__(
            self, "u_iso_A2", None if self.u_iso_A2 is None else float(self.u_iso_A2)
        )
        object.__setattr__(self, "source_multiplicity", multiplicity)


@dataclass(frozen=True, slots=True)
class CrystalStructure:
    """Immutable cell and its symmetry-expanded sites."""

    phase_id: str
    spacegroup_hm: str
    direct_basis_A: NDArray[np.float64]
    volume_A3: float
    sites: tuple[CrystalSite, ...]
    source_path: Path
    provenance: str
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        basis = np.array(self.direct_basis_A, dtype=np.float64, copy=True, order="C")
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("direct_basis_A must be a finite 3 by 3 matrix")
        determinant = float(np.linalg.det(basis))
        if determinant <= 0.0 or not np.isclose(
            determinant, self.volume_A3, rtol=2e-12, atol=1e-12
        ):
            raise ValueError("direct basis must be right-handed and agree with cell volume")
        if not np.isfinite(self.volume_A3) or self.volume_A3 <= 0.0:
            raise ValueError("cell volume must be finite and positive")
        if not self.phase_id or not self.spacegroup_hm or not self.sites or not self.provenance:
            raise ValueError("phase, space group, sites, and provenance are required")
        source_sha256 = self.source_sha256
        if source_sha256 is not None and (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or source_sha256 != source_sha256.lower()
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError("source_sha256 must be one lowercase SHA-256 digest")
        basis.setflags(write=False)
        object.__setattr__(self, "direct_basis_A", basis)
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "source_sha256", source_sha256)


def crystal_structure_revision(crystal: CrystalStructure) -> str:
    """Hash resolved cell and expanded-site physics, excluding filesystem provenance."""

    if not isinstance(crystal, CrystalStructure):
        raise TypeError("crystal must be CrystalStructure")
    u_known = np.asarray(
        [site.u_iso_A2 is not None for site in crystal.sites],
        dtype=np.bool_,
    )
    u_value = np.asarray(
        [0.0 if site.u_iso_A2 is None else site.u_iso_A2 for site in crystal.sites],
        dtype=np.float64,
    )
    return canonical_revision_sha256(
        ("definition_id", "resolved_expanded_crystal_structure.v1"),
        ("phase_id", crystal.phase_id),
        ("spacegroup_hm", crystal.spacegroup_hm),
        ("direct_basis_A", crystal.direct_basis_A),
        ("volume_A3", np.asarray(crystal.volume_A3, dtype=np.float64)),
        ("source_label", tuple(site.source_label for site in crystal.sites)),
        ("species", tuple(site.species for site in crystal.sites)),
        ("element", tuple(site.element for site in crystal.sites)),
        ("charge", np.asarray([site.charge for site in crystal.sites], dtype=np.int64)),
        (
            "occupancy",
            np.asarray([site.occupancy for site in crystal.sites], dtype=np.float64),
        ),
        (
            "fractional",
            np.asarray([site.fractional for site in crystal.sites], dtype=np.float64),
        ),
        ("u_iso_known", u_known),
        ("u_iso_A2", u_value),
        (
            "source_multiplicity",
            np.asarray(
                [site.source_multiplicity for site in crystal.sites],
                dtype=np.int64,
            ),
        ),
    )


def _readonly_finite_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class AffineCifSiteBasis:
    """Declarative affine coordinates, occupancies, and isotropic U on expanded CIF rows."""

    reference_crystal: CrystalStructure
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    reference_parameters: NDArray[np.float64]
    fractional_coefficients: NDArray[np.float64]
    occupancy_coefficients: NDArray[np.float64]
    u_iso_A2_coefficients: NDArray[np.float64]
    unknown_u_iso_A2: float | None = None
    reference_crystal_revision: str = field(init=False)
    basis_revision: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_crystal, CrystalStructure):
            raise TypeError("reference_crystal must be CrystalStructure")
        names = tuple(self.parameter_names)
        units = tuple(self.parameter_units)
        if (
            not names
            or any(not isinstance(value, str) or not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("parameter_names must contain unique nonempty strings")
        if len(units) != len(names) or any(
            not isinstance(value, str) or not value for value in units
        ):
            raise ValueError("parameter_units must align with the named parameters")
        parameter_count = len(names)
        site_count = len(self.reference_crystal.sites)
        reference = _readonly_finite_array(
            self.reference_parameters,
            (parameter_count,),
            "reference_parameters",
        )
        fractional = _readonly_finite_array(
            self.fractional_coefficients,
            (site_count, 3, parameter_count),
            "fractional_coefficients",
        )
        occupancy = _readonly_finite_array(
            self.occupancy_coefficients,
            (site_count, parameter_count),
            "occupancy_coefficients",
        )
        u_iso = _readonly_finite_array(
            self.u_iso_A2_coefficients,
            (site_count, parameter_count),
            "u_iso_A2_coefficients",
        )
        unknown = self.unknown_u_iso_A2
        if unknown is not None:
            unknown = float(unknown)
            if not math.isfinite(unknown) or unknown < 0.0:
                raise ValueError("unknown_u_iso_A2 must be finite and nonnegative")
        unknown_rows = np.asarray(
            [site.u_iso_A2 is None for site in self.reference_crystal.sites],
            dtype=np.bool_,
        )
        if np.any((np.linalg.norm(u_iso, axis=1) > 0.0) & unknown_rows) and unknown is None:
            raise ValueError("varying an unknown Uiso row requires unknown_u_iso_A2")

        fractional_without_origin = fractional - np.mean(
            fractional,
            axis=0,
            keepdims=True,
        )
        design = np.concatenate(
            (
                fractional_without_origin.reshape(-1, parameter_count),
                occupancy,
                u_iso,
            ),
            axis=0,
        )
        column_norm = np.linalg.norm(design, axis=0)
        if np.any(column_norm == 0.0):
            raise ValueError("every parameter must change structure beyond a common origin shift")
        normalized = design / column_norm[None, :]
        singular = np.linalg.svd(normalized, full_matrices=False, compute_uv=False)
        tolerance = 128.0 * np.finfo(np.float64).eps * max(normalized.shape) * float(singular[0])
        if int(np.count_nonzero(singular > tolerance)) != parameter_count:
            raise ValueError("affine CIF parameter columns are linearly dependent")

        reference_revision = crystal_structure_revision(self.reference_crystal)
        unknown_policy = "explicit" if unknown is not None else "preserve_none"
        basis_revision = canonical_revision_sha256(
            ("definition_id", "affine_expanded_cif_site_basis.v1"),
            ("reference_crystal_revision", reference_revision),
            ("parameter_names", names),
            ("parameter_units", units),
            ("reference_parameters", reference),
            ("fractional_coefficients", fractional),
            ("occupancy_coefficients", occupancy),
            ("u_iso_A2_coefficients", u_iso),
            ("unknown_u_iso_policy", unknown_policy),
            (
                "unknown_u_iso_A2",
                np.asarray(0.0 if unknown is None else unknown, dtype=np.float64),
            ),
        )
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "parameter_units", units)
        object.__setattr__(self, "reference_parameters", reference)
        object.__setattr__(self, "fractional_coefficients", fractional)
        object.__setattr__(self, "occupancy_coefficients", occupancy)
        object.__setattr__(self, "u_iso_A2_coefficients", u_iso)
        object.__setattr__(self, "unknown_u_iso_A2", unknown)
        object.__setattr__(self, "reference_crystal_revision", reference_revision)
        object.__setattr__(self, "basis_revision", basis_revision)

    def apply(self, parameters: ArrayLike) -> CrystalStructure:
        """Apply one candidate vector without wrapping, clipping, or topology changes."""

        value = _readonly_finite_array(
            parameters,
            (len(self.parameter_names),),
            "parameters",
        )
        delta = value - self.reference_parameters
        reference_fractional = np.asarray(
            [site.fractional for site in self.reference_crystal.sites],
            dtype=np.float64,
        )
        reference_occupancy = np.asarray(
            [site.occupancy for site in self.reference_crystal.sites],
            dtype=np.float64,
        )
        fractional = reference_fractional + np.einsum(
            "scp,p->sc",
            self.fractional_coefficients,
            delta,
        )
        occupancy = reference_occupancy + self.occupancy_coefficients @ delta
        if np.any((occupancy < 0.0) | (occupancy > 1.0)):
            raise ValueError("affine CIF occupancy lies outside [0, 1]")

        sites: list[CrystalSite] = []
        for site_index, site in enumerate(self.reference_crystal.sites):
            u_change = float(self.u_iso_A2_coefficients[site_index] @ delta)
            if (
                site.u_iso_A2 is None
                and self.unknown_u_iso_A2 is None
                and np.all(self.u_iso_A2_coefficients[site_index] == 0.0)
            ):
                u_iso_A2 = None
            else:
                baseline_u = self.unknown_u_iso_A2 if site.u_iso_A2 is None else site.u_iso_A2
                assert baseline_u is not None
                u_iso_A2 = baseline_u + u_change
                if u_iso_A2 < 0.0:
                    raise ValueError("affine CIF isotropic displacement became negative")
            sites.append(
                replace(
                    site,
                    fractional=tuple(float(item) for item in fractional[site_index]),
                    occupancy=float(occupancy[site_index]),
                    u_iso_A2=u_iso_A2,
                )
            )
        return CrystalStructure(
            phase_id=self.reference_crystal.phase_id,
            spacegroup_hm="P 1",
            direct_basis_A=self.reference_crystal.direct_basis_A,
            volume_A3=self.reference_crystal.volume_A3,
            sites=tuple(sites),
            source_path=self.reference_crystal.source_path,
            provenance=(
                f"{self.reference_crystal.provenance}; affine expanded-CIF site basis "
                f"{self.basis_revision}"
            ),
            source_sha256=self.reference_crystal.source_sha256,
        )


def crystal_with_direct_basis(
    crystal: CrystalStructure,
    direct_basis_A: NDArray[np.float64],
    *,
    provenance: str,
) -> CrystalStructure:
    """Return the same fractional structure in an explicitly supplied direct basis."""

    if not isinstance(crystal, CrystalStructure):
        raise TypeError("crystal must be CrystalStructure")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must be nonempty")
    basis = np.asarray(direct_basis_A, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("direct_basis_A must be a finite 3 by 3 matrix")
    volume_A3 = float(np.linalg.det(basis))
    if volume_A3 <= 0.0:
        raise ValueError("direct_basis_A must be right-handed")
    return CrystalStructure(
        phase_id=crystal.phase_id,
        spacegroup_hm=crystal.spacegroup_hm,
        direct_basis_A=basis,
        volume_A3=volume_A3,
        sites=crystal.sites,
        source_path=crystal.source_path,
        provenance=f"{crystal.provenance}; {provenance.strip()}",
        source_sha256=crystal.source_sha256,
    )


def _direct_basis(cell: gemmi.UnitCell) -> NDArray[np.float64]:
    vectors = (
        cell.orthogonalize(gemmi.Fractional(1.0, 0.0, 0.0)),
        cell.orthogonalize(gemmi.Fractional(0.0, 1.0, 0.0)),
        cell.orthogonalize(gemmi.Fractional(0.0, 0.0, 1.0)),
    )
    return np.column_stack(tuple((vector.x, vector.y, vector.z) for vector in vectors))


def _source_displacements(
    block: gemmi.cif.Block, source_sites: list[gemmi.SmallStructure.Site]
) -> dict[str, float | None]:
    site_count = len(source_sites)

    def required_values(tag: str, field: str) -> list[str]:
        values = list(block.find_values(tag))
        if len(values) != site_count or any(gemmi.cif.is_null(value) for value in values):
            raise ValueError(f"CIF atom-site {field} must be explicitly present for every site")
        return values

    labels = [gemmi.cif.as_string(value) for value in required_values("_atom_site_label", "label")]
    required_values("_atom_site_occupancy", "occupancy")
    required_values("_atom_site_fract_x", "fractional x coordinate")
    required_values("_atom_site_fract_y", "fractional y coordinate")
    required_values("_atom_site_fract_z", "fractional z coordinate")
    required_values("_atom_site_type_symbol", "species")
    u_values = list(block.find_values("_atom_site_U_iso_or_equiv"))
    b_values = list(block.find_values("_atom_site_B_iso_or_equiv"))
    if len(set(labels)) != len(labels) or any(
        label != site.label for label, site in zip(labels, source_sites, strict=True)
    ):
        raise ValueError("CIF atom-site labels must be present and unique")
    if u_values and b_values:
        raise ValueError("CIF contains ambiguous isotropic U and B displacement columns")
    raw_values = u_values or b_values
    if raw_values and len(raw_values) != len(source_sites):
        raise ValueError("isotropic displacement column does not align with atom sites")

    result: dict[str, float | None] = {}
    for site_index, (label, site) in enumerate(zip(labels, source_sites, strict=True)):
        raw = raw_values[site_index] if raw_values else "?"
        result[label] = None if raw in {"?", "."} else float(site.u_iso)
    return result


def _canonical_fractional(position: gemmi.Fractional) -> tuple[float, float, float]:
    fractional = np.mod((position.x, position.y, position.z), 1.0)
    fractional[np.isclose(fractional, 1.0, rtol=0.0, atol=1e-12)] = 0.0
    return tuple(float(value) for value in fractional)


def read_crystal(
    path: str | Path,
    *,
    phase_id: str | None = None,
    expected_sha256: str | None = None,
) -> CrystalStructure:
    """Read exactly one CIF structure and expand its symmetry exactly once."""

    source_path = Path(path)
    try:
        source_bytes = source_path.read_bytes()
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if expected_sha256 is not None:
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
            ):
                raise ValueError("expected_sha256 must be one lowercase SHA-256 digest")
            if actual_sha256 != expected_sha256:
                raise ValueError("CIF content changed after its configuration revision was frozen")
        document = gemmi.cif.read_string(source_bytes.decode("utf-8"))
        document.check_for_missing_values()
        document.check_for_duplicates()
        for source_block in document:
            if source_block.name == " ":
                raise RuntimeError("missing block name (bare data_)")
        pending_blocks = list(document)
        while pending_blocks:
            source_block = pending_blocks.pop()
            for item in source_block:
                loop = item.loop
                if loop is not None and loop.length() == 0:
                    raise RuntimeError(f"empty loop with {loop.tags[0]}")
                frame = item.frame
                if frame is not None:
                    pending_blocks.append(frame)
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        raise ValueError(f"failed to read CIF {source_path}: {error}") from error
    if len(document) != 1:
        raise ValueError("CIF must contain exactly one data block")
    block = document.sole_block()
    try:
        small = gemmi.make_small_structure_from_block(block)
    except RuntimeError as error:
        raise ValueError(f"failed to interpret CIF structure: {error}") from error
    if small.spacegroup is None:
        raise ValueError("CIF space group is required and must resolve unambiguously")
    spacegroup_error = small.check_spacegroup()
    if spacegroup_error:
        raise ValueError(f"CIF space-group declarations conflict: {spacegroup_error}")
    if not small.cell.is_compatible_with_spacegroup(small.spacegroup):
        raise ValueError("CIF cell metric is incompatible with its space group")

    source_sites = list(small.sites)
    if not source_sites:
        raise ValueError("CIF must contain at least one atom site")
    anisotropic_tags = tuple(
        f"_atom_site_aniso_{kind}_{suffix}"
        for kind in ("U", "B")
        for suffix in ("11", "22", "33", "12", "13", "23")
    )
    has_anisotropic_metadata = bool(list(block.find_values("_atom_site_aniso_label"))) or any(
        any(not gemmi.cif.is_null(value) for value in block.find_values(tag))
        for tag in anisotropic_tags
    )
    if has_anisotropic_metadata or any(site.aniso.nonzero() for site in source_sites):
        raise NotImplementedError("anisotropic displacement metadata is not supported")
    u_iso_by_label = _source_displacements(block, source_sites)

    expanded_sites = list(small.get_all_unit_cell_sites())
    multiplicities = Counter(site.label for site in expanded_sites)
    sites: list[CrystalSite] = []
    for site in expanded_sites:
        if site.element.atomic_number <= 0:
            raise ValueError(f"unrecognized atomic species {site.type_symbol!r}")
        sites.append(
            CrystalSite(
                source_label=site.label,
                species=site.type_symbol,
                element=site.element.name,
                charge=int(site.charge),
                occupancy=float(site.occ),
                fractional=_canonical_fractional(site.fract),
                u_iso_A2=u_iso_by_label[site.label],
                source_multiplicity=multiplicities[site.label],
            )
        )

    resolved_phase_id = phase_id or block.name
    return CrystalStructure(
        phase_id=resolved_phase_id,
        spacegroup_hm=small.spacegroup.xhm(),
        direct_basis_A=_direct_basis(small.cell),
        volume_A3=float(small.cell.volume),
        sites=tuple(sites),
        source_path=source_path,
        provenance=(
            f"Gemmi {gemmi.__version__}; symmetry expanded once; "
            "unknown isotropic displacement preserved as None"
        ),
        source_sha256=actual_sha256,
    )
