"""CIF parsing with one explicit symmetry-expansion boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from operator import index
from pathlib import Path

import gemmi
import numpy as np
from numpy.typing import NDArray


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
        basis.setflags(write=False)
        object.__setattr__(self, "direct_basis_A", basis)
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "source_path", Path(self.source_path))


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


def read_crystal(path: str | Path, *, phase_id: str | None = None) -> CrystalStructure:
    """Read exactly one CIF structure and expand its symmetry exactly once."""

    source_path = Path(path)
    try:
        document = gemmi.cif.read_file(str(source_path))
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
    except (OSError, RuntimeError, ValueError) as error:
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
    )
