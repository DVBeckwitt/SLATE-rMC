"""Proof-only Bi2Se3 quintuple-layer reconstruction and VESTA parity."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import gemmi
import numpy as np
from numpy.typing import NDArray

from rasim_next.materials import CrystalSite, CrystalStructure, read_crystal
from rasim_next.ordered.amplitudes import unit_cell_amplitude
from rasim_next.proof.traces import Measure, QuantityKind, TraceRecord, compare_traces

_WAVELENGTH_A = 1.540592925
_D_MIN_A = 0.7
_CENTERING_TRANSLATIONS = (
    (0.0, 0.0, 0.0),
    (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
)
_ABSENT_HKL = (
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (-1, 0, 0),
    (1, -1, 0),
    (2, 0, 0),
)
_REQUIRED_MUTATION_STAGES = (
    ("omit_outer_Se", "ordered.unit_cell_amplitude"),
    ("wrong_R_centering_translation", "ordered.unit_cell_amplitude"),
    ("dropped_centering_phase", "ordered.unit_cell_amplitude"),
    ("ignored_integer_image_shift_phase", "ordered.unit_cell_amplitude"),
    ("duplicated_periodic_boundary_atom", "ordered.unit_cell_amplitude"),
)
_VESTA_CORRECTIONS_E = (
    ("Bi", -4.23706, 8.83640, -0.018084),
    ("Se", -0.787865, 1.13462, -0.0080314),
)


@dataclass(frozen=True, slots=True)
class _Atom:
    source_label: str
    species: str
    element: str
    charge: int
    occupancy: float
    fractional: tuple[float, float, float]
    u_iso_A2: float


def _atom_property_key(atom: _Atom) -> tuple[str, str, int, float, float]:
    return atom.species, atom.element, atom.charge, atom.occupancy, atom.u_iso_A2


@dataclass(frozen=True, slots=True)
class _VestaTable:
    hkl: NDArray[np.int64]
    d_A: NDArray[np.float64]
    amplitude_e: NDArray[np.complex128]
    magnitude_e: NDArray[np.float64]
    two_theta_deg: NDArray[np.float64]
    intensity_is_nan: NDArray[np.bool_]
    multiplicity: NDArray[np.int64]


def _atoms_from_crystal(crystal: CrystalStructure) -> tuple[_Atom, ...]:
    atoms: list[_Atom] = []
    for site in crystal.sites:
        if site.u_iso_A2 is None:
            raise ValueError("Bi2Se3 proof requires explicit isotropic displacement")
        atoms.append(
            _Atom(
                source_label=site.source_label,
                species=site.species,
                element=site.element,
                charge=site.charge,
                occupancy=site.occupancy,
                fractional=site.fractional,
                u_iso_A2=site.u_iso_A2,
            )
        )
    return tuple(atoms)


def _required_values(block: gemmi.cif.Block, tag: str) -> list[str]:
    values = list(block.find_values(tag))
    if not values or any(gemmi.cif.is_null(value) for value in values):
        raise ValueError(f"explicit P1 CIF requires {tag}")
    return values


def _basis_from_block(block: gemmi.cif.Block) -> NDArray[np.float64]:
    a, b, c = (
        float(block.find_value(tag))
        for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c")
    )
    alpha, beta, gamma = np.radians(
        [
            float(block.find_value(tag))
            for tag in ("_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
        ]
    )
    sin_gamma = math.sin(float(gamma))
    c_x = c * math.cos(float(beta))
    c_y = c * (math.cos(float(alpha)) - math.cos(float(beta)) * math.cos(float(gamma))) / sin_gamma
    c_z = math.sqrt(max(0.0, c * c - c_x * c_x - c_y * c_y))
    return np.asarray(
        [
            [a, b * math.cos(float(gamma)), c_x],
            [0.0, b * sin_gamma, c_y],
            [0.0, 0.0, c_z],
        ],
        dtype=np.float64,
    )


def _read_explicit_p1(path: Path) -> tuple[NDArray[np.float64], tuple[_Atom, ...]]:
    document = gemmi.cif.read_file(str(path), check_level=2)
    if len(document) != 1:
        raise ValueError("explicit P1 proof CIF must contain one data block")
    block = document.sole_block()
    spacegroup = gemmi.SpaceGroup(
        gemmi.cif.as_string(block.find_value("_space_group_name_H-M_alt"))
    )
    if spacegroup.number != 1:
        raise ValueError("expanded proof structure must be P1")
    columns = [
        _required_values(block, tag)
        for tag in (
            "_atom_site_label",
            "_atom_site_occupancy",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_U_iso_or_equiv",
            "_atom_site_type_symbol",
        )
    ]
    if len({len(column) for column in columns}) != 1:
        raise ValueError("explicit P1 atom columns do not align")
    atoms: list[_Atom] = []
    for label, occupancy, x, y, z, u_iso, symbol in zip(*columns, strict=True):
        species = gemmi.cif.as_string(symbol)
        atoms.append(
            _Atom(
                source_label=gemmi.cif.as_string(label),
                species=species,
                element=gemmi.Element(species).name,
                charge=0,
                occupancy=float(occupancy),
                fractional=(float(x), float(y), float(z)),
                u_iso_A2=float(u_iso),
            )
        )
    return _basis_from_block(block), tuple(atoms)


def _canonical_xy(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.mod(values, 1.0)
    result[np.isclose(result, 1.0, rtol=0.0, atol=1e-12)] = 0.0
    return result


def _extract_quintuple_layers(
    atoms: tuple[_Atom, ...],
    *,
    center_source_label: str | None = None,
) -> tuple[tuple[_Atom, ...], tuple[_Atom, ...], tuple[tuple[float, float, float], ...], float]:
    bi_count = sum(atom.element == "Bi" for atom in atoms)
    se_count = sum(atom.element == "Se" for atom in atoms)
    if bi_count % 2 or se_count % 3 or bi_count // 2 != se_count // 3:
        raise ValueError("expanded Bi2Se3 must contain complete Bi2Se3 quintuple layers")
    ql_count = bi_count // 2
    if len(atoms) != 5 * ql_count:
        raise ValueError("Bi2Se3 proof accepts only Bi and Se sites")

    motif_blocks: list[tuple[_Atom, ...]] = []
    complete_blocks: list[tuple[_Atom, ...]] = []
    centers: list[tuple[float, float, float]] = []
    covered: list[int] = []
    center_candidates = [
        (index, atom)
        for index, atom in enumerate(atoms)
        if atom.element == "Se"
        and (center_source_label is None or atom.source_label == center_source_label)
    ]
    for center_index, center_atom in center_candidates:
        neighboring_images: list[tuple[float, int, _Atom]] = []
        for atom_index, atom in enumerate(atoms):
            if atom_index == center_index:
                continue
            delta_z = atom.fractional[2] - center_atom.fractional[2]
            relative_z = delta_z - math.floor(delta_z + 0.5)
            neighboring_images.append((relative_z, atom_index, atom))
        lower = sorted((row for row in neighboring_images if row[0] < 0.0), reverse=True)[:2]
        upper = sorted(row for row in neighboring_images if row[0] > 0.0)[:2]
        if len(lower) != 2 or len(upper) != 2:
            raise ValueError("each Se1 center must have two lower and two upper QL sites")
        block = [*sorted(lower), (0.0, center_index, center_atom), *upper]
        if tuple(atom.element for _, _, atom in block) != ("Se", "Bi", "Se", "Bi", "Se"):
            if center_source_label is not None:
                raise ValueError("Se1-centered quintuple-layer order must be Se-Bi-Se-Bi-Se")
            continue
        if center_source_label is not None and (
            block[0][2].source_label != "Se2" or block[-1][2].source_label != "Se2"
        ):
            raise ValueError("Se1-centered quintuple-layer outer sites must be source-labelled Se2")
        covered.extend(index for _, index, _ in block)

        center = np.asarray(
            [
                center_atom.fractional[0],
                center_atom.fractional[1],
                np.mod(center_atom.fractional[2], 1.0),
            ],
            dtype=np.float64,
        )
        center[np.isclose(center, 1.0, rtol=0.0, atol=1e-12)] = 0.0
        motif: list[_Atom] = []
        complete: list[_Atom] = []
        for relative_z, _, atom in block:
            offset_xy = _canonical_xy(np.asarray(atom.fractional[:2]) - center[:2])
            offset = (float(offset_xy[0]), float(offset_xy[1]), float(relative_z))
            motif_atom = replace(atom, fractional=offset)
            motif.append(motif_atom)
            complete.append(
                replace(
                    atom,
                    fractional=tuple(float(value) for value in np.asarray(offset) + center),
                )
            )
        motif_blocks.append(tuple(motif))
        complete_blocks.append(tuple(complete))
        centers.append(tuple(float(value) for value in center))

    if len(motif_blocks) != ql_count or sorted(covered) != list(range(len(atoms))):
        raise ValueError("quintuple-layer extraction must cover every site exactly once")
    canonical_index = min(range(ql_count), key=lambda index: centers[index])
    canonical = motif_blocks[canonical_index]
    maximum_spread = 0.0
    reference_signature = tuple(_atom_property_key(atom) for atom in canonical)
    reference_coordinates = np.asarray([atom.fractional for atom in canonical])
    for motif in motif_blocks:
        signature = tuple(_atom_property_key(atom) for atom in motif)
        if signature != reference_signature:
            raise ValueError("all extracted quintuple layers must preserve atom properties")
        maximum_spread = max(
            maximum_spread,
            float(
                np.max(
                    np.abs(np.asarray([atom.fractional for atom in motif]) - reference_coordinates)
                )
            ),
        )
    return (
        canonical,
        tuple(atom for block in complete_blocks for atom in block),
        tuple(centers),
        maximum_spread,
    )


def _translated_atoms(
    motif: tuple[_Atom, ...], translations: tuple[tuple[float, float, float], ...]
) -> tuple[_Atom, ...]:
    return tuple(
        replace(
            atom,
            fractional=tuple(
                float(value) for value in np.asarray(atom.fractional) + np.asarray(translation)
            ),
        )
        for translation in translations
        for atom in motif
    )


def _crystal_from_atoms(
    template: CrystalStructure, atoms: tuple[_Atom, ...], *, phase_id: str
) -> CrystalStructure:
    return CrystalStructure(
        phase_id=phase_id,
        spacegroup_hm="P 1",
        direct_basis_A=template.direct_basis_A,
        volume_A3=template.volume_A3,
        sites=tuple(
            CrystalSite(
                source_label=atom.source_label,
                species=atom.species,
                element=atom.element,
                charge=atom.charge,
                occupancy=atom.occupancy,
                fractional=atom.fractional,
                u_iso_A2=atom.u_iso_A2,
                source_multiplicity=1,
            )
            for atom in atoms
        ),
        source_path=template.source_path,
        provenance=f"single-QL proof reconstruction from {template.source_path.name}",
    )


def _waasmaier_kirfel_f0(element: str, s_Ainv: NDArray[np.float64]) -> NDArray[np.float64]:
    if element == "Bi":
        a = (16.282274, 32.725136, 6.678302, 2.694750, 20.576559)
        b = (0.101180, 1.002287, 25.714146, 77.057549, 6.291882)
        c = 4.040914
    elif element == "Se":
        a = (17.354071, 4.653248, 4.259489, 4.136455, 6.749163)
        b = (2.349787, 0.002550, 15.579460, 45.181202, 0.177432)
        c = -3.160982
    else:
        raise ValueError(f"VESTA Bi2Se3 proof has no factor for {element}")
    squared = np.asarray(s_Ainv, dtype=np.float64) ** 2
    return (
        np.sum(
            np.asarray(a)[:, None] * np.exp(-np.asarray(b)[:, None] * squared[None, :]),
            axis=0,
        )
        + c
    )


def _vesta_factor(element: str, s_Ainv: NDArray[np.float64]) -> NDArray[np.complex128]:
    correction = next((values for values in _VESTA_CORRECTIONS_E if values[0] == element), None)
    if correction is None:
        raise ValueError(f"VESTA Bi2Se3 proof has no factor for {element}")
    _, f_prime, f_double_prime, nuclear_thomson = correction
    return np.asarray(
        _waasmaier_kirfel_f0(element, s_Ainv) + f_prime + nuclear_thomson + 1.0j * f_double_prime,
        dtype=np.complex128,
    )


def _direct_amplitudes(
    atoms: tuple[_Atom, ...],
    direct_basis_A: NDArray[np.float64],
    hkl: NDArray[np.float64],
) -> NDArray[np.complex128]:
    indices = np.asarray(hkl, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct_basis_A).T
    q_magnitude = np.linalg.norm(indices @ reciprocal.T, axis=1)
    s_Ainv = q_magnitude / (4.0 * np.pi)
    fractional = np.asarray([atom.fractional for atom in atoms], dtype=np.float64)
    phase = np.exp(2.0j * np.pi * (indices @ fractional.T))
    occupancy = np.asarray([atom.occupancy for atom in atoms], dtype=np.float64)
    u_iso_A2 = np.asarray([atom.u_iso_A2 for atom in atoms], dtype=np.float64)
    amplitude = np.zeros(indices.shape[0], dtype=np.complex128)
    for element in ("Bi", "Se"):
        mask = np.fromiter(
            (atom.element == element for atom in atoms), dtype=np.bool_, count=len(atoms)
        )
        if not np.any(mask):
            continue
        damping = np.exp(-0.5 * q_magnitude[:, None] ** 2 * u_iso_A2[None, mask])
        amplitude += _vesta_factor(element, s_Ainv) * np.sum(
            phase[:, mask] * damping * occupancy[None, mask], axis=1
        )
    return amplitude


def _centering_factor(hkl: NDArray[np.float64]) -> NDArray[np.complex128]:
    h, k, ell = np.asarray(hkl, dtype=np.float64).T
    return (
        1.0
        + np.exp(2.0j * np.pi * (2.0 * h + k + ell) / 3.0)
        + np.exp(2.0j * np.pi * (h + 2.0 * k + 2.0 * ell) / 3.0)
    )


def _read_vesta_table(path: Path) -> _VestaTable:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:] if line]
    if any(len(row) != 10 for row in rows):
        raise ValueError("VESTA structure-factor table must contain ten columns")
    return _VestaTable(
        hkl=np.asarray([[int(value) for value in row[:3]] for row in rows], dtype=np.int64),
        d_A=np.asarray([float(row[3]) for row in rows], dtype=np.float64),
        amplitude_e=np.asarray(
            [complex(float(row[4]), float(row[5])) for row in rows], dtype=np.complex128
        ),
        magnitude_e=np.asarray([float(row[6]) for row in rows], dtype=np.float64),
        two_theta_deg=np.asarray(
            [np.nan if "nan" in row[7].lower() else float(row[7]) for row in rows],
            dtype=np.float64,
        ),
        intensity_is_nan=np.asarray(["nan" in row[8].lower() for row in rows], dtype=np.bool_),
        multiplicity=np.asarray([int(row[9]) for row in rows], dtype=np.int64),
    )


def _reflection_set(
    direct_basis_A: NDArray[np.float64], d_min_A: float
) -> tuple[tuple[tuple[int, int, int], ...], int]:
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct_basis_A).T
    bounds = [
        math.ceil(float(np.linalg.norm(direct_basis_A[:, axis])) / d_min_A) for axis in range(3)
    ]
    reflections: list[tuple[int, int, int]] = []
    geometric_candidate_count = 0
    for h in range(bounds[0] + 1):
        for k in range(bounds[1] + 1):
            for ell in range(bounds[2] + 1):
                if (h, k, ell) == (0, 0, 0) or (ell == 0 and h < k):
                    continue
                q_magnitude = float(np.linalg.norm(reciprocal @ (h, k, ell)))
                if 2.0 * np.pi / q_magnitude + 1e-12 < d_min_A:
                    continue
                geometric_candidate_count += 1
                if (-h + k + ell) % 3 == 0:
                    reflections.append((h, k, ell))
    return tuple(reflections), geometric_candidate_count


def _multiplicities(path: Path, hkl: NDArray[np.int64]) -> NDArray[np.int64]:
    small = gemmi.make_small_structure_from_block(gemmi.cif.read_file(str(path)).sole_block())
    if small.spacegroup is None:
        raise ValueError("R-3m space group is required")
    operations = small.spacegroup.operations()
    return np.asarray(
        [
            len(
                {
                    tuple(operation.apply_to_hkl(tuple(int(value) for value in miller)))
                    for operation in operations
                }
            )
            for miller in hkl
        ],
        dtype=np.int64,
    )


def _complex_pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _maximum(error: NDArray[np.complex128] | NDArray[np.float64]) -> float:
    return float(np.max(np.abs(error), initial=0.0))


def _coordinate_payload(atoms: tuple[_Atom, ...]) -> list[dict[str, object]]:
    return [
        {
            "source_label": atom.source_label,
            "species": atom.species,
            "element": atom.element,
            "charge": atom.charge,
            "occupancy": atom.occupancy,
            "u_iso_A2": atom.u_iso_A2,
            "fractional": list(atom.fractional),
        }
        for atom in atoms
    ]


def _periodic_coordinate_error(first: tuple[_Atom, ...], second: tuple[_Atom, ...]) -> float:
    available = set(range(len(second)))
    maximum = 0.0
    for atom in first:
        candidates: list[tuple[float, int]] = []
        for index in available:
            other = second[index]
            if _atom_property_key(other) != _atom_property_key(atom):
                continue
            difference = np.asarray(atom.fractional) - np.asarray(other.fractional)
            difference -= np.rint(difference)
            candidates.append((float(np.max(np.abs(difference))), index))
        if not candidates:
            raise ValueError("expanded structures do not contain matching atom species")
        error, matched = min(candidates)
        available.remove(matched)
        maximum = max(maximum, error)
    if available:
        raise ValueError("expanded structures have different atom counts")
    return maximum


def _coordinate_serialization_bound_e(
    atoms: tuple[_Atom, ...],
    direct_basis_A: NDArray[np.float64],
    hkl: NDArray[np.float64],
    maximum_fractional_error: float,
) -> NDArray[np.float64]:
    """Triangle bound for phase error from finite-precision fractional coordinates."""

    indices = np.asarray(hkl, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct_basis_A).T
    q_magnitude = np.linalg.norm(indices @ reciprocal.T, axis=1)
    s_Ainv = q_magnitude / (4.0 * np.pi)
    coefficient_sum = np.zeros(indices.shape[0], dtype=np.float64)
    for atom in atoms:
        damping = np.exp(-0.5 * atom.u_iso_A2 * q_magnitude**2)
        coefficient_sum += atom.occupancy * np.abs(_vesta_factor(atom.element, s_Ainv)) * damping
    phase_bound = np.minimum(
        2.0,
        2.0 * np.pi * maximum_fractional_error * np.sum(np.abs(indices), axis=1),
    )
    floating_margin = 256.0 * np.finfo(np.float64).eps * coefficient_sum
    return coefficient_sum * phase_bound + floating_margin


def _image_shifts(
    wrapped: tuple[_Atom, ...], complete: tuple[_Atom, ...]
) -> tuple[list[dict[str, object]], tuple[tuple[int, int, int], ...], float]:
    available = set(range(len(wrapped)))
    payload: list[dict[str, object]] = []
    shifts: list[tuple[int, int, int]] = []
    maximum_error = 0.0
    for atom in complete:
        candidates: list[tuple[float, int, NDArray[np.float64]]] = []
        for index in available:
            source = wrapped[index]
            if _atom_property_key(source) != _atom_property_key(atom):
                continue
            difference = np.asarray(atom.fractional) - np.asarray(source.fractional)
            residual = difference - np.rint(difference)
            candidates.append((float(np.max(np.abs(residual))), index, difference))
        if not candidates:
            raise ValueError("complete QL atom has no identity-preserving wrapped-cell match")
        error, matched, difference = min(candidates, key=lambda item: item[0])
        available.remove(matched)
        maximum_error = max(maximum_error, error)
        image_shift = tuple(int(value) for value in np.rint(difference))
        shifts.append(image_shift)
        payload.append(
            {
                "source_label": wrapped[matched].source_label,
                "element": atom.element,
                "wrapped_fractional": list(wrapped[matched].fractional),
                "complete_fractional": list(atom.fractional),
                "integer_image_shift": list(image_shift),
            }
        )
    if available:
        raise ValueError("complete QL reconstruction must match every wrapped site exactly once")
    return payload, tuple(shifts), maximum_error


def _mutation_results(
    canonical: tuple[_Atom, ...],
    direct_basis_A: NDArray[np.float64],
    integer_hkl: NDArray[np.float64],
    noninteger_hkl: NDArray[np.float64],
    noninteger_wk_correct: NDArray[np.complex128],
    noninteger_wk_without_image_phase: NDArray[np.complex128],
) -> list[dict[str, object]]:
    correct_atoms = _translated_atoms(canonical, _CENTERING_TRANSLATIONS)
    correct_integer = _direct_amplitudes(correct_atoms, direct_basis_A, integer_hkl)
    integer_ql_amplitude = _direct_amplitudes(canonical, direct_basis_A, integer_hkl)
    noninteger_ql_amplitude = _direct_amplitudes(canonical, direct_basis_A, noninteger_hkl)
    phase_by_translation = np.exp(
        2.0j * np.pi * (integer_hkl @ np.asarray(_CENTERING_TRANSLATIONS).T)
    )
    boundary_duplicate = replace(canonical[2], fractional=(0.0, 0.0, 1.0))
    cases = (
        (
            "omit_outer_Se",
            integer_ql_amplitude,
            correct_integer,
            _direct_amplitudes(
                _translated_atoms(canonical[1:], _CENTERING_TRANSLATIONS),
                direct_basis_A,
                integer_hkl,
            ),
        ),
        (
            "wrong_R_centering_translation",
            integer_ql_amplitude,
            correct_integer,
            _direct_amplitudes(
                _translated_atoms(
                    canonical,
                    (
                        _CENTERING_TRANSLATIONS[0],
                        (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
                        _CENTERING_TRANSLATIONS[2],
                    ),
                ),
                direct_basis_A,
                integer_hkl,
            ),
        ),
        (
            "dropped_centering_phase",
            integer_ql_amplitude,
            integer_ql_amplitude * np.sum(phase_by_translation, axis=1),
            integer_ql_amplitude * np.sum(phase_by_translation[:, :2], axis=1),
        ),
        (
            "ignored_integer_image_shift_phase",
            noninteger_ql_amplitude,
            noninteger_wk_correct,
            noninteger_wk_without_image_phase,
        ),
        (
            "duplicated_periodic_boundary_atom",
            integer_ql_amplitude,
            correct_integer,
            _direct_amplitudes((*correct_atoms, boundary_duplicate), direct_basis_A, integer_hkl),
        ),
    )
    results: list[dict[str, object]] = []
    first_stage = "ordered.unit_cell_amplitude"
    for mutation_id, prior_atomic_amplitude, correct, mutated in cases:
        fixture_id = f"ordered.bi2se3_ql.{mutation_id}"
        reference_records: list[TraceRecord] = []
        candidate_records: list[TraceRecord] = []
        for records in (reference_records, candidate_records):
            records.append(
                TraceRecord(
                    fixture_id,
                    "ordered.atomic_amplitude",
                    prior_atomic_amplitude,
                    "e",
                    "crystal",
                    Measure.NONE,
                    QuantityKind.AMPLITUDE,
                    "t04-bi2se3-proof-v1",
                    "unchanged single-QL atomic amplitude",
                )
            )
        for records, value in ((reference_records, correct), (candidate_records, mutated)):
            records.append(
                TraceRecord(
                    fixture_id,
                    first_stage,
                    value,
                    "e",
                    "crystal",
                    Measure.NONE,
                    QuantityKind.AMPLITUDE,
                    "t04-bi2se3-proof-v1",
                    "in-memory Bi2Se3 proof mutation",
                )
            )
        comparison = compare_traces(reference_records, candidate_records)
        error = _maximum(mutated - correct)
        results.append(
            {
                "mutation_id": mutation_id,
                "fixture_id": fixture_id,
                "expected_first_stage": first_stage,
                "expected_failure_metric": "numeric_value",
                "observed_first_stage": comparison.first_failing_stage,
                "observed_failure_metric": comparison.failure_metric,
                "prior_stages_identical": comparison.first_failing_stage == first_stage,
                "detected": bool(
                    comparison.first_failing_stage == first_stage
                    and comparison.failure_metric == "numeric_value"
                    and error > 1e-10
                ),
                "maximum_error_e": error,
            }
        )
    return results


def run_bi2se3_ql_proof(
    root: str,
    tolerances: dict[str, dict[str, float]],
    *,
    direct_atom_amplitudes: Callable[
        [CrystalStructure, NDArray[np.float64], NDArray[np.float64]],
        NDArray[np.complex128],
    ],
) -> dict[str, object]:
    """Prove direct R-3m, explicit P1, one-QL, and VESTA routes independently."""

    repository = Path(root)
    structure_root = repository / "examples" / "bi2se3" / "structures"
    reference_root = repository / "examples" / "bi2se3" / "reference"
    r3m_path = structure_root / "Bi2Se3_vesta.cif"
    legacy_path = structure_root / "Bi2Se3_legacy.cif"
    p1_path = structure_root / "Bi2Se3_expanded_P1.cif"
    table_path = reference_root / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"
    metadata = json.loads(
        (reference_root / "Bi2Se3_vesta_cu_ka1_dmin_0p7.metadata.json").read_text(encoding="utf-8")
    )

    r3m = read_crystal(r3m_path, phase_id="bi2se3-vesta-r3m")
    r3m_atoms = _atoms_from_crystal(r3m)
    legacy = read_crystal(legacy_path, phase_id="bi2se3-legacy-r3m")
    legacy_atoms = _atoms_from_crystal(legacy)
    p1_basis, p1_atoms = _read_explicit_p1(p1_path)
    canonical, complete_r3m, centers, r3m_motif_spread = _extract_quintuple_layers(
        r3m_atoms, center_source_label="Se1"
    )
    legacy_canonical, _, _, legacy_motif_spread = _extract_quintuple_layers(
        legacy_atoms, center_source_label="Se1"
    )
    _, complete_p1, _, p1_motif_spread = _extract_quintuple_layers(p1_atoms)
    reconstructed = _translated_atoms(canonical, _CENTERING_TRANSLATIONS)
    image_shift_payload, image_shifts, image_integrality_error = _image_shifts(
        r3m_atoms, reconstructed
    )
    wrapped_reconstructed = tuple(
        replace(
            atom,
            fractional=tuple(
                float(value) for value in np.asarray(atom.fractional) - np.asarray(image_shift)
            ),
        )
        for atom, image_shift in zip(reconstructed, image_shifts, strict=True)
    )
    ql_crystal = _crystal_from_atoms(r3m, canonical, phase_id="bi2se3-single-ql")
    wrapped_ql_crystal = _crystal_from_atoms(
        r3m, wrapped_reconstructed, phase_id="bi2se3-single-ql-wrapped-cell"
    )
    unwrapped_ql_crystal = _crystal_from_atoms(
        r3m, reconstructed, phase_id="bi2se3-single-ql-complete-images"
    )
    coordinate_error = _periodic_coordinate_error(r3m_atoms, p1_atoms)
    legacy_coordinate_error = _periodic_coordinate_error(r3m_atoms, legacy_atoms)
    legacy_ql_coordinate_error = _periodic_coordinate_error(canonical, legacy_canonical)
    reconstruction_coordinate_error = _periodic_coordinate_error(r3m_atoms, reconstructed)
    p1_reconstruction_coordinate_error = _periodic_coordinate_error(reconstructed, p1_atoms)
    legacy_reconstruction_coordinate_error = _periodic_coordinate_error(reconstructed, legacy_atoms)
    basis_error = _maximum(r3m.direct_basis_A - p1_basis)
    legacy_basis_error = _maximum(r3m.direct_basis_A - legacy.direct_basis_A)

    expected_centers = np.asarray(_CENTERING_TRANSLATIONS)
    observed_centers = np.asarray(sorted(centers, key=lambda value: value[2]))
    center_error = _maximum(observed_centers - expected_centers)
    table = _read_vesta_table(table_path)
    hkl = np.asarray(table.hkl, dtype=np.float64)
    absent_hkl = np.asarray(_ABSENT_HKL, dtype=np.float64)
    proof_hkl = np.vstack((hkl, absent_hkl))

    integer_wavelength_A = np.resize(
        np.asarray((_WAVELENGTH_A, 1.1, 1.8, 1.3), dtype=np.float64), proof_hkl.shape[0]
    )
    integer_ql_amplitude = unit_cell_amplitude(
        ql_crystal, proof_hkl, integer_wavelength_A
    ).amplitude_e * _centering_factor(proof_hkl)
    integer_direct_amplitude = direct_atom_amplitudes(r3m, proof_hkl, integer_wavelength_A)
    integer_production_amplitude = unit_cell_amplitude(
        r3m, proof_hkl, integer_wavelength_A
    ).amplitude_e
    maximum_direct_amplitude_residual = _maximum(integer_ql_amplitude - integer_direct_amplitude)
    maximum_production_amplitude_residual = _maximum(
        integer_ql_amplitude - integer_production_amplitude
    )

    direct_r3m = _direct_amplitudes(r3m_atoms, r3m.direct_basis_A, proof_hkl)
    direct_p1 = _direct_amplitudes(p1_atoms, p1_basis, proof_hkl)
    direct_ql = _direct_amplitudes(reconstructed, r3m.direct_basis_A, proof_hkl)
    ql_amplitude = _direct_amplitudes(canonical, r3m.direct_basis_A, proof_hkl)
    analytic = ql_amplitude * _centering_factor(proof_hkl)
    table_rows = table.hkl.shape[0]
    error_ab = direct_r3m[:table_rows] - direct_p1[:table_rows]
    p1_absence_residual = direct_p1[table_rows:]
    error_ac = direct_r3m - direct_ql
    error_c_analytic = direct_ql - analytic

    continuous_hkl = np.asarray(
        [(0.0, 0.0, 0.37), (1.0, -1.0, 1.25), (-2.0, 1.0, -0.4)],
        dtype=np.float64,
    )
    continuous_a = _direct_amplitudes(complete_r3m, r3m.direct_basis_A, continuous_hkl)
    continuous_b = _direct_amplitudes(complete_p1, p1_basis, continuous_hkl)
    continuous_c = _direct_amplitudes(reconstructed, r3m.direct_basis_A, continuous_hkl)
    continuous_wrapped = _direct_amplitudes(r3m_atoms, r3m.direct_basis_A, continuous_hkl)
    continuous_analytic = _direct_amplitudes(
        canonical, r3m.direct_basis_A, continuous_hkl
    ) * _centering_factor(continuous_hkl)
    continuous_error = max(
        _maximum(continuous_a - continuous_b),
        _maximum(continuous_a - continuous_c),
        _maximum(continuous_c - continuous_analytic),
    )
    continuous_reconstruction_error = max(
        _maximum(continuous_a - continuous_c),
        _maximum(continuous_c - continuous_analytic),
    )
    continuous_wavelength_A = np.asarray((_WAVELENGTH_A, 1.1, 1.8), dtype=np.float64)
    continuous_image_phase_amplitude = direct_atom_amplitudes(
        wrapped_ql_crystal, continuous_hkl, continuous_wavelength_A
    )
    continuous_ql_production = unit_cell_amplitude(
        wrapped_ql_crystal, continuous_hkl, continuous_wavelength_A
    ).amplitude_e
    continuous_production = unit_cell_amplitude(
        r3m, continuous_hkl, continuous_wavelength_A
    ).amplitude_e
    continuous_without_image_phase = direct_atom_amplitudes(
        unwrapped_ql_crystal, continuous_hkl, continuous_wavelength_A
    )
    maximum_noninteger_image_shift_residual = max(
        _maximum(continuous_image_phase_amplitude - continuous_production),
        _maximum(continuous_ql_production - continuous_production),
    )
    image_shift_phase_sensitivity = _maximum(
        continuous_without_image_phase - continuous_image_phase_amplitude
    )
    p1_absence_bound = _coordinate_serialization_bound_e(
        r3m_atoms, r3m.direct_basis_A, absent_hkl, coordinate_error
    )
    p1_continuous_bound = _coordinate_serialization_bound_e(
        r3m_atoms, r3m.direct_basis_A, continuous_hkl, coordinate_error
    )
    p1_serialization_passed = bool(
        np.all(np.abs(p1_absence_residual) <= p1_absence_bound)
        and np.all(np.abs(continuous_a - continuous_b) <= p1_continuous_bound)
    )

    calculated = direct_r3m[:table_rows]
    amplitude_error = calculated - table.amplitude_e
    magnitude_error = np.abs(calculated) - table.magnitude_e
    reciprocal = 2.0 * np.pi * np.linalg.inv(r3m.direct_basis_A).T
    q_magnitude = np.linalg.norm(hkl @ reciprocal.T, axis=1)
    calculated_d = 2.0 * np.pi / q_magnitude
    d_error = calculated_d - table.d_A
    arcsin_argument = _WAVELENGTH_A / (2.0 * calculated_d)
    finite_two_theta = arcsin_argument <= 1.0
    calculated_two_theta = np.full(table.d_A.shape, np.nan)
    calculated_two_theta[finite_two_theta] = 2.0 * np.degrees(
        np.arcsin(arcsin_argument[finite_two_theta])
    )
    table_finite = np.isfinite(table.two_theta_deg)
    two_theta_mask_disagreements = int(np.count_nonzero(finite_two_theta != table_finite))
    two_theta_error = calculated_two_theta[table_finite] - table.two_theta_deg[table_finite]

    generated, geometric_candidate_count = _reflection_set(r3m.direct_basis_A, _D_MIN_A)
    generated_reflections = set(generated)
    table_reflections = {tuple(int(value) for value in row) for row in table.hkl}
    hkl_disagreements = len(generated_reflections.symmetric_difference(table_reflections))
    calculated_multiplicity = _multiplicities(r3m_path, table.hkl)
    multiplicity_disagreements = int(
        np.count_nonzero(calculated_multiplicity != table.multiplicity)
    )
    absent_factor = _centering_factor(absent_hkl)
    absent_amplitude = direct_ql[-len(_ABSENT_HKL) :]
    absence_disagreements = int(
        np.count_nonzero(np.abs(absent_factor) > 1e-12)
        + np.count_nonzero(np.abs(absent_amplitude) > 1e-10)
        + sum((-int(h) + int(k) + int(ell)) % 3 != 0 for h, k, ell in table.hkl)
    )

    row_003 = int(np.flatnonzero(np.all(table.hkl == (0, 0, 3), axis=1))[0])
    target_003 = 104.983515 + 14.466758j
    hkl_003 = np.asarray(((0.0, 0.0, 3.0),), dtype=np.float64)
    wavelength_003_A = np.asarray((_WAVELENGTH_A,), dtype=np.float64)
    f003_ql = (
        unit_cell_amplitude(ql_crystal, hkl_003, wavelength_003_A).amplitude_e[0]
        * _centering_factor(hkl_003)[0]
    )
    f003_direct = direct_atom_amplitudes(r3m, hkl_003, wavelength_003_A)[0]
    f003_production = unit_cell_amplitude(r3m, hkl_003, wavelength_003_A).amplitude_e[0]
    p1_003 = direct_p1[row_003]
    ql_003 = direct_ql[row_003]
    production_result = unit_cell_amplitude(r3m, table.hkl, np.full(table_rows, _WAVELENGTH_A))
    production = production_result.amplitude_e
    production_error = production - table.amplitude_e
    production_003 = production[row_003]

    reconstruction_tolerance = tolerances["bi2se3_reconstruction_e"]
    reconstruction_passed = all(
        np.allclose(first, second, **reconstruction_tolerance)
        for first, second in (
            (direct_r3m[:table_rows], direct_p1[:table_rows]),
            (direct_r3m, direct_ql),
            (direct_ql, analytic),
            (continuous_a, continuous_c),
            (continuous_c, continuous_analytic),
        )
    )
    direct_tolerance = tolerances["direct_atom_e"]
    amplitude_reconstruction_passed = all(
        np.allclose(first, second, **direct_tolerance)
        for first, second in (
            (integer_ql_amplitude, integer_direct_amplitude),
            (integer_ql_amplitude, integer_production_amplitude),
            (integer_direct_amplitude, integer_production_amplitude),
            (continuous_image_phase_amplitude, continuous_production),
            (continuous_ql_production, continuous_production),
            (np.asarray((f003_ql,)), np.asarray((f003_direct,))),
            (np.asarray((f003_ql,)), np.asarray((f003_production,))),
        )
    )
    relative_magnitude_error = np.abs(magnitude_error) / np.maximum(table.magnitude_e, 1.0)
    rms_complex_error = float(np.sqrt(np.mean(np.abs(amplitude_error) ** 2)))
    table_passed = (
        table_rows == int(metadata["row_count"]) == 206
        and np.isclose(
            float(metadata["inferred_lambda_angstrom"]), _WAVELENGTH_A, rtol=0.0, atol=0.0
        )
        and np.isclose(float(metadata["dmin_angstrom"]), _D_MIN_A, rtol=0.0, atol=0.0)
        and table.amplitude_e[row_003] == target_003
        and _maximum(amplitude_error.real) <= tolerances["bi2se3_vesta_component_e"]["atol"]
        and _maximum(amplitude_error.imag) <= tolerances["bi2se3_vesta_component_e"]["atol"]
        and rms_complex_error <= tolerances["bi2se3_vesta_rms_e"]["atol"]
        and _maximum(magnitude_error) <= tolerances["bi2se3_vesta_magnitude_e"]["atol"]
        and _maximum(relative_magnitude_error)
        <= tolerances["bi2se3_vesta_relative_magnitude"]["atol"]
        and _maximum(d_error) <= tolerances["bi2se3_vesta_d_A"]["atol"]
        and _maximum(two_theta_error) <= tolerances["bi2se3_vesta_two_theta_deg"]["atol"]
        and two_theta_mask_disagreements == 0
        and bool(np.all(table.intensity_is_nan))
        and hkl_disagreements == 0
        and multiplicity_disagreements == 0
        and absence_disagreements == 0
        and max(
            abs(calculated[row_003].real - target_003.real),
            abs(calculated[row_003].imag - target_003.imag),
        )
        <= tolerances["vesta_003_component_e"]["atol"]
    )

    mutations = _mutation_results(
        canonical,
        r3m.direct_basis_A,
        proof_hkl,
        continuous_hkl,
        continuous_wrapped,
        continuous_c,
    )
    mutation_signature = tuple(
        (str(item["mutation_id"]), str(item["expected_first_stage"])) for item in mutations
    )
    mutations_passed = mutation_signature == _REQUIRED_MUTATION_STAGES and all(
        bool(item["detected"]) for item in mutations
    )
    weak_row = int(np.argmin(np.abs(calculated)))
    strong_row = int(np.argmax(np.abs(calculated)))
    atom_counts = {
        "Bi": sum(atom.element == "Bi" for atom in reconstructed),
        "Se": sum(atom.element == "Se" for atom in reconstructed),
        "total": len(reconstructed),
    }
    source_multiplicities = {
        label: sum(atom.source_label == label for atom in r3m_atoms)
        for label in sorted({atom.source_label for atom in r3m_atoms})
    }
    p1_motif_passed = bool(p1_motif_spread <= 1e-12 + 4.0 * np.finfo(np.float64).eps)
    structural_passed = (
        r3m.spacegroup_hm == "R -3 m:H"
        and atom_counts == {"Bi": 6, "Se": 9, "total": 15}
        and source_multiplicities == {"Bi": 6, "Se1": 3, "Se2": 6}
        and canonical[2].source_label == "Se1"
        and canonical[0].source_label == canonical[-1].source_label == "Se2"
        and coordinate_error <= 1e-12
        and legacy_coordinate_error <= 1e-12
        and legacy_ql_coordinate_error <= 1e-12
        and legacy_reconstruction_coordinate_error <= 1e-12
        and reconstruction_coordinate_error <= 1e-12
        and p1_reconstruction_coordinate_error <= 1e-12
        and basis_error <= 1e-12
        and legacy_basis_error <= 1e-12
        and center_error <= 1e-12
        and image_integrality_error <= 1e-12
        and r3m_motif_spread <= 1e-12
        and legacy_motif_spread <= 1e-12
        and p1_motif_passed
    )
    single_ql_passed = (
        structural_passed
        and amplitude_reconstruction_passed
        and image_shift_phase_sensitivity > direct_tolerance["atol"]
        and mutations_passed
    )
    passed = single_ql_passed and reconstruction_passed and p1_serialization_passed and table_passed

    return {
        "status": "PASS" if passed else "FAIL",
        "terminology": "Bi2Se3 R-3m; complete Se-Bi-Se-Bi-Se quintuple layer (QL)",
        "normalization": "raw complex electron amplitude; no scaling, rounding, pruning, or r_e^2",
        "wavelength_A": _WAVELENGTH_A,
        "d_min_A": _D_MIN_A,
        "single_ql_reconstruction": {
            "status": "PASS" if single_ql_passed else "FAIL",
            "ql_count": len(_CENTERING_TRANSLATIONS),
            "atoms_per_ql": len(canonical),
            "stoichiometry": {
                element: sum(atom.element == element for atom in canonical)
                for element in ("Bi", "Se")
            },
            "source_label_roles": {
                "center": canonical[2].source_label,
                "outer": canonical[0].source_label,
            },
            "site_coverage_exact": (
                len(reconstructed) == len(r3m_atoms) and reconstruction_coordinate_error <= 1e-12
            ),
            "periodic_boundary_unique": (
                len(image_shift_payload) == len(r3m_atoms) == len(reconstructed)
                and image_integrality_error <= 1e-12
            ),
            "ql_property_identity_exact": (
                r3m_motif_spread <= 1e-12 and legacy_motif_spread <= 1e-12 and p1_motif_passed
            ),
            "centering_translations": [list(value) for value in _CENTERING_TRANSLATIONS],
            "maximum_coordinate_residual_fractional": reconstruction_coordinate_error,
            "legacy_maximum_coordinate_residual_fractional": max(
                legacy_coordinate_error,
                legacy_ql_coordinate_error,
                legacy_reconstruction_coordinate_error,
            ),
            "p1_maximum_coordinate_residual_fractional": max(
                coordinate_error, p1_reconstruction_coordinate_error
            ),
            "integer_reflection_events": int(proof_hkl.shape[0]),
            "wavelengths_A": sorted({float(value) for value in integer_wavelength_A}),
            "maximum_direct_amplitude_residual_e": maximum_direct_amplitude_residual,
            "maximum_production_amplitude_residual_e": maximum_production_amplitude_residual,
            "F003_ql_e": _complex_pair(f003_ql),
            "F003_direct_e": _complex_pair(f003_direct),
            "F003_production_e": _complex_pair(f003_production),
            "maximum_noninteger_L_image_shift_residual_e": (
                maximum_noninteger_image_shift_residual
            ),
            "noninteger_image_phase": "u=w+n; exp(2pi*i*H.u)*exp(-2pi*i*H.n)",
            "ignored_image_shift_phase_difference_e": image_shift_phase_sensitivity,
            "mutations": mutations,
        },
        "canonical_QL": {
            "gauge": "central Se at (0,0,0); bottom-to-top; xy offsets modulo one; signed z; positive phase",
            "atoms": [
                {"role": role, **payload}
                for role, payload in zip(
                    ("outer_lower", "lower", "center", "upper", "outer_upper"),
                    _coordinate_payload(canonical),
                    strict=True,
                )
            ],
            "center_translations": [list(value) for value in _CENTERING_TRANSLATIONS],
            "layer_repeat_A": float(np.linalg.norm(r3m.direct_basis_A[:, 2]) / 3.0),
            "QL_thickness_A": float(
                (canonical[-1].fractional[2] - canonical[0].fractional[2])
                * np.linalg.norm(r3m.direct_basis_A[:, 2])
            ),
        },
        "structure": {
            "source_multiplicities": source_multiplicities,
            "reconstructed_atom_counts": atom_counts,
            "direct_basis_A": r3m.direct_basis_A.tolist(),
            "expanded_R3m_coordinates": _coordinate_payload(r3m_atoms),
            "reconstructed_complete_coordinates": _coordinate_payload(reconstructed),
            "integer_image_shifts": image_shift_payload,
            "maximum_image_integrality_error_fractional": image_integrality_error,
            "R3m_vs_P1_coordinate_error_fractional": coordinate_error,
            "R3m_vs_legacy_coordinate_error_fractional": legacy_coordinate_error,
            "QL_vs_legacy_QL_coordinate_error_fractional": legacy_ql_coordinate_error,
            "R3m_vs_reconstruction_coordinate_error_fractional": reconstruction_coordinate_error,
            "reconstruction_vs_P1_coordinate_error_fractional": (
                p1_reconstruction_coordinate_error
            ),
            "reconstruction_vs_legacy_coordinate_error_fractional": (
                legacy_reconstruction_coordinate_error
            ),
            "R3m_vs_P1_basis_error_A": basis_error,
            "R3m_vs_legacy_basis_error_A": legacy_basis_error,
            "center_translation_error_fractional": center_error,
            "R3m_motif_spread_fractional": r3m_motif_spread,
            "legacy_motif_spread_fractional": legacy_motif_spread,
            "P1_motif_spread_fractional": p1_motif_spread,
        },
        "reconstruction": {
            "events": int(proof_hkl.shape[0]),
            "continuous_gauge_events": continuous_hkl.tolist(),
            "maximum_A_vs_B_error_e": _maximum(error_ab),
            "expanded_P1_decimal_absence_floor_e": _maximum(p1_absence_residual),
            "expanded_P1_absence_serialization_bound_e": _maximum(p1_absence_bound),
            "maximum_A_vs_C_error_e": _maximum(error_ac),
            "maximum_C_vs_analytic_error_e": _maximum(error_c_analytic),
            "maximum_continuous_complete_gauge_error_e": continuous_error,
            "maximum_continuous_P1_serialization_bound_e": _maximum(p1_continuous_bound),
            "maximum_continuous_A_C_analytic_error_e": continuous_reconstruction_error,
            "wrapped_vs_complete_continuous_difference_e": _maximum(
                continuous_wrapped - continuous_a
            ),
            "P1_serialization_bound_passed": p1_serialization_passed,
            "centering_formula": ("1+exp(2pi*i*(2h+k+l)/3)+exp(2pi*i*(h+2k+2l)/3)"),
            "absent_hkl": [list(value) for value in _ABSENT_HKL],
            "maximum_absent_amplitude_e": _maximum(absent_amplitude),
            "maximum_absent_centering_factor": _maximum(absent_factor),
            "absence_disagreements": absence_disagreements,
        },
        "vesta_parity": {
            "factor_convention": (
                "proof-only Waasmaier-Kirfel f0 + tracked Cu Kalpha1 f' + i*f'' "
                "+ VESTA nuclear Thomson term"
            ),
            "factor_terms_e": {
                element: {
                    "f_prime": f_prime,
                    "f_double_prime": f_double_prime,
                    "f_NT": nuclear_thomson,
                }
                for element, f_prime, f_double_prime, nuclear_thomson in _VESTA_CORRECTIONS_E
            },
            "factor_sources": {
                "VESTA_formula": "https://jp-minerals.org/vesta/en/doc/VESTAch14.html",
                "Waasmaier_Kirfel": "https://doi.org/10.1107/S0108767394013292",
            },
            "row_count": int(table_rows),
            "generated_reflection_count": len(generated_reflections),
            "geometric_candidate_count": geometric_candidate_count,
            "systematically_absent_candidate_count": (
                geometric_candidate_count - len(generated_reflections)
            ),
            "hkl_disagreements": hkl_disagreements,
            "multiplicity_disagreements": multiplicity_disagreements,
            "multiplicity_distribution": {
                str(value): int(np.count_nonzero(calculated_multiplicity == value))
                for value in np.unique(calculated_multiplicity)
            },
            "finite_two_theta_count": int(np.count_nonzero(table_finite)),
            "nan_two_theta_count": int(np.count_nonzero(~table_finite)),
            "nan_intensity_count": int(np.count_nonzero(table.intensity_is_nan)),
            "two_theta_mask_disagreements": two_theta_mask_disagreements,
            "maximum_real_residual_e": _maximum(amplitude_error.real),
            "maximum_imaginary_residual_e": _maximum(amplitude_error.imag),
            "RMS_complex_residual_e": rms_complex_error,
            "maximum_magnitude_residual_e": _maximum(magnitude_error),
            "maximum_relative_magnitude_residual_above_1e": _maximum(relative_magnitude_error),
            "maximum_d_residual_A": _maximum(d_error),
            "maximum_finite_two_theta_residual_deg": _maximum(two_theta_error),
            "absence_disagreements": absence_disagreements,
            "intensity_column_classification": "NO_ORACLE; VESTA export contains only NaN",
            "weak_case": {
                "hkl": table.hkl[weak_row].tolist(),
                "amplitude_e": _complex_pair(calculated[weak_row]),
            },
            "strong_case": {
                "hkl": table.hkl[strong_row].tolist(),
                "amplitude_e": _complex_pair(calculated[strong_row]),
            },
        },
        "F003_e": {
            "direct_R3m": _complex_pair(calculated[row_003]),
            "expanded_P1": _complex_pair(p1_003),
            "canonical_QL_motif": _complex_pair(ql_amplitude[row_003]),
            "centering_factor": _complex_pair(_centering_factor(hkl[[row_003]])[0]),
            "QL_reconstructed_cell": _complex_pair(ql_003),
            "VESTA_target": _complex_pair(table.amplitude_e[row_003]),
            "production_xraydb": _complex_pair(production_003),
        },
        "production_xraydb": {
            "classification": "independent table comparison; no VESTA equality requirement",
            "provenance": production_result.provenance,
            "maximum_complex_difference_e": _maximum(production_error),
            "maximum_real_difference_e": _maximum(production_error.real),
            "maximum_imaginary_difference_e": _maximum(production_error.imag),
            "RMS_complex_difference_e": float(np.sqrt(np.mean(np.abs(production_error) ** 2))),
            "F003_difference_e": _complex_pair(production_003 - table.amplitude_e[row_003]),
        },
        "handoff_relations": [
            {
                "comparison": "direct_R3m versus expanded_P1",
                "relation": "independent analytic/numerical proof",
            },
            {
                "comparison": "direct_R3m versus single_QL reconstruction",
                "relation": "independent material reconstruction proof",
                "observable": "15-atom conventional cell generated only from one canonical 5-atom QL",
            },
            {
                "comparison": "single_QL versus VESTA",
                "relation": "VESTA_PARITY",
                "observable": "15-atom conventional cell generated only from one canonical 5-atom QL",
            },
            {
                "comparison": "production XrayDB versus VESTA",
                "relation": "independent table comparison",
            },
        ],
        "mutation_gate": {
            "required": len(_REQUIRED_MUTATION_STAGES),
            "detected": sum(bool(item["detected"]) for item in mutations),
            "exact_id_stage_set": mutation_signature == _REQUIRED_MUTATION_STAGES,
        },
        "mutations": mutations,
        "limitations": [
            "noninteger-L QL reconstruction explicitly applies each complete-image integer shift before comparison with the wrapped production cell",
            "the P1 file serializes fractional coordinates to 12 decimals; forbidden and continuous residuals are gated against the analytic coordinate-serialization bound",
            "VESTA intensity values are NO_ORACLE because all 206 exported entries are NaN",
        ],
        "t05_transition_implementation_called": False,
    }
