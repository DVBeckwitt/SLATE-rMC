"""Proof-only exact-2H and relaxed-target PbI2 polytype atom sums."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import xraydb
from numpy.typing import NDArray

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.materials import CrystalStructure, read_crystal
from rasim_next.materials.optics import HC_EV_A
from rasim_next.ordered.amplitudes import unit_cell_amplitude
from rasim_next.ordered.motifs import (
    MotifAtom,
    PbI2Motif,
    extract_pbi2_motifs,
    pbi2_layer_amplitudes,
)
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog

_REGISTRIES = (
    ("A", (0.0, 0.0)),
    ("B", (1.0 / 3.0, 2.0 / 3.0)),
    ("C", (2.0 / 3.0, 1.0 / 3.0)),
)
_EVENT_ID = (41, 7, 93, 18)
_EVENT_H = (0, 1, 0, 2)
_EVENT_K = (0, 0, 1, -1)
_EVENT_QZ_AINV = (0.37, 0.113, 0.251, 0.509)
_EVENT_WAVELENGTH_A = (1.540592925, 1.1, 1.8, 1.3)
_IDEAL_PARENT_TOPOLOGIES = (
    ("2H", (("plus", "A"),)),
    ("4H+", (("plus", "A"), ("minus", "B"))),
    ("4H-", (("plus", "A"), ("minus", "C"))),
    ("6H+", (("plus", "A"), ("plus", "B"), ("plus", "C"))),
    ("6H-", (("plus", "A"), ("plus", "C"), ("plus", "B"))),
)
_PERIOD_MULTIPLES = (1, 2, 5)


def _registry(registry_name: str) -> tuple[float, float]:
    for name, offset in _REGISTRIES:
        if name == registry_name:
            return offset
    raise ValueError(f"unknown registry {registry_name!r}")


@dataclass(frozen=True, slots=True)
class _ExplicitAtom:
    element: str
    charge: int
    occupancy: float
    u_iso_A2: float | None
    position_A: tuple[float, float, float]


def _complex_pairs(values: NDArray[np.complex128]) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def _q_vectors(
    crystal: CrystalStructure,
    h: NDArray[np.int32],
    k: NDArray[np.int32],
    qz_Ainv: NDArray[np.float64],
) -> NDArray[np.float64]:
    lattice = ReciprocalLattice.from_crystal(crystal)
    c_axis = crystal.direct_basis_A[:, 2] / np.linalg.norm(crystal.direct_basis_A[:, 2])
    return (
        h[:, None] * lattice.basis_Ainv[:, 0]
        + k[:, None] * lattice.basis_Ainv[:, 1]
        + qz_Ainv[:, None] * c_axis
    )


def _direct_atom_sum(
    atoms: tuple[_ExplicitAtom, ...],
    q_vectors_Ainv: NDArray[np.float64],
    wavelength_A: NDArray[np.float64],
    *,
    unknown_u_iso_A2: float,
) -> NDArray[np.complex128]:
    """Scalar direct enumeration independent of every production amplitude helper."""

    q_magnitude = np.linalg.norm(q_vectors_Ainv, axis=1)
    energy_eV = HC_EV_A / wavelength_A
    unique_energy, inverse = np.unique(energy_eV, return_inverse=True)
    factors: list[NDArray[np.complex128]] = []
    for atom in atoms:
        requested = atom.element
        if atom.charge:
            sign = "+" if atom.charge > 0 else "-"
            ionic = f"{atom.element}{abs(atom.charge)}{sign}"
            if ionic in xraydb.f0_ions(atom.element):
                requested = ionic
        f0 = np.asarray(xraydb.f0(requested, q_magnitude / (4.0 * np.pi)), dtype=np.float64)
        f1 = np.asarray(
            [xraydb.f1_chantler(atom.element, float(energy)) for energy in unique_energy]
        )[inverse]
        f2 = np.asarray(
            [xraydb.f2_chantler(atom.element, float(energy)) for energy in unique_energy]
        )[inverse]
        factors.append(np.asarray(f0 + f1 + 1.0j * f2, dtype=np.complex128))

    result = np.empty(q_vectors_Ainv.shape[0], dtype=np.complex128)
    for event_index, q_vector in enumerate(q_vectors_Ainv):
        contributions: list[complex] = []
        for atom_index, atom in enumerate(atoms):
            u_iso = unknown_u_iso_A2 if atom.u_iso_A2 is None else atom.u_iso_A2
            damping = math.exp(-0.5 * u_iso * q_magnitude[event_index] ** 2)
            phase = cmath.exp(1.0j * float(np.dot(q_vector, atom.position_A)))
            contributions.append(
                atom.occupancy * factors[atom_index][event_index] * damping * phase
            )
        result[event_index] = complex(
            math.fsum(value.real for value in contributions),
            math.fsum(value.imag for value in contributions),
        )
    return result


def _reflected(atoms: tuple[MotifAtom, MotifAtom, MotifAtom]) -> tuple[MotifAtom, ...]:
    return tuple(
        replace(
            atom,
            fractional_offset=(
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                -atom.fractional_offset[2],
            ),
        )
        for atom in atoms
    )


def _source_orientations(
    source: CrystalStructure,
) -> tuple[dict[str, tuple[MotifAtom, ...]], PbI2Motif]:
    motifs = extract_pbi2_motifs(source)
    if len(motifs) != 1:
        raise ValueError("exact 2H source must expand to one complete PbI2 layer")
    actual = motifs[0]
    opposite = _reflected(actual.atoms)
    if actual.orientation == "minus":
        return {"plus": opposite, "minus": actual.atoms}, actual
    return {"plus": actual.atoms, "minus": opposite}, actual


def _source_layer_atoms(
    source: CrystalStructure, atoms: tuple[MotifAtom, ...]
) -> tuple[_ExplicitAtom, ...]:
    return tuple(
        _ExplicitAtom(
            element=atom.element,
            charge=atom.charge,
            occupancy=atom.occupancy,
            u_iso_A2=atom.u_iso_A2,
            position_A=tuple(
                float(value) for value in source.direct_basis_A @ np.asarray(atom.fractional_offset)
            ),
        )
        for atom in atoms
    )


def _motif_coordinates(
    source: CrystalStructure, atoms: tuple[MotifAtom, ...]
) -> list[dict[str, object]]:
    return [
        {
            "element": atom.element,
            "fractional_offset": [float(value) for value in atom.fractional_offset],
            "cartesian_offset_A": [
                float(value) for value in source.direct_basis_A @ np.asarray(atom.fractional_offset)
            ],
        }
        for atom in atoms
    ]


def _ideal_atoms(
    source: CrystalStructure,
    orientations: dict[str, tuple[MotifAtom, ...]],
    topology: tuple[tuple[str, str], ...],
    layer_repeat_A: float,
) -> tuple[tuple[_ExplicitAtom, ...], list[dict[str, object]]]:
    c_axis = source.direct_basis_A[:, 2] / np.linalg.norm(source.direct_basis_A[:, 2])
    atoms: list[_ExplicitAtom] = []
    coordinates: list[dict[str, object]] = []
    for layer_index, (orientation, registry_name) in enumerate(topology):
        registry = np.asarray(_registry(registry_name))
        depth_A = layer_index * layer_repeat_A
        for atom in orientations[orientation]:
            fractional_xy = registry + np.asarray(atom.fractional_offset[:2])
            position = source.direct_basis_A[:, :2] @ fractional_xy + c_axis * (
                depth_A + atom.fractional_offset[2] * layer_repeat_A
            )
            atoms.append(
                _ExplicitAtom(
                    atom.element,
                    atom.charge,
                    atom.occupancy,
                    atom.u_iso_A2,
                    tuple(float(value) for value in position),
                )
            )
            coordinates.append(
                {
                    "layer_index": layer_index,
                    "orientation": orientation,
                    "registry": registry_name,
                    "element": atom.element,
                    "fractional_xy": [float(value) for value in fractional_xy],
                    "z_A": float(depth_A + atom.fractional_offset[2] * layer_repeat_A),
                }
            )
    return tuple(atoms), coordinates


def _coherent_layers(
    f_plus: NDArray[np.complex128],
    f_minus: NDArray[np.complex128],
    topology: tuple[tuple[str, str], ...],
    repeat_A: float,
    event_h: NDArray[np.int32],
    event_k: NDArray[np.int32],
    event_qz_Ainv: NDArray[np.float64],
) -> NDArray[np.complex128]:
    amplitude = np.zeros(event_h.size, dtype=np.complex128)
    for layer_index, (orientation, registry_name) in enumerate(topology):
        registry = _registry(registry_name)
        phase = np.exp(
            1.0j
            * (
                2.0 * np.pi * (event_h * registry[0] + event_k * registry[1])
                + event_qz_Ainv * layer_index * repeat_A
            )
        )
        amplitude += (f_plus if orientation == "plus" else f_minus) * phase
    return amplitude


def _complete_target_atoms(
    crystal: CrystalStructure, motifs: tuple[PbI2Motif, ...]
) -> tuple[tuple[_ExplicitAtom, ...], list[dict[str, object]]]:
    atoms: list[_ExplicitAtom] = []
    coordinates: list[dict[str, object]] = []
    for layer_index, motif in enumerate(motifs):
        pb_atom = next(atom for atom in motif.atoms if atom.element == "Pb")
        center = np.asarray(crystal.sites[pb_atom.site_index].fractional)
        for atom in motif.atoms:
            fractional = center + np.asarray(atom.fractional_offset)
            position = crystal.direct_basis_A @ fractional
            atoms.append(
                _ExplicitAtom(
                    atom.element,
                    atom.charge,
                    atom.occupancy,
                    atom.u_iso_A2,
                    tuple(float(value) for value in position),
                )
            )
            coordinates.append(
                {
                    "site_index": atom.site_index,
                    "layer_index": layer_index,
                    "element": atom.element,
                    "fractional": [float(value) for value in fractional],
                    "cartesian_A": [float(value) for value in position],
                }
            )
    return tuple(atoms), coordinates


def _canonical_target_atoms(
    crystal: CrystalStructure,
) -> tuple[tuple[_ExplicitAtom, ...], list[dict[str, object]]]:
    atoms: list[_ExplicitAtom] = []
    coordinates: list[dict[str, object]] = []
    for site_index, site in enumerate(crystal.sites):
        position = crystal.direct_basis_A @ np.asarray(site.fractional)
        atoms.append(
            _ExplicitAtom(
                site.element,
                site.charge,
                site.occupancy,
                site.u_iso_A2,
                tuple(float(value) for value in position),
            )
        )
        coordinates.append(
            {
                "site_index": site_index,
                "element": site.element,
                "fractional": [float(value) for value in site.fractional],
                "cartesian_A": [float(value) for value in position],
            }
        )
    return tuple(atoms), coordinates


def _registry_name(fractional_xy: tuple[float, float]) -> str:
    for name, registry in _REGISTRIES:
        difference = np.asarray(fractional_xy) - np.asarray(registry)
        difference -= np.rint(difference)
        if np.max(np.abs(difference)) <= 3e-5:
            return name
    raise ValueError(f"Pb registry {fractional_xy!r} is not A, B, or C")


def _target_topology(
    crystal: CrystalStructure, motifs: tuple[PbI2Motif, ...]
) -> tuple[tuple[str, str], ...]:
    topology: list[tuple[str, str]] = []
    for motif in motifs:
        pb_atom = next(atom for atom in motif.atoms if atom.element == "Pb")
        pb = crystal.sites[pb_atom.site_index]
        coordination_orientation = "+" if motif.orientation == "minus" else "-"
        topology.append((coordination_orientation, _registry_name(pb.fractional[:2])))
    return tuple(topology)


def _topology_payload(topology: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [
        {"coordination_orientation": orientation, "registry": registry}
        for orientation, registry in topology
    ]


def _expected_image_shifts(
    polytype: str,
) -> tuple[tuple[int, tuple[int, int, int]], ...]:
    if polytype == "2H":
        return ((0, (0, 0, 0)), (1, (0, 0, 0)), (2, (0, 0, -1)))
    if polytype == "4H":
        return (
            (0, (0, 0, 0)),
            (1, (0, 0, 0)),
            (2, (1, 1, 0)),
            (3, (1, 1, 0)),
            (4, (0, 1, -1)),
            (5, (1, 0, 0)),
        )
    if polytype == "6H":
        return (
            (0, (0, 0, 0)),
            (1, (0, 0, 0)),
            (2, (0, 0, 0)),
            (3, (1, 1, 0)),
            (4, (1, 1, 0)),
            (5, (0, 1, 0)),
            (6, (0, 0, 0)),
            (7, (0, 0, -1)),
            (8, (1, 0, 0)),
        )
    raise ValueError(f"unsupported tracked polytype {polytype!r}")


def _image_shift_evidence(
    polytype: str,
    canonical_coordinates: list[dict[str, object]],
    layer_coordinates: list[dict[str, object]],
) -> tuple[bool, float, list[dict[str, object]]]:
    canonical_by_site = {
        int(coordinate["site_index"]): np.asarray(coordinate["fractional"], dtype=np.float64)
        for coordinate in canonical_coordinates
    }
    observed: list[tuple[int, tuple[int, int, int]]] = []
    integrality_error = 0.0
    for coordinate in layer_coordinates:
        site_index = int(coordinate["site_index"])
        delta = (
            np.asarray(coordinate["fractional"], dtype=np.float64) - canonical_by_site[site_index]
        )
        rounded = np.rint(delta)
        integrality_error = max(
            integrality_error, float(np.max(np.abs(delta - rounded), initial=0.0))
        )
        observed.append((site_index, tuple(int(value) for value in rounded)))
    observed.sort()
    expected = _expected_image_shifts(polytype)
    passed = (
        len(canonical_by_site) == len(layer_coordinates)
        and tuple(observed) == expected
        and integrality_error <= 1e-12
    )
    payload = [
        {"site_index": site_index, "integer_image_shift": list(shift)}
        for site_index, shift in observed
    ]
    return passed, integrality_error, payload


def run_pbi2_polytype_proof(root: str) -> dict[str, object]:
    """Return integration-ready raw amplitudes without calling any T05 implementation."""

    structure_root = Path(root) / "examples" / "pbi2" / "structures"
    source = read_crystal(structure_root / "PbI2_2H.cif", phase_id="pbi2-2h")
    orientations, source_motif = _source_orientations(source)
    event_id = np.asarray(_EVENT_ID, dtype=np.int64)
    event_h = np.asarray(_EVENT_H, dtype=np.int32)
    event_k = np.asarray(_EVENT_K, dtype=np.int32)
    event_qz_Ainv = np.asarray(_EVENT_QZ_AINV, dtype=np.float64)
    event_wavelength_A = np.asarray(_EVENT_WAVELENGTH_A, dtype=np.float64)
    source_repeat_A = float(np.linalg.norm(source.direct_basis_A[:, 2]))
    lattice = ReciprocalLattice.from_crystal(source)
    catalog = build_rod_catalog(
        source,
        h_bounds=(int(np.min(event_h)), int(np.max(event_h))),
        k_bounds=(int(np.min(event_k)), int(np.max(event_k))),
    )
    rod_by_hk = {
        (int(h), int(k)): int(rod_id)
        for rod_id, h, k in zip(catalog.rod_id, catalog.h, catalog.k, strict=True)
    }
    query = RodQueryBatch(
        event_id=event_id,
        rod_id=np.asarray(
            [rod_by_hk[(int(h), int(k))] for h, k in zip(event_h, event_k, strict=True)],
            dtype=np.int64,
        ),
        phase_id=(source.phase_id,) * event_id.size,
        h=event_h,
        k=event_k,
        qz_Ainv=event_qz_Ainv,
        l_coordinate=event_qz_Ainv / lattice.basis_Ainv[2, 2],
        wavelength_A=event_wavelength_A,
    )
    layer = pbi2_layer_amplitudes(source, query, unknown_u_iso_A2=0.0)
    q_vectors = _q_vectors(source, event_h, event_k, event_qz_Ainv)
    direct_plus = _direct_atom_sum(
        _source_layer_atoms(source, orientations["plus"]),
        q_vectors,
        event_wavelength_A,
        unknown_u_iso_A2=0.0,
    )
    direct_minus = _direct_atom_sum(
        _source_layer_atoms(source, orientations["minus"]),
        q_vectors,
        event_wavelength_A,
        unknown_u_iso_A2=0.0,
    )
    layer_error = max(
        float(np.max(np.abs(layer.f_plus - direct_plus))),
        float(np.max(np.abs(layer.f_minus - direct_minus))),
    )

    ideal_direct_amplitudes: dict[tuple[str, int], NDArray[np.complex128]] = {}
    ideal_hands: dict[str, object] = {}
    ideal_hand_names = {
        "4H+": "4H_plus",
        "4H-": "4H_minus",
        "6H+": "6H_plus",
        "6H-": "6H_minus",
    }
    ideal_error = 0.0
    for parent, period_topology in _IDEAL_PARENT_TOPOLOGIES:
        for period_multiple in _PERIOD_MULTIPLES:
            topology = period_topology * period_multiple
            atoms, coordinates = _ideal_atoms(source, orientations, topology, source_repeat_A)
            direct = _direct_atom_sum(atoms, q_vectors, event_wavelength_A, unknown_u_iso_A2=0.0)
            coherent = _coherent_layers(
                layer.f_plus,
                layer.f_minus,
                topology,
                source_repeat_A,
                event_h,
                event_k,
                event_qz_Ainv,
            )
            error = float(np.max(np.abs(direct - coherent)))
            ideal_error = max(ideal_error, error)
            ideal_direct_amplitudes[(parent, period_multiple)] = direct
            output_name = ideal_hand_names.get(parent)
            if output_name is not None and period_multiple == 1:
                ideal_hands[output_name] = {
                    "topology_t04": [list(value) for value in period_topology],
                    "direct_amplitude_e": _complex_pairs(direct),
                    "coherent_f_layer_amplitude_e": _complex_pairs(coherent),
                    "maximum_factorization_error_e": error,
                    "explicit_coordinates": coordinates,
                }

    required_topologies = {
        "2H": (("+", "A"),),
        "4H": (("-", "B"), ("+", "C")),
        "6H": (("-", "A"), ("-", "C"), ("-", "B")),
    }
    opposite_topologies = {
        "2H": (("+", "A"),),
        "4H": (("-", "B"), ("+", "A")),
        "6H": (("-", "A"), ("-", "B"), ("-", "C")),
    }
    targets: dict[str, object] = {}
    topology_passed = True
    target_crosscheck_error = 0.0
    integer_gauge_error = 0.0
    image_integrality_error = 0.0
    image_gauge_passed = True
    arbitrary_qz_image_effect_passed = True
    for polytype, filename in (
        ("2H", "PbI2_2H.cif"),
        ("4H", "PbI2_4H.cif"),
        ("6H", "PbI2_6H.cif"),
    ):
        target = read_crystal(structure_root / filename, phase_id=f"pbi2-{polytype.lower()}")
        motifs = extract_pbi2_motifs(target)
        canonical_atoms, canonical_coordinates = _canonical_target_atoms(target)
        layer_atoms, layer_coordinates = _complete_target_atoms(target, motifs)
        target_q = _q_vectors(target, event_h, event_k, event_qz_Ainv)
        canonical_direct = _direct_atom_sum(
            canonical_atoms,
            target_q,
            event_wavelength_A,
            unknown_u_iso_A2=0.0,
        )
        layer_direct = _direct_atom_sum(
            layer_atoms,
            target_q,
            event_wavelength_A,
            unknown_u_iso_A2=0.0,
        )
        target_lattice = ReciprocalLattice.from_crystal(target)
        target_hkl = np.column_stack(
            (event_h, event_k, event_qz_Ainv / target_lattice.basis_Ainv[2, 2])
        )
        production = unit_cell_amplitude(
            target,
            target_hkl,
            event_wavelength_A,
            unknown_u_iso_A2=0.0,
        ).amplitude_e
        crosscheck = float(np.max(np.abs(canonical_direct - production)))
        target_crosscheck_error = max(target_crosscheck_error, crosscheck)
        image_passed, image_error, image_shifts = _image_shift_evidence(
            polytype, canonical_coordinates, layer_coordinates
        )
        image_gauge_passed &= image_passed
        image_integrality_error = max(image_integrality_error, image_error)
        image_residual_maximum = float(np.max(np.abs(canonical_direct - layer_direct)))
        arbitrary_qz_image_effect_passed &= image_residual_maximum > 1.0

        target_repeat_A = float(np.linalg.norm(target.direct_basis_A[:, 2])) / len(motifs)
        actual_topology = _target_topology(target, motifs)
        topology_passed &= actual_topology == required_topologies[polytype]
        heights = [
            float(
                np.mean(
                    [abs(atom.fractional_offset[2]) for atom in motif.atoms if atom.element == "I"]
                )
                * np.linalg.norm(target.direct_basis_A[:, 2])
            )
            for motif in motifs
        ]

        primary_t04 = tuple(
            ("minus" if sign == "+" else "plus", registry)
            for sign, registry in required_topologies[polytype]
        )
        opposite_t04 = tuple(
            ("minus" if sign == "+" else "plus", registry)
            for sign, registry in opposite_topologies[polytype]
        )
        primary_atoms, primary_coordinates = _ideal_atoms(
            source, orientations, primary_t04, source_repeat_A
        )
        opposite_atoms, opposite_coordinates = _ideal_atoms(
            source, orientations, opposite_t04, source_repeat_A
        )
        ideal_primary = _direct_atom_sum(
            primary_atoms, q_vectors, event_wavelength_A, unknown_u_iso_A2=0.0
        )
        ideal_opposite = _direct_atom_sum(
            opposite_atoms, q_vectors, event_wavelength_A, unknown_u_iso_A2=0.0
        )

        integer_qz = np.asarray([2.0 * np.pi / np.linalg.norm(target.direct_basis_A[:, 2])])
        integer_q = _q_vectors(
            target,
            np.asarray([1], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            integer_qz,
        )
        canonical_integer = _direct_atom_sum(
            canonical_atoms,
            integer_q,
            np.asarray([1.540592925]),
            unknown_u_iso_A2=0.0,
        )
        layer_integer = _direct_atom_sum(
            layer_atoms,
            integer_q,
            np.asarray([1.540592925]),
            unknown_u_iso_A2=0.0,
        )
        integer_gauge_error = max(
            integer_gauge_error,
            float(np.max(np.abs(canonical_integer - layer_integer))),
        )
        targets[polytype] = {
            "required_coordination_topology": _topology_payload(required_topologies[polytype]),
            "actual_coordination_topology": _topology_payload(actual_topology),
            "opposite_hand_coordination_topology": _topology_payload(opposite_topologies[polytype]),
            "ideal_2h_derived_direct_amplitude_e": _complex_pairs(ideal_primary),
            "ideal_opposite_hand_direct_amplitude_e": _complex_pairs(ideal_opposite),
            "ideal_primary_coordinates": primary_coordinates,
            "ideal_opposite_coordinates": opposite_coordinates,
            "target_cif_complete_layer_direct_amplitude_e": _complex_pairs(layer_direct),
            "target_cif_canonical_image_direct_amplitude_e": _complex_pairs(canonical_direct),
            "target_canonical_production_crosscheck_error_e": crosscheck,
            "target_expanded_coordinates": canonical_coordinates,
            "target_complete_layer_coordinates": layer_coordinates,
            "target_complete_layer_integer_image_shifts": image_shifts,
            "target_direct_basis_A": target.direct_basis_A.tolist(),
            "target_reciprocal_basis_Ainv": target_lattice.basis_Ainv.tolist(),
            "ideal_layer_repeat_A": source_repeat_A,
            "target_layer_repeat_A": target_repeat_A,
            "target_intralayer_heights_A": heights,
            "canonical_image_vs_complete_layer_residual_e": _complex_pairs(
                canonical_direct - layer_direct
            ),
            "canonical_image_vs_complete_layer_maximum_residual_e": image_residual_maximum,
            "ideal_vs_relaxed_complete_layer_residual_e": _complex_pairs(
                ideal_primary - layer_direct
            ),
        }

    orientation_label_mapping = {
        "LayerAmplitudeResult.f_plus": "T04 manuscript F_plus",
        "LayerAmplitudeResult.f_minus": "T04 manuscript F_minus; exact expanded 2H orientation",
        "coordination_plus": "T04 manuscript F_minus",
        "coordination_minus": "T04 manuscript F_plus",
    }
    events_payload = [
        {
            "event_id": int(event_id_value),
            "h": int(h),
            "k": int(k),
            "qz_Ainv": float(qz),
            "wavelength_A": float(wavelength),
        }
        for event_id_value, h, k, qz, wavelength in zip(
            event_id,
            event_h,
            event_k,
            event_qz_Ainv,
            event_wavelength_A,
            strict=True,
        )
    ]
    deterministic_material_payload = {
        "schema_version": 1,
        "events": events_payload,
        "layer": {
            "f_plus_e": _complex_pairs(layer.f_plus),
            "f_minus_e": _complex_pairs(layer.f_minus),
            "layer_repeat_A": source_repeat_A,
        },
        "conventions": {
            "complex_encoding": "[real,imag] float64 electron amplitude",
            "normalization": "raw complex electron amplitude; no scaling",
            "phase_sign": "positive",
            "orientation_label_mapping": orientation_label_mapping,
            "registry_offsets_fractional": {
                name: [float(value) for value in offset] for name, offset in _REGISTRIES
            },
            "registry_phase_model": "exp[2pi*i*(h*x+k*y)]; B gives exp[2pi*i*(h+2k)/3]",
            "unknown_u_iso_A2": 0.0,
        },
        "parents": {
            parent: {
                "topology_t04": [list(value) for value in topology],
                "direct_ideal_amplitude_e_by_period_multiple": {
                    str(period_multiple): _complex_pairs(
                        ideal_direct_amplitudes[(parent, period_multiple)]
                    )
                    for period_multiple in _PERIOD_MULTIPLES
                },
            }
            for parent, topology in _IDEAL_PARENT_TOPOLOGIES
        },
    }
    maximum_error = max(layer_error, ideal_error, target_crosscheck_error, integer_gauge_error)
    return {
        "status": "PASS"
        if topology_passed
        and image_gauge_passed
        and arbitrary_qz_image_effect_passed
        and maximum_error <= 1e-10
        else "FAIL",
        "normalization": "raw complex electron amplitude; no strongest-reflection normalization",
        "registry_phase_in_f_values": False,
        "unknown_u_iso_A2": 0.0,
        "events": events_payload,
        "orientation_label_mapping": orientation_label_mapping,
        "source_2h": {
            "expanded_orientation_t04": source_motif.orientation,
            "registry": _registry_name(
                source.sites[
                    next(atom for atom in source_motif.atoms if atom.element == "Pb").site_index
                ].fractional[:2]
            ),
            "layer_repeat_A": source_repeat_A,
            "intralayer_height_A": float(
                np.mean(
                    [
                        abs(atom.fractional_offset[2])
                        for atom in source_motif.atoms
                        if atom.element == "I"
                    ]
                )
                * source_repeat_A
            ),
            "f_plus_e": _complex_pairs(layer.f_plus),
            "f_minus_e": _complex_pairs(layer.f_minus),
            "direct_f_plus_e": _complex_pairs(direct_plus),
            "direct_f_minus_e": _complex_pairs(direct_minus),
            "f_plus_pb_centered_coordinates": _motif_coordinates(source, orientations["plus"]),
            "f_minus_pb_centered_coordinates": _motif_coordinates(source, orientations["minus"]),
            "expanded_coordinates": _canonical_target_atoms(source)[1],
        },
        "canonical_ideal_hands": ideal_hands,
        "deterministic_material_payload": deterministic_material_payload,
        "polytypes": targets,
        "maximum_errors_e": {
            "layer_amplitude_vs_direct": layer_error,
            "ideal_explicit_vs_coherent_factors": ideal_error,
            "target_canonical_direct_vs_production": target_crosscheck_error,
            "integer_l_canonical_vs_complete_layer_gauge": integer_gauge_error,
        },
        "target_vs_ideal_is_descriptive_relaxation_evidence": True,
        "complete_layer_image_shift_gate": {
            "passed": image_gauge_passed,
            "maximum_integrality_error_fractional": image_integrality_error,
            "arbitrary_qz_image_effect_passed": arbitrary_qz_image_effect_passed,
        },
        "t05_transition_implementation_called": False,
    }
