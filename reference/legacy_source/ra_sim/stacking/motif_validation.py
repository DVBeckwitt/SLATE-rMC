"""Deterministic validation helpers for CIF-derived stacking motifs."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ra_sim.stacking.motif_form_factor import (
    CellGeometry,
    ExactLayerMotif,
    ExpandedSite,
    ExpandedStructure,
    Matrix3,
    Vector3,
    extract_motif_by_indices,
)


@dataclass(frozen=True, slots=True)
class ExtractedBlock:
    """One unwrapped structural block and its diagnostic metadata."""

    name: str
    motif: ExactLayerMotif
    site_indices: tuple[int, ...]
    image_shifts: tuple[tuple[int, int, int], ...]
    stoichiometry: tuple[tuple[str, int], ...]
    stacking_normal: Vector3

    @property
    def center_cartesian(self) -> Vector3:
        return self.motif.center_cartesian


@dataclass(frozen=True, slots=True)
class RigidOrientationMatch:
    """Species-preserving rigid operation from one motif orientation to another."""

    matrix: Matrix3
    removed_registry_translation: Vector3
    permutation: tuple[int, ...]
    species_rmsd: tuple[tuple[str, float], ...]
    max_rmsd: float
    status: str


def stacking_normal_from_cell(cell: CellGeometry) -> Vector3:
    c_axis = cell.direct[:, 2]
    norm = float(np.linalg.norm(c_axis))
    if norm <= 0.0:
        raise ValueError("cell c-axis must be non-zero")
    normal = c_axis / norm
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def extract_pbi2_blocks(
    structure: ExpandedStructure,
    *,
    allow_generated_repeats: bool = True,
) -> tuple[ExtractedBlock, ...]:
    if allow_generated_repeats:
        generated_blocks = _extract_generated_repeat_blocks(
            structure,
            lambda base_structure: extract_pbi2_blocks(base_structure, allow_generated_repeats=False),
        )
        if generated_blocks is not None:
            return generated_blocks

    normal = stacking_normal_from_cell(structure.cell)
    pb_sites = _indices_for_element(structure, "Pb")
    iodine_sites = _indices_for_element(structure, "I")
    if not pb_sites or len(iodine_sites) != 2 * len(pb_sites):
        raise ValueError("PbI2 block extraction requires one Pb and two I sites per block")

    blocks: list[ExtractedBlock] = []
    iodine_by_pb = _assign_pbi2_iodine_to_pb(structure, pb_sites, iodine_sites)
    for block_index, pb_index in enumerate(_sort_indices_along_normal(structure, pb_sites, normal)):
        block = _block_from_selection(
            structure,
            name=f"PbI2-{block_index}",
            selected=[(pb_index, (0, 0, 0)), *iodine_by_pb[pb_index]],
            normal=normal,
        )
        _assert_stoichiometry(block, {"I": 2, "Pb": 1})
        blocks.append(block)
    _assert_site_coverage(structure, blocks)
    return tuple(blocks)


def extract_bi2se3_quintuple_blocks(
    structure: ExpandedStructure,
    *,
    allow_generated_repeats: bool = True,
) -> tuple[ExtractedBlock, ...]:
    if allow_generated_repeats:
        generated_blocks = _extract_generated_repeat_blocks(
            structure,
            lambda base_structure: extract_bi2se3_quintuple_blocks(base_structure, allow_generated_repeats=False),
        )
        if generated_blocks is not None:
            return generated_blocks

    normal = stacking_normal_from_cell(structure.cell)
    se1_sites = [
        index
        for index, site in enumerate(structure.sites)
        if site.element == "Se" and _label_base(site.source_label) == "Se1"
    ]
    bi_sites = _indices_for_element(structure, "Bi")
    se2_sites = [
        index
        for index, site in enumerate(structure.sites)
        if site.element == "Se" and _label_base(site.source_label) != "Se1"
    ]
    if not se1_sites or len(bi_sites) != 2 * len(se1_sites) or len(se2_sites) != 2 * len(se1_sites):
        raise ValueError("Bi2Se3 quintuple extraction requires Se1 centers plus two Bi/two Se2 sites")

    blocks: list[ExtractedBlock] = []
    used_bi: set[int] = set()
    used_se2: set[int] = set()
    for block_index, se1_index in enumerate(_sort_indices_along_normal(structure, se1_sites, normal)):
        center_cart = structure.sites[se1_index].cart
        available_bi = _same_in_plane_repeat_tile_indices(
            structure,
            se1_index,
            [idx for idx in bi_sites if idx not in used_bi],
        )
        available_se2 = _same_in_plane_repeat_tile_indices(
            structure,
            se1_index,
            [idx for idx in se2_sites if idx not in used_se2],
        )
        selected_bi = _nearest_unique_site_images(
            structure,
            center_cart,
            available_bi,
            count=2,
        )
        selected_se2 = _nearest_unique_site_images(
            structure,
            center_cart,
            available_se2,
            count=2,
        )
        used_bi.update(site_index for site_index, _shift in selected_bi)
        used_se2.update(site_index for site_index, _shift in selected_se2)
        block = _block_from_selection(
            structure,
            name=f"Bi2Se3-QL-{block_index}",
            selected=[(se1_index, (0, 0, 0)), *selected_bi, *selected_se2],
            normal=normal,
        )
        _assert_stoichiometry(block, {"Bi": 2, "Se": 3})
        blocks.append(block)
    _assert_site_coverage(structure, blocks)
    return tuple(blocks)


def match_species_preserving_rigid_operation(
    source: ExactLayerMotif,
    target: ExactLayerMotif,
    *,
    source_center: Vector3 = (0.0, 0.0, 0.0),
    target_center: Vector3 = (0.0, 0.0, 0.0),
    tolerance: float = 1e-8,
) -> RigidOrientationMatch:
    if len(source.atoms) != len(target.atoms):
        raise ValueError("motifs must contain the same number of atoms")
    if Counter(atom.element for atom in source.atoms) != Counter(atom.element for atom in target.atoms):
        raise ValueError("species-mismatched atom mapping")
    _raise_if_duplicate_species_positions(source, tolerance=tolerance)
    _raise_if_duplicate_species_positions(target, tolerance=tolerance)

    source_coords = np.asarray([atom.local_cartesian for atom in source.atoms], dtype=float)
    target_coords = np.asarray([atom.local_cartesian for atom in target.atoms], dtype=float)
    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    occupancy_mismatch_seen = False
    for permutation in _species_preserving_permutations(source, target):
        if any(
            abs(source.atoms[source_index].occupancy - target_atom.occupancy) > tolerance
            for source_index, target_atom in zip(permutation, target.atoms, strict=True)
        ):
            occupancy_mismatch_seen = True
            continue
        permuted = source_coords[list(permutation)]
        matrix, residuals = _orthogonal_fit(permuted, target_coords)
        rmsd = float(np.sqrt(np.mean(residuals * residuals)))
        if best is None or rmsd < best[0] - tolerance:
            best = (rmsd, permutation, matrix, residuals)
    if best is None:
        if occupancy_mismatch_seen:
            raise ValueError("occupancy-mismatched atom mapping")
        raise ValueError("no species-preserving permutation exists")

    rmsd, permutation, matrix, residuals = best
    species_rmsd = []
    target_species = [atom.element for atom in target.atoms]
    for element in sorted(set(target_species)):
        mask = np.asarray([species == element for species in target_species], dtype=bool)
        species_rmsd.append((element, float(np.sqrt(np.mean(residuals[mask] * residuals[mask])))))
    source_center_arr = np.asarray(source_center, dtype=float)
    target_center_arr = np.asarray(target_center, dtype=float)
    registry = target_center_arr - matrix @ source_center_arr
    if rmsd > tolerance:
        status = "INDEPENDENT_COORDINATE_SETS"
    elif np.allclose(matrix, np.eye(3), atol=tolerance, rtol=0.0):
        status = "EQUIVALENT_UP_TO_TRANSLATION"
    else:
        status = "DISTINCT_ORIENTATIONS"
    return RigidOrientationMatch(
        matrix=_matrix3(matrix),
        removed_registry_translation=_vector3(registry),
        permutation=permutation,
        species_rmsd=tuple(species_rmsd),
        max_rmsd=float(np.max(residuals)),
        status=status,
    )


def _wrap_fractional_xy(value) -> tuple[float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError("fractional xy value must be a 2-vector")
    wrapped = ((arr + 0.5) % 1.0) - 0.5
    return (float(wrapped[0]), float(wrapped[1]))


def _nearest_unique_site_images(
    structure: ExpandedStructure,
    center_cartesian,
    candidate_indices: list[int],
    *,
    count: int,
) -> list[tuple[int, tuple[int, int, int]]]:
    candidates = []
    for site_index in candidate_indices:
        distance, shift, key = _best_site_image_to_center(structure, center_cartesian, site_index)
        candidates.append((distance, key, site_index, shift))
    if len(candidates) < count:
        raise ValueError("not enough candidate sites for block extraction")
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if len(candidates) > count and abs(candidates[count][0] - candidates[count - 1][0]) <= 1e-10:
        raise ValueError("ambiguous nearest-neighbor block extraction")
    return [(site_index, shift) for _distance, _key, site_index, shift in candidates[:count]]


def _best_site_image_to_center(
    structure: ExpandedStructure,
    center_cartesian,
    site_index: int,
) -> tuple[float, tuple[int, int, int], tuple[float, ...]]:
    center = np.asarray(center_cartesian, dtype=float)
    center_frac = np.linalg.inv(structure.cell.direct) @ center
    site = structure.sites[int(site_index)]
    best = None
    for shift in itertools.product((-1, 0, 1), repeat=3):
        delta_frac = np.asarray(site.frac, dtype=float) + np.asarray(shift, dtype=float) - center_frac
        cart_delta = structure.cell.direct @ delta_frac
        distance = float(np.linalg.norm(cart_delta))
        wrapped_xy = np.asarray(_wrap_fractional_xy(delta_frac[:2]), dtype=float)
        canonical_delta = np.asarray([wrapped_xy[0], wrapped_xy[1], delta_frac[2]], dtype=float)
        wrap_penalty = float(np.linalg.norm(delta_frac[:2] - wrapped_xy))
        fractional_norm = float(np.linalg.norm(canonical_delta))
        delta_key = tuple(float(np.round(value, 12)) for value in canonical_delta)
        distance_key = float(np.round(distance, 10))
        item = (
            distance_key,
            wrap_penalty,
            fractional_norm,
            *delta_key,
            int(site_index),
            distance,
            shift,
        )
        if best is None or item < best:
            best = item
    if best is None:
        raise ValueError("could not find a periodic image for site")
    *key_values, shift = best
    distance = float(key_values[-1])
    key = tuple(float(value) for value in key_values[1:])
    return distance, shift, key


def _block_from_selection(
    structure: ExpandedStructure,
    *,
    name: str,
    selected: list[tuple[int, tuple[int, int, int]]],
    normal: Vector3,
) -> ExtractedBlock:
    direct = structure.cell.direct
    normal_arr = np.asarray(normal, dtype=float)
    selected_sorted = sorted(
        selected,
        key=lambda item: (
            float(np.dot(np.asarray(structure.sites[item[0]].cart) + direct @ np.asarray(item[1], dtype=float), normal_arr)),
            structure.sites[item[0]].element,
            item[0],
            item[1],
        ),
    )
    indices = tuple(site_index for site_index, _shift in selected_sorted)
    shifts = tuple(shift for _site_index, shift in selected_sorted)
    motif = extract_motif_by_indices(structure, indices, image_shifts=shifts, center="centroid")
    stoich = tuple(sorted(Counter(atom.element for atom in motif.atoms).items()))
    return ExtractedBlock(
        name=name,
        motif=motif,
        site_indices=indices,
        image_shifts=shifts,
        stoichiometry=stoich,
        stacking_normal=normal,
    )


def _extract_generated_repeat_blocks(
    structure: ExpandedStructure,
    extractor,
) -> tuple[ExtractedBlock, ...] | None:
    mapping = _generated_repeat_base_mapping(structure, range(len(structure.sites)))
    if mapping is None:
        return None
    base_structure, _generated_to_base, base_repeat_to_generated, repeats, repeat_set = mapping
    base_blocks = extractor(base_structure)
    if not base_blocks:
        return None

    normal = stacking_normal_from_cell(structure.cell)
    blocks: list[ExtractedBlock] = []
    for repeat in sorted(repeat_set):
        for base_block in base_blocks:
            selected = []
            for base_index, primitive_shift in zip(
                base_block.site_indices,
                base_block.image_shifts,
                strict=True,
            ):
                absolute_repeat = tuple(repeat[axis] + primitive_shift[axis] for axis in range(3))
                supercell_shift = tuple(absolute_repeat[axis] // repeats[axis] for axis in range(3))
                generated_repeat = tuple(absolute_repeat[axis] % repeats[axis] for axis in range(3))
                selected.append((base_repeat_to_generated[(base_index, generated_repeat)], supercell_shift))
            blocks.append(
                _block_from_selection(
                    structure,
                    name=f"{base_block.name}_{repeat[0]}_{repeat[1]}_{repeat[2]}",
                    selected=selected,
                    normal=normal,
                )
            )
    blocks.sort(
        key=lambda block: (
            float(np.dot(np.asarray(block.center_cartesian, dtype=float), np.asarray(normal, dtype=float))),
            block.name,
        )
    )
    _assert_site_coverage(structure, blocks)
    return tuple(
        ExtractedBlock(
            name=f"{block.name.rsplit('_', 3)[0]}-{index}",
            motif=block.motif,
            site_indices=block.site_indices,
            image_shifts=block.image_shifts,
            stoichiometry=block.stoichiometry,
            stacking_normal=block.stacking_normal,
        )
        for index, block in enumerate(blocks)
    )


def _indices_for_element(structure: ExpandedStructure, element: str) -> list[int]:
    return [index for index, site in enumerate(structure.sites) if site.element == element]


def _same_in_plane_repeat_tile_indices(
    structure: ExpandedStructure,
    center_index: int,
    candidate_indices: list[int],
) -> list[int]:
    suffix = _in_plane_repeat_suffix(structure.sites[center_index].source_label)
    if suffix is None:
        return candidate_indices
    filtered = [
        index
        for index in candidate_indices
        if _in_plane_repeat_suffix(structure.sites[index].source_label) == suffix
    ]
    return filtered if len(filtered) >= 2 else candidate_indices


def _in_plane_repeat_suffix(label: str) -> tuple[int, int] | None:
    suffix = _repeat_suffix(label)
    if suffix is None:
        return None
    return suffix[:2]


def _repeat_suffix(label: str) -> tuple[int, int, int] | None:
    parts = str(label).rsplit("_", 3)
    if len(parts) != 4:
        return None
    try:
        return int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _label_without_repeat_suffix(label: str) -> str:
    return str(label).rsplit("_", 3)[0]


def _generated_repeat_base_mapping(
    structure: ExpandedStructure,
    involved_indices,
):
    involved = list(involved_indices)
    suffix_by_index = {index: _repeat_suffix(structure.sites[index].source_label) for index in involved}
    if any(suffix is None for suffix in suffix_by_index.values()):
        return None
    suffixes = tuple(suffix for suffix in suffix_by_index.values() if suffix is not None)
    repeats = tuple(max(suffix[axis] for suffix in suffixes) + 1 for axis in range(3))
    if repeats == (1, 1, 1):
        return None

    repeat_set = set(itertools.product(*(range(repeat) for repeat in repeats)))
    repeats_arr = np.asarray(repeats, dtype=float)
    base_direct = structure.cell.direct.copy()
    for axis, repeat in enumerate(repeats):
        base_direct[:, axis] /= float(repeat)

    base_sites: list[ExpandedSite] = []
    base_key_to_index: dict[tuple[object, ...], int] = {}
    generated_to_base: dict[int, int] = {}
    base_repeat_to_generated: dict[tuple[int, tuple[int, int, int]], int] = {}
    for generated_index in sorted(involved):
        site = structure.sites[generated_index]
        suffix = suffix_by_index[generated_index]
        if suffix is None:
            return None
        primitive_frac = (np.asarray(site.frac, dtype=float) * repeats_arr - np.asarray(suffix, dtype=float)) % 1.0
        base_label = _label_without_repeat_suffix(site.source_label)
        key = (
            site.element,
            base_label,
            tuple(float(np.round(value, 12)) for value in primitive_frac),
            float(np.round(site.occupancy, 12)),
            None if site.uiso is None else float(np.round(site.uiso, 12)),
            None if site.b_iso is None else float(np.round(site.b_iso, 12)),
        )
        base_index = base_key_to_index.get(key)
        if base_index is None:
            base_index = len(base_sites)
            base_key_to_index[key] = base_index
            base_sites.append(
                ExpandedSite(
                    site.element,
                    site.occupancy,
                    _vector3(primitive_frac),
                    _vector3(base_direct @ primitive_frac),
                    site.uiso,
                    base_label,
                    b_iso=site.b_iso,
                )
            )
        generated_to_base[generated_index] = base_index
        base_repeat_to_generated[(base_index, suffix)] = generated_index

    base_indices = set(generated_to_base.values())
    if any((base_index, repeat) not in base_repeat_to_generated for base_index in base_indices for repeat in repeat_set):
        return None

    return (
        ExpandedStructure(cell=CellGeometry(base_direct), sites=tuple(base_sites)),
        generated_to_base,
        base_repeat_to_generated,
        repeats,
        repeat_set,
    )


def _assign_pbi2_iodine_to_pb(
    structure: ExpandedStructure,
    pb_sites: list[int],
    iodine_sites: list[int],
) -> dict[int, list[tuple[int, tuple[int, int, int]]]]:
    pb_order = _sort_indices_along_normal(structure, pb_sites, stacking_normal_from_cell(structure.cell))
    iodine_order = sorted(iodine_sites, key=lambda index: _site_fractional_key(structure.sites[index]))
    pair_options = {
        (iodine_index, pb_index): _best_site_image_to_center(
            structure,
            structure.sites[pb_index].cart,
            iodine_index,
        )
        for iodine_index in iodine_order
        for pb_index in pb_order
    }
    memo: dict[
        tuple[int, tuple[int, ...]],
        tuple[float, tuple[tuple[float, ...], ...], tuple[int, ...], bool] | None,
    ] = {}

    def solve(
        iodine_position: int,
        counts: tuple[int, ...],
    ) -> tuple[float, tuple[tuple[float, ...], ...], tuple[int, ...], bool] | None:
        state = (iodine_position, counts)
        if state in memo:
            return memo[state]
        if iodine_position == len(iodine_order):
            result = (0.0, (), (), False) if counts == (2,) * len(pb_order) else None
            memo[state] = result
            return result

        iodine_index = iodine_order[iodine_position]
        best_result = None
        for pb_position, pb_index in enumerate(pb_order):
            if counts[pb_position] >= 2:
                continue
            next_counts = list(counts)
            next_counts[pb_position] += 1
            tail = solve(iodine_position + 1, tuple(next_counts))
            if tail is None:
                continue
            tail_cost, tail_key, tail_assignment, tail_ambiguous = tail
            distance, _shift, key = pair_options[(iodine_index, pb_index)]
            item = (
                float(distance * distance + tail_cost),
                (key, *tail_key),
                (pb_position, *tail_assignment),
                bool(tail_ambiguous),
            )
            if best_result is None or item[0] < best_result[0] - 1e-12:
                best_result = item
            elif abs(item[0] - best_result[0]) <= 1e-12:
                ambiguous = best_result[3] or item[3] or item[1] != best_result[1] or item[2] != best_result[2]
                best_result = (*min(item[:3], best_result[:3]), ambiguous)
        memo[state] = best_result
        return best_result

    best = solve(0, tuple([0] * len(pb_order)))
    if best is None:
        raise ValueError("could not assign PbI2 iodine sites to Pb centers")
    if best[3]:
        raise ValueError("ambiguous PbI2 block extraction")

    _cost, _assignment_key, assignment, _ambiguous = best
    by_pb: dict[int, list[tuple[int, tuple[int, int, int]]]] = {pb_index: [] for pb_index in pb_order}
    for iodine_index, pb_position in zip(iodine_order, assignment, strict=True):
        pb_index = pb_order[pb_position]
        _distance, shift, _key = pair_options[(iodine_index, pb_index)]
        by_pb[pb_index].append((iodine_index, shift))
    return {
        pb_index: sorted(items, key=lambda item: _site_fractional_key(structure.sites[item[0]]))
        for pb_index, items in by_pb.items()
    }


def _site_fractional_key(site: ExpandedSite) -> tuple[float, float, float, str]:
    return (
        float(np.round(site.frac[2], 12)),
        float(np.round(site.frac[0], 12)),
        float(np.round(site.frac[1], 12)),
        site.source_label,
    )


def _sort_indices_along_normal(
    structure: ExpandedStructure,
    indices: list[int],
    normal: Vector3,
) -> list[int]:
    normal_arr = np.asarray(normal, dtype=float)
    return sorted(indices, key=lambda index: (float(np.dot(structure.sites[index].cart, normal_arr)), index))


def _assert_stoichiometry(block: ExtractedBlock, expected: dict[str, int]) -> None:
    actual = dict(block.stoichiometry)
    if actual != expected:
        raise ValueError(f"{block.name} stoichiometry {actual} != {expected}")


def _assert_site_coverage(structure: ExpandedStructure, blocks: tuple[ExtractedBlock, ...] | list[ExtractedBlock]) -> None:
    counts = Counter(index for block in blocks for index in block.site_indices)
    expected = set(range(len(structure.sites)))
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError("extracted blocks must cover every P1 site exactly once")


def _species_preserving_permutations(
    source: ExactLayerMotif,
    target: ExactLayerMotif,
) -> tuple[tuple[int, ...], ...]:
    source_by_species: dict[str, list[int]] = {}
    for index, atom in enumerate(source.atoms):
        source_by_species.setdefault(atom.element, []).append(index)
    target_species = [atom.element for atom in target.atoms]
    per_species = [
        tuple(itertools.permutations(source_by_species[element], target_species.count(element)))
        for element in sorted(source_by_species)
    ]
    permutations = []
    for combo in itertools.product(*per_species):
        mapping_by_species = {
            element: list(values)
            for element, values in zip(sorted(source_by_species), combo, strict=True)
        }
        permutations.append(tuple(mapping_by_species[element].pop(0) for element in target_species))
    return tuple(permutations)


def _raise_if_duplicate_species_positions(
    motif: ExactLayerMotif,
    *,
    tolerance: float,
) -> None:
    for left_index, left_atom in enumerate(motif.atoms):
        left = np.asarray(left_atom.local_cartesian, dtype=float)
        for right_atom in motif.atoms[left_index + 1 :]:
            if left_atom.element != right_atom.element:
                continue
            if abs(left_atom.occupancy - right_atom.occupancy) > tolerance:
                continue
            right = np.asarray(right_atom.local_cartesian, dtype=float)
            if np.linalg.norm(left - right) <= tolerance:
                raise ValueError("ambiguous species-preserving atom mapping")


def _orthogonal_fit(source_coords: np.ndarray, target_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_centered = source_coords - np.mean(source_coords, axis=0)
    target_centered = target_coords - np.mean(target_coords, axis=0)
    left, _singular, right_t = np.linalg.svd(source_centered.T @ target_centered)
    row_rotation = left @ right_t
    matrix = row_rotation.T
    predicted = source_centered @ row_rotation
    residuals = np.linalg.norm(predicted - target_centered, axis=1)
    return matrix, residuals


def _label_base(label: str) -> str:
    return str(label).split("_", 1)[0]


def _vector3(value) -> Vector3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("value must be a 3-vector")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _matrix3(value) -> Matrix3:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError("value must be a 3x3 matrix")
    return (
        (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
        (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
    )


