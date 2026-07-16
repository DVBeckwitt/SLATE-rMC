"""Compact T04 scientific proof for ordered amplitudes and reflectivity."""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import platform
import subprocess
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import xraydb
from numpy.typing import NDArray

from rasim_next.core.contracts import CONTRACT_API_VERSION, RodQueryBatch
from rasim_next.materials import CrystalStructure, read_crystal
from rasim_next.materials.optics import (
    CLASSICAL_ELECTRON_RADIUS_A,
    HC_EV_A,
    material_optics,
)
from rasim_next.ordered.amplitudes import ordered_event_result, unit_cell_amplitude
from rasim_next.ordered.bi2se3_proof import run_bi2se3_ql_proof
from rasim_next.ordered.finite_stack import coherent_finite_stack, uniform_finite_stack
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity.parratt import parratt_reflectivity
from rasim_next.reflectivity.specular import manuscript_specular_composite

ROOT = Path(__file__).resolve().parents[3]
PACK_PATH = ROOT / "reference" / "rasim_reference_v1.npz"
WAVELENGTH_A = 1.540592925
PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
TRACE_SCHEMA_VERSION = 4
TOLERANCES = {
    "amplitude_e": {"atol": 1e-10, "rtol": 1e-12},
    "coordinate_fractional": {"atol": 2e-15, "rtol": 0.0},
    "parratt": {"atol": 5e-11, "rtol": 0.0},
    "reference_d_A": {"atol": 2e-12, "rtol": 2e-12},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _check(check_id: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "evidence": json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False),
    }


def _reference_pack() -> tuple[dict[str, NDArray[Any]], str]:
    manifest = tomllib.loads(
        (ROOT / "reference" / "reference_manifest.toml").read_text(encoding="utf-8")
    )
    if manifest["contract_api_version"] != f"rasim-contracts-v{CONTRACT_API_VERSION}":
        raise ValueError("reference manifest contract version mismatch")
    if manifest["trace_schema_version"] != f"rasim-stage-trace-v{TRACE_SCHEMA_VERSION}":
        raise ValueError("reference manifest trace-schema version mismatch")
    expected_hash = str(manifest["reference_pack"]["sha256"])
    observed_hash = _sha256(PACK_PATH)
    if observed_hash != expected_hash:
        raise ValueError("immutable reference-pack hash mismatch")
    names = {
        "ordered_hkl",
        "ordered_sim_d",
        "ordered_sim_f_real",
        "ordered_sim_f_imag",
        "parratt_qz",
        "parratt_reflectivity",
        "parratt_parameters",
        "parratt_indices",
    }
    with np.load(PACK_PATH, allow_pickle=False) as source:
        return {name: np.array(source[name], copy=True) for name in names}, observed_hash


def _atomic_factor_e(
    species: str,
    element: str,
    charge: int,
    q_magnitude_Ainv: float,
    wavelength_A: float,
) -> complex:
    requested = species
    if charge:
        sign = "+" if charge > 0 else "-"
        ionic = f"{element}{abs(charge)}{sign}"
        if ionic in xraydb.f0_ions(element):
            requested = ionic
    f0 = float(np.asarray(xraydb.f0(requested, q_magnitude_Ainv / (4.0 * math.pi))).item())
    energy_eV = HC_EV_A / wavelength_A
    return complex(
        f0 + xraydb.f1_chantler(element, energy_eV),
        xraydb.f2_chantler(element, energy_eV),
    )


def _direct_atom_amplitudes(
    crystal: CrystalStructure,
    hkl: NDArray[np.float64],
    wavelength_A: NDArray[np.float64],
    *,
    unknown_u_iso_A2: float | None = None,
) -> NDArray[np.complex128]:
    """Independent scalar atom sum with explicit positive phase and damping."""

    reciprocal = 2.0 * math.pi * np.linalg.inv(crystal.direct_basis_A).T
    result: list[complex] = []
    for miller, wavelength in zip(hkl, wavelength_A, strict=True):
        q_vector = reciprocal @ miller
        q_magnitude = float(np.linalg.norm(q_vector))
        contributions: list[complex] = []
        for site in crystal.sites:
            if site.u_iso_A2 is None and unknown_u_iso_A2 is None:
                raise ValueError("scalar atom oracle requires an explicit unknown Uiso")
            u_iso = unknown_u_iso_A2 if site.u_iso_A2 is None else site.u_iso_A2
            factor = _atomic_factor_e(
                site.species,
                site.element,
                site.charge,
                q_magnitude,
                float(wavelength),
            )
            phase = cmath.exp(2.0j * math.pi * float(np.dot(miller, site.fractional)))
            damping = math.exp(-0.5 * float(u_iso) * q_magnitude**2)
            contributions.append(site.occupancy * factor * damping * phase)
        result.append(
            complex(
                math.fsum(value.real for value in contributions),
                math.fsum(value.imag for value in contributions),
            )
        )
    return np.asarray(result, dtype=np.complex128)


def _scalar_and_raw_check(
    crystal: CrystalStructure, pack: dict[str, NDArray[Any]]
) -> tuple[dict[str, object], dict[str, object]]:
    pack_hkl = np.asarray(pack["ordered_hkl"], dtype=np.float64)
    row_003 = int(np.flatnonzero(np.all(pack_hkl == (0.0, 0.0, 3.0), axis=1))[0])
    hkl = np.asarray(((0.0, 0.0, 3.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, -1.0, 0.37)))
    wavelength = np.asarray((WAVELENGTH_A, WAVELENGTH_A, 1.1, 1.8))
    production = unit_cell_amplitude(crystal, hkl, wavelength).amplitude_e
    direct = _direct_atom_amplitudes(crystal, hkl, wavelength)
    direct_error = float(np.max(np.abs(production - direct)))
    absence_error_e = float(abs(production[1]))
    small_atom_errors = []
    for atom_count in (1, 2):
        small = replace(
            crystal,
            phase_id=f"bi2se3-{atom_count}-atom",
            sites=crystal.sites[:atom_count],
        )
        small_atom_errors.append(
            float(
                np.max(
                    np.abs(
                        unit_cell_amplitude(small, hkl, wavelength).amplitude_e
                        - _direct_atom_amplitudes(small, hkl, wavelength)
                    )
                )
            )
        )
    small_atom_error_e = max(small_atom_errors)

    partial = replace(
        crystal,
        phase_id="bi2se3-half-bi",
        sites=tuple(
            replace(site, occupancy=0.5) if site.source_label == "Bi" else site
            for site in crystal.sites
        ),
    )
    partial_production = unit_cell_amplitude(partial, hkl, wavelength).amplitude_e
    partial_direct = _direct_atom_amplitudes(partial, hkl, wavelength)
    partial_error_e = float(np.max(np.abs(partial_production - partial_direct)))
    occupancy_effect_e = float(np.max(np.abs(partial_production - production)))

    lattice = ReciprocalLattice.from_crystal(crystal)
    reciprocal_duality_error = float(
        np.max(np.abs(crystal.direct_basis_A.T @ lattice.basis_Ainv - 2.0 * math.pi * np.eye(3)))
    )
    nonorthogonal_cell = not np.isclose(
        np.dot(crystal.direct_basis_A[:, 0], crystal.direct_basis_A[:, 1]),
        0.0,
    )
    d_003_A = 2.0 * math.pi / float(np.linalg.norm(lattice.q_cartesian_Ainv(hkl[0])))
    d_error_A = abs(d_003_A - float(pack["ordered_sim_d"][row_003]))
    historical = complex(
        float(pack["ordered_sim_f_real"][row_003]),
        float(pack["ordered_sim_f_imag"][row_003]),
    )
    historical_residual_e = abs(complex(production[0]) - historical)

    vesta_row = (
        (ROOT / "examples" / "bi2se3" / "reference" / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt")
        .read_text(encoding="utf-8")
        .splitlines()[1]
        .split()
    )
    vesta_003 = complex(float(vesta_row[4]), float(vesta_row[5]))
    vesta_component_error_e = float(
        np.max(
            np.abs(
                (
                    production[0].real - vesta_003.real,
                    production[0].imag - vesta_003.imag,
                )
            )
        )
    )

    optical_wavelength_A = np.asarray((1.1, WAVELENGTH_A, 1.8))
    optics = material_optics(crystal, optical_wavelength_A)
    forward = unit_cell_amplitude(
        crystal,
        np.zeros((optical_wavelength_A.size, 3)),
        optical_wavelength_A,
    ).amplitude_e
    prefactor = (
        CLASSICAL_ELECTRON_RADIUS_A * optical_wavelength_A**2 / (2.0 * math.pi * crystal.volume_A3)
    )
    optics_consistency_error = max(
        float(np.max(np.abs(optics.delta - prefactor * forward.real))),
        float(np.max(np.abs(optics.beta - prefactor * forward.imag))),
        float(np.max(np.abs(optics.mu_Ainv - 4.0 * math.pi * optics.beta / optical_wavelength_A))),
        float(np.max(np.abs(optics.n_complex - (1.0 - optics.delta + 1.0j * optics.beta)))),
    )

    catalog = build_rod_catalog(crystal, h_bounds=(0, 1), k_bounds=(0, 0))
    catalog_hk = set(zip(catalog.h.tolist(), catalog.k.tolist(), strict=True))
    catalog_complete = catalog_hk == {(0, 0), (1, 0)} and np.unique(catalog.rod_id).size == 2
    rows = [int(np.flatnonzero((catalog.h == h_value) & (catalog.k == 0))[0]) for h_value in (0, 1)]
    l_coordinate = np.asarray((0.5, 1.25))
    query = RodQueryBatch(
        event_id=np.asarray((42, 7)),
        rod_id=catalog.rod_id[rows],
        phase_id=(crystal.phase_id,) * 2,
        h=np.asarray((0, 1), dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        qz_Ainv=np.asarray((0.123, -0.456)),
        l_coordinate=l_coordinate,
        wavelength_A=np.full(2, WAVELENGTH_A),
    )
    ordered = ordered_event_result(crystal, catalog, query)
    expected_event = unit_cell_amplitude(
        crystal,
        np.column_stack((query.h, query.k, query.l_coordinate)),
        query.wavelength_A,
    ).amplitude_e
    event_amplitude_error_e = float(np.max(np.abs(ordered.amplitude_e - expected_event)))
    raw_electron2 = np.abs(ordered.amplitude_e) ** 2
    protected_field_error_electron2 = float(
        np.max(np.abs(ordered.intensity.intensity_per_sr - raw_electron2))
    )
    passed = bool(
        direct_error <= TOLERANCES["amplitude_e"]["atol"]
        and small_atom_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and partial_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and occupancy_effect_e > 1.0
        and absence_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and reciprocal_duality_error <= 1e-12
        and nonorthogonal_cell
        and optics_consistency_error <= 1e-12
        and np.all(optics.beta > 0.0)
        and d_error_A <= TOLERANCES["reference_d_A"]["atol"]
        and vesta_component_error_e <= 0.01
        and event_amplitude_error_e == 0.0
        and protected_field_error_electron2 <= 5e-12
        and np.all(np.isfinite(raw_electron2))
        and catalog_complete
        and np.array_equal(ordered.event_id, query.event_id)
        and np.array_equal(ordered.intensity.event_id, query.event_id)
        and ordered.intensity.normalization == "|F_e|^2; electron2; no external factors"
        and not ordered.amplitude_e.flags.writeable
    )
    check = _check(
        "cif_atom_optics_and_raw_amplitude",
        passed,
        {
            "maximum_scalar_oracle_error_e": direct_error,
            "one_two_atom_scalar_oracle_error_e": small_atom_error_e,
            "partial_occupancy_oracle_error_e": partial_error_e,
            "partial_occupancy_effect_e": occupancy_effect_e,
            "F001_systematic_absence_e": absence_error_e,
            "reciprocal_duality_error": reciprocal_duality_error,
            "nonorthogonal_cell": nonorthogonal_cell,
            "material_optics_consistency_error": optics_consistency_error,
            "minimum_beta": float(np.min(optics.beta)),
            "F003_reciprocal_d_error_A": d_error_A,
            "F003_vesta_component_error_e": vesta_component_error_e,
            "F003_historical_pack_residual_e": historical_residual_e,
            "arbitrary_L_event_amplitude_error_e": event_amplitude_error_e,
            "bounded_rod_catalog_complete": catalog_complete,
            "protected_intensity_per_sr_field_error_electron2": (protected_field_error_electron2),
            "protected_field_normalization": ordered.intensity.normalization,
            "raw_electron2_minimum": float(np.min(raw_electron2)),
            "raw_electron2_maximum": float(np.max(raw_electron2)),
        },
    )
    comparison = {
        "case_id": "ordered.bi2se3_vesta",
        "classification": "CORRECTED",
        "prior_matching_evidence": "F003 reciprocal d-spacing",
        "prior_stage_maximum_error_A": d_error_A,
        "first_divergence_stage": "ordered.unit_cell_amplitude",
        "historical_pack_residual_e": historical_residual_e,
        "independent_oracle_check_id": check["check_id"],
    }
    return check, comparison


def _pbi2_check() -> dict[str, object]:
    crystal = read_crystal(
        ROOT / "examples" / "pbi2" / "structures" / "PbI2_2H.cif",
        phase_id="pbi2",
    )
    motifs = extract_pbi2_motifs(crystal)
    motif = motifs[0]
    plus_offsets = [
        (
            atom,
            (
                atom.fractional_offset[0],
                atom.fractional_offset[1],
                -atom.fractional_offset[2],
            )
            if motif.orientation == "minus"
            else atom.fractional_offset,
        )
        for atom in motif.atoms
    ]
    minus_offsets = [(atom, (offset[0], offset[1], -offset[2])) for atom, offset in plus_offsets]

    catalog = build_rod_catalog(crystal, h_bounds=(1, 1), k_bounds=(0, 0))
    l_coordinate = np.asarray((0.75, -0.75))
    lattice = ReciprocalLattice.from_crystal(crystal)
    query = RodQueryBatch(
        event_id=np.asarray((9, 2)),
        rod_id=np.repeat(catalog.rod_id, 2),
        phase_id=(crystal.phase_id,) * 2,
        h=np.ones(2, dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        qz_Ainv=np.asarray((0.2, 0.3)),
        l_coordinate=l_coordinate,
        wavelength_A=np.full(2, WAVELENGTH_A),
    )
    production = pbi2_layer_amplitudes(crystal, query, unknown_u_iso_A2=0.0)

    def scalar(offsets: list[tuple[Any, tuple[float, float, float]]]) -> NDArray[np.complex128]:
        values: list[complex] = []
        for h_value, k_value, l_value, wavelength in zip(
            query.h,
            query.k,
            query.l_coordinate,
            query.wavelength_A,
            strict=True,
        ):
            q_vector = lattice.q_cartesian_Ainv((h_value, k_value, l_value))
            q_magnitude = float(np.linalg.norm(q_vector))
            terms = []
            for atom, offset in offsets:
                factor = _atomic_factor_e(
                    atom.species,
                    atom.element,
                    atom.charge,
                    q_magnitude,
                    float(wavelength),
                )
                phase = cmath.exp(
                    2.0j
                    * math.pi
                    * (
                        float(h_value) * offset[0]
                        + float(k_value) * offset[1]
                        + float(l_value) * offset[2]
                    )
                )
                terms.append(atom.occupancy * factor * phase)
            values.append(
                complex(
                    math.fsum(value.real for value in terms),
                    math.fsum(value.imag for value in terms),
                )
            )
        return np.asarray(values, dtype=np.complex128)

    expected_plus = scalar(plus_offsets)
    expected_minus = scalar(minus_offsets)
    maximum_error_e = max(
        float(np.max(np.abs(production.f_plus - expected_plus))),
        float(np.max(np.abs(production.f_minus - expected_minus))),
    )
    covered = sorted(atom.site_index for item in motifs for atom in item.atoms)
    passed = bool(
        len(motifs) == 1
        and motif.orientation == "minus"
        and covered == list(range(len(crystal.sites)))
        and maximum_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and np.allclose(production.f_minus[0], production.f_plus[1], rtol=1e-12, atol=1e-10)
        and not production.f_plus.flags.writeable
    )
    return _check(
        "pbi2_motif_layer_boundary",
        passed,
        {
            "motif_count": len(motifs),
            "orientation": motif.orientation,
            "covered_sites": len(covered),
            "maximum_scalar_oracle_error_e": maximum_error_e,
            "gauge": "Pb-centered; registry-free; positive phase",
        },
    )


def _finite_stack_check() -> dict[str, object]:
    event_id = np.asarray((17, 3))
    qz_Ainv = np.asarray((0.0, 0.73))
    amplitudes = np.asarray(
        (
            (1.0 + 0.5j, 2.0 - 0.25j, -0.4 + 0.7j),
            (0.5j, 1.2 - 0.3j, 0.8 + 0.1j),
        )
    )
    depths_A = np.asarray((0.0, 2.5, 7.25))
    production = coherent_finite_stack(event_id, qz_Ainv, amplitudes, depths_A)
    direct = np.sum(amplitudes * np.exp(1.0j * qz_Ainv[:, None] * depths_A), axis=1)
    direct_error_e = float(np.max(np.abs(production.amplitude_e - direct)))
    raw_error_e2 = float(
        np.max(np.abs(production.intensity.intensity_per_sr - np.abs(direct) ** 2))
    )

    repeat = np.asarray((1.2 + 0.4j,))
    off_bragg_qz = np.asarray((0.31,))
    off_bragg = uniform_finite_stack(
        event_id[:1],
        off_bragg_qz,
        repeat,
        repeat_spacing_A=2.5,
        repeat_count=5,
    )
    enumerated = repeat * np.sum(
        np.exp(1.0j * off_bragg_qz[:, None] * 2.5 * np.arange(5)),
        axis=1,
    )
    off_bragg_error_e = float(np.max(np.abs(off_bragg.amplitude_e - enumerated)))
    bragg = uniform_finite_stack(
        event_id[:1],
        np.asarray((2.0 * math.pi / 2.5,)),
        repeat,
        repeat_spacing_A=2.5,
        repeat_count=7,
    )
    bragg_error_e = float(np.max(np.abs(bragg.amplitude_e - 7.0 * repeat)))
    passed = bool(
        direct_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and raw_error_e2 <= 5e-12
        and off_bragg_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and bragg_error_e <= TOLERANCES["amplitude_e"]["atol"]
        and production.intensity.normalization.endswith("electron2")
    )
    return _check(
        "finite_stack_direct_sum",
        passed,
        {
            "maximum_direct_error_e": direct_error_e,
            "maximum_raw_measure_error_electron2": raw_error_e2,
            "five_repeat_off_Bragg_error_e": off_bragg_error_e,
            "seven_repeat_Bragg_error_e": bragg_error_e,
            "normalization": production.intensity.normalization,
        },
    )


def _direct_parratt(
    qz_Ainv: NDArray[np.float64],
    wavelength_A: float,
    indices: NDArray[np.complex128],
    thickness_A: tuple[float | None, ...],
    roughness_A: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    k0 = 2.0 * math.pi / wavelength_A
    kz_rows: list[list[complex]] = []
    interface_rows: list[list[complex]] = []
    amplitudes: list[complex] = []
    for qz in qz_Ainv:
        kz = [
            cmath.sqrt(
                (complex(index) ** 2 - complex(indices[0]) ** 2) * k0**2 + (0.5 * float(qz)) ** 2
            )
            for index in indices
        ]
        kz = [
            -value if value.imag < 0.0 or (value.imag == 0.0 and value.real < 0.0) else value
            for value in kz
        ]
        interfaces = [
            (kz[row] - kz[row + 1])
            / (kz[row] + kz[row + 1])
            * cmath.exp(-2.0 * kz[row] * kz[row + 1] * float(roughness_A[row]) ** 2)
            for row in range(len(indices) - 1)
        ]
        recursion = interfaces[-1]
        for row in range(len(indices) - 3, -1, -1):
            phase = cmath.exp(2.0j * kz[row + 1] * float(thickness_A[row + 1]))
            recursion = (interfaces[row] + recursion * phase) / (
                1.0 + interfaces[row] * recursion * phase
            )
        kz_rows.append(kz)
        interface_rows.append(interfaces)
        amplitudes.append(recursion)
    return (
        np.asarray(kz_rows, dtype=np.complex128),
        np.asarray(interface_rows, dtype=np.complex128),
        np.asarray(amplitudes, dtype=np.complex128),
    )


def _parratt_check(
    pack: dict[str, NDArray[Any]],
) -> tuple[dict[str, object], dict[str, object]]:
    wavelength, _, thickness, _, sigma_top, sigma_bottom = np.asarray(
        pack["parratt_parameters"], dtype=np.float64
    )
    sample_rows = np.asarray((0, 128, 256))
    qz = np.asarray(pack["parratt_qz"], dtype=np.float64)[sample_rows]
    indices = np.asarray(pack["parratt_indices"], dtype=np.complex128)
    production = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=indices,
        thickness_A=(None, float(thickness), None),
        roughness_A=(float(sigma_top), float(sigma_bottom)),
    )
    direct_kz, direct_interfaces, direct_amplitude = _direct_parratt(
        qz,
        float(wavelength),
        indices,
        (None, float(thickness), None),
        np.asarray((sigma_top, sigma_bottom)),
    )
    scalar_error = max(
        float(np.max(np.abs(production.kz_Ainv - direct_kz))),
        float(np.max(np.abs(production.interface_amplitude - direct_interfaces))),
        float(np.max(np.abs(production.amplitude - direct_amplitude))),
    )
    pack_error = float(
        np.max(
            np.abs(
                production.reflectivity
                - np.asarray(pack["parratt_reflectivity"], dtype=np.float64)[sample_rows]
            )
        )
    )
    collapsed = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=indices,
        thickness_A=(None, 0.0, None),
        roughness_A=(0.0, 0.0),
    )
    bare = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=indices[[0, 2]],
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    collapse_error = float(np.max(np.abs(collapsed.amplitude - bare.amplitude)))
    thick = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=indices,
        thickness_A=(None, 1.0e7, None),
        roughness_A=(0.0, 0.0),
    )
    thick_limit_error = float(np.max(np.abs(thick.amplitude - thick.interface_amplitude[:, 0])))

    k0 = 2.0 * math.pi / float(wavelength)
    ambient_qz = np.asarray((2.2 * k0,))
    ambient_indices = np.asarray((1.2 + 0.0j, 1.5 + 0.0j))
    ambient = parratt_reflectivity(
        ambient_qz,
        float(wavelength),
        refractive_index=ambient_indices,
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    ambient_kz, ambient_interfaces, ambient_amplitude = _direct_parratt(
        ambient_qz,
        float(wavelength),
        ambient_indices,
        (None, None),
        np.asarray((0.0,)),
    )
    ambient_error = max(
        float(np.max(np.abs(ambient.kz_Ainv - ambient_kz))),
        float(np.max(np.abs(ambient.interface_amplitude - ambient_interfaces))),
        float(np.max(np.abs(ambient.amplitude - ambient_amplitude))),
    )
    passed = bool(
        scalar_error <= TOLERANCES["parratt"]["atol"]
        and pack_error <= TOLERANCES["parratt"]["atol"]
        and collapse_error <= TOLERANCES["parratt"]["atol"]
        and thick_limit_error <= TOLERANCES["parratt"]["atol"]
        and ambient_error <= TOLERANCES["parratt"]["atol"]
        and production.normalization == "dimensionless pure Parratt reflectivity"
    )
    check = _check(
        "parratt_scalar_and_pack_sample",
        passed,
        {
            "sample_count": int(sample_rows.size),
            "maximum_scalar_stage_error": scalar_error,
            "maximum_pack_reflectivity_error": pack_error,
            "zero_thickness_collapse_error": collapse_error,
            "thick_film_top_interface_error": thick_limit_error,
            "arbitrary_lossless_ambient_stage_error": ambient_error,
        },
    )
    comparison = {
        "case_id": "reflectivity.parratt_three_layer",
        "classification": "MATCH",
        "first_divergence_stage": None,
        "maximum_pack_reflectivity_error": pack_error,
        "independent_oracle_check_id": check["check_id"],
    }
    return check, comparison


def _specular_check(pack: dict[str, NDArray[Any]]) -> dict[str, object]:
    wavelength, c_A, thickness, qc_Ainv, sigma_top, sigma_bottom = np.asarray(
        pack["parratt_parameters"], dtype=np.float64
    )
    qz = np.linspace(2.0, 11.0, 19) * qc_Ainv
    parratt = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=np.asarray(pack["parratt_indices"], dtype=np.complex128),
        thickness_A=(None, float(thickness), None),
        roughness_A=(float(sigma_top), float(sigma_bottom)),
    )
    result = manuscript_specular_composite(
        parratt,
        lambda layer: 3.0 + 0.05 * layer**2,
        c_A=float(c_A),
        qc_Ainv=float(qc_Ainv),
        film_layer_index=1,
    )
    external_l = qz * c_A / (2.0 * np.pi)
    expected_phase_l = 2.0 * np.maximum(parratt.kz_Ainv[:, 1].real, 0.0) * c_A / (2.0 * np.pi)
    expected_raw = 3.0 + 0.05 * external_l**2
    expected_shape = ((3.0 + 0.05 * expected_phase_l**2) / 3.0) / qz**2
    scale_points = (qz / qc_Ainv > 5.0) & (qz / qc_Ainv < 10.0)
    expected_scale = np.exp(
        np.median(np.log(parratt.reflectivity[scale_points]) - np.log(expected_shape[scale_points]))
    )
    phase_error = float(np.max(np.abs(result.phase_l_coordinate - expected_phase_l)))
    raw_error = float(np.max(np.abs(result.raw_kinematic_e2 - expected_raw)))
    high_branch_error = float(
        np.max(np.abs(result.scaled_high_branch - expected_scale * expected_shape))
    )
    passed = bool(
        np.array_equal(result.parratt_reflectivity, parratt.reflectivity)
        and phase_error == 0.0
        and raw_error == 0.0
        and high_branch_error <= 2e-15
        and np.all(np.isfinite(result.composite_reflectivity))
        and result.raw_kinematic_normalization == "raw finite-stack electron2"
        and result.parratt_normalization == "dimensionless pure Parratt reflectivity"
        and result.composite_normalization == "dimensionless manuscript specular composite"
        and not result.composite_reflectivity.flags.writeable
    )
    return _check(
        "named_specular_output_contract",
        passed,
        {
            "sample_count": int(qz.size),
            "pure_parratt_preserved": np.array_equal(
                result.parratt_reflectivity, parratt.reflectivity
            ),
            "maximum_internal_phase_L_error": phase_error,
            "maximum_raw_kinematic_error_electron2": raw_error,
            "maximum_normalized_qz2_high_branch_error": high_branch_error,
            "raw_normalization": result.raw_kinematic_normalization,
            "parratt_normalization": result.parratt_normalization,
            "composite_normalization": result.composite_normalization,
        },
    )


def _proof_metadata() -> dict[str, object]:
    environment = {
        "gemmi": gemmi.__version__,
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "xraydb": xraydb.__version__,
    }
    encoded_environment = json.dumps(
        environment,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "schema_version": 1,
        "task_id": "T04",
        "base_sha": PROOF_BASE_SHA,
        "commit_sha": _git("rev-parse", "HEAD"),
        "contract_version": CONTRACT_API_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "environment_sha256": hashlib.sha256(encoded_environment).hexdigest(),
        "environment": environment,
    }


def _contract_requests() -> list[dict[str, object]]:
    return [
        {
            "request_id": "IR-T04-MEASURE-GAUGE",
            "status": "BLOCKING",
            "owner": "future reviewed shared-contract/integration change",
            "t06_role": "record reviewed acceptance; do not silently change production",
            "problems": [
                (
                    "EventIntensityResult.intensity_per_sr carries raw |F_e|^2 electron2 "
                    "without r_e^2 or detector solid angle"
                ),
                (
                    "LayerAmplitudeResult cannot declare electron units, motif normalization, "
                    "positive Fourier sign, Pb-centered origin/gauge, or event/rod alignment"
                ),
                (
                    "the finite-stack qz-times-depth interface does not identify the qz frame or "
                    "the owner of layer projection and registry phase"
                ),
                "T04 has no reviewed versioned tolerance artifact or recorded tolerance hash",
            ],
            "decision": (
                "freeze whether EventIntensityResult is raw electron2 or true per_sr and assign "
                "r_e^2 and detector solid angle exactly once; freeze LayerAmplitudeResult as "
                "event/rod-aligned complex electron amplitudes for one Pb-centered I-Pb-I motif "
                "with occupancy, anomalous, and displacement factors, exp(+i Q dot r), a fixed "
                "F-plus/F-minus mapping, and no registry/population/optical/solid-angle factor; "
                "freeze a frame-tagged layer projection or event-aligned phase owner and publish "
                "reviewed tolerances with version and hash"
            ),
            "acceptance": (
                "T04 and T05 preserve query event and rod order and interoperate without an "
                "adapter, false per-sr labeling, hidden phase/scaling, or duplicate factors; T04 "
                "loads the reviewed tolerance artifact and records its version/hash"
            ),
        }
    ]


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    """Run the compact permanent T04 proof without generating artifacts."""

    metadata = _proof_metadata()
    if not PACK_PATH.is_file():
        if allow_missing_pack:
            return {
                **metadata,
                "status": "BLOCKED",
                "reference_pack_sha256s": {},
                "checks": [
                    {
                        "check_id": "reference_pack",
                        "status": "SKIP",
                        "evidence": "immutable reference pack is missing",
                    }
                ],
                "classifications": [],
                "convergence": [],
                "limitations": ["immutable reference pack is missing"],
                "contract_requests": _contract_requests(),
            }
        raise FileNotFoundError(PACK_PATH)

    pack, pack_hash = _reference_pack()
    crystal = read_crystal(
        ROOT / "examples" / "bi2se3" / "structures" / "Bi2Se3_vesta.cif",
        phase_id="bi2se3",
    )
    scalar_check, ordered_comparison = _scalar_and_raw_check(crystal, pack)
    ql_evidence = run_bi2se3_ql_proof(str(ROOT))
    ql_check = _check(
        "bi2se3_ql_and_rod_identity",
        ql_evidence["status"] == "PASS",
        ql_evidence,
    )
    parratt_check, parratt_comparison = _parratt_check(pack)
    checks = [
        scalar_check,
        ql_check,
        _pbi2_check(),
        _finite_stack_check(),
        parratt_check,
        _specular_check(pack),
    ]
    status = "BLOCKED" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    return {
        **metadata,
        "status": status,
        "reference_pack_sha256s": {"rasim_reference_v1": pack_hash},
        "checks": checks,
        "classifications": [
            {**ordered_comparison, "ledger_ids": ["PHY-ORD-009"]},
            {**parratt_comparison, "ledger_ids": ["PHY-REF-002"]},
            {
                "case_id": "reflectivity.manuscript_specular_composite",
                "classification": "NO_ORACLE",
                "ledger_ids": ["PHY-REF-007"],
                "first_divergence_stage": None,
                "reason": "the immutable pack contains pure Parratt but no composite observable",
            },
        ],
        "convergence": [],
        "contract_requests": _contract_requests(),
        "limitations": [
            "T04 does not implement stacking disorder, mosaic, detector geometry, fitting, CLI, or GUI",
            "the protected EventIntensityResult.intensity_per_sr output contains raw electron2 and is not a true per-steradian measure",
            "layer phase/projection ownership, motif gauge metadata, and reviewed tolerance provenance remain unresolved shared-contract gates",
            "this exact numerical core has no approximation or refinement variable, so convergence is not fabricated",
            "benchmarks and negative controls are disposable handoff evidence, not permanent proof frameworks",
        ],
    }
