"""Compact T04 analytic, reference, convergence, mutation, and benchmark proof."""

# ruff: noqa: E402

from __future__ import annotations

import cmath
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import tomllib
import tracemalloc
from pathlib import Path
from statistics import median
from typing import Any

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_variable in _THREAD_VARIABLES:
    os.environ[_thread_variable] = "1"

import gemmi
import numpy as np
import xraydb
from numpy.typing import NDArray

from rasim_next.core.contracts import RodQueryBatch
from rasim_next.materials import (
    CLASSICAL_ELECTRON_RADIUS_A,
    CrystalSite,
    CrystalStructure,
    material_optics,
    read_crystal,
)
from rasim_next.materials.optics import HC_EV_A
from rasim_next.ordered.amplitudes import ordered_event_result, unit_cell_amplitude
from rasim_next.ordered.bi2se3_proof import run_bi2se3_ql_proof
from rasim_next.ordered.finite_stack import coherent_finite_stack, uniform_finite_stack
from rasim_next.ordered.motifs import extract_pbi2_motifs, pbi2_layer_amplitudes
from rasim_next.ordered.pbi2_proof import run_pbi2_polytype_proof
from rasim_next.proof.traces import Measure, QuantityKind, TraceRecord, compare_traces
from rasim_next.reciprocal.lattice import ReciprocalLattice
from rasim_next.reciprocal.rods import build_rod_catalog
from rasim_next.reflectivity.parratt import parratt_reflectivity
from rasim_next.reflectivity.specular import manuscript_specular_composite

ROOT = Path(__file__).resolve().parents[3]
PACK_PATH = ROOT / "reference" / "rasim_reference_v1.npz"
WAVELENGTH_A = 1.540592925
TOLERANCES: dict[str, dict[str, float]] = {
    "exact_identity": {"atol": 0.0, "rtol": 0.0},
    "direct_atom_e": {"atol": 1e-10, "rtol": 1e-12},
    "vesta_003_component_e": {"atol": 0.01, "rtol": 0.0},
    "bi2se3_reconstruction_e": {"atol": 1e-10, "rtol": 1e-12},
    "bi2se3_vesta_component_e": {"atol": 1e-4, "rtol": 0.0},
    "bi2se3_vesta_rms_e": {"atol": 5e-5, "rtol": 0.0},
    "bi2se3_vesta_magnitude_e": {"atol": 6e-4, "rtol": 0.0},
    "bi2se3_vesta_relative_magnitude": {"atol": 5e-6, "rtol": 0.0},
    "bi2se3_vesta_d_A": {"atol": 6e-7, "rtol": 0.0},
    "bi2se3_vesta_two_theta_deg": {"atol": 1.2e-4, "rtol": 0.0},
    "reciprocal_d_A": {"atol": 2e-12, "rtol": 2e-12},
    "finite_stack_amplitude_e": {"atol": 1e-10, "rtol": 1e-12},
    "parratt_qz_Ainv": {"atol": 5e-12, "rtol": 0.0},
    "parratt_phase_L": {"atol": 1e-11, "rtol": 0.0},
    "parratt_reflectivity": {"atol": 5e-11, "rtol": 0.0},
    "composite_convergence": {"atol": 1e-10, "rtol": 0.002},
}
TOLERANCE_VERSION = "ordered-reflectivity-tolerances-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any, *, allow_nan: bool = True) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=allow_nan,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _environment_sha256() -> str:
    payload = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xraydb": xraydb.__version__,
        "xraydb_database": xraydb.get_xraydb().get_version(),
        "gemmi": gemmi.__version__,
        "threads": {name: os.environ[name] for name in _THREAD_VARIABLES},
    }
    return _canonical_json_sha256(payload)


def _stats(
    error: NDArray[np.floating[Any]] | NDArray[np.complexfloating[Any, Any]],
) -> dict[str, float]:
    absolute = np.abs(np.asarray(error)).ravel()
    return {
        "maximum": float(np.max(absolute, initial=0.0)),
        "rms": float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        "percentile_95": float(np.percentile(absolute, 95)) if absolute.size else 0.0,
    }


def _check(
    check_id: str,
    passed: bool,
    evidence: object,
    stage_ids: list[str],
    *,
    maximum_error: float = 0.0,
    percentile_95_error: float = 0.0,
    tolerance: object = None,
) -> dict[str, object]:
    return {
        "check_id": check_id,
        "status": "PASS" if passed else "FAIL",
        "evidence": evidence,
        "stage_ids": stage_ids,
        "maximum_error": float(maximum_error),
        "percentile_95_error": float(percentile_95_error),
        "failing_element": None,
        "tolerance": tolerance,
    }


def _direct_atom_amplitudes(
    crystal: CrystalStructure,
    hkl: NDArray[np.float64],
    wavelength_A: NDArray[np.float64],
    *,
    include_occupancy: bool = True,
    include_anomalous: bool = True,
    include_displacement: bool = True,
    unknown_u_iso_A2: float | None = None,
) -> NDArray[np.complex128]:
    """Independent proof-only scalar atom sum using direct XrayDB calls."""

    indices = np.asarray(hkl, dtype=np.float64)
    wavelength = np.broadcast_to(np.asarray(wavelength_A, dtype=np.float64), (indices.shape[0],))
    reciprocal_basis = 2.0 * np.pi * np.linalg.inv(crystal.direct_basis_A).T
    q_vectors = indices @ reciprocal_basis.T
    q_magnitude = np.linalg.norm(q_vectors, axis=1)
    energy = HC_EV_A / wavelength
    if include_anomalous:
        unique_energy, inverse = np.unique(energy, return_inverse=True)
    factors: list[NDArray[np.complex128]] = []
    for site in crystal.sites:
        requested = site.element
        if site.charge:
            sign = "+" if site.charge > 0 else "-"
            ionic = f"{site.element}{abs(site.charge)}{sign}"
            if ionic in xraydb.f0_ions(site.element):
                requested = ionic
        f0 = np.asarray(xraydb.f0(requested, q_magnitude / (4.0 * np.pi)), dtype=np.float64)
        if include_anomalous:
            f1 = np.asarray(
                [xraydb.f1_chantler(site.element, float(value)) for value in unique_energy]
            )[inverse]
            f2 = np.asarray(
                [xraydb.f2_chantler(site.element, float(value)) for value in unique_energy]
            )[inverse]
            factors.append(np.asarray(f0 + f1 + 1.0j * f2, dtype=np.complex128))
        else:
            factors.append(np.asarray(f0, dtype=np.complex128))

    result = np.empty(indices.shape[0], dtype=np.complex128)
    for event_index, miller in enumerate(indices):
        contributions: list[complex] = []
        for site_index, site in enumerate(crystal.sites):
            occupancy = site.occupancy if include_occupancy else 1.0
            if site.u_iso_A2 is None:
                if unknown_u_iso_A2 is None:
                    raise ValueError("direct oracle requires explicit unknown Uiso")
                u_iso = unknown_u_iso_A2
            else:
                u_iso = site.u_iso_A2
            damping = (
                math.exp(-0.5 * u_iso * q_magnitude[event_index] ** 2)
                if include_displacement
                else 1.0
            )
            phase = cmath.exp(2.0j * math.pi * float(np.dot(miller, np.asarray(site.fractional))))
            contributions.append(occupancy * factors[site_index][event_index] * damping * phase)
        result[event_index] = complex(
            math.fsum(value.real for value in contributions),
            math.fsum(value.imag for value in contributions),
        )
    return result


def _direct_parratt(
    qz_Ainv: NDArray[np.float64],
    wavelength_A: float,
    indices: NDArray[np.complex128],
    thickness_A: tuple[float | None, ...],
    roughness_A: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    kz_rows: list[list[complex]] = []
    interface_rows: list[list[complex]] = []
    amplitudes: list[complex] = []
    k0 = 2.0 * math.pi / wavelength_A
    for qz in qz_Ainv:
        kz: list[complex] = []
        for refractive_index in indices:
            index = complex(refractive_index)
            root = cmath.sqrt((index * index - 1.0) * k0**2 + (0.5 * float(qz)) ** 2)
            if root.imag < 0.0 or (root.imag == 0.0 and root.real < 0.0):
                root = -root
            kz.append(root)
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


def _load_pack() -> tuple[dict[str, NDArray[Any]], str]:
    manifest = tomllib.loads(
        (ROOT / "reference" / "reference_manifest.toml").read_text(encoding="utf-8")
    )
    expected_hash = manifest["reference_pack"]["sha256"]
    observed_hash = _sha256(PACK_PATH)
    if observed_hash != expected_hash:
        raise ValueError("immutable reference-pack hash mismatch")
    names = {
        "ordered_hkl",
        "ordered_sim_d",
        "ordered_sim_f_real",
        "ordered_sim_f_imag",
        "parratt_qz",
        "parratt_phase_qz",
        "parratt_phase_L",
        "parratt_reflectivity",
        "parratt_parameters",
        "parratt_indices",
    }
    with np.load(PACK_PATH, allow_pickle=False) as source:
        return {name: np.array(source[name], copy=True) for name in names}, observed_hash


def _vesta_table() -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    path = ROOT / "examples" / "bi2se3" / "reference" / "Bi2Se3_vesta_cu_ka1_dmin_0p7.txt"
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:] if line]
    hkl = np.asarray([[float(value) for value in row[:3]] for row in rows], dtype=np.float64)
    amplitude = np.asarray(
        [complex(float(row[4]), float(row[5])) for row in rows], dtype=np.complex128
    )
    return hkl, amplitude


def _reference_checks(
    crystal: CrystalStructure, pack: dict[str, NDArray[Any]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    hkl = np.asarray(pack["ordered_hkl"], dtype=np.float64)
    wavelength = np.full(hkl.shape[0], WAVELENGTH_A)
    production = unit_cell_amplitude(crystal, hkl, wavelength).amplitude_e
    oracle = _direct_atom_amplitudes(crystal, hkl, wavelength)
    direct_error = production - oracle
    direct_stats = _stats(direct_error)
    direct_passed = bool(np.allclose(production, oracle, **TOLERANCES["direct_atom_e"]))

    lattice = ReciprocalLattice.from_crystal(crystal)
    d_A = 2.0 * np.pi / np.linalg.norm(lattice.q_cartesian_Ainv(hkl), axis=1)
    d_error = d_A - np.asarray(pack["ordered_sim_d"])
    d_stats = _stats(d_error)
    historical = np.asarray(pack["ordered_sim_f_real"]) + 1.0j * np.asarray(
        pack["ordered_sim_f_imag"]
    )
    historical_stats = _stats(production - historical)
    pack_row_003 = int(np.flatnonzero(np.all(hkl == (0.0, 0.0, 3.0), axis=1))[0])

    vesta_hkl, vesta_amplitude = _vesta_table()
    vesta_production = unit_cell_amplitude(
        crystal, vesta_hkl, np.full(vesta_hkl.shape[0], WAVELENGTH_A)
    ).amplitude_e
    vesta_stats = _stats(vesta_production - vesta_amplitude)
    row_003 = int(np.flatnonzero(np.all(vesta_hkl == (0.0, 0.0, 3.0), axis=1))[0])
    component_error_003 = np.abs(
        np.asarray(
            [
                vesta_production[row_003].real - vesta_amplitude[row_003].real,
                vesta_production[row_003].imag - vesta_amplitude[row_003].imag,
            ]
        )
    )

    checks = [
        _check(
            "direct_atom_oracle",
            direct_passed,
            {
                "events": int(hkl.shape[0]),
                "equation": "B=2*pi*inv(A).T; scalar XrayDB atom sum; positive phase",
            },
            ["ordered.atomic_amplitude", "ordered.unit_cell_amplitude"],
            maximum_error=direct_stats["maximum"],
            percentile_95_error=direct_stats["percentile_95"],
            tolerance=TOLERANCES["direct_atom_e"],
        ),
        _check(
            "vesta_fixed_checkpoint",
            float(np.max(component_error_003)) <= 0.01,
            {
                "target_F003_e": [
                    float(vesta_amplitude[row_003].real),
                    float(vesta_amplitude[row_003].imag),
                ],
                "production_F003_e": [
                    float(vesta_production[row_003].real),
                    float(vesta_production[row_003].imag),
                ],
                "component_errors_e": component_error_003.tolist(),
                "broad_206_row_descriptive_residual_e": vesta_stats,
            },
            ["ordered.unit_cell_amplitude"],
            maximum_error=float(np.max(component_error_003)),
            percentile_95_error=float(np.percentile(component_error_003, 95)),
            tolerance=TOLERANCES["vesta_003_component_e"],
        ),
        _check(
            "ordered_corrected_reference",
            d_stats["maximum"] <= 2e-12,
            {
                "prior_reciprocal_d_error_A": d_stats,
                "historical_pack_amplitude_residual_e": historical_stats,
                "historical_pack_F003_e": [
                    float(historical[pack_row_003].real),
                    float(historical[pack_row_003].imag),
                ],
                "first_divergence": "ordered.unit_cell_amplitude",
                "pack_has_atomic_trace": False,
                "downstream_oracle": "direct_atom_oracle",
            },
            ["ordered.unit_cell_amplitude"],
            maximum_error=historical_stats["maximum"],
            percentile_95_error=historical_stats["percentile_95"],
            tolerance={"classification": "CORRECTED", "reciprocal_d_A": 2e-12},
        ),
    ]
    comparisons = [
        {
            "case_id": "ordered.bi2se3_vesta",
            "immutable_pack_classification": "MATCH",
            "classification": "CORRECTED",
            "prior_matching_evidence": "reciprocal d-spacing",
            "prior_stage_maximum_error_A": d_stats["maximum"],
            "first_divergence_stage": "ordered.unit_cell_amplitude",
            "historical_pack_residual_e": historical_stats,
            "independent_oracle_check_id": "direct_atom_oracle",
            "atomic_stage_claimed": False,
        }
    ]
    return checks, comparisons


def _foundation_checks(crystal: CrystalStructure) -> list[dict[str, object]]:
    lattice = ReciprocalLattice.from_crystal(crystal)
    dual_error = crystal.direct_basis_A.T @ lattice.basis_Ainv - 2.0 * np.pi * np.eye(3)
    dual_stats = _stats(dual_error)

    optics = material_optics(crystal, np.asarray([WAVELENGTH_A]))
    optics_reference = np.asarray(
        [
            7.663983929529866,
            1.9282298010759586e-5,
            1.5800513246371108e-6,
            1.2888226482734936e-5,
        ]
    )
    density = float(optics.provenance.split("density_g_cm3=")[1].split(";")[0])
    optics_actual = np.asarray([density, optics.delta[0], optics.beta[0], optics.mu_Ainv[0]])
    optics_relative = np.abs((optics_actual - optics_reference) / optics_reference)
    optics_identity = np.max(np.abs(optics.n_complex - (1.0 - optics.delta + 1.0j * optics.beta)))

    catalog = build_rod_catalog(crystal, h_bounds=(-1, 1), k_bounds=(-1, 1))
    selected_hk = ((1, 0), (0, 1), (1, -1))
    rows = [int(np.flatnonzero((catalog.h == h) & (catalog.k == k))[0]) for h, k in selected_hk]
    l_coordinate = np.asarray([0.5, 1.25, -0.75])
    query = RodQueryBatch(
        event_id=np.asarray([8, 3, 12]),
        rod_id=catalog.rod_id[rows],
        phase_id=(crystal.phase_id,) * 3,
        h=np.asarray([value[0] for value in selected_hk], dtype=np.int32),
        k=np.asarray([value[1] for value in selected_hk], dtype=np.int32),
        qz_Ainv=l_coordinate * lattice.basis_Ainv[2, 2],
        l_coordinate=l_coordinate,
        wavelength_A=np.full(3, WAVELENGTH_A),
    )
    ordered = ordered_event_result(crystal, catalog, query)
    raw_scale_error = float(
        np.max(np.abs(ordered.intensity.intensity_per_sr - np.abs(ordered.amplitude_e) ** 2))
    )

    expected_orientations = {
        "PbI2_2H.cif": ("minus",),
        "PbI2_4H.cif": ("plus", "minus"),
        "PbI2_6H.cif": ("plus", "plus", "plus"),
    }
    motif_evidence: dict[str, object] = {}
    for filename, expected in expected_orientations.items():
        pbi2 = read_crystal(
            ROOT / "examples" / "pbi2" / "structures" / filename,
            phase_id=filename.removesuffix(".cif"),
        )
        motifs = extract_pbi2_motifs(pbi2)
        orientations = tuple(motif.orientation for motif in motifs)
        motif_evidence[filename] = {
            "orientations": list(orientations),
            "covered_site_count": len(
                {atom.site_index for motif in motifs for atom in motif.atoms}
            ),
        }
        if orientations != expected:
            motif_evidence[filename]["mismatch"] = True

    pbi2 = read_crystal(ROOT / "examples" / "pbi2" / "structures" / "PbI2_4H.cif", phase_id="pbi2")
    pbi2_catalog = build_rod_catalog(pbi2, h_bounds=(1, 1), k_bounds=(0, 0))
    pbi2_l = np.asarray([0.75, -0.75])
    pbi2_query = RodQueryBatch(
        event_id=np.asarray([5, 1]),
        rod_id=np.repeat(pbi2_catalog.rod_id, 2),
        phase_id=("pbi2", "pbi2"),
        h=np.ones(2, dtype=np.int32),
        k=np.zeros(2, dtype=np.int32),
        qz_Ainv=pbi2_l * ReciprocalLattice.from_crystal(pbi2).basis_Ainv[2, 2],
        l_coordinate=pbi2_l,
        wavelength_A=np.full(2, WAVELENGTH_A),
    )
    layer = pbi2_layer_amplitudes(pbi2, pbi2_query, unknown_u_iso_A2=0.0)
    motif_relation_error = float(abs(layer.f_minus[0] - layer.f_plus[1]))

    event_id = np.arange(3, dtype=np.int64)
    qz = np.asarray([0.0, 0.37, 2.0 * np.pi / 4.2])
    repeat = np.asarray([1.2 + 0.3j, -0.4 + 0.8j, 0.7 - 0.1j])
    count = 7
    uniform = uniform_finite_stack(event_id, qz, repeat, 4.2, count)
    explicit = coherent_finite_stack(
        event_id,
        qz,
        np.repeat(repeat[:, None], count, axis=1),
        4.2 * np.arange(count),
    )
    finite_error = _stats(uniform.amplitude_e - explicit.amplitude_e)

    return [
        _check(
            "crystallographic_invariants",
            dual_stats["maximum"] <= 2e-12,
            {"dual_basis_error": dual_stats},
            ["ordered.atomic_amplitude"],
            maximum_error=dual_stats["maximum"],
            percentile_95_error=dual_stats["percentile_95"],
            tolerance=TOLERANCES["reciprocal_d_A"],
        ),
        _check(
            "material_optics_consistency",
            float(np.max(optics_relative)) <= 2e-12 and optics_identity == 0.0,
            {
                "density_delta_beta_mu": optics_actual.tolist(),
                "maximum_relative_reference_error": float(np.max(optics_relative)),
                "n_identity_error": float(optics_identity),
            },
            ["ordered.atomic_amplitude", "reflectivity.layer_kz"],
            maximum_error=float(np.max(optics_relative)),
            tolerance={"relative": 2e-12},
        ),
        _check(
            "rod_identity_and_raw_event_scale",
            catalog.rod_id.size == 9
            and np.unique(catalog.rod_id).size == 9
            and len({catalog.family_id[row] for row in rows}) == 1
            and raw_scale_error == 0.0,
            {
                "rod_count": int(catalog.rod_id.size),
                "event_ids": ordered.event_id.tolist(),
                "normalization": ordered.intensity.normalization,
                "universal_scale_applied": False,
            },
            ["reciprocal.rod_id", "ordered.event_intensity"],
            maximum_error=raw_scale_error,
            tolerance=TOLERANCES["exact_identity"],
        ),
        _check(
            "motif_and_layer_oracle",
            not any("mismatch" in value for value in motif_evidence.values())
            and motif_relation_error <= 1e-10,
            {
                "tracked_structures": motif_evidence,
                "Fminus_hkL_minus_Fplus_hkminusL_e": motif_relation_error,
                "gauge": "Pb-centered; registry-free; occupancy and Uiso included",
            },
            ["ordered.layer_amplitude"],
            maximum_error=motif_relation_error,
            tolerance=TOLERANCES["direct_atom_e"],
        ),
        _check(
            "finite_stack_limits",
            finite_error["maximum"] <= 1e-10
            and uniform.intensity.normalization.endswith("electron2"),
            {
                "direct_vs_stable": finite_error,
                "zero_phase_intensity_e2": float(uniform.intensity.intensity_per_sr[0]),
                "normalization": uniform.intensity.normalization,
            },
            ["ordered.finite_stack_amplitude", "ordered.event_intensity"],
            maximum_error=finite_error["maximum"],
            percentile_95_error=finite_error["percentile_95"],
            tolerance=TOLERANCES["finite_stack_amplitude_e"],
        ),
    ]


def _parratt_checks(
    pack: dict[str, NDArray[Any]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    wavelength, c_A, thickness, _, sigma_top, sigma_bottom = np.asarray(
        pack["parratt_parameters"], dtype=np.float64
    )
    qz = np.asarray(pack["parratt_qz"], dtype=np.float64)
    indices = np.asarray(pack["parratt_indices"], dtype=np.complex128)
    production = parratt_reflectivity(
        qz,
        float(wavelength),
        refractive_index=indices,
        thickness_A=(None, float(thickness), None),
        roughness_A=(float(sigma_top), float(sigma_bottom)),
    )
    phase_qz = 2.0 * np.maximum(production.kz_Ainv[:, 1].real, 0.0)
    phase_l = phase_qz * float(c_A) / (2.0 * np.pi)
    reflectivity_stats = _stats(production.reflectivity - pack["parratt_reflectivity"])
    phase_qz_stats = _stats(phase_qz - pack["parratt_phase_qz"])
    phase_l_stats = _stats(phase_l - pack["parratt_phase_L"])
    direct_kz, direct_interfaces, direct_amplitude = _direct_parratt(
        qz,
        float(wavelength),
        indices,
        (None, float(thickness), None),
        np.asarray([sigma_top, sigma_bottom]),
    )
    direct_error = max(
        _stats(production.kz_Ainv - direct_kz)["maximum"],
        _stats(production.interface_amplitude - direct_interfaces)["maximum"],
        _stats(production.amplitude - direct_amplitude)["maximum"],
    )

    zero_thickness = parratt_reflectivity(
        qz[::64],
        float(wavelength),
        refractive_index=indices,
        thickness_A=(None, 0.0, None),
        roughness_A=(0.0, 0.0),
    )
    bare = parratt_reflectivity(
        qz[::64],
        float(wavelength),
        refractive_index=indices[[0, 2]],
        thickness_A=(None, None),
        roughness_A=(0.0,),
    )
    collapse_error = _stats(zero_thickness.amplitude - bare.amplitude)
    passed = (
        reflectivity_stats["maximum"] <= TOLERANCES["parratt_reflectivity"]["atol"]
        and phase_qz_stats["maximum"] <= TOLERANCES["parratt_qz_Ainv"]["atol"]
        and phase_l_stats["maximum"] <= TOLERANCES["parratt_phase_L"]["atol"]
        and collapse_error["maximum"] <= 2e-11
        and direct_error <= TOLERANCES["parratt_reflectivity"]["atol"]
    )
    checks = [
        _check(
            "parratt_analytic_and_pack_match",
            passed,
            {
                "pack_reflectivity_error": reflectivity_stats,
                "pack_phase_qz_error_Ainv": phase_qz_stats,
                "pack_phase_L_error": phase_l_stats,
                "zero_thickness_collapse_error": collapse_error,
                "direct_scalar_stage_maximum_error": direct_error,
            },
            [
                "reflectivity.layer_kz",
                "reflectivity.interface_amplitude",
                "reflectivity.recursion_amplitude",
                "reflectivity.parratt_intensity",
            ],
            maximum_error=reflectivity_stats["maximum"],
            percentile_95_error=reflectivity_stats["percentile_95"],
            tolerance=TOLERANCES["parratt_reflectivity"],
        )
    ]
    comparisons = [
        {
            "case_id": "reflectivity.parratt_three_layer",
            "immutable_pack_classification": "MATCH",
            "classification": "MATCH",
            "first_divergence_stage": None,
            "reflectivity_error": reflectivity_stats,
            "phase_qz_error_Ainv": phase_qz_stats,
            "phase_L_error": phase_l_stats,
            "independent_oracle_check_id": "parratt_analytic_and_pack_match",
        },
        {
            "case_id": "reflectivity.manuscript_specular_composite",
            "immutable_pack_classification": None,
            "classification": "NO_ORACLE",
            "first_divergence_stage": None,
            "reason": "immutable pack contains pure Parratt arrays but no composite arrays",
            "independent_oracle_check_id": "nested_grid_convergence",
        },
    ]
    return checks, comparisons


def _convergence(pack: dict[str, NDArray[Any]]) -> dict[str, object]:
    wavelength, c_A, _, qc_Ainv, sigma_top, sigma_bottom = np.asarray(
        pack["parratt_parameters"], dtype=np.float64
    )
    indices = np.asarray(pack["parratt_indices"], dtype=np.complex128)
    sizes = (1025, 2049, 4097)
    outputs = []
    for size in sizes:
        q_over_qc = np.linspace(2.0, 11.0, size)
        parratt = parratt_reflectivity(
            q_over_qc * qc_Ainv,
            wavelength,
            refractive_index=indices,
            thickness_A=(None, 500.0, None),
            roughness_A=(sigma_top, sigma_bottom),
        )
        specular = manuscript_specular_composite(
            parratt,
            lambda layer: 1.0 + 0.05 * layer**2,
            c_A=c_A,
            qc_Ainv=qc_Ainv,
            film_layer_index=1,
        )
        outputs.append(specular)
    parratt_errors = [
        float(
            np.max(
                np.abs(
                    outputs[row].parratt_reflectivity - outputs[row + 1].parratt_reflectivity[::2]
                )
            )
        )
        for row in range(2)
    ]
    composite_errors = [
        float(
            np.max(
                np.abs(
                    outputs[row].composite_reflectivity
                    - outputs[row + 1].composite_reflectivity[::2]
                )
            )
        )
        for row in range(2)
    ]
    blend_bounds = [list(output.blend_bounds_q_over_qc) for output in outputs]
    bound_deltas = [
        float(np.max(np.abs(np.asarray(blend_bounds[row]) - np.asarray(blend_bounds[row + 1]))))
        for row in range(2)
    ]
    scale_factors = [float(output.scale_factor) for output in outputs]
    scale_relative_deltas = [
        abs(scale_factors[row + 1] / scale_factors[row] - 1.0) for row in range(2)
    ]
    fine_spacing = 9.0 / (sizes[-1] - 1)
    fine_scale = float(np.max(outputs[-1].composite_reflectivity))
    passed = (
        max(parratt_errors) <= 5e-13
        and max(composite_errors) <= 1e-10 + 0.002 * fine_scale
        and bound_deltas[-1] <= 2.0 * fine_spacing + 1e-12
        and scale_relative_deltas[-1] <= 0.002
    )
    return {
        "grid_sizes": list(sizes),
        "q_over_qc_spacings": [9.0 / (size - 1) for size in sizes],
        "parratt_shared_point_max_errors": parratt_errors,
        "composite_shared_point_max_errors": composite_errors,
        "blend_bounds_q_over_qc": blend_bounds,
        "blend_bound_deltas": bound_deltas,
        "scale_factors": scale_factors,
        "scale_relative_deltas": scale_relative_deltas,
        "arbitrary_l_evaluation": "direct callback; no interpolation grid",
        "passed": passed,
    }


def _benchmark(crystal: CrystalStructure, pack: dict[str, NDArray[Any]]) -> dict[str, object]:
    event_count = 10_000
    parratt_count = 4_096
    source_hkl = np.asarray(pack["ordered_hkl"], dtype=np.float64)
    repeats = math.ceil(event_count / source_hkl.shape[0])
    hkl = np.tile(source_hkl, (repeats, 1))[:event_count]
    wavelength = np.full(event_count, WAVELENGTH_A)
    catalog = build_rod_catalog(
        crystal,
        h_bounds=(int(np.min(hkl[:, 0])), int(np.max(hkl[:, 0]))),
        k_bounds=(int(np.min(hkl[:, 1])), int(np.max(hkl[:, 1]))),
    )
    rod_by_hk = {
        (int(h), int(k)): int(rod_id)
        for rod_id, h, k in zip(catalog.rod_id, catalog.h, catalog.k, strict=True)
    }
    lattice = ReciprocalLattice.from_crystal(crystal)
    query = RodQueryBatch(
        event_id=np.arange(event_count, dtype=np.int64),
        rod_id=np.asarray([rod_by_hk[(int(row[0]), int(row[1]))] for row in hkl], dtype=np.int64),
        phase_id=(crystal.phase_id,) * event_count,
        h=np.asarray(hkl[:, 0], dtype=np.int32),
        k=np.asarray(hkl[:, 1], dtype=np.int32),
        qz_Ainv=hkl[:, 2] * lattice.basis_Ainv[2, 2],
        l_coordinate=hkl[:, 2],
        wavelength_A=wavelength,
    )
    parameters = np.asarray(pack["parratt_parameters"], dtype=np.float64)
    parratt_qz = np.resize(np.asarray(pack["parratt_qz"], dtype=np.float64), parratt_count)
    indices = np.asarray(pack["parratt_indices"], dtype=np.complex128)

    def production_work() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        ordered = ordered_event_result(crystal, catalog, query)
        reflectivity = parratt_reflectivity(
            parratt_qz,
            parameters[0],
            refractive_index=indices,
            thickness_A=(None, parameters[2], None),
            roughness_A=(parameters[4], parameters[5]),
        )
        return ordered.amplitude_e, ordered.intensity.intensity_per_sr, reflectivity.reflectivity

    def oracle_work() -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
        amplitude = _direct_atom_amplitudes(crystal, hkl, wavelength)
        _, _, reflection_amplitude = _direct_parratt(
            parratt_qz,
            float(parameters[0]),
            indices,
            (None, float(parameters[2]), None),
            np.asarray(parameters[[4, 5]]),
        )
        return amplitude, np.abs(amplitude) ** 2, np.abs(reflection_amplitude) ** 2

    production_work()
    oracle_work()

    def time_samples(function: Any) -> tuple[list[float], tuple[NDArray[Any], ...]]:
        samples: list[float] = []
        result: tuple[NDArray[Any], ...] = ()
        for _ in range(3):
            gc.collect()
            started = time.perf_counter()
            result = function()
            samples.append(time.perf_counter() - started)
        return samples, result

    production_samples, production_result = time_samples(production_work)
    oracle_samples, oracle_result = time_samples(oracle_work)

    def peak_bytes(function: Any) -> int:
        gc.collect()
        tracemalloc.start()
        result = function()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del result
        return int(peak)

    production_peak = peak_bytes(production_work)
    oracle_peak = peak_bytes(oracle_work)
    structure_error = _stats(production_result[0] - oracle_result[0])
    intensity_error = _stats(production_result[1] - oracle_result[1])
    parratt_error = _stats(production_result[2] - oracle_result[2])
    production_median = float(median(production_samples))
    oracle_median = float(median(oracle_samples))
    direct_limit = 1e-10 + 1e-12 * float(np.max(np.abs(oracle_result[0])))
    passed = (
        structure_error["maximum"] <= direct_limit
        and intensity_error["maximum"]
        <= 2.0 * float(np.max(np.abs(oracle_result[0]))) * direct_limit + direct_limit**2
        and parratt_error["maximum"] <= 5e-11
        and all(value > 0.0 and np.isfinite(value) for value in production_samples + oracle_samples)
    )
    return {
        "equivalent_work": {
            "event_aligned_structure_amplitudes": event_count,
            "three_layer_parratt_points": parratt_count,
            "parratt_distribution": "tracked 257-point qz case repeated to equivalent work count",
        },
        "thread_environment": {name: os.environ[name] for name in _THREAD_VARIABLES},
        "warmup_runs": 1,
        "timed_repeats": 3,
        "production_seconds_samples": production_samples,
        "production_seconds_median": production_median,
        "oracle_seconds_samples": oracle_samples,
        "oracle_seconds_median": oracle_median,
        "speedup": oracle_median / production_median,
        "production_peak_bytes": production_peak,
        "oracle_peak_bytes": oracle_peak,
        "structure_max_error_e": structure_error["maximum"],
        "raw_intensity_max_error_electron2": intensity_error["maximum"],
        "parratt_max_error": parratt_error["maximum"],
        "passed": passed,
    }


def _mutation_record(
    mutation_id: str,
    fixture_id: str,
    stage: str,
    metric: str,
    reference_value: NDArray[Any],
    candidate_value: NDArray[Any],
    unit: str,
    quantity_kind: QuantityKind,
    *,
    prior_stage: str | None = None,
    prior_value: NDArray[Any] | None = None,
    prior_unit: str = "1",
    prior_kind: QuantityKind = QuantityKind.SCALAR,
) -> dict[str, object]:
    reference: list[TraceRecord] = []
    candidate: list[TraceRecord] = []
    if prior_stage is not None:
        if prior_value is None:
            raise ValueError("mutation prior stage requires an explicit value")
        for target in (reference, candidate):
            target.append(
                TraceRecord(
                    fixture_id,
                    prior_stage,
                    prior_value,
                    prior_unit,
                    "crystal",
                    Measure.NONE,
                    prior_kind,
                    "t04-proof-v1",
                    "unmutated prior stage",
                )
            )
    for target, value in ((reference, reference_value), (candidate, candidate_value)):
        target.append(
            TraceRecord(
                fixture_id,
                stage,
                value,
                unit,
                "crystal",
                Measure.NONE,
                quantity_kind,
                "t04-proof-v1",
                "in-memory proof mutation",
            )
        )
    comparison = compare_traces(reference, candidate)
    detected = comparison.first_failing_stage == stage and comparison.failure_metric == metric
    return {
        "mutation_id": mutation_id,
        "fixture_id": fixture_id,
        "expected_first_stage": stage,
        "expected_failure_metric": metric,
        "observed_first_stage": comparison.first_failing_stage,
        "observed_failure_metric": comparison.failure_metric,
        "prior_stages_identical": prior_stage is None or comparison.first_failing_stage == stage,
        "detected": bool(detected),
    }


def _mutations(crystal: CrystalStructure) -> list[dict[str, object]]:
    raw_intensity = np.asarray([0.37, 2.81, 19.4])
    catalog = build_rod_catalog(crystal, h_bounds=(-1, 1), k_bounds=(-1, 1))
    partial = CrystalStructure(
        phase_id="partial",
        spacegroup_hm="P 1",
        direct_basis_A=np.diag([4.0, 5.0, 6.0]),
        volume_A3=120.0,
        sites=(CrystalSite("C1", "C", "C", 0, 0.35, (0.2, 0.3, 0.4), 0.02, 1),),
        source_path=Path("partial.cif"),
        provenance="proof-only analytic fixture",
    )
    partial_hkl = np.asarray([[2.0, 1.0, 0.5]])
    partial_wavelength = np.asarray([WAVELENGTH_A])
    occupied = _direct_atom_amplitudes(partial, partial_hkl, partial_wavelength)
    omitted_occupancy = _direct_atom_amplitudes(
        partial, partial_hkl, partial_wavelength, include_occupancy=False
    )
    high_q = np.asarray([[3.0, 2.0, 10.0]])
    full = _direct_atom_amplitudes(crystal, high_q, np.asarray([WAVELENGTH_A]))
    no_u = _direct_atom_amplitudes(
        crystal, high_q, np.asarray([WAVELENGTH_A]), include_displacement=False
    )
    anomalous = _direct_atom_amplitudes(
        crystal, np.asarray([[0.0, 0.0, 3.0]]), np.asarray([WAVELENGTH_A])
    )
    no_anomalous = _direct_atom_amplitudes(
        crystal,
        np.asarray([[0.0, 0.0, 3.0]]),
        np.asarray([WAVELENGTH_A]),
        include_anomalous=False,
    )
    nonorthogonal = np.asarray([[4.0, 0.7, 0.3], [0.0, 5.0, 0.2], [0.0, 0.0, 6.0]])
    right_basis = 2.0 * np.pi * np.linalg.inv(nonorthogonal).T
    wrong_basis = 2.0 * np.pi * np.linalg.inv(nonorthogonal)
    nonorthogonal_hkl = np.asarray([1.0, 2.0, 0.5])
    atom_position_A = nonorthogonal @ np.asarray([0.23, 0.37, 0.41])

    def atomic_amplitude(reciprocal_basis_Ainv: NDArray[np.float64]) -> NDArray[np.complex128]:
        q_vector = reciprocal_basis_Ainv @ nonorthogonal_hkl
        q_magnitude = float(np.linalg.norm(q_vector))
        energy_eV = HC_EV_A / WAVELENGTH_A
        factor = complex(
            float(np.asarray(xraydb.f0("C", q_magnitude / (4.0 * np.pi))).item())
            + xraydb.f1_chantler("C", energy_eV),
            xraydb.f2_chantler("C", energy_eV),
        )
        value = (
            0.73
            * factor
            * math.exp(-0.5 * 0.02 * q_magnitude**2)
            * cmath.exp(1.0j * float(np.dot(q_vector, atom_position_A)))
        )
        return np.asarray([value], dtype=np.complex128)

    right_atomic_amplitude = atomic_amplitude(right_basis)
    wrong_atomic_amplitude = atomic_amplitude(wrong_basis)
    par = parratt_reflectivity(
        np.asarray([0.025]),
        WAVELENGTH_A,
        refractive_index=np.asarray([1.0 + 0.0j, 0.999979 + 3.2e-7j, 0.99999 + 1e-8j]),
        thickness_A=(None, 500.0, None),
        roughness_A=(0.0, 0.0),
    )
    low = np.asarray([0.8, 0.5])
    high = np.asarray([0.2, 0.1])
    raw_amplitude = np.sqrt(raw_intensity).astype(np.complex128)
    registry_correct = coherent_finite_stack(
        np.asarray([0]),
        np.asarray([0.0]),
        np.ones((1, 2), dtype=np.complex128),
        np.zeros(2),
        registry_phase_rad=np.asarray([[0.0, 2.0 * np.pi / 3.0]]),
    ).amplitude_e
    registry_omitted = coherent_finite_stack(
        np.asarray([0]),
        np.asarray([0.0]),
        np.ones((1, 2), dtype=np.complex128),
        np.zeros(2),
    ).amplitude_e
    return [
        _mutation_record(
            "strongest_normalized_to_100",
            "ordered.varied_raw_intensity",
            "ordered.event_intensity",
            "numeric_value",
            raw_intensity,
            100.0 * raw_intensity / np.max(raw_intensity),
            "electron2",
            QuantityKind.INTENSITY,
            prior_stage="ordered.unit_cell_amplitude",
            prior_value=raw_amplitude,
            prior_unit="e",
            prior_kind=QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "rounded_intensity",
            "ordered.noninteger_raw_intensity",
            "ordered.event_intensity",
            "numeric_value",
            raw_intensity,
            np.round(raw_intensity),
            "electron2",
            QuantityKind.INTENSITY,
            prior_stage="ordered.unit_cell_amplitude",
            prior_value=raw_amplitude,
            prior_unit="e",
            prior_kind=QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "premature_classical_electron_scale",
            "ordered.raw_electron2",
            "ordered.event_intensity",
            "numeric_value",
            raw_intensity,
            CLASSICAL_ELECTRON_RADIUS_A**2 * raw_intensity,
            "electron2",
            QuantityKind.INTENSITY,
            prior_stage="ordered.unit_cell_amplitude",
            prior_value=raw_amplitude,
            prior_unit="e",
            prior_kind=QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "weak_nonzero_rod_pruned",
            "rod.complete_integer_catalog",
            "reciprocal.rod_id",
            "shape",
            catalog.rod_id,
            catalog.rod_id[:-1],
            "1",
            QuantityKind.INDEX,
        ),
        _mutation_record(
            "equal_qr_rods_collapsed",
            "rod.hexagonal_family",
            "reciprocal.rod_id",
            "shape",
            catalog.rod_id,
            np.arange(np.unique(np.round(catalog.qr_Ainv, 12)).size, dtype=np.int64),
            "1",
            QuantityKind.INDEX,
        ),
        _mutation_record(
            "fractional_rod_fabricated",
            "rod.integer_catalog",
            "reciprocal.rod_id",
            "shape",
            catalog.rod_id,
            np.append(catalog.rod_id, catalog.rod_id.size),
            "1",
            QuantityKind.INDEX,
        ),
        _mutation_record(
            "occupancy_omitted",
            "ordered.partial_occupancy",
            "ordered.atomic_amplitude",
            "numeric_value",
            occupied,
            omitted_occupancy,
            "e",
            QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "anomalous_term_omitted",
            "ordered.bi2se3_cu_ka1",
            "ordered.atomic_amplitude",
            "numeric_value",
            anomalous,
            no_anomalous,
            "e",
            QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "isotropic_displacement_omitted",
            "ordered.bi2se3_high_q",
            "ordered.atomic_amplitude",
            "numeric_value",
            full,
            no_u,
            "e",
            QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "reciprocal_transpose_wrong",
            "ordered.nonorthogonal_cell",
            "ordered.atomic_amplitude",
            "numeric_value",
            right_atomic_amplitude,
            wrong_atomic_amplitude,
            "e",
            QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "opposite_parratt_branch",
            "reflectivity.absorbing_subcritical_film",
            "reflectivity.layer_kz",
            "numeric_value",
            par.kz_Ainv[:, 1],
            -par.kz_Ainv[:, 1],
            "A^-1",
            QuantityKind.SCALAR,
        ),
        _mutation_record(
            "registry_translation_omitted",
            "ordered.two_layer_registry",
            "ordered.finite_stack_amplitude",
            "numeric_value",
            registry_correct,
            registry_omitted,
            "e",
            QuantityKind.AMPLITUDE,
            prior_stage="ordered.layer_amplitude",
            prior_value=np.ones(2, dtype=np.complex128),
            prior_unit="e",
            prior_kind=QuantityKind.AMPLITUDE,
        ),
        _mutation_record(
            "blend_outside_handoff",
            "reflectivity.distinct_branches",
            "reflectivity.composite_intensity",
            "numeric_value",
            low,
            0.5 * (low + high),
            "1",
            QuantityKind.INTENSITY,
            prior_stage="reflectivity.parratt_intensity",
            prior_value=low,
            prior_unit="1",
            prior_kind=QuantityKind.INTENSITY,
        ),
    ]


def run_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    if not PACK_PATH.is_file():
        if allow_missing_pack:
            return {
                "schema_version": "rasim-ordered-reflectivity-proof-v1",
                "task_id": "T04",
                "status": "BLOCKED",
                "limitations": ["immutable reference pack is absent"],
            }
        raise FileNotFoundError(PACK_PATH)

    pack, pack_hash = _load_pack()
    crystal = read_crystal(
        ROOT / "examples" / "bi2se3" / "structures" / "Bi2Se3_legacy.cif",
        phase_id="bi2se3",
    )
    checks = _foundation_checks(crystal)
    pbi2_polytype = run_pbi2_polytype_proof(str(ROOT))
    deterministic_material_payload = pbi2_polytype["deterministic_material_payload"]
    deterministic_material_payload_sha256 = _canonical_json_sha256(
        deterministic_material_payload, allow_nan=False
    )
    pbi2_maximum_error = max(float(value) for value in pbi2_polytype["maximum_errors_e"].values())
    checks.append(
        _check(
            "pbi2_polytype_direct_atom_sum",
            pbi2_polytype["status"] == "PASS",
            {
                "maximum_errors_e": pbi2_polytype["maximum_errors_e"],
                "topologies": {
                    name: payload["actual_coordination_topology"]
                    for name, payload in pbi2_polytype["polytypes"].items()
                },
                "deterministic_material_payload_sha256": deterministic_material_payload_sha256,
                "deterministic_parent_period_multiples": [1, 2, 5],
            },
            [
                "ordered.layer_amplitude",
                "ordered.finite_stack_amplitude",
                "ordered.unit_cell_amplitude",
            ],
            maximum_error=pbi2_maximum_error,
            tolerance=TOLERANCES["direct_atom_e"],
        )
    )
    bi2se3_ql = run_bi2se3_ql_proof(
        str(ROOT), TOLERANCES, direct_atom_amplitudes=_direct_atom_amplitudes
    )
    bi2se3_single_ql = bi2se3_ql["single_ql_reconstruction"]
    checks.append(
        _check(
            "bi2se3_single_ql_reconstruction",
            bi2se3_ql["status"] == "PASS" and bi2se3_single_ql["status"] == "PASS",
            bi2se3_single_ql,
            ["ordered.unit_cell_amplitude"],
            maximum_error=max(
                float(bi2se3_single_ql["maximum_direct_amplitude_residual_e"]),
                float(bi2se3_single_ql["maximum_production_amplitude_residual_e"]),
                float(bi2se3_single_ql["maximum_noninteger_L_image_shift_residual_e"]),
            ),
            tolerance=TOLERANCES["direct_atom_e"],
        )
    )
    reference_checks, reference_comparisons = _reference_checks(crystal, pack)
    checks.extend(reference_checks)
    parratt_checks, parratt_comparisons = _parratt_checks(pack)
    checks.extend(parratt_checks)
    reference_comparisons.extend(parratt_comparisons)

    convergence = _convergence(pack)
    checks.append(
        _check(
            "nested_grid_convergence",
            bool(convergence["passed"]),
            convergence,
            ["reflectivity.parratt_intensity", "reflectivity.composite_intensity"],
            maximum_error=max(convergence["composite_shared_point_max_errors"]),
            tolerance=TOLERANCES["composite_convergence"],
        )
    )
    benchmark = _benchmark(crystal, pack)
    checks.append(
        _check(
            "optimized_proof_agreement_and_benchmark",
            bool(benchmark["passed"]),
            {
                "equivalent_work": benchmark["equivalent_work"],
                "structure_max_error_e": benchmark["structure_max_error_e"],
                "raw_intensity_max_error_electron2": benchmark["raw_intensity_max_error_electron2"],
                "parratt_max_error": benchmark["parratt_max_error"],
            },
            [
                "ordered.unit_cell_amplitude",
                "ordered.event_intensity",
                "reflectivity.parratt_intensity",
            ],
            maximum_error=float(benchmark["structure_max_error_e"]),
            tolerance=TOLERANCES["direct_atom_e"],
        )
    )
    mutations = _mutations(crystal)
    mutation_passed = all(item["detected"] for item in mutations)
    checks.append(
        _check(
            "error_injection",
            mutation_passed,
            {
                "detected": sum(bool(item["detected"]) for item in mutations),
                "total": len(mutations),
            },
            sorted({str(item["expected_first_stage"]) for item in mutations}),
            tolerance={"required_detection_fraction": 1.0},
        )
    )
    checks.append(
        _check(
            "legacy_classification_coverage",
            {item["classification"] for item in reference_comparisons}
            == {"MATCH", "CORRECTED", "NO_ORACLE"},
            {
                "cases": [item["case_id"] for item in reference_comparisons],
                "classifications": [item["classification"] for item in reference_comparisons],
            },
            [
                "ordered.unit_cell_amplitude",
                "reflectivity.parratt_intensity",
                "reflectivity.composite_intensity",
            ],
            tolerance={"required": ["MATCH", "CORRECTED", "NO_ORACLE"]},
        )
    )

    classifications = [
        {
            "case_id": item["case_id"],
            "immutable_pack_classification": item["immutable_pack_classification"],
            "classification": item["classification"],
            "first_divergence_stage": item["first_divergence_stage"],
            "independent_oracle_check_id": item["independent_oracle_check_id"],
        }
        for item in reference_comparisons
    ]
    passed = all(check["status"] == "PASS" for check in checks)
    return {
        "schema_version": "rasim-ordered-reflectivity-proof-v1",
        "task_id": "T04",
        "status": "READY" if passed else "FAIL",
        "base_sha": _git("merge-base", "HEAD", "main"),
        "commit_sha": _git("rev-parse", "HEAD"),
        "contract_version": 4,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": pack_hash},
        "environment_sha256": _environment_sha256(),
        "owned_paths": [
            "src/rasim_next/materials/",
            "src/rasim_next/reciprocal/lattice.py",
            "src/rasim_next/reciprocal/rods.py",
            "src/rasim_next/ordered/",
            "src/rasim_next/reflectivity/",
            "tests/test_ordered_reflectivity.py",
            "tasks/04_ordered_reflectivity.md#handoff",
        ],
        "tolerance_version": TOLERANCE_VERSION,
        "tolerance_policy_sha256": _canonical_json_sha256(TOLERANCES),
        "tolerances": TOLERANCES,
        "checks": checks,
        "classifications": classifications,
        "reference_comparisons": reference_comparisons,
        "convergence": convergence,
        "benchmark": benchmark,
        "mutations": mutations,
        "deterministic_material_payload": deterministic_material_payload,
        "deterministic_material_payload_sha256": deterministic_material_payload_sha256,
        "pbi2_polytype_proof": pbi2_polytype,
        "bi2se3_single_ql_reconstruction": bi2se3_single_ql,
        "bi2se3_ql_proof": bi2se3_ql,
        "limitations": [
            "layered L/Qz validation is limited to the declared c-axis-normal crystallographic slice",
            "only isotropic displacement is supported; unknown Uiso requires an explicit calculation value",
            "Bi2Se3 continuous-L QL reconstruction applies explicit per-atom integer image phases before comparison with the wrapped production cell",
            "the VESTA intensity column is NO_ORACLE because every exported value is NaN",
            "the named composite is specular-only and is not an off-specular optical model",
        ],
        "contract_requests": [],
    }
