"""Numerical helper functions used by the simulator."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
import math
from numba import njit

R_E = 2.8179403267e-15
LAMBDA_DEFAULT = 1.54e-10
N_A = 6.022e23
_HC_ANGSTROM_EV = 12398.419843320026
_ELEMENT_SYMBOL_RE = re.compile(r"[A-Z][a-z]?")


# Function to calculate d-spacing for hexagonal crystals
def d_spacing(h, k, l, av, cv):
    if (h, k, l) == (0, 0, 0):
        return None
    term1 = 4 / 3 * (h**2 + h * k + k**2) / av**2
    term2 = (l**2) / cv**2
    return 1 / np.sqrt(term1 + term2)


# Function to calculate 2theta using Bragg's law
def two_theta(d, wavelength):
    if d is None:
        return None
    sin_theta = wavelength / (2 * d)
    if sin_theta > 1:  # This means the reflection is not physically possible
        return None
    theta = np.arcsin(sin_theta)
    return 2 * np.degrees(theta)


SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD = np.deg2rad(1.0e-3)
DEFAULT_BLEND_X2_Q_OVER_QC = 6.0
DEFAULT_BLEND_SEARCH_MIN_Q_OVER_QC = 3.0
DEFAULT_BLEND_SEARCH_MAX_Q_OVER_QC = 10.0
DEFAULT_BLEND_MAX_ABS_LOG10_RATIO = 0.10
DEFAULT_BLEND_MIN_WIDTH_Q_OVER_QC = 0.2


def source_branch_index_from_phi_rad(phi_rad):
    """Return stable detector-side branch label ``0``/``1`` from signed azimuth."""

    try:
        value = float(phi_rad)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    wrapped = float(((value + np.pi) % (2.0 * np.pi)) - np.pi)
    if wrapped <= -np.pi + 1.0e-12:
        wrapped = float(np.pi)
    if abs(float(wrapped)) <= SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD:
        return None
    return 0 if float(wrapped) < 0.0 else 1


def _entry_hkl_00l_l_value(value: object) -> int | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        h_val = int(round(float(value[0])))
        k_val = int(round(float(value[1])))
        l_val = int(round(float(value[2])))
    except Exception:
        return None
    if h_val != 0 or k_val != 0:
        return None
    return int(l_val)


def _entry_q_group_00l_l_value(value: object) -> int | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 4:
        return None
    try:
        if str(value[0]) != "q_group" or int(round(float(value[2]))) != 0:
            return None
        return int(round(float(value[3])))
    except Exception:
        return None


def entry_is_nonzero_00l_reflection(entry: Mapping[str, object] | None) -> bool:
    """Return True when an entry has positive-L evidence for collapsed 00L."""

    if not isinstance(entry, Mapping):
        return False
    hkl_l = _entry_hkl_00l_l_value(entry.get("hkl"))
    if hkl_l is not None:
        return hkl_l > 0
    q_group_l = _entry_q_group_00l_l_value(entry.get("q_group_key"))
    return q_group_l is not None and q_group_l > 0


def resolve_canonical_branch(
    entry: Mapping[str, object] | None,
) -> tuple[int | None, str | None, str | None]:
    """Return the explicit canonical ``source_branch_index`` identity."""

    if not isinstance(entry, Mapping):
        return None, None, "missing_source_branch_index"
    if entry_is_nonzero_00l_reflection(entry):
        return None, "00l_collapsed", "00l_collapsed"
    try:
        branch_index = int(entry["source_branch_index"])
    except (KeyError, TypeError, ValueError):
        return None, None, "missing_source_branch_index"
    if branch_index not in {0, 1}:
        return None, "source_branch_index", "invalid_source_branch_index"
    return branch_index, "source_branch_index", None


def _validate_lambda_m(lambda_m) -> float:
    try:
        value = float(lambda_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wavelength must be numeric.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Wavelength must be finite and positive.")
    return value


def _validate_lambda_array(lambda_m_array) -> np.ndarray:
    try:
        arr = np.asarray(lambda_m_array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Wavelength values must be numeric.") from exc
    out = np.array(arr, copy=True, dtype=np.float64)
    invalid = (~np.isfinite(out)) | (out <= 0.0)
    if np.any(invalid):
        raise ValueError("Wavelength values must be finite and positive.")
    return out


def _n2_wavelength_snapshot_from_angstrom(wavelength_angstrom_array) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(wavelength_angstrom_array, dtype=np.float64).reshape(-1),
        dtype=np.float64,
    )


def _normalize_n2_source_meta(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        kind, payload = value
        if kind == "cif_path" and payload:
            return ("cif_path", str(Path(payload).expanduser().resolve()))
    return None


def _normalize_element_symbol(raw_value) -> str | None:
    text = str(raw_value).strip()
    if not text:
        return None
    match = _ELEMENT_SYMBOL_RE.search(text)
    if match is None:
        return None
    return match.group(0)


def _complex_index_from_density(lambda_m, electron_density_m3, mu_m):
    lambda_arr = _validate_lambda_array(lambda_m)
    delta = (R_E * np.square(lambda_arr) * float(electron_density_m3)) / (2.0 * np.pi)
    beta = (np.asarray(mu_m, dtype=np.float64) * lambda_arr) / (4.0 * np.pi)
    return (1.0 - delta).astype(np.complex128) + 1.0j * beta


@lru_cache(maxsize=32)
def _cif_optics_properties(cif_path: str) -> dict[str, object]:
    import Dans_Diffraction as dif
    import xraydb

    resolved_path = str(Path(str(cif_path)).expanduser().resolve())
    xtl = dif.Crystal(resolved_path)
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    structure = xtl.Structure

    raw_types = np.asarray(getattr(structure, "type", []), dtype=object).reshape(-1)
    if raw_types.size <= 0:
        raise ValueError(f"Could not determine atom types from CIF {resolved_path!r}.")

    occupancies = np.asarray(
        getattr(structure, "occupancy", np.ones(raw_types.shape[0], dtype=np.float64)),
        dtype=np.float64,
    ).reshape(-1)
    if occupancies.size != raw_types.size:
        raise ValueError(
            f"CIF atom-type and occupancy counts differ in {resolved_path!r}."
        )
    if np.any(~np.isfinite(occupancies)) or np.any(occupancies < 0.0):
        raise ValueError(f"CIF occupancies must be finite and nonnegative in {resolved_path!r}.")

    composition_counts: dict[str, float] = {}
    for raw_type, occ in zip(raw_types, occupancies):
        symbol = _normalize_element_symbol(raw_type)
        if symbol is None:
            continue
        occ_value = float(occ)
        if occ_value == 0.0:
            continue
        composition_counts[symbol] = composition_counts.get(symbol, 0.0) + occ_value

    if not composition_counts:
        raise ValueError(f"Could not derive unit-cell composition from CIF {resolved_path!r}.")

    element_symbols = tuple(sorted(composition_counts))
    counts = np.array([composition_counts[sym] for sym in element_symbols], dtype=np.float64)
    atomic_masses = np.array(
        [float(xraydb.atomic_mass(sym)) for sym in element_symbols],
        dtype=np.float64,
    )
    atomic_numbers = np.array(
        [float(xraydb.atomic_number(sym)) for sym in element_symbols],
        dtype=np.float64,
    )

    cell_volume_ang3 = float(xtl.Cell.volume())
    if not np.isfinite(cell_volume_ang3) or cell_volume_ang3 <= 0.0:
        raise ValueError(f"CIF {resolved_path!r} has invalid unit-cell volume.")

    cell_molar_mass_g = float(np.dot(counts, atomic_masses))
    if not np.isfinite(cell_molar_mass_g) or cell_molar_mass_g <= 0.0:
        raise ValueError(f"CIF {resolved_path!r} has invalid unit-cell mass.")

    mass_fractions = (counts * atomic_masses) / cell_molar_mass_g
    density_g_cm3 = (cell_molar_mass_g / N_A) / (cell_volume_ang3 * 1.0e-24)
    electron_density_m3 = float(np.dot(counts, atomic_numbers)) / (cell_volume_ang3 * 1.0e-30)

    return {
        "path": resolved_path,
        "density_g_cm3": float(density_g_cm3),
        "electron_density_m3": float(electron_density_m3),
        "element_symbols": element_symbols,
        "mass_fractions": mass_fractions,
    }


def _weighted_mass_attenuation_from_cif(props: dict[str, object], energy_kev) -> np.ndarray:
    import xraydb

    weighted = np.zeros_like(np.asarray(energy_kev, dtype=np.float64), dtype=np.float64)
    symbols = tuple(props["element_symbols"])
    fractions = np.asarray(props["mass_fractions"], dtype=np.float64)
    for symbol, fraction in zip(symbols, fractions):
        weighted += float(fraction) * np.asarray(
            xraydb.mu_elam(str(symbol), energy_kev),
            dtype=np.float64,
        )
    return weighted


def _index_of_refraction_array_from_cif_props(
    lambda_m_array,
    props: dict[str, object],
) -> np.ndarray:
    lambda_arr = _validate_lambda_array(lambda_m_array)
    energy_ev = _HC_ANGSTROM_EV / (lambda_arr * 1.0e10)
    weighted_mass_attn = _weighted_mass_attenuation_from_cif(props, energy_ev)
    mu_m = float(props["density_g_cm3"]) * weighted_mass_attn * 1.0e2
    return _complex_index_from_density(
        lambda_arr,
        float(props["electron_density_m3"]),
        mu_m,
    )


def resolve_index_of_refraction_array(
    lambda_m_array,
    *,
    cif_path: str,
) -> np.ndarray:
    """Return wavelength-specific complex indices from the active CIF."""

    if not str(cif_path).strip():
        raise ValueError("cif_path is required for exact material optics.")
    return _index_of_refraction_array_from_cif_props(
        lambda_m_array,
        _cif_optics_properties(str(cif_path)),
    )


def resolve_index_of_refraction(
    lambda_m=LAMBDA_DEFAULT,
    *,
    cif_path: str,
) -> complex:
    """Return the complex index of refraction for one wavelength."""

    lambda_value = _validate_lambda_m(lambda_m)
    return complex(
        resolve_index_of_refraction_array(
            np.array([lambda_value], dtype=np.float64),
            cif_path=cif_path,
        )[0]
    )


def critical_qz_from_refractive_index(
    refractive_index: complex,
    lambda_angstrom: float,
) -> float:
    """Return the total-reflection critical Qz for ``n = 1 - delta + i beta``."""

    try:
        wavelength = float(lambda_angstrom)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        return 0.0

    n_value = complex(refractive_index)
    delta = max(0.0, 1.0 - float(n_value.real))
    if delta <= 0.0:
        return 0.0
    alpha_c = math.sqrt(2.0 * delta)
    return float((4.0 * math.pi / wavelength) * math.sin(alpha_c))


def _parratt_kz_for_layer(
    qz_angstrom_inv,
    *,
    lambda_angstrom: float,
    refractive_index: complex,
):
    qz = np.asarray(qz_angstrom_inv, dtype=np.float64)
    wavelength = float(lambda_angstrom)
    k0 = 2.0 * math.pi / wavelength
    alpha = np.arcsin(np.clip(qz * wavelength / (4.0 * math.pi), 0.0, 1.0))
    kz = k0 * np.sqrt(complex(refractive_index) ** 2 - np.cos(alpha) ** 2 + 0.0j)

    # Use the branch that propagates/decays away from the interface.
    flip = np.imag(kz) < 0.0
    if np.any(flip):
        kz[flip] *= -1.0
    flip = (np.abs(np.imag(kz)) < 1.0e-15) & (np.real(kz) < 0.0)
    if np.any(flip):
        kz[flip] *= -1.0
    return kz


@dataclass(slots=True)
class _PreparedSfLookup:
    L: np.ndarray
    intensity: np.ndarray
    left: float
    right: float


@dataclass(slots=True)
class _ParrattSfOpticalBasis:
    qz: np.ndarray
    qz_over_qc: np.ndarray
    phase_qz: np.ndarray
    phase_L: np.ndarray
    pure_parratt: np.ndarray


def _prepare_sf_lookup(L_values, intensity_values) -> _PreparedSfLookup:
    L_vals = np.asarray(L_values, dtype=np.float64)
    intensity = np.asarray(intensity_values, dtype=np.float64)
    valid_lookup = np.isfinite(L_vals) & np.isfinite(intensity)
    if not np.any(valid_lookup):
        empty = np.array([], dtype=np.float64)
        return _PreparedSfLookup(empty, empty, np.nan, np.nan)

    order = np.argsort(L_vals[valid_lookup])
    L_sorted = L_vals[valid_lookup][order]
    intensity_sorted = intensity[valid_lookup][order]
    L_unique, unique_index = np.unique(L_sorted, return_index=True)
    intensity_unique = intensity_sorted[unique_index]
    return _PreparedSfLookup(
        L_unique,
        intensity_unique,
        float(intensity_unique[0]),
        float(intensity_unique[-1]),
    )


def _interpolate_prepared_sf(lookup: _PreparedSfLookup, phase_L_values) -> np.ndarray:
    phase_L = np.asarray(phase_L_values, dtype=np.float64)
    if lookup.L.size == 0:
        return np.full_like(phase_L, np.nan, dtype=np.float64)
    if lookup.L.size == 1:
        return np.full_like(phase_L, lookup.left, dtype=np.float64)
    return np.interp(
        phase_L,
        lookup.L,
        lookup.intensity,
        left=lookup.left,
        right=lookup.right,
    )


def _parratt_reflectivity_from_layer_kz(
    kz_layers: list[np.ndarray],
    thicknesses_angstrom: list[float | None],
    roughnesses_angstrom: list[float],
) -> np.ndarray:
    if len(kz_layers) < 2 or len(thicknesses_angstrom) != len(kz_layers):
        raise ValueError("Parratt layers require matching kz and thickness entries.")
    if len(roughnesses_angstrom) != len(kz_layers) - 1:
        raise ValueError("Parratt layers require one roughness per interface.")

    tiny = np.finfo(np.float64).tiny
    r_eff = np.zeros_like(kz_layers[0], dtype=np.complex128)
    for layer_index in range(len(kz_layers) - 2, -1, -1):
        kz_j = kz_layers[layer_index]
        kz_next = kz_layers[layer_index + 1]
        r_j = (kz_j - kz_next) / (kz_j + kz_next + tiny)
        sigma = float(roughnesses_angstrom[layer_index])
        if sigma > 0.0:
            r_j = r_j * np.exp(-2.0 * kz_j * kz_next * sigma**2)
        thickness_next = thicknesses_angstrom[layer_index + 1]
        phase = 1.0 if thickness_next is None else np.exp(2.0j * kz_next * float(thickness_next))
        r_eff = (r_j + r_eff * phase) / (1.0 + r_j * r_eff * phase + tiny)

    reflectivity = np.abs(r_eff) ** 2
    if not np.all(np.isfinite(reflectivity)):
        raise FloatingPointError("Parratt reflectivity produced non-finite values.")
    return np.maximum(reflectivity, 0.0)


def _compute_parratt_sf_optical_basis(
    L_values,
    *,
    c_lattice_angstrom: float,
    lambda_angstrom: float,
    refractive_index: complex,
    thickness_angstrom: float,
    substrate_refractive_index: complex = 1.0 + 0.0j,
    top_roughness_angstrom: float = 0.0,
    bottom_roughness_angstrom: float = 0.0,
    critical_qz: float,
) -> _ParrattSfOpticalBasis:
    L_vals = np.asarray(L_values, dtype=np.float64)
    c_value = float(c_lattice_angstrom)
    wavelength = float(lambda_angstrom)
    n_value = complex(refractive_index)
    thickness = float(thickness_angstrom)
    qz = (2.0 * math.pi / c_value) * L_vals
    qz_nonnegative = np.maximum(qz, 0.0)
    qc = float(critical_qz)
    kz_air = _parratt_kz_for_layer(
        np.ravel(qz_nonnegative),
        lambda_angstrom=wavelength,
        refractive_index=1.0 + 0.0j,
    ).reshape(qz.shape)
    kz_film = _parratt_kz_for_layer(
        np.ravel(qz_nonnegative),
        lambda_angstrom=wavelength,
        refractive_index=n_value,
    ).reshape(qz.shape)
    phase_qz = np.maximum(np.real(2.0 * kz_film), 0.0)
    phase_L = phase_qz * c_value / (2.0 * math.pi)
    kz_substrate = _parratt_kz_for_layer(
        np.ravel(qz_nonnegative),
        lambda_angstrom=wavelength,
        refractive_index=complex(substrate_refractive_index),
    ).reshape(qz.shape)
    pure_parratt = _parratt_reflectivity_from_layer_kz(
        [kz_air, kz_film, kz_substrate],
        [None, thickness, None],
        [float(top_roughness_angstrom), float(bottom_roughness_angstrom)],
    )
    return _ParrattSfOpticalBasis(
        qz=qz,
        qz_over_qc=qz / qc,
        phase_qz=phase_qz,
        phase_L=phase_L,
        pure_parratt=pure_parratt,
    )


def smooth_log_blend(
    low_curve,
    high_curve,
    qz_over_qc,
    *,
    x1: float,
    x2: float,
    floor: float = 1.0e-300,
) -> tuple[np.ndarray, dict[str, float | str]]:
    left = float(x1)
    right = float(x2)
    value_floor = float(floor)
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError("blend x1 and x2 must be finite.")
    if right <= left:
        raise ValueError("blend x2 must be greater than x1.")
    if not np.isfinite(value_floor) or value_floor <= 0.0:
        raise ValueError("blend floor must be finite and positive.")

    low, high, x = np.broadcast_arrays(
        np.asarray(low_curve, dtype=np.float64),
        np.asarray(high_curve, dtype=np.float64),
        np.asarray(qz_over_qc, dtype=np.float64),
    )
    out = np.array(low, dtype=np.float64, copy=True)
    high_side = x >= right
    blend = (x > left) & (x < right) & np.isfinite(x)
    out[high_side] = high[high_side]

    if np.any(blend):
        t_values = np.clip((x[blend] - left) / (right - left), 0.0, 1.0)
        weight = t_values**3 * (10.0 - 15.0 * t_values + 6.0 * t_values**2)
        low_for_log = np.maximum(low[blend], value_floor)
        high_for_log = np.maximum(high[blend], value_floor)
        log_low = np.log10(low_for_log)
        log_high = np.log10(high_for_log)
        out[blend] = 10.0 ** ((1.0 - weight) * log_low + weight * log_high)

    return out, {
        "blend_method": "smooth_log_quintic",
        "blend_x1_q_over_qc": left,
        "blend_x2_q_over_qc": right,
        "blend_floor": value_floor,
    }


def choose_auto_blend_window(
    qz_over_qc,
    low_curve,
    high_curve,
    *,
    search_min: float = DEFAULT_BLEND_SEARCH_MIN_Q_OVER_QC,
    search_max: float = DEFAULT_BLEND_SEARCH_MAX_Q_OVER_QC,
    max_abs_log10_ratio: float = DEFAULT_BLEND_MAX_ABS_LOG10_RATIO,
    min_width: float = DEFAULT_BLEND_MIN_WIDTH_Q_OVER_QC,
) -> tuple[float, float, dict[str, object]]:
    x_values = np.asarray(qz_over_qc, dtype=np.float64)
    low = np.asarray(low_curve, dtype=np.float64)
    high = np.asarray(high_curve, dtype=np.float64)
    ratio_mask = (
        np.isfinite(high) & np.isfinite(low) & np.isfinite(x_values) & (high > 0.0) & (low > 0.0)
    )
    log_ratio = np.full_like(high, np.nan, dtype=np.float64)
    log_ratio[ratio_mask] = np.log10(high[ratio_mask] / low[ratio_mask])
    valid = (
        ratio_mask
        & (x_values >= float(search_min))
        & (x_values <= float(search_max))
        & (np.abs(log_ratio) <= float(max_abs_log10_ratio))
    )

    candidates = []
    edges = np.diff(np.r_[False, np.asarray(valid, dtype=bool), False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        width = float(x_values[stop - 1] - x_values[start])
        if width < float(min_width):
            continue
        median_abs = float(np.median(np.abs(log_ratio[start:stop])))
        candidates.append((-width, median_abs, start, stop))

    base_metadata = {
        "blend_window_mode": "auto",
        "blend_search_min_q_over_qc": float(search_min),
        "blend_search_max_q_over_qc": float(search_max),
        "blend_max_abs_log10_ratio": float(max_abs_log10_ratio),
        "blend_min_width_q_over_qc": float(min_width),
    }
    if not candidates:
        raise ValueError(
            "No valid Parratt/HT overlap satisfies the automatic blend-window criteria."
        )

    negative_width, median_abs, start, stop = min(candidates)
    left = float(x_values[start])
    right = float(x_values[stop - 1])
    return (
        left,
        right,
        {
            **base_metadata,
            "blend_x1_q_over_qc": left,
            "blend_x2_q_over_qc": right,
            "auto_blend_region_width_q_over_qc": -float(negative_width),
            "auto_blend_median_abs_log10_ratio": float(median_abs),
        },
    )


def stitch_parratt_reflectivity_to_intensity(
    L_values,
    intensity_values,
    *,
    c_lattice_angstrom: float,
    lambda_angstrom: float,
    refractive_index: complex,
    thickness_angstrom: float,
    substrate_refractive_index: complex = 1.0 + 0.0j,
    top_roughness_angstrom: float = 0.0,
    bottom_roughness_angstrom: float = 0.0,
    enabled: bool = True,
    q_start_factor: float = 3.0,
    q_end_factor: float = DEFAULT_BLEND_X2_Q_OVER_QC,
) -> tuple[np.ndarray, dict[str, object]]:
    """Smoothly blend pure Parratt reflectivity to scaled HT/Qz^2 intensity."""

    intensity = np.asarray(intensity_values, dtype=np.float64)
    metadata: dict[str, object] = {
        "active": False,
        "reason": "disabled" if not enabled else "inactive",
    }
    if not enabled:
        return intensity.copy(), metadata

    L_vals = np.asarray(L_values, dtype=np.float64)
    if L_vals.shape != intensity.shape:
        raise ValueError("Parratt stitch L and intensity arrays must have the same shape.")

    try:
        c_value = float(c_lattice_angstrom)
        wavelength = float(lambda_angstrom)
        thickness = float(thickness_angstrom)
        top_roughness = float(top_roughness_angstrom)
        bottom_roughness = float(bottom_roughness_angstrom)
        start_factor = float(q_start_factor)
        end_factor = float(q_end_factor)
        substrate_n = complex(substrate_refractive_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Parratt stitch inputs must be numeric.") from exc

    if (
        L_vals.size == 0
        or not np.isfinite(c_value)
        or c_value <= 0.0
        or not np.isfinite(wavelength)
        or wavelength <= 0.0
        or not np.isfinite(thickness)
        or thickness <= 0.0
        or not np.isfinite(top_roughness)
        or top_roughness < 0.0
        or not np.isfinite(bottom_roughness)
        or bottom_roughness < 0.0
        or not np.isfinite(start_factor)
        or not np.isfinite(end_factor)
        or start_factor <= 0.0
        or end_factor <= start_factor
        or not np.isfinite(substrate_n.real)
        or not np.isfinite(substrate_n.imag)
        or substrate_n.real <= 0.0
    ):
        raise ValueError("Parratt stitch inputs are outside their valid ranges.")

    n_value = complex(refractive_index)
    qc = critical_qz_from_refractive_index(n_value, wavelength)
    if qc <= 0.0:
        raise ValueError("Parratt stitch requires a positive critical edge.")

    q_start = start_factor * qc
    q_end = end_factor * qc
    lookup = _prepare_sf_lookup(L_vals, intensity)
    basis = _compute_parratt_sf_optical_basis(
        L_vals,
        c_lattice_angstrom=c_value,
        lambda_angstrom=wavelength,
        refractive_index=n_value,
        substrate_refractive_index=substrate_n,
        thickness_angstrom=thickness,
        top_roughness_angstrom=top_roughness,
        bottom_roughness_angstrom=bottom_roughness,
        critical_qz=qc,
    )

    phase_intensity = _interpolate_prepared_sf(lookup, basis.phase_L)
    ht_over_q2 = np.full_like(intensity, np.nan, dtype=np.float64)
    np.divide(
        phase_intensity,
        basis.qz**2,
        out=ht_over_q2,
        where=np.isfinite(phase_intensity) & np.isfinite(basis.qz) & (basis.qz > 0.0),
    )
    finite_positive = (
        np.isfinite(ht_over_q2)
        & np.isfinite(basis.pure_parratt)
        & np.isfinite(basis.qz_over_qc)
        & (ht_over_q2 > 0.0)
        & (basis.pure_parratt > 0.0)
    )
    fit_start_factor = max(DEFAULT_BLEND_SEARCH_MIN_Q_OVER_QC, end_factor)
    fit_end_factor = max(
        DEFAULT_BLEND_SEARCH_MAX_Q_OVER_QC,
        fit_start_factor + np.finfo(np.float64).eps,
    )
    overlap = (
        finite_positive
        & (basis.qz_over_qc >= fit_start_factor)
        & (basis.qz_over_qc <= fit_end_factor)
    )
    if not np.any(overlap):
        raise ValueError("Parratt stitch has no positive normalization overlap.")

    # Normalize Parratt upward to the kinematic HT/Qz^2 scale. The former
    # implementation applied the reciprocal factor to HT/Qz^2, which put the
    # 00L simulation on dimensionless reflectivity scale while every other
    # Bragg rod remained in the original kinematic intensity units.
    log_parratt_to_kinematic_scale = float(
        np.median(np.log(ht_over_q2[overlap]) - np.log(basis.pure_parratt[overlap]))
    )
    parratt_to_kinematic_scale = float(np.exp(log_parratt_to_kinematic_scale))
    if not np.isfinite(parratt_to_kinematic_scale) or parratt_to_kinematic_scale <= 0.0:
        raise ValueError("Parratt stitch normalization is not finite and positive.")
    kinematic_to_reflectivity_scale = 1.0 / parratt_to_kinematic_scale

    kinematic_parratt_over_q2 = basis.pure_parratt * parratt_to_kinematic_scale

    blend_x1, blend_x2, window_metadata = choose_auto_blend_window(
        basis.qz_over_qc,
        kinematic_parratt_over_q2,
        ht_over_q2,
    )
    kinematic_stitched_over_q2, blend_metadata = smooth_log_blend(
        kinematic_parratt_over_q2,
        ht_over_q2,
        basis.qz_over_qc,
        x1=blend_x1,
        x2=blend_x2,
    )

    # Convert the blended reflectivity-form curve back to the same kinematic
    # intensity units supplied by the structure-factor model. Above the blend
    # window this recovers phase_intensity exactly, apart from roundoff.
    kinematic_stitched_intensity = kinematic_stitched_over_q2 * basis.qz**2
    kinematic_stitched_intensity = np.where(
        np.isfinite(kinematic_stitched_intensity),
        kinematic_stitched_intensity,
        0.0,
    )
    kinematic_stitched_intensity = np.maximum(kinematic_stitched_intensity, 0.0)

    # Keep dimensionless Parratt-scale curves for the Film/Fig. 2 diagnostics.
    # These are diagnostic views only and are no longer passed to the detector
    # simulation as reflection intensities.
    scaled_ht_over_q2 = ht_over_q2 * kinematic_to_reflectivity_scale
    stitched_reflectivity = kinematic_stitched_over_q2 * kinematic_to_reflectivity_scale
    stitched_reflectivity = np.where(
        np.isfinite(stitched_reflectivity),
        stitched_reflectivity,
        0.0,
    )

    metadata.update(
        {
            "active": True,
            "reason": "applied",
            "stitch_mode": "smooth_log_auto_parratt_to_kinematic",
            "normalization_direction": "parratt_to_kinematic",
            "simulation_intensity_units": "input_kinematic_intensity",
            "n_real": float(n_value.real),
            "n_imag": float(n_value.imag),
            "substrate_n_real": float(substrate_n.real),
            "substrate_n_imag": float(substrate_n.imag),
            "top_roughness_angstrom": float(top_roughness),
            "bottom_roughness_angstrom": float(bottom_roughness),
            "critical_qz_angstrom_inv": float(qc),
            "critical_L": float(qc * c_value / (2.0 * math.pi)),
            "sf_phase_qz_formula": "real(2*kz_film_from_Parratt)",
            "sf_phase_coordinate": "film-internal",
            "q_start_factor": float(start_factor),
            "q_end_factor": float(end_factor),
            "q_start_angstrom_inv": float(q_start),
            "q_end_angstrom_inv": float(q_end),
            "stitch_cut_q_over_qc": float(blend_x2),
            "q_fit_start_factor": float(fit_start_factor),
            "q_fit_end_factor": float(fit_end_factor),
            "scale_factor": float(parratt_to_kinematic_scale),
            "parratt_to_kinematic_scale_factor": float(parratt_to_kinematic_scale),
            "kinematic_to_parratt_scale_factor": float(kinematic_to_reflectivity_scale),
            "thickness_angstrom": float(thickness),
            "L_values": np.asarray(L_vals, dtype=np.float64).copy(),
            "qz_angstrom_inv": np.asarray(basis.qz, dtype=np.float64).copy(),
            "qz_over_qc": np.asarray(basis.qz_over_qc, dtype=np.float64).copy(),
            "pure_parratt": np.asarray(basis.pure_parratt, dtype=np.float64).copy(),
            "scaled_ht_over_q2": np.asarray(scaled_ht_over_q2, dtype=np.float64).copy(),
            "stitched_curve": np.asarray(stitched_reflectivity, dtype=np.float64).copy(),
            "kinematic_phase_intensity": np.asarray(phase_intensity, dtype=np.float64).copy(),
            "kinematic_ht_over_q2": np.asarray(ht_over_q2, dtype=np.float64).copy(),
            "kinematic_parratt_over_q2": np.asarray(
                kinematic_parratt_over_q2, dtype=np.float64
            ).copy(),
            "kinematic_stitched_over_q2": np.asarray(
                kinematic_stitched_over_q2, dtype=np.float64
            ).copy(),
            "kinematic_stitched_intensity": np.asarray(
                kinematic_stitched_intensity, dtype=np.float64
            ).copy(),
            "simulation_intensity_curve": np.asarray(
                kinematic_stitched_intensity, dtype=np.float64
            ).copy(),
            **window_metadata,
            **blend_metadata,
        }
    )
    return kinematic_stitched_intensity, metadata


@njit
def complex_sqrt(z):
    """
    Compute the principal square root of a complex number z
    in a Numba-friendly way using polar form.
    """
    r = math.hypot(z.real, z.imag)  # sqrt(x^2 + y^2)
    phi = math.atan2(z.imag, z.real)
    sqrt_r = math.sqrt(r)
    half_phi = 0.5 * phi
    return complex(sqrt_r * math.cos(half_phi), sqrt_r * math.sin(half_phi))
