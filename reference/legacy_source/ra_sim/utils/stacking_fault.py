# stacking_fault.py - analytical HT with diffuse-consistent F^2, C2H, and factors

import ast
from collections.abc import Mapping
import os
import re
import numpy as np
from typing import Protocol


# Global constants
P_CLAMP = 1e-6
N_P, A_C = 3, 17.98e-10    # number of sublayers, real-space area (m^2)
AREA    = (2*np.pi)**2 / A_C * N_P
_TWO_PI = 2.0 * np.pi

DEFAULT_PHASE_DELTA_EXPRESSION = "2*pi*((2*h + k)/3)"
DEFAULT_PHI_L_DIVISOR = 1.0

RICH_PHASE_COMPONENTS = ("2H", "4H", "6H")
RICH_PHASE_COMPONENT_ORDER = (*RICH_PHASE_COMPONENTS, "mix")
_PBI2_CIF_RE = re.compile(r"pbi2", re.IGNORECASE)
_RICH_PHASE_PARENT_TEMPLATES = {
    "2H": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    "4H": np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=float),
    "6H": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=float),
}
_RICH_PHASE_FAULT_TEMPLATES = {
    "2H": np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=float) / 4.0,
    "4H": np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=float) / 4.0,
    "6H": np.array([1.0, 0.0, 1.0, 1.0, 1.0], dtype=float) / 4.0,
}
_RICH_PHASE_REGISTRY_ROOTS = (
    1.0 + 0.0j,
    complex(-0.5, np.sqrt(3.0) / 2.0),
    complex(-0.5, -np.sqrt(3.0) / 2.0),
)
class LayerFormFactorProviderLike(Protocol):
    """Provider contract accepted by rich-phase basis builders."""

    def evaluate(self, q_cartesian) -> tuple[np.ndarray, np.ndarray]:
        ...

def _first_positive_float(*candidates, name: str) -> float:
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0.0:
            return parsed
    raise ValueError(f"{name} must be positive")


def hexagonal_q_cartesian(
    h: int,
    k: int,
    l_vals,
    *,
    a_axis: float,
    c_axis: float,
) -> np.ndarray:
    """Return Cartesian Q vectors for the hexagonal HKL convention used here."""

    a_val = _first_positive_float(a_axis, name="a_axis")
    c_val = _first_positive_float(c_axis, name="c_axis")
    l_arr = np.asarray(l_vals, dtype=float)
    if not np.all(np.isfinite(l_arr)):
        raise ValueError("l_vals must contain only finite values")

    q_vals = np.empty(l_arr.shape + (3,), dtype=float)
    q_vals[..., 0] = _TWO_PI * float(int(h)) / a_val
    q_vals[..., 1] = _TWO_PI * (float(int(h)) + 2.0 * float(int(k))) / (
        np.sqrt(3.0) * a_val
    )
    q_vals[..., 2] = _TWO_PI * l_arr / c_val
    return q_vals


# Cache of base L grids and F2 values keyed by geometry and occupancy mapping.
# Each entry: {(h,k): {"L": array, "F2": array}}
_HT_BASE_CACHE: dict[tuple, dict] = {}
_HT_BASE_CACHE_MAX_ENTRIES = 24

_PHASE_DELTA_EXPR_CACHE: dict[str, object] = {}
_ALLOWED_PHASE_DELTA_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "real": np.real,
    "imag": np.imag,
    "angle": np.angle,
}
_ALLOWED_PHASE_DELTA_NAMES = {"h", "k", "L", "p", "pi"} | set(
    _ALLOWED_PHASE_DELTA_FUNCS.keys()
)
_ALLOWED_PHASE_DELTA_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)


def _normalize_nonnegative(values) -> np.ndarray:
    """Return non-negative values normalized to unit sum."""

    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("At least one non-negative component weight must be positive")
    return arr / total


def rich_phase_probabilities(phase: str, epsilon: float) -> np.ndarray:
    """Return theta=(a,b+,b-,d+,d-) for one rich-phase population."""

    phase_key = str(phase)
    if phase_key not in _RICH_PHASE_PARENT_TEMPLATES:
        raise ValueError(f"Unknown rich phase: {phase}")
    eps = float(np.clip(float(epsilon), 0.0, 1.0))
    theta = (
        (1.0 - eps) * _RICH_PHASE_PARENT_TEMPLATES[phase_key]
        + eps * _RICH_PHASE_FAULT_TEMPLATES[phase_key]
    )
    total = float(theta.sum())
    if total <= 0.0:
        raise ValueError(f"Rich-phase probabilities for {phase_key} have zero total")
    return theta / total


def rich_phase_diffuse_parent_fractions(component) -> np.ndarray:
    """Return normalized 2H/4H/6H character for the diffuse-mix component."""

    comp = component if isinstance(component, dict) else {}
    return _normalize_nonnegative(
        [comp.get("f2", 1.0), comp.get("f4", 1.0), comp.get("f6", 1.0)]
    )


def rich_phase_component_probabilities(component_key: str, components) -> np.ndarray:
    """Return direction-resolved probabilities for a rich-phase component."""

    key = str(component_key)
    comp = components.get(key, {}) if isinstance(components, dict) else {}
    eps = float(np.clip(float(comp.get("epsilon", 0.0)), 0.0, 1.0))
    if key in RICH_PHASE_COMPONENTS:
        return rich_phase_probabilities(key, eps)
    if key != "mix":
        raise ValueError(f"Unknown rich-phase component: {component_key}")

    f2, f4, f6 = rich_phase_diffuse_parent_fractions(comp)
    theta = (
        f2 * rich_phase_probabilities("2H", eps)
        + f4 * rich_phase_probabilities("4H", eps)
        + f6 * rich_phase_probabilities("6H", eps)
    )
    total = float(theta.sum())
    if total <= 0.0:
        raise ValueError("Rich-phase mixture probabilities have zero total")
    return theta / total


def normalize_rich_phase_component_weights(components) -> dict[str, float]:
    """Normalize enabled rich-phase component weights."""

    vals = []
    for key in RICH_PHASE_COMPONENT_ORDER:
        comp = components.get(key, {}) if isinstance(components, dict) else {}
        value = comp.get("w", 0.0)
        if not comp.get("enabled", True):
            value = 0.0
        vals.append(value)

    arr = np.clip(np.asarray(vals, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return {key: 0.0 for key in RICH_PHASE_COMPONENT_ORDER}
    arr /= total
    return {
        key: float(arr[idx])
        for idx, key in enumerate(RICH_PHASE_COMPONENT_ORDER)
    }


def _rich_phase_cesaro_orientation_distribution(d_total: float) -> np.ndarray:
    if abs(float(d_total)) < 1e-15:
        return np.array([1.0, 0.0], dtype=float)
    return np.array([0.5, 0.5], dtype=float)


def _rich_phase_geom_sum_1_to_n(x_vals: np.ndarray, n_layers: int) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=complex)
    if int(n_layers) <= 0:
        return np.zeros_like(x_vals)
    out = np.empty_like(x_vals)
    mask = np.isclose(x_vals, 1.0 + 0.0j, rtol=1e-12, atol=1e-12)
    out[mask] = float(n_layers)
    xm = x_vals[~mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[~mask] = xm * (1.0 - xm**n_layers) / (1.0 - xm)
    return out


def _rich_phase_weighted_finite_sum(x_vals: np.ndarray, n_layers: int) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=complex)
    if int(n_layers) <= 1:
        return np.zeros_like(x_vals)
    out = np.zeros_like(x_vals)
    power = np.ones_like(x_vals)
    for distance in range(1, int(n_layers)):
        power = power * x_vals
        out += float(int(n_layers) - distance) * power
    return out


def _rich_phase_cross_geometric_sum(
    x_vals: np.ndarray,
    ratio: float,
    n_layers: int,
) -> np.ndarray:
    x_vals = np.asarray(x_vals, dtype=complex)
    if int(n_layers) <= 1:
        return np.zeros_like(x_vals)
    ratio = float(ratio)
    out = np.empty_like(x_vals)
    mask = np.isclose(x_vals, ratio + 0.0j, rtol=1e-12, atol=1e-12)
    out[mask] = (n_layers - 1) * (complex(ratio) ** n_layers)
    xm = x_vals[~mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[~mask] = xm * ratio * (ratio ** (n_layers - 1) - xm ** (n_layers - 1)) / (
            ratio - xm
        )
    return out


def _rich_phase_contrast_finite_sum(
    x_vals: np.ndarray,
    ratio: float,
    n_layers: int,
) -> np.ndarray:
    if abs(1.0 - float(ratio)) < 1e-12:
        return _rich_phase_weighted_finite_sum(x_vals, n_layers)
    return (
        _rich_phase_geom_sum_1_to_n(x_vals, int(n_layers) - 1)
        - _rich_phase_cross_geometric_sum(x_vals, ratio, n_layers)
    ) / (1.0 - ratio)


def _rich_phase_orientation_contrast_sum(ratio: float, n_layers: int) -> float:
    if int(n_layers) <= 0:
        return 0.0
    if abs(1.0 - float(ratio)) < 1e-12:
        return float(n_layers)
    return float((1.0 - ratio**n_layers) / (1.0 - ratio))


def rich_phase_intensity_from_aggregated_basis(
    l_vals,
    omega: complex,
    u_vals,
    v_vals,
    w_vals,
    theta,
    *,
    finite_layers=None,
    f2_only=False,
    phi_l_divisor: float | None = None,
    apply_ht_pole_clamp: bool = False,
) -> np.ndarray:
    """Evaluate the rich-phase 2x2 transition model for grouped rods."""

    l_vals = np.asarray(l_vals, dtype=float)
    u_vals = np.asarray(u_vals, dtype=float)
    v_vals = np.asarray(v_vals, dtype=float)
    w_vals = np.asarray(w_vals, dtype=complex)
    a_val, bp_val, bm_val, dp_val, dm_val = map(float, theta)
    d_total = dp_val + dm_val
    abs2 = np.vstack([u_vals, v_vals])

    a_term = a_val + bp_val * omega + bm_val * np.conj(omega)
    b_term = dp_val * omega + dm_val * np.conj(omega)
    q_term = float(abs(b_term))
    eta = b_term / q_term if q_term > 1e-15 else (1.0 + 0.0j)
    x_vals = eta * w_vals
    y_vals = np.conj(x_vals)

    if bool(f2_only):
        p_stat = _rich_phase_cesaro_orientation_distribution(d_total)
        return AREA * np.maximum(np.real(p_stat @ abs2), 0.0)

    phi_div = normalize_phi_l_divisor(
        3.0 if phi_l_divisor is None else phi_l_divisor,
    )
    z_unit = np.exp(1j * _TWO_PI * (l_vals / phi_div))
    if finite_layers is not None:
        n_layers = int(max(1, finite_layers))
        ratio = 1.0 - 2.0 * d_total
        contrast_n = _rich_phase_orientation_contrast_sum(ratio, n_layers)

        u0 = 0.5 * (u_vals + v_vals)
        u1 = 0.5 * (u_vals - v_vals)
        x0 = 0.5 * (x_vals + y_vals)
        x1 = 0.5 * (x_vals - y_vals)
        i_self = u0 + (contrast_n / float(n_layers)) * u1
        if n_layers == 1:
            return AREA * np.maximum(np.real(i_self), 0.0)

        z_finite = (1.0 - float(P_CLAMP)) * z_unit if bool(apply_ht_pole_clamp) else z_unit
        lam_plus = a_term + q_term
        lam_minus = a_term - q_term
        zp = z_finite * lam_plus
        zm = z_finite * lam_minus
        s_plus = _rich_phase_weighted_finite_sum(zp, n_layers)
        s_minus = _rich_phase_weighted_finite_sum(zm, n_layers)
        d_plus = _rich_phase_contrast_finite_sum(zp, ratio, n_layers)
        d_minus = _rich_phase_contrast_finite_sum(zm, ratio, n_layers)
        pair_sum = 0.5 * (
            (u0 + x0) * s_plus
            + (u1 + x1) * d_plus
            + (u0 - x0) * s_minus
            + (u1 - x1) * d_minus
        )
        intensity = AREA * (i_self + (2.0 / float(n_layers)) * np.real(pair_sum))
        return np.maximum(np.real(intensity), 0.0)

    p_stat = _rich_phase_cesaro_orientation_distribution(d_total)
    i_self = p_stat @ abs2
    lam_plus = a_term + q_term
    lam_minus = a_term - q_term
    z_damped = (1.0 - float(P_CLAMP)) * z_unit
    s_plus = (z_damped * lam_plus) / (1.0 - z_damped * lam_plus)
    s_minus = (z_damped * lam_minus) / (1.0 - z_damped * lam_minus)
    a_coef = 0.5 * (s_plus + s_minus)
    b_coef = 0.5 * (s_plus - s_minus)
    p_plus, p_minus = float(p_stat[0]), float(p_stat[1])
    pair_sum = (
        a_coef * (p_plus * u_vals + p_minus * v_vals)
        + b_coef * (p_plus * x_vals + p_minus * y_vals)
    )
    intensity = AREA * (i_self + 2.0 * np.real(pair_sum))
    return np.maximum(np.real(intensity), 0.0)


def rich_phase_intensity_from_basis(
    l_vals,
    omega: complex,
    f_plus,
    f_minus,
    theta,
    *,
    finite_layers=None,
    f2_only=False,
    phi_l_divisor: float | None = None,
    apply_ht_pole_clamp: bool = False,
) -> np.ndarray:
    """Evaluate the rich-phase 2x2 transition model for one rod."""

    f_plus = np.asarray(f_plus, dtype=complex)
    f_minus = np.asarray(f_minus, dtype=complex)
    return rich_phase_intensity_from_aggregated_basis(
        l_vals,
        omega,
        np.abs(f_plus) ** 2,
        np.abs(f_minus) ** 2,
        np.conj(f_plus) * f_minus,
        theta,
        finite_layers=finite_layers,
        f2_only=f2_only,
        phi_l_divisor=phi_l_divisor,
        apply_ht_pole_clamp=apply_ht_pole_clamp,
    )


def _rich_phase_registry_channel(h_val, k_val) -> int:
    return (int(h_val) + 2 * int(k_val)) % 3


def default_rich_phase_components() -> dict[str, dict[str, float | bool]]:
    return {
        "2H": {"epsilon": 0.01, "w": 1.0, "enabled": True},
        "4H": {"epsilon": 0.01, "w": 0.0, "enabled": False},
        "6H": {"epsilon": 0.01, "w": 0.0, "enabled": False},
        "mix": {
            "epsilon": 0.01,
            "f2": 1.0,
            "f4": 1.0,
            "f6": 1.0,
            "w": 0.0,
            "enabled": False,
        },
    }


def canonicalize_rich_phase_components(
    components: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, float | bool]]:
    """Validate and copy the complete current rich-phase component map."""

    if components is None:
        return default_rich_phase_components()
    if not isinstance(components, Mapping):
        raise TypeError("rich-phase components must be a mapping")
    if set(components) != set(RICH_PHASE_COMPONENT_ORDER):
        raise ValueError(
            f"rich-phase components must contain exactly {RICH_PHASE_COMPONENT_ORDER!r}"
        )

    result: dict[str, dict[str, float | bool]] = {}
    for key in RICH_PHASE_COMPONENT_ORDER:
        raw = components[key]
        if not isinstance(raw, Mapping):
            raise TypeError(f"rich-phase component {key!r} must be a mapping")
        required = (
            {"epsilon", "f2", "f4", "f6", "w", "enabled"}
            if key == "mix"
            else {"epsilon", "w", "enabled"}
        )
        if set(raw) != required:
            raise ValueError(
                f"rich-phase component {key!r} must contain exactly {sorted(required)!r}"
            )

        numeric: dict[str, float | bool] = {}
        for field in required - {"enabled"}:
            try:
                value = float(raw[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"rich-phase component {key!r} field {field!r} must be numeric"
                ) from exc
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"rich-phase component {key!r} field {field!r} "
                    "must be finite and non-negative"
                )
            if field == "epsilon" and value > 1.0:
                raise ValueError(
                    f"rich-phase component {key!r} epsilon must be between 0 and 1"
                )
            numeric[field] = value
        if not isinstance(raw["enabled"], bool):
            raise TypeError(f"rich-phase component {key!r} enabled must be boolean")
        numeric["enabled"] = raw["enabled"]
        result[key] = numeric
    return result


def _as_cif_sequence(raw_value) -> list[object]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return [raw_value]


def _formula_counts_from_text(text: object) -> dict[str, float]:
    counts: dict[str, float] = {}
    for element, count_text in re.findall(
        r"([A-Z][a-z]?)(?:\s*([0-9]+(?:\.[0-9]+)?))?",
        str(text).strip().strip("'\""),
    ):
        count = float(count_text) if count_text else 1.0
        counts[element] = counts.get(element, 0.0) + count
    return counts


def _formula_counts_look_like_pbi2(counts: Mapping[str, float]) -> bool:
    if set(counts) != {"Pb", "I"}:
        return False
    pb_count = float(counts.get("Pb", 0.0))
    i_count = float(counts.get("I", 0.0))
    return pb_count > 0.0 and np.isclose(i_count / pb_count, 2.0, rtol=0.0, atol=1e-6)


_PBI2_CONTENT_CACHE: dict[tuple[str, int | None, int | None], bool] = {}
_BI2SE3_CONTENT_CACHE: dict[tuple[str, int | None, int | None], bool] = {}


def cif_looks_like_pbi2(cif_path) -> bool:
    """Return True when a CIF path/name or parseable CIF content identifies PbI2."""

    if _PBI2_CIF_RE.search(os.path.basename(str(cif_path))):
        return True

    signature = _cif_cache_signature(str(cif_path))
    cached = _PBI2_CONTENT_CACHE.get(signature)
    if cached is not None:
        return bool(cached)

    looks_like = False
    try:
        import CifFile

        cf = CifFile.ReadCif(str(cif_path))
        keys = list(cf.keys())
        if keys:
            block = cf[keys[0]]
            for tag in (
                "_chemical_formula_sum",
                "_chemical_formula_structural",
                "_chemical_name_common",
            ):
                for raw_formula in _as_cif_sequence(block.get(tag)):
                    if _formula_counts_look_like_pbi2(_formula_counts_from_text(raw_formula)):
                        looks_like = True
                        break
                if looks_like:
                    break
            if not looks_like:
                raw_symbols = block.get("_atom_site_type_symbol")
                if raw_symbols is None:
                    raw_symbols = block.get("_atom_site_label")
                elements = {
                    _element_key(symbol)
                    for symbol in _as_cif_sequence(raw_symbols)
                    if str(symbol).strip()
                }
                looks_like = elements == {"Pb", "I"}
    except Exception:
        looks_like = False

    _PBI2_CONTENT_CACHE[signature] = bool(looks_like)
    return bool(looks_like)


def cif_looks_like_bi2se3(cif_path) -> bool:
    """Return True only when parseable CIF content has Bi2Se3 stoichiometry."""

    signature = _cif_cache_signature(str(cif_path))
    if signature not in _BI2SE3_CONTENT_CACHE:
        try:
            elements = [
                _element_key(site[3])
                for site in _sites_from_cif_with_factors(str(cif_path), occ_factors=1.0)
            ]
            _BI2SE3_CONTENT_CACHE[signature] = set(elements) == {"Bi", "Se"} and (
                2 * elements.count("Se") == 3 * elements.count("Bi")
            )
        except Exception:
            _BI2SE3_CONTENT_CACHE[signature] = False
    return _BI2SE3_CONTENT_CACHE[signature]


def _reject_pbi2_standard_ht(cif_path) -> None:
    if cif_looks_like_pbi2(cif_path):
        raise ValueError(
            "PbI2 standard HT is disabled; use rich-phase simulation with a "
            "2H-derived layer_form_factor_provider."
        )


def rich_phase_basis_curve_map(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    L_step: float = 0.005,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    *,
    layer_form_factor_provider: LayerFormFactorProviderLike | None,
    phase_z_divisor: float = DEFAULT_PHI_L_DIVISOR,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Return rich-phase basis curves from the layer form-factor provider."""

    provider = layer_form_factor_provider
    if provider is None or not callable(getattr(provider, "evaluate", None)):
        raise ValueError(
            "rich-phase basis construction requires a layer_form_factor_provider."
        )
    a_axis = _first_positive_float(a_lattice, name="a_lattice")
    c_axis = _first_positive_float(c_lattice, name="c_lattice")

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        import itertools

        hk_list = [(h, k) for h, k in itertools.product(range(-mx + 1, mx), repeat=2)]
    hk_values = [(int(h), int(k)) for h, k in hk_list]
    if not np.isfinite(L_step) or L_step <= 0.0:
        raise ValueError("L_step must be > 0")
    if not np.isfinite(L_max) or L_max < 0.0:
        raise ValueError("L_max must be nonnegative")
    L_step = max(float(L_step), 1.0e-4)
    base_l = (
        np.arange(0.0, float(L_max) + L_step / 2.0, L_step, dtype=float)
        if two_theta_max is None
        else None
    )
    q_max = (
        None
        if two_theta_max is None
        else (4.0 * np.pi / float(lambda_))
        * np.sin(np.radians(float(two_theta_max) / 2.0))
    )
    q_builder = getattr(provider, "q_cartesian_for_hkl", None)
    curve_map: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for h, k in hk_values:
        if base_l is None:
            radial = (4.0 / 3.0) * float(_hk_radial_index(h, k)) / (a_axis**2)
            l_sq = (float(q_max) / (2.0 * np.pi)) ** 2 - radial
            l_values = (
                np.empty(0, dtype=float)
                if l_sq <= 0.0
                else np.arange(
                    0.0,
                    c_axis * np.sqrt(l_sq) + L_step / 2.0,
                    L_step,
                    dtype=float,
                )
            )
        else:
            l_values = base_l
        if l_values.size == 0:
            continue
        q_cartesian = (
            q_builder(h, k, l_values)
            if callable(q_builder)
            else hexagonal_q_cartesian(
                h,
                k,
                l_values,
                a_axis=a_axis,
                c_axis=c_axis,
            )
        )
        f_plus, f_minus = provider.evaluate(q_cartesian)
        f_plus_array = np.asarray(f_plus, dtype=complex).reshape(-1)
        f_minus_array = np.asarray(f_minus, dtype=complex).reshape(-1)
        if f_plus_array.size != l_values.size or f_minus_array.size != l_values.size:
            raise ValueError("Layer form-factor provider returned an invalid curve length.")
        curve_map[(h, k)] = {
            "L": np.asarray(l_values, dtype=float).copy(),
            "F_plus": f_plus_array,
            "F_minus": f_minus_array,
        }
    return curve_map


def rich_phase_I_dict_from_basis_curve_map(
    curve_map,
    *,
    components=None,
    finite_stack: bool = False,
    stack_layers: int = 50,
    f2_only: bool = False,
    phi_l_divisor: float | None = None,
    apply_ht_pole_clamp: bool = False,
):
    """Combine cached F+/F- basis curves with rich-phase component weights."""

    comp_map = canonicalize_rich_phase_components(components)
    weights = normalize_rich_phase_component_weights(comp_map)
    finite_layers = int(max(1, stack_layers)) if bool(finite_stack) else None

    out = {}
    for (h, k), data in dict(curve_map or {}).items():
        l_vals = np.asarray(data.get("L", []), dtype=float)
        if l_vals.size == 0:
            out[(int(h), int(k))] = {"L": l_vals.copy(), "I": np.asarray([], dtype=float)}
            continue
        f_plus = np.asarray(data["F_plus"], dtype=complex)
        f_minus = np.asarray(data["F_minus"], dtype=complex)
        omega = _RICH_PHASE_REGISTRY_ROOTS[_rich_phase_registry_channel(h, k)]
        intensity = np.zeros_like(l_vals, dtype=float)
        for comp_key in RICH_PHASE_COMPONENT_ORDER:
            weight = float(weights.get(comp_key, 0.0))
            if weight <= 0.0:
                continue
            theta = rich_phase_component_probabilities(comp_key, comp_map)
            intensity += weight * rich_phase_intensity_from_basis(
                l_vals,
                omega,
                f_plus,
                f_minus,
                theta,
                finite_layers=finite_layers,
                f2_only=f2_only,
                phi_l_divisor=phi_l_divisor,
                apply_ht_pole_clamp=apply_ht_pole_clamp,
            )
        out[(int(h), int(k))] = {"L": l_vals.copy(), "I": intensity}
    return out


def rich_phase_I_dict(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    components=None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    *,
    finite_stack: bool = False,
    stack_layers: int = 50,
    f2_only: bool = False,
    phi_l_divisor: float | None = None,
    apply_ht_pole_clamp: bool = False,
    layer_form_factor_provider: LayerFormFactorProviderLike | None = None,
):
    """Return detector-native rich-phase intensities in the HT dict shape."""

    curve_map = rich_phase_basis_curve_map(
        cif_path=cif_path,
        hk_list=hk_list,
        mx=mx,
        L_step=L_step,
        L_max=L_max,
        two_theta_max=two_theta_max,
        lambda_=lambda_,
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        layer_form_factor_provider=layer_form_factor_provider,
    )
    return rich_phase_I_dict_from_basis_curve_map(
        curve_map,
        components=components,
        finite_stack=finite_stack,
        stack_layers=stack_layers,
        f2_only=f2_only,
        phi_l_divisor=phi_l_divisor,
        apply_ht_pole_clamp=apply_ht_pole_clamp,
    )


def normalize_phase_delta_expression(expression: str) -> str:
    """Return a non-empty phase-delta expression string."""

    if expression is None:
        raise ValueError("phase_delta_expression is required")
    text = str(expression).strip()
    if not text:
        raise ValueError("phase_delta_expression must not be empty")
    return text


def normalize_phi_l_divisor(value: float | str) -> float:
    """Return a finite positive divisor used in the HT out-of-plane phase."""

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("phi_l_divisor must be numeric") from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("phi_l_divisor must be finite and positive")
    return out


class _PhaseDeltaExprValidator(ast.NodeVisitor):
    """Validate expression AST against a restricted whitelist."""

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_PHASE_DELTA_NODES):
            raise ValueError(
                f"Unsupported expression construct: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in _ALLOWED_PHASE_DELTA_NAMES:
            raise ValueError(f"Unsupported name '{node.id}' in phase expression")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in _ALLOWED_PHASE_DELTA_FUNCS:
            raise ValueError(
                f"Function '{node.func.id}' is not allowed in phase expression"
            )
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in phase expression")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            sign = 1.0
            if isinstance(exponent, ast.UnaryOp) and isinstance(
                exponent.op, (ast.USub, ast.UAdd)
            ):
                sign = -1.0 if isinstance(exponent.op, ast.USub) else 1.0
                exponent = exponent.operand
            if (
                not isinstance(exponent, ast.Constant)
                or isinstance(exponent.value, bool)
                or not isinstance(exponent.value, (int, float))
                or not np.isfinite(float(exponent.value))
                or abs(sign * float(exponent.value)) > 16.0
            ):
                raise ValueError("Phase-expression powers require a constant exponent <= 16")
        self.generic_visit(node)


def _compile_phase_delta_expression(expression: str | None):
    """Compile and cache a validated phase-delta expression."""

    normalized = normalize_phase_delta_expression(expression)
    if len(normalized) > 256:
        raise ValueError("phase_delta_expression must not exceed 256 characters")
    cached = _PHASE_DELTA_EXPR_CACHE.get(normalized)
    if cached is not None:
        return cached, normalized

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid phase expression syntax: {exc.msg}") from exc

    if sum(1 for _node in ast.walk(tree)) > 64:
        raise ValueError("phase_delta_expression must not exceed 64 syntax nodes")
    _PhaseDeltaExprValidator().visit(tree)
    compiled = compile(tree, "<phase_delta_expression>", "eval")
    _PHASE_DELTA_EXPR_CACHE[normalized] = compiled
    while len(_PHASE_DELTA_EXPR_CACHE) > 32:
        _PHASE_DELTA_EXPR_CACHE.pop(next(iter(_PHASE_DELTA_EXPR_CACHE)))
    return compiled, normalized


def validate_phase_delta_expression(expression: str | None) -> str:
    """Validate expression and return normalized text."""

    compiled, normalized = _compile_phase_delta_expression(expression)
    L_test = np.asarray([0.0, 0.5], dtype=float)
    namespace = dict(_ALLOWED_PHASE_DELTA_FUNCS)
    namespace.update(
        {"h": 1.0, "k": 0.0, "L": L_test, "p": 0.25, "pi": np.pi}
    )
    try:
        raw = eval(compiled, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError(f"Phase expression evaluation failed: {exc}") from exc

    arr = np.asarray(raw, dtype=float)
    try:
        np.broadcast_to(arr, L_test.shape)
    except ValueError as exc:
        raise ValueError(
            "Phase expression must evaluate to a scalar or match L shape"
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("Phase expression must produce finite values")
    return normalized


def evaluate_phase_delta_expression(
    expression: str | None,
    h: int,
    k: int,
    L: np.ndarray,
    p: float,
) -> np.ndarray:
    """Evaluate custom delta(h, k, L, p) expression in radians."""

    compiled, normalized = _compile_phase_delta_expression(expression)
    L_vals = np.asarray(L, dtype=float)
    namespace = dict(_ALLOWED_PHASE_DELTA_FUNCS)
    namespace.update(
        {
            "h": float(h),
            "k": float(k),
            "L": L_vals,
            "p": float(p),
            "pi": np.pi,
        }
    )
    try:
        raw = eval(compiled, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate phase expression '{normalized}': {exc}"
        ) from exc

    delta = np.asarray(raw, dtype=float)
    try:
        delta = np.broadcast_to(delta, L_vals.shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(
            "Phase expression must evaluate to a scalar or match L shape"
        ) from exc
    if not np.all(np.isfinite(delta)):
        raise ValueError("Phase expression must produce finite values")
    return delta

# -------------------------- low-level physics helpers -----------------------
def _parse_cif_num(raw) -> float:
    """Parse CIF numeric values, including uncertainty suffixes."""

    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    txt = str(raw).strip()
    m = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", txt)
    if m is None:
        raise ValueError(f"Unable to parse CIF numeric value: {raw!r}")
    return float(m.group(0))


def _cell_a_c_from_cif(cif_path: str) -> tuple[float, float]:
    """Return (a, c) lattice parameters from a CIF using PyCifRW."""

    import CifFile

    cf = CifFile.ReadCif(cif_path)
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {cif_path}")
    blk = cf[keys[0]]
    a_raw = blk.get("_cell_length_a")
    c_raw = blk.get("_cell_length_c")
    if a_raw is None or c_raw is None:
        raise ValueError(f"_cell_length_a/_cell_length_c not found in {cif_path}")
    return _parse_cif_num(a_raw), _parse_cif_num(c_raw)


def _cif_cache_signature(cif_path: str) -> tuple[str, int | None, int | None]:
    """Return a cache signature that changes when the CIF file content changes."""

    abs_path = os.path.abspath(str(cif_path))
    try:
        st = os.stat(abs_path)
        return abs_path, int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        return abs_path, None, None


def infer_iodine_z_from_cif(cif_path: str) -> float | None:
    """Return the I1 fractional z coordinate from the CIF atom-site loop."""

    import CifFile

    cf = CifFile.ReadCif(cif_path)
    keys = list(cf.keys())
    if not keys:
        raise ValueError("CIF contains no data block")
    block = cf[keys[0]]
    labels = block.get("_atom_site_label")
    z_values = block.get("_atom_site_fract_z")
    if labels is None or z_values is None:
        raise ValueError("CIF requires _atom_site_label and _atom_site_fract_z")
    label_rows = list(labels) if isinstance(labels, (list, tuple)) else [labels]
    z_rows = list(z_values) if isinstance(z_values, (list, tuple)) else [z_values]
    if len(label_rows) != len(z_rows):
        raise ValueError("CIF atom-site labels and fractional-z values have different lengths")
    matches = [
        _parse_cif_num(z_raw)
        for label_raw, z_raw in zip(label_rows, z_rows, strict=True)
        if str(label_raw).strip().strip("'\"").startswith("I1")
    ]
    if len(matches) > 1:
        raise ValueError("CIF contains more than one I1 atom-site row")
    return float(matches[0]) if matches else None

def _normalize_occ_factors(occ_factors, n_sites: int) -> np.ndarray:
    """Return one occupancy scale factor per generated structure site."""

    n_sites = int(max(1, n_sites))
    if isinstance(occ_factors, (list, tuple, np.ndarray)):
        values = [float(v) for v in occ_factors]
        if not values:
            values = [1.0]
        if len(values) < n_sites:
            values.extend([values[-1]] * (n_sites - len(values)))
        else:
            values = values[:n_sites]
        return np.asarray(values, dtype=np.float64)

    return np.full(n_sites, float(occ_factors), dtype=np.float64)


def _sites_from_cif_with_factors(cif_path: str, occ_factors=1.0):
    """Return atomic sites with per-generated-site occupancy factors applied."""
    import Dans_Diffraction as dif
    xtl = dif.Crystal(str(cif_path))
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    st = xtl.Structure
    n_sites = len(st.u)
    occ_vals = np.ones(n_sites, dtype=np.float64)  # match diffuse_cif_toggle: ignore CIF occupancy
    site_factors = _normalize_occ_factors(occ_factors, n_sites)
    return [
        (
            float(st.u[i]),
            float(st.v[i]),
            float(st.w[i]),
            str(st.type[i]),
            float(occ_vals[i]) * float(site_factors[i]),
        )
        for i in range(n_sites)
    ]


def _energy_kev_from_lambda(lambda_a: float) -> float:
    """Convert wavelength in Å to energy in keV."""
    return (12398.4193 / float(lambda_a)) / 1000.0


# Prefer known ionic scattering-factor labels when available; fall back to
# neutral-element labels for all other species.
IONIC_F0_LABELS = {
    "Pb": "Pb2+",
    "I": "I1-",
}

def _element_key(sym: str) -> str:
    """Map CIF site symbol to a periodic-table element key."""

    text = str(sym).strip()
    m = re.match(r"([A-Za-z]{1,2})", text)
    if m:
        token = m.group(1)
    else:
        letters = "".join(ch for ch in text if ch.isalpha())
        if not letters:
            raise KeyError(f"Unknown element symbol in CIF type '{sym}'")
        token = letters[:2]

    token = token[0].upper() + (token[1].lower() if len(token) > 1 else "")
    return token

def f_comp(el_sym: str, Q: np.ndarray, energy_kev: float) -> np.ndarray:
    """
    Composite atomic form factor f = f0 + f' + i f''.

    Parameters
    ----------
    el_sym : str
        CIF site symbol (e.g. 'Pb', 'I', 'Pb1').
    Q : ndarray
        |Q| magnitude in Å^-1.
    energy_kev : float
        Photon energy in keV.
    """
    from Dans_Diffraction.functions_crystallography import (
        xray_scattering_factor,
        xray_dispersion_corrections,
    )

    element = _element_key(el_sym)
    q = np.asarray(Q, dtype=float).reshape(-1)

    f0_label = IONIC_F0_LABELS.get(element, element)
    try:
        f0 = xray_scattering_factor([f0_label], q)[:, 0]
    except Exception as exc:
        raise ValueError(f"Unsupported x-ray form-factor label: {f0_label!r}") from exc

    try:
        f1, f2 = xray_dispersion_corrections([element], energy_kev=[float(energy_kev)])
        f1 = float(f1[0, 0])
        f2 = float(f2[0, 0])
    except Exception as exc:
        raise ValueError(
            f"Unable to compute x-ray dispersion corrections for {element!r}"
        ) from exc

    out = f0 + f1 + 1j * f2
    return out.reshape(Q.shape)


def analytical_ht_intensity_for_pair(
    L_vals,
    F2_vals,
    h: int,
    k: int,
    p: float,
    *,
    phase_delta_expression: str | None = None,
    phi_l_divisor: float = DEFAULT_PHI_L_DIVISOR,
    finite_layers: int | None = None,
    f2_only: bool = False,
) -> np.ndarray:
    """Return analytical HT intensity for one (h, k) rod.

    Matches the algebraic HT implementation used in diffuse_cif_toggle.py:
      - p is flipped (p -> 1 - p) before forming z
      - R uses the same infinite- or finite-layer closed forms
      - no extra clipping/regularization beyond P_CLAMP
    """

    F2_vals = np.asarray(F2_vals, dtype=float)
    if f2_only:
        return F2_vals

    L_vals = np.asarray(L_vals, dtype=float)

    # Match diffuse_cif_toggle.py convention: flip p -> 1 - p
    p_flipped = 1.0 - float(p)

    # Allow custom delta(h,k,L,p) but pass the *flipped* p so delta uses the same p
    # that appears in z, consistent with the diffuse algebra.
    delta = evaluate_phase_delta_expression(
        phase_delta_expression,
        h,
        k,
        L_vals,
        p_flipped,
    )

    z = (1.0 - p_flipped) + p_flipped * np.exp(1j * delta)
    f_val = np.minimum(np.abs(z), 1.0 - float(P_CLAMP))
    psi = np.angle(z)
    phi_div = normalize_phi_l_divisor(phi_l_divisor)
    phi = delta + _TWO_PI * L_vals * (1.0 / phi_div)

    if finite_layers is None:
        R = (1.0 - f_val**2) / (1.0 + f_val**2 - 2.0 * f_val * np.cos(phi - psi))
    else:
        t = f_val * np.exp(1j * (phi - psi))
        n_layers = int(max(1, finite_layers))
        if n_layers == 1:
            R = np.ones_like(np.real(t), dtype=float)
        else:
            series = _rich_phase_weighted_finite_sum(t, n_layers)
            R = np.maximum((float(n_layers) + 2.0 * np.real(series)) / float(n_layers), 0.0)

    return float(AREA) * F2_vals * R


# ----------------------------- base curve builder ---------------------------
def _hk_radial_index(h: int, k: int) -> int:
    """Hexagonal in-plane radial class index m = h^2 + hk + k^2."""

    return int(h * h + h * k + k * k)


def _normalize_complex_phase_vector(
    values: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize complex coefficients by a stable anchor phase.

    This turns globally phase-shifted coefficient vectors into the same canonical
    representation, so we can safely reuse |F|^2 when a reflection differs only
    by an overall complex phase factor.
    """

    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        return arr

    nz = np.flatnonzero(np.abs(arr) > float(eps))
    if nz.size == 0:
        return np.zeros_like(arr, dtype=np.complex128)

    anchor = arr[int(nz[0])]
    if abs(anchor) <= float(eps):
        return arr.copy()
    return arr / anchor


def _complex_phase_signature_key(values: np.ndarray, *, digits: int = 12) -> bytes:
    """Quantized byte-key for complex vectors used in reflection dedup lookup."""

    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    packed = np.empty((arr.size, 2), dtype=np.float64)
    packed[:, 0] = np.round(np.real(arr), int(digits))
    packed[:, 1] = np.round(np.imag(arr), int(digits))
    return packed.tobytes()


def _get_base_curves(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    phase_z_divisor: float = DEFAULT_PHI_L_DIVISOR,
    iodine_z: float | None = None,
):
    """
    Return cached {(h,k): {"L": ..., "F2": ...}} for a given occupancy mapping.

    Conventions align with diffuse_cif_toggle:
      - a/c used for |Q| (and optional two-theta clipping) are a_lattice/c_lattice when
        provided, otherwise the CIF a/c (no 3x scaling).
      - Iodine contributions are collapsed onto one shared z-plane (iodine_z or the
        first iodine z found in the CIF), matching diffuse_cif_toggle.
      - Phase uses z/phase_z_divisor in the vertical factor.
    """
    import itertools
    import math

    _reject_pbi2_standard_ht(cif_path)

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        hk_list = [(h, k) for h, k in itertools.product(range(-mx + 1, mx), repeat=2)]
    hk_list = [(int(h), int(k)) for (h, k) in hk_list]

    cif_signature = _cif_cache_signature(cif_path)
    key = (
        cif_signature,
        tuple(hk_list),
        float(L_step),
        float(L_max),
        two_theta_max,
        float(lambda_),
        None if a_lattice is None else float(a_lattice),
        None if c_lattice is None else float(c_lattice),
        float(phase_z_divisor),
        None if iodine_z is None else float(iodine_z),
    )
    cached = _HT_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    a_cif, c_cif = _cell_a_c_from_cif(cif_path)
    # Match diffuse_cif_toggle.py: ignore CIF occupancy and any external occupancy scaling in F².
    sites = _sites_from_cif_with_factors(cif_path, occ_factors=1.0)
    if L_step < 1e-4:
        raise ValueError("L_step must be at least 1e-4")

    a_effective = float(a_cif if a_lattice is None else a_lattice)
    if not math.isfinite(a_effective) or a_effective <= 0.0:
        raise ValueError("a_lattice must be finite and positive")

    c_effective = float(c_cif if c_lattice is None else c_lattice)
    if not math.isfinite(c_effective) or c_effective <= 0.0:
        raise ValueError("c_lattice must be finite and positive")

    a_window_factor = float(a_effective)
    c_window_factor = float(c_effective)
    a_form_factor = float(a_effective)
    c_form_factor = float(c_effective)
    energy_kev = _energy_kev_from_lambda(lambda_)

    z_div = float(phase_z_divisor)
    if (not np.isfinite(z_div)) or abs(z_div) < 1e-14:
        raise ValueError("phase_z_divisor must be finite and nonzero")

    site_count = len(sites)
    site_x = np.empty(site_count, dtype=np.float64)
    site_y = np.empty(site_count, dtype=np.float64)
    site_z = np.empty(site_count, dtype=np.float64)
    site_element: list[str] = []
    for idx, (x, y, z, sym, _occ) in enumerate(sites):
        site_x[idx] = float(x)
        site_y[idx] = float(y)
        site_z[idx] = float(z)
        site_element.append(_element_key(sym))
    site_is_iodine = np.asarray([el == "I" for el in site_element], dtype=bool)

    # Match diffuse_cif_toggle.py F² behaviour: treat iodine as a single plane at
    # z = iodine_z (default inferred from the CIF) by removing iodine z from the
    # per-site phase and applying one shared exp(2π i L * (iodine_z/phase_z_divisor)).
    iodine_z_eff = None
    iodine_active = False
    if bool(np.any(site_is_iodine)):
        iodine_z_eff = iodine_z
        if iodine_z_eff is None:
            iodine_z_eff = infer_iodine_z_from_cif(cif_path)
        if iodine_z_eff is not None:
            iodine_active = True

    # Preserve site-encounter order for deterministic behavior.
    unique_elements = list(dict.fromkeys(site_element))
    if iodine_active and "I" not in unique_elements:
        unique_elements.append("I")
    site_indices_by_element: dict[str, np.ndarray] = {}
    for elem in unique_elements:
        indices = [idx for idx, site_el in enumerate(site_element) if site_el == elem]
        site_indices_by_element[elem] = np.asarray(indices, dtype=np.int64)
    if iodine_active:
        iodine_indices = site_indices_by_element.get("I", np.empty((0,), dtype=np.int64))
    else:
        iodine_indices = np.empty((0,), dtype=np.int64)
    non_iodine_elements = [
        elem for elem in unique_elements if not (iodine_active and elem == "I")
    ]

    base_L = None
    q_max = None
    if two_theta_max is None:
        base_L = np.arange(0.0, L_max + L_step / 2.0, L_step, dtype=float)
    else:
        q_max = (4.0 * math.pi / lambda_) * math.sin(math.radians(two_theta_max / 2.0))

    L_grid_cache: dict[int, np.ndarray] = {}

    def _L_for_m(m: int) -> np.ndarray:
        cached_L = L_grid_cache.get(m)
        if cached_L is not None:
            return cached_L

        if base_L is not None:
            L_vals = base_L
        else:
            const = (4.0 / 3.0) * float(m) / (a_window_factor**2)
            l_sq = (float(q_max) / (2.0 * math.pi))**2 - const
            if l_sq <= 0.0:
                L_vals = np.array([], dtype=float)
            else:
                L_max_local = c_window_factor * math.sqrt(l_sq)
                L_vals = np.arange(0.0, L_max_local + L_step / 2.0, L_step, dtype=float)

        L_grid_cache[m] = L_vals
        return L_vals

    # Per-radial-class cache:
    # - form factors ff(Q) by element
    # - z-phase factors by site
    # - optional iodine z-phase
    # - reflection-level dedup cache by normalized in-plane phase signature
    m_state_cache: dict[int, dict] = {}

    def _state_for_m(m: int) -> dict:
        state = m_state_cache.get(m)
        if state is not None:
            return state

        L_vals = _L_for_m(m)
        state = {
            "L": L_vals,
            "ff_by_element": {},
            "phase_z_stack_by_element": {},
            "phase_z_iodine": None,
            "signature_cache": {},
        }
        if L_vals.size > 0:
            q_term = (4.0 / 3.0) * float(m) / (a_form_factor**2)
            Q_vals = 2.0 * np.pi * np.sqrt(q_term + (L_vals * L_vals) / (c_form_factor**2))

            ff_by_element = {}
            for elem in unique_elements:
                ff_by_element[elem] = np.asarray(
                    f_comp(elem, Q_vals, energy_kev),
                    dtype=np.complex128,
                )
            state["ff_by_element"] = ff_by_element

            phase_z_stack_by_element = {}
            phase_z_cache: dict[float, np.ndarray] = {}
            for elem in non_iodine_elements:
                elem_indices = site_indices_by_element[elem]
                phase_z_stack = np.empty(
                    (elem_indices.size, L_vals.size),
                    dtype=np.complex128,
                )
                for row_idx, site_idx in enumerate(elem_indices):
                    z_val = site_z[site_idx]
                    z_key = round(float(z_val), 12)
                    phase_z = phase_z_cache.get(z_key)
                    if phase_z is None:
                        phase_z = np.exp(1j * _TWO_PI * (L_vals * (float(z_val) / z_div)))
                        phase_z_cache[z_key] = phase_z
                    phase_z_stack[row_idx, :] = phase_z
                phase_z_stack_by_element[elem] = phase_z_stack
            state["phase_z_stack_by_element"] = phase_z_stack_by_element

            if iodine_active:
                state["phase_z_iodine"] = np.exp(
                    1j * _TWO_PI * (L_vals * (float(iodine_z_eff) / z_div))
                )

        m_state_cache[m] = state
        return state

    out: dict[tuple, dict] = {}
    for h, k in hk_list:
        m = _hk_radial_index(h, k)
        state = _state_for_m(m)
        L_vals = state["L"]

        if L_vals.size == 0:
            L_empty = np.array([], dtype=float)
            out[(h, k)] = {"L": L_empty, "F2": L_empty}
            continue

        phase_xy = np.exp(1j * _TWO_PI * (float(h) * site_x + float(k) * site_y))
        coeff_norm = _normalize_complex_phase_vector(phase_xy)
        signature_key = _complex_phase_signature_key(coeff_norm)

        reused_curve = None
        candidates = state["signature_cache"].get(signature_key)
        if candidates is not None:
            for prev_norm, prev_curve in candidates:
                if np.allclose(coeff_norm, prev_norm, rtol=0.0, atol=1e-11):
                    reused_curve = prev_curve
                    break

        if reused_curve is None:
            F = np.zeros(L_vals.shape, dtype=np.complex128)
            ff_by_element = state["ff_by_element"]
            phase_z_stack_by_element = state["phase_z_stack_by_element"]

            for elem in non_iodine_elements:
                elem_indices = site_indices_by_element[elem]
                if elem_indices.size == 0:
                    continue
                F += ff_by_element[elem] * np.dot(
                    phase_xy[elem_indices],
                    phase_z_stack_by_element[elem],
                )

            if iodine_active and state["phase_z_iodine"] is not None and iodine_indices.size > 0:
                iodine_coeff = np.sum(phase_xy[iodine_indices])
                F += ff_by_element["I"] * iodine_coeff * state["phase_z_iodine"]

            curve = {"F2": np.abs(F) ** 2}

            entries = state["signature_cache"].setdefault(signature_key, [])
            entries.append((coeff_norm.copy(), curve))
            reused_curve = curve

        out_entry = {
            "L": np.asarray(L_vals, dtype=float).copy(),
            "F2": np.asarray(reused_curve["F2"], dtype=float).copy(),
        }
        out[(h, k)] = out_entry

    _HT_BASE_CACHE[key] = out
    # Bound cache growth because occupancy sliders can produce many unique keys.
    while len(_HT_BASE_CACHE) > _HT_BASE_CACHE_MAX_ENTRIES:
        try:
            _HT_BASE_CACHE.pop(next(iter(_HT_BASE_CACHE)))
        except StopIteration:
            break
    return out


# ------------------------------- public routine -----------------------------
def ht_Iinf_dict(
    cif_path: str,
    hk_list=None,                 # explicit list or None
    mx: int | None = None,        # generate -mx+1..mx-1 if hk_list is None
    p: float = 0.1,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    phase_z_divisor: float | None = None,
    iodine_z: float | None = None,
    phase_delta_expression: str | None = None,
    phi_l_divisor: float = DEFAULT_PHI_L_DIVISOR,
    *,
    finite_stack: bool = False,
    stack_layers: int = 50,
):
    """
    Hendricks–Teller intensities using the analytical HT expression.

    Returns {(h,k): {'L':..., 'I':...}} with F² and C2H conventions identical
    to diffuse_cif_toggle.py.
    When ``c_lattice`` is provided it defines the active L-axis convention:
    both the two-theta clipping window and the Qz scaling inside F² use that
    effective c-axis length instead of the raw 2H value from the CIF.
    ``a_lattice`` optionally overrides the active in-plane lattice constant
    used for |Q| and two-theta clipping.
    ``phase_z_divisor`` controls vertical phase scaling in F². When omitted it
    defaults to ``phi_l_divisor`` so the F² vertical phase and HT correlation
    phase use the same L-axis convention.
    ``iodine_z`` optionally pins the iodine z-plane used in F². When ``None``,
    the value is inferred from the CIF in the same way as diffuse_cif_toggle.
    ``phase_delta_expression`` must define delta(h, k, L, p) in radians for
    the analytical HT correlation term.
    ``phi_l_divisor`` sets the out-of-plane term in the HT phase as
    ``phi = delta + 2*pi*L/phi_l_divisor``.
    When ``finite_stack`` is ``True`` the per-layer finite-thickness factor for
    ``stack_layers`` layers is applied instead of the infinite-domain limit.
    """
    _reject_pbi2_standard_ht(cif_path)

    phase_expr = validate_phase_delta_expression(phase_delta_expression)
    phi_div = normalize_phi_l_divisor(phi_l_divisor)
    if phase_z_divisor is None:
        phase_z_div = phi_div
    else:
        phase_z_div = normalize_phi_l_divisor(phase_z_divisor)

    base = _get_base_curves(
        cif_path=cif_path,
        hk_list=hk_list,
        mx=mx,
        L_step=L_step,
        L_max=L_max,
        two_theta_max=two_theta_max,
        lambda_=lambda_,
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        phase_z_divisor=phase_z_div,
        iodine_z=iodine_z,
    )

    out = {}
    finite_layers = int(max(1, stack_layers)) if finite_stack else None

    for (h, k), data in base.items():
        L_vals = data["L"]
        F2 = data["F2"]
        I = analytical_ht_intensity_for_pair(
            L_vals,
            F2,
            h,
            k,
            p,
            phase_delta_expression=phase_expr,
            phi_l_divisor=phi_div,
            finite_layers=finite_layers,
        )
        out[(h, k)] = {"L": L_vals.copy(), "I": I}
    return out


# ------------------------- array and rod grouping helpers -------------------
def ht_dict_to_arrays(ht_curves):
    """
    Convert the dict output of ht_Iinf_dict to arrays compatible with
    miller_generator style consumers.
    """
    total = sum(len(c["L"]) for c in ht_curves.values())
    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.ones(total, dtype=np.int32)
    details = []

    idx = 0
    for (h, k), curve in ht_curves.items():
        L_vals = curve["L"]
        I_vals = curve["I"]
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details


def ht_dict_to_qr_dict(ht_curves):
    """
    Combine HT curves with identical radial index m = h^2 + hk + k^2.
    """
    rods = {}
    for (h, k), curve in ht_curves.items():
        L_vals = np.asarray(curve["L"], dtype=float)
        I_vals = np.asarray(curve["I"], dtype=float)
        m = h*h + h*k + k*k
        if m not in rods:
            rods[m] = {"L": L_vals.copy(), "I": I_vals.copy(), "hk": (h, k), "deg": 1}
            continue

        entry = rods[m]
        if entry["L"].shape != L_vals.shape or not np.allclose(entry["L"], L_vals):
            union = np.union1d(entry["L"], L_vals)
            entry_I = np.interp(union, entry["L"], entry["I"], left=0.0, right=0.0)
            add_I   = np.interp(union, L_vals, I_vals,      left=0.0, right=0.0)
            entry["L"] = union
            entry["I"] = entry_I + add_I
        else:
            entry["I"] += I_vals
        entry["deg"] += 1

    return rods


def qr_dict_to_arrays(qr_dict):
    """Convert a qr_dict from ht_dict_to_qr_dict into arrays."""
    total = sum(len(v["L"]) for v in qr_dict.values())
    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.empty(total, dtype=np.int32)
    details = []

    idx = 0
    for m, data in sorted(qr_dict.items()):
        h, k = data["hk"]
        L_vals = data["L"]
        I_vals = data["I"]
        deg = int(data.get("deg", 1))
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals
        degeneracy[idx:idx+n] = deg

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details
