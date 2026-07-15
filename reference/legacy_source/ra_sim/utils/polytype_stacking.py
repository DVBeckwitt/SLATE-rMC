"""Reduced 2x2 transition-matrix stacking model for PbI2 polytypes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from ra_sim.utils import stacking_fault as sf

COMPONENT_ORDER: tuple[str, ...] = ("mix", "2H", "4H", "6H")
COMPONENT_LABELS: dict[str, str] = {
    "mix": "Mix / diffuse",
    "2H": "2H-rich",
    "4H": "4H-rich",
    "6H": "6H-rich",
}
POLYTYPE_PHASE_DELTA_EXPRESSION = "2*pi*((h + 2*k)/3)"
PBI2_MODEL_BOUNDARY_ERROR = "Reduced 2x2 polytype stacking requires a PbI2 CIF."


_ORIENTATION_BASIS_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_ORIENTATION_BASIS_CACHE_MAX = 8
_COMPONENT_INTENSITY_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_COMPONENT_INTENSITY_CACHE_MAX = 48


DEFAULT_COMPONENTS: dict[str, dict[str, float]] = {
    "mix": {"a": 0.56, "b": 0.26, "d": 0.18, "weight": 15.0},
    "2H": {"a": 1.00, "b": 0.00, "d": 0.00, "weight": 70.0},
    "4H": {"a": 0.00, "b": 0.00, "d": 1.00, "weight": 5.0},
    "6H": {"a": 0.00, "b": 1.00, "d": 0.00, "weight": 10.0},
}


def default_components() -> dict[str, dict[str, float]]:
    """Return a deep copy of the default four-component model."""

    return {key: dict(value) for key, value in DEFAULT_COMPONENTS.items()}


def _finite_nonnegative(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("polytype component values must be numeric") from exc
    if not np.isfinite(out) or out < 0.0:
        raise ValueError("polytype component values must be finite and non-negative")
    return out


def canonicalize_components(
    components: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    """Return the current complete four-component parameter mapping."""

    if components is None:
        return default_components()
    if not isinstance(components, Mapping):
        raise TypeError("polytype components must be a mapping")
    if set(components) != set(COMPONENT_ORDER):
        raise ValueError(f"polytype components must contain exactly {COMPONENT_ORDER!r}")
    result: dict[str, dict[str, float]] = {}
    for key in COMPONENT_ORDER:
        raw = components[key]
        if not isinstance(raw, Mapping):
            raise TypeError(f"polytype component {key!r} must be a mapping")
        if set(raw) != {"a", "b", "d", "weight"}:
            raise ValueError(f"polytype component {key!r} must contain a, b, d, and weight")
        result[key] = {
            "a": _finite_nonnegative(raw["a"]),
            "b": _finite_nonnegative(raw["b"]),
            "d": _finite_nonnegative(raw["d"]),
            "weight": _finite_nonnegative(raw["weight"]),
        }
    return result


def normalize_abd(component: Mapping[str, Any]) -> tuple[float, float, float]:
    """Normalize one component's raw ``a,b,d`` values to probabilities."""

    values = [
        _finite_nonnegative(component["a"]),
        _finite_nonnegative(component["b"]),
        _finite_nonnegative(component["d"]),
    ]
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("polytype component a, b, and d cannot all be zero")
    return tuple(float(value / total) for value in values)


def normalized_component_weights(
    components: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, float]:
    """Return normalized non-negative intensity weights for all components."""

    canon = canonicalize_components(components)
    values = [float(canon[key]["weight"]) for key in COMPONENT_ORDER]
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("polytype component weights cannot all be zero")
    return {key: values[idx] / total for idx, key in enumerate(COMPONENT_ORDER)}


def component_signature(
    components: Mapping[str, Mapping[str, Any]] | None,
    *,
    digits: int = 12,
) -> tuple[tuple[str, float, float, float, float], ...]:
    """Stable signature for cache invalidation and session comparisons."""

    canon = canonicalize_components(components)
    return tuple(
        (
            key,
            round(canon[key]["a"], digits),
            round(canon[key]["b"], digits),
            round(canon[key]["d"], digits),
            round(canon[key]["weight"], digits),
        )
        for key in COMPONENT_ORDER
    )


def normalized_component_summary(
    components: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, float]]:
    """Return normalized probabilities and mixture fractions for display."""

    canon = canonicalize_components(components)
    weights = normalized_component_weights(canon)
    out: dict[str, dict[str, float]] = {}
    for key in COMPONENT_ORDER:
        a, b, d = normalize_abd(canon[key])
        out[key] = {"a": a, "b": b, "d": d, "weight": weights[key]}
    return out


def _geom_sum_1_to_n(x: np.ndarray, n: int) -> np.ndarray:
    """Return ``sum(x**j, j=1..n)`` with the removable x=1 pole handled."""

    x = np.asarray(x, dtype=complex)
    if n <= 0:
        return np.zeros_like(x)
    out = np.empty_like(x)
    mask = np.isclose(x, 1.0 + 0.0j, rtol=1e-12, atol=1e-12)
    out[mask] = float(n)
    xm = x[~mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[~mask] = xm * (1.0 - xm**n) / (1.0 - xm)
    return out


def _weighted_finite_sum(x: np.ndarray, N: int) -> np.ndarray:
    """Return ``sum((N-dd)*x**dd, dd=1..N-1)`` exactly."""

    x = np.asarray(x, dtype=complex)
    if N <= 1:
        return np.zeros_like(x)
    out = np.empty_like(x)
    mask = np.isclose(x, 1.0 + 0.0j, rtol=1e-12, atol=1e-12)
    out[mask] = float(N * (N - 1) / 2)
    xm = x[~mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = 1.0 - xm
        s1 = xm * (1.0 - xm ** (N - 1)) / denom
        s2 = xm * (1.0 - N * xm ** (N - 1) + (N - 1) * xm**N) / denom**2
        out[~mask] = N * s1 - s2
    return out


def _cross_geometric_sum(x: np.ndarray, r: float, N: int) -> np.ndarray:
    """Return ``sum(x**dd * r**(N-dd), dd=1..N-1)``."""

    x = np.asarray(x, dtype=complex)
    if N <= 1:
        return np.zeros_like(x)
    out = np.empty_like(x)
    mask = np.isclose(x, complex(r), rtol=1e-12, atol=1e-12)
    out[mask] = (N - 1) * (complex(r) ** N)
    xm = x[~mask]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out[~mask] = xm * r * (r ** (N - 1) - xm ** (N - 1)) / (r - xm)
    return out


def _contrast_finite_sum(x: np.ndarray, r: float, N: int) -> np.ndarray:
    """Finite sum weighted by the transient orientation contrast."""

    if abs(1.0 - r) < 1e-12:
        return _weighted_finite_sum(x, N)
    return (_geom_sum_1_to_n(x, N - 1) - _cross_geometric_sum(x, r, N)) / (1.0 - r)


def _orientation_contrast_sum(r: float, N: int) -> float:
    """Return ``sum(r**m, m=0..N-1)``."""

    if N <= 0:
        return 0.0
    if abs(1.0 - r) < 1e-12:
        return float(N)
    return float((1.0 - r**N) / (1.0 - r))


def _intensity_from_basis(
    *,
    xi: np.ndarray,
    omega: np.ndarray,
    F_plus: np.ndarray,
    F_minus: np.ndarray,
    abd: tuple[float, float, float],
    finite_layers: int | None,
) -> np.ndarray:
    """Return the per-layer intensity for one reflection and one component."""

    a, b, d = abd
    xi = np.asarray(xi, dtype=complex)
    omega = np.asarray(omega, dtype=complex)
    Fp = np.asarray(F_plus, dtype=complex)
    Fm = np.asarray(F_minus, dtype=complex)

    U = np.abs(Fp) ** 2
    V = np.abs(Fm) ** 2
    X = omega * np.conj(Fp) * Fm
    Y = np.conj(X)

    if finite_layers is not None:
        N = int(max(1, finite_layers))
        r = 1.0 - 2.0 * d
        contrast_N = _orientation_contrast_sum(r, N)

        u0 = 0.5 * (U + V)
        u1 = 0.5 * (U - V)
        x0 = 0.5 * (X + Y)
        x1 = 0.5 * (X - Y)
        I_self = u0 + (contrast_N / float(N)) * u1
        if N == 1:
            return sf.AREA * np.maximum(np.real(I_self), 0.0)

        C = a + b * omega
        lam_plus = C + d
        lam_minus = C - d
        xp = xi * lam_plus
        xm = xi * lam_minus
        S_plus = _weighted_finite_sum(xp, N)
        S_minus = _weighted_finite_sum(xm, N)
        D_plus = _contrast_finite_sum(xp, r, N)
        D_minus = _contrast_finite_sum(xm, r, N)
        pair_sum = 0.5 * (
            (u0 + x0) * S_plus + (u1 + x1) * D_plus + (u0 - x0) * S_minus + (u1 - x1) * D_minus
        )
        intensity = sf.AREA * (I_self + (2.0 / float(N)) * np.real(pair_sum))
        return np.maximum(np.real(intensity), 0.0)

    p0, p1 = (1.0, 0.0) if abs(float(d)) < 1e-15 else (0.5, 0.5)
    I_self = p0 * U + p1 * V
    C = a + b * omega
    lam_plus = C + d
    lam_minus = C - d
    z = (1.0 - float(sf.P_CLAMP)) * xi
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        S_plus = (z * lam_plus) / (1.0 - z * lam_plus)
        S_minus = (z * lam_minus) / (1.0 - z * lam_minus)
    Acoef = 0.5 * (S_plus + S_minus)
    Bcoef = 0.5 * (S_plus - S_minus)
    pair_sum = Acoef * (p0 * U + p1 * V) + Bcoef * (p0 * X + p1 * Y)
    intensity = sf.AREA * (I_self + 2.0 * np.real(pair_sum))
    return np.maximum(np.real(intensity), 0.0)


def _periodic_delta(value: float, origin: float) -> float:
    return ((float(value) - float(origin) + 0.5) % 1.0) - 0.5


def _orientation_basis(
    *,
    cif_path: str,
    mx: int,
    L_step: float,
    L_max: float,
    two_theta_max: float | None,
    lambda_: float,
    a_lattice: float | None,
    c_lattice: float | None,
    phase_delta_expression: str,
    phi_l_divisor: float,
    layer_form_factor_provider: object | None,
) -> tuple[tuple, dict[tuple[int, int], dict[str, np.ndarray]]]:
    """Return cached reflection-dependent ``L, xi, omega, F+, F-`` arrays."""

    auto_pbi2_provider = layer_form_factor_provider is None and sf.cif_looks_like_pbi2(
        str(cif_path)
    )
    provider_key = (
        ("auto_pbi2",)
        if auto_pbi2_provider
        else (
            None
            if layer_form_factor_provider is None
            else ("explicit", id(layer_form_factor_provider))
        )
    )
    cache_key = (
        sf._cif_cache_signature(str(cif_path)),
        int(mx),
        round(float(L_step), 12),
        round(float(L_max), 12),
        None if two_theta_max is None else round(float(two_theta_max), 12),
        round(float(lambda_), 12),
        None if a_lattice is None else round(float(a_lattice), 12),
        None if c_lattice is None else round(float(c_lattice), 12),
        str(phase_delta_expression),
        round(float(phi_l_divisor), 12),
        provider_key,
    )
    cached = _ORIENTATION_BASIS_CACHE.get(cache_key)
    if cached is not None:
        _ORIENTATION_BASIS_CACHE.move_to_end(cache_key)
        return cache_key, cached

    provider = layer_form_factor_provider
    if auto_pbi2_provider:
        from ra_sim.stacking.motif_form_factor import GenericOrientationFormFactorProvider
        from ra_sim.structure_factors.options import StructureFactorOptions

        provider = GenericOrientationFormFactorProvider.from_pbi2_2h_cif(
            str(cif_path),
            wavelength_angstrom=float(lambda_),
            options=StructureFactorOptions.package_default(),
        )

    base = sf.rich_phase_basis_curve_map(
        cif_path=str(cif_path),
        mx=int(mx),
        L_step=float(L_step),
        L_max=float(L_max),
        two_theta_max=two_theta_max,
        lambda_=float(lambda_),
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        layer_form_factor_provider=provider,
    )
    result: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for (h, k), data in base.items():
        L = np.asarray(data.get("L", ()), dtype=float)
        if L.size == 0:
            result[(h, k)] = {
                "L": L,
                "xi": L.astype(complex),
                "omega": L.astype(complex),
                "F_plus": L.astype(complex),
                "F_minus": L.astype(complex),
            }
            continue
        delta = sf.evaluate_phase_delta_expression(phase_delta_expression, h, k, L, 0.0)
        omega = np.exp(1j * delta)
        xi = np.exp(1j * sf._TWO_PI * L / float(phi_l_divisor))
        Fp = np.asarray(data.get("F_plus", ()), dtype=complex)
        Fm = np.asarray(data.get("F_minus", ()), dtype=complex)
        result[(h, k)] = {
            "L": L.copy(),
            "xi": xi,
            "omega": omega,
            "F_plus": Fp,
            "F_minus": Fm,
        }

    _ORIENTATION_BASIS_CACHE[cache_key] = result
    while len(_ORIENTATION_BASIS_CACHE) > _ORIENTATION_BASIS_CACHE_MAX:
        _ORIENTATION_BASIS_CACHE.popitem(last=False)
    return cache_key, result


def _cached_component_intensities(
    *,
    basis_key: tuple,
    basis: Mapping[tuple[int, int], Mapping[str, np.ndarray]],
    abd: tuple[float, float, float],
    finite_layers: int | None,
) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    key = (
        basis_key,
        tuple(round(float(value), 12) for value in abd),
        None if finite_layers is None else int(finite_layers),
    )
    cached = _COMPONENT_INTENSITY_CACHE.get(key)
    if cached is not None:
        _COMPONENT_INTENSITY_CACHE.move_to_end(key)
        return cached
    curves: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for hk, item in basis.items():
        L = np.asarray(item["L"], dtype=float)
        if L.size == 0:
            curves[hk] = {"L": L, "I": L.copy()}
            continue
        I = _intensity_from_basis(
            xi=np.asarray(item["xi"], dtype=complex),
            omega=np.asarray(item["omega"], dtype=complex),
            F_plus=np.asarray(item["F_plus"], dtype=complex),
            F_minus=np.asarray(item["F_minus"], dtype=complex),
            abd=abd,
            finite_layers=finite_layers,
        )
        curves[hk] = {"L": L.copy(), "I": I}
    _COMPONENT_INTENSITY_CACHE[key] = curves
    while len(_COMPONENT_INTENSITY_CACHE) > _COMPONENT_INTENSITY_CACHE_MAX:
        _COMPONENT_INTENSITY_CACHE.popitem(last=False)
    return curves


def polytype_component_curves(
    *,
    cif_path: str,
    mx: int,
    components: Mapping[str, Mapping[str, Any]] | None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    phase_delta_expression: str | None = None,
    phi_l_divisor: float = sf.DEFAULT_PHI_L_DIVISOR,
    finite_stack: bool = False,
    stack_layers: int = 50,
    layer_form_factor_provider: object | None = None,
) -> tuple[
    dict[tuple[int, int], dict[str, np.ndarray]],
    dict[str, dict[tuple[int, int], dict[str, np.ndarray]]],
]:
    """Calculate the weighted total and the four unweighted component curves."""

    if not sf.cif_looks_like_pbi2(str(cif_path)):
        raise ValueError(PBI2_MODEL_BOUNDARY_ERROR)

    canon = canonicalize_components(components)
    weights = normalized_component_weights(canon)
    phase_expr = sf.validate_phase_delta_expression(
        sf.normalize_phase_delta_expression(
            POLYTYPE_PHASE_DELTA_EXPRESSION
            if phase_delta_expression is None
            else phase_delta_expression,
        )
    )
    phi_div = sf.normalize_phi_l_divisor(phi_l_divisor)
    finite_layers = int(max(1, stack_layers)) if finite_stack else None

    basis_key, basis = _orientation_basis(
        cif_path=str(cif_path),
        mx=int(mx),
        L_step=float(L_step),
        L_max=float(L_max),
        two_theta_max=two_theta_max,
        lambda_=float(lambda_),
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        phase_delta_expression=phase_expr,
        phi_l_divisor=phi_div,
        layer_form_factor_provider=layer_form_factor_provider,
    )

    component_curves = {
        key: _cached_component_intensities(
            basis_key=basis_key,
            basis=basis,
            abd=normalize_abd(canon[key]),
            finite_layers=finite_layers,
        )
        for key in COMPONENT_ORDER
    }
    total: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for hk, basis_item in basis.items():
        L = np.asarray(basis_item["L"], dtype=float)
        total_I = np.zeros_like(L, dtype=float)
        for key in COMPONENT_ORDER:
            total_I += float(weights[key]) * np.asarray(component_curves[key][hk]["I"], dtype=float)
        total[hk] = {"L": L.copy(), "I": total_I}

    return total, component_curves
