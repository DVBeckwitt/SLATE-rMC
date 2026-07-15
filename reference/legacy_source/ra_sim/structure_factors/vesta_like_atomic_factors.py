"""Atomic factor helpers for VESTA-parity diagnostics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


VESTA_CU_KA1_ANOMALOUS_TERMS = {
    "Bi": (-4.23706, 8.83640),
    "I": (-0.533926829535952, 6.804613238987644),
    "Pb": (-4.307716547050297, 8.417749608175468),
    "Se": (-0.787865, 1.13462),
}


def _normalize_elements(elements: str | Iterable[str]) -> np.ndarray:
    return np.asarray([elements] if isinstance(elements, str) else list(elements), dtype=str)


def f0(elements: str | Iterable[str], s_values, *, table: str = "waaskirf") -> np.ndarray:
    import Dans_Diffraction.functions_crystallography as fc

    element_array = _normalize_elements(elements)
    qmag = 4.0 * np.pi * np.asarray(s_values, dtype=float).reshape(-1)
    if table == "waaskirf":
        return fc.xray_scattering_factor_WaasKirf(element_array, qmag)
    if table == "itc":
        return fc.xray_scattering_factor(element_array, qmag)
    raise ValueError(f"Unsupported scattering table: {table}")


def anomalous_terms(
    elements: str | Iterable[str],
    wavelength_angstrom: float,
    *,
    mode: str = "vesta_cu_ka1",
) -> tuple[np.ndarray, np.ndarray]:
    element_array = _normalize_elements(elements)
    if mode == "off":
        return np.zeros(len(element_array)), np.zeros(len(element_array))
    if mode == "vesta_cu_ka1":
        missing = sorted({el for el in element_array if el not in VESTA_CU_KA1_ANOMALOUS_TERMS})
        if missing:
            raise ValueError(f"No VESTA Cu Kalpha anomalous terms for: {missing}")
        f_prime = np.array([VESTA_CU_KA1_ANOMALOUS_TERMS[el][0] for el in element_array])
        f_double_prime = np.array([VESTA_CU_KA1_ANOMALOUS_TERMS[el][1] for el in element_array])
        return f_prime, f_double_prime
    if mode == "xraydb":
        import Dans_Diffraction.functions_crystallography as fc

        energy_kev = 12.398419843320026 / float(wavelength_angstrom)
        f_prime, package_f2 = fc.xray_dispersion_corrections(
            element_array,
            np.array([energy_kev]),
        )
        return np.asarray(f_prime[0], dtype=float), -np.asarray(package_f2[0], dtype=float)
    raise ValueError(f"Unsupported anomalous mode: {mode}")
