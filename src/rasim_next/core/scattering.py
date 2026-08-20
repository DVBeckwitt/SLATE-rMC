from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

CLASSICAL_ELECTRON_RADIUS_A = 2.8179403262e-5
UNITY_APPROXIMATION = "UNITY_APPROXIMATION"
THOMSON_UNPOLARIZED_UNANALYSED = "THOMSON_UNPOLARIZED_UNANALYSED"
POLARIZATION_MODEL_CODES = {
    UNITY_APPROXIMATION: 0,
    THOMSON_UNPOLARIZED_UNANALYSED: 1,
}


def polarization_model_code(model_id: str) -> int:
    """Return the compiled code for one declared scattering-polarization model."""

    try:
        return POLARIZATION_MODEL_CODES[model_id]
    except (KeyError, TypeError) as error:
        supported = ", ".join(POLARIZATION_MODEL_CODES)
        raise ValueError(
            f"unsupported polarization model {model_id!r}; expected {supported}"
        ) from error


def scattering_polarization_weight(
    incident_air_direction: ArrayLike,
    outgoing_air_direction: ArrayLike,
    *,
    model_id: str,
) -> NDArray[np.float64]:
    """Return the once-only event intensity factor for external-air directions."""

    incident, outgoing = np.broadcast_arrays(
        np.asarray(incident_air_direction, dtype=np.float64),
        np.asarray(outgoing_air_direction, dtype=np.float64),
    )
    if incident.ndim < 1 or incident.shape[-1] != 3 or not np.all(np.isfinite(incident)):
        raise ValueError("incident_air_direction must contain finite three-vectors")
    if outgoing.shape[-1] != 3 or not np.all(np.isfinite(outgoing)):
        raise ValueError("outgoing_air_direction must contain finite three-vectors")
    norms = (np.linalg.norm(incident, axis=-1), np.linalg.norm(outgoing, axis=-1))
    if any(not np.allclose(value, 1.0, rtol=0.0, atol=1.0e-12) for value in norms):
        raise ValueError("scattering-polarization directions must be unit vectors")
    code = polarization_model_code(model_id)
    if code == 0:
        result = np.ones(incident.shape[:-1], dtype=np.float64)
    else:
        cosine = np.einsum("...i,...i->...", incident, outgoing, optimize=True)
        cosine = np.clip(cosine, -1.0, 1.0)
        result = 0.5 * (1.0 + cosine * cosine)
    result.setflags(write=False)
    return result


def electron_squared_to_scattering_strength_A2(
    raw_electron_squared: ArrayLike,
) -> NDArray[np.float64]:
    """Return polarization-neutral ``r_e**2 * electron**2`` in angstrom²."""

    supplied = np.asarray(raw_electron_squared)
    if supplied.dtype.kind not in "fiu":
        raise TypeError("raw_electron_squared must be a real numeric array")
    values = np.array(raw_electron_squared, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("raw_electron_squared must be finite and nonnegative")
    values *= CLASSICAL_ELECTRON_RADIUS_A**2
    values.setflags(write=False)
    return values
