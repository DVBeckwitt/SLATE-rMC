"""Scalar planar-interface modes with shared branch and amplitude primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import MaterialOptics
from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.validity import ValidityCode
from rasim_next.core.wave_modes import normal_wavevector, select_normal_wavevector


def _vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be a real phase vector")
    array = np.array(supplied.real, dtype=np.float64, copy=True)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    array.setflags(write=False)
    return array


def _vectors3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must contain real phase vectors")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (N, 3) with finite values")
    return array


def _wavelengths(value: ArrayLike, size: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"wavelength_A must contain {size} finite positive values")
    return array


def _material_indices(
    material: MaterialOptics,
    wavelength_A: NDArray[np.float64],
) -> NDArray[np.intp]:
    if not isinstance(material, MaterialOptics):
        raise TypeError("material must be MaterialOptics")
    unique, first_indices, counts = np.unique(
        material.wavelength_A,
        return_index=True,
        return_counts=True,
    )
    positions = np.searchsorted(unique, wavelength_A)
    bounded = positions < unique.size
    matched = np.zeros(wavelength_A.size, dtype=np.bool_)
    matched[bounded] = unique[positions[bounded]] == wavelength_A[bounded]
    unique_match = np.zeros(wavelength_A.size, dtype=np.bool_)
    unique_match[bounded] = counts[positions[bounded]] == 1
    if not np.all(matched & unique_match):
        raise ValueError("material must contain the exact wavelength exactly once")
    return np.asarray(first_indices[positions], dtype=np.intp)


def _normal_wavevectors(
    k0_Ainv: NDArray[np.float64],
    refractive_index: NDArray[np.complex128],
    k_parallel_Ainv: NDArray[np.float64],
    propagation_direction: NDArray[np.int8],
) -> NDArray[np.complex128]:
    result = np.empty(k0_Ainv.size, dtype=np.complex128)
    for direction in (-1, 1):
        selected = propagation_direction == direction
        if np.any(selected):
            result[selected] = normal_wavevector(
                k0_Ainv=k0_Ainv[selected],
                refractive_index=refractive_index[selected],
                k_parallel_Ainv=k_parallel_Ainv[selected],
                propagation_direction=direction,
            )
    squared_parallel = np.sum(k_parallel_Ainv * k_parallel_Ainv, axis=1)
    squared_bulk = (refractive_index * k0_Ainv) ** 2
    cancellation_scale = np.maximum(np.abs(squared_bulk), squared_parallel)
    cancellation = np.abs(squared_bulk - squared_parallel) <= 1e-3 * cancellation_scale
    for row in np.flatnonzero(cancellation):
        result[row] = normal_wavevector(
            k0_Ainv=float(k0_Ainv[row]),
            refractive_index=complex(refractive_index[row]),
            k_parallel_Ainv=k_parallel_Ainv[row],
            propagation_direction=int(propagation_direction[row]),
        )
    return result


@dataclass(frozen=True, slots=True)
class _IncidentModeArrays:
    k_air_sample_Ainv: NDArray[np.float64]
    k_parallel_sample_Ainv: NDArray[np.float64]
    k_film_phase_sample_Ainv: NDArray[np.float64]
    kz_air_Ainv: NDArray[np.complex128]
    kz_film_Ainv: NDArray[np.complex128]
    entrance_amplitude: NDArray[np.complex128]
    propagation_direction: NDArray[np.int8]
    status: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class _ExitModeArrays:
    k_film_phase_sample_Ainv: NDArray[np.float64]
    k_parallel_sample_Ainv: NDArray[np.float64]
    k_air_phase_sample_Ainv: NDArray[np.float64]
    kz_film_Ainv: NDArray[np.complex128]
    kz_air_Ainv: NDArray[np.complex128]
    exit_amplitude: NDArray[np.complex128]
    propagation_direction: NDArray[np.int8]
    status: NDArray[np.str_]


def _solve_incident_mode_arrays(
    direction_sample: ArrayLike,
    wavelength_A: ArrayLike,
    material: MaterialOptics,
) -> _IncidentModeArrays:
    direction = _vectors3(direction_sample, "direction_sample")
    if not np.allclose(
        np.linalg.norm(direction, axis=1),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("direction_sample must contain unit vectors")
    wavelengths = _wavelengths(wavelength_A, direction.shape[0])
    indices = _material_indices(material, wavelengths)
    k0_Ainv = 2.0 * np.pi / wavelengths
    k_air = k0_Ainv[:, None] * direction
    k_parallel = k_air.copy()
    k_parallel[:, 2] = 0.0
    kz_air = k_air[:, 2].astype(np.complex128)
    propagation_direction = np.where(kz_air.real >= 0.0, 1, -1).astype(np.int8)
    kz_film = _normal_wavevectors(
        k0_Ainv,
        material.n_complex[indices],
        k_parallel,
        propagation_direction,
    )
    k_film_phase = k_parallel.copy()
    k_film_phase[:, 2] = kz_film.real
    defined = kz_air + kz_film != 0.0
    amplitude = np.zeros(direction.shape[0], dtype=np.complex128)
    if np.any(defined):
        amplitude[defined] = scalar_interface_amplitude(kz_air[defined], kz_film[defined])
    status = np.full(direction.shape[0], ValidityCode.VALID, dtype="U16")
    status[~defined] = ValidityCode.NUMERIC_FAILURE
    return _IncidentModeArrays(
        k_air,
        k_parallel,
        k_film_phase,
        kz_air,
        kz_film,
        amplitude,
        propagation_direction,
        status,
    )


def _solve_exit_mode_arrays(
    k_film_phase_sample_Ainv: ArrayLike,
    wavelength_A: ArrayLike,
    material: MaterialOptics,
) -> _ExitModeArrays:
    k_film_phase = _vectors3(
        k_film_phase_sample_Ainv,
        "k_film_phase_sample_Ainv",
    )
    wavelengths = _wavelengths(wavelength_A, k_film_phase.shape[0])
    indices = _material_indices(material, wavelengths)
    k0_Ainv = 2.0 * np.pi / wavelengths
    k_parallel = k_film_phase.copy()
    k_parallel[:, 2] = 0.0
    propagation_direction = np.where(k_film_phase[:, 2] >= 0.0, 1, -1).astype(np.int8)
    kz_film = _normal_wavevectors(
        k0_Ainv,
        material.n_complex[indices],
        k_parallel,
        propagation_direction,
    )
    squared_parallel = np.sum(k_parallel**2, axis=1)
    squared_air_normal = k0_Ainv**2 - squared_parallel
    roundoff_tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * np.maximum.reduce((k0_Ainv**2, squared_parallel, np.ones(k0_Ainv.size)))
    )
    nonpropagating = squared_air_normal < -roundoff_tolerance
    kz_air = _normal_wavevectors(
        k0_Ainv,
        np.ones(k0_Ainv.size, dtype=np.complex128),
        k_parallel,
        propagation_direction,
    )
    rounded_to_critical = (squared_air_normal < 0.0) & ~nonpropagating
    for direction in (-1, 1):
        selected = rounded_to_critical & (propagation_direction == direction)
        if np.any(selected):
            kz_air[selected] = select_normal_wavevector(0.0, direction)
    defined = kz_film + kz_air != 0.0
    amplitude = np.zeros(k0_Ainv.size, dtype=np.complex128)
    amplitude_rows = ~nonpropagating & defined
    if np.any(amplitude_rows):
        amplitude[amplitude_rows] = scalar_interface_amplitude(
            kz_film[amplitude_rows],
            kz_air[amplitude_rows],
        )
    k_air_phase = k_parallel.copy()
    k_air_phase[:, 2] = kz_air.real
    k_air_phase[nonpropagating] = 0.0
    status = np.full(k0_Ainv.size, ValidityCode.VALID, dtype="U16")
    status[nonpropagating] = ValidityCode.NON_PROPAGATING
    status[~nonpropagating & ~defined] = ValidityCode.NUMERIC_FAILURE
    return _ExitModeArrays(
        k_film_phase,
        k_parallel,
        k_air_phase,
        kz_film,
        kz_air,
        amplitude,
        propagation_direction,
        status,
    )


@dataclass(frozen=True, slots=True)
class IncidentMode:
    """Air-to-film mode; the film phase vector excludes its decay component."""

    k_air_sample_Ainv: NDArray[np.float64]
    k_parallel_sample_Ainv: NDArray[np.float64]
    k_film_phase_sample_Ainv: NDArray[np.float64]
    kz_air_Ainv: complex
    kz_film_Ainv: complex
    entrance_amplitude: complex
    propagation_direction: int
    status: ValidityCode


@dataclass(frozen=True, slots=True)
class ExitMode:
    """Film-to-air mode; a non-propagating ambient result has no air phase vector."""

    k_film_phase_sample_Ainv: NDArray[np.float64]
    k_parallel_sample_Ainv: NDArray[np.float64]
    k_air_phase_sample_Ainv: NDArray[np.float64]
    kz_film_Ainv: complex
    kz_air_Ainv: complex
    exit_amplitude: complex
    propagation_direction: int
    status: ValidityCode


def solve_incident_mode(
    direction_sample: ArrayLike,
    wavelength_A: float,
    material: MaterialOptics,
) -> IncidentMode:
    """Solve one incident mode for sample-local surface normal ``+z``."""

    direction = _vector3(direction_sample, "direction_sample")
    modes = _solve_incident_mode_arrays(
        direction[None, :],
        np.array([wavelength_A], dtype=np.float64),
        material,
    )
    return IncidentMode(
        _vector3(modes.k_air_sample_Ainv[0], "k_air_sample_Ainv"),
        _vector3(modes.k_parallel_sample_Ainv[0], "k_parallel_sample_Ainv"),
        _vector3(modes.k_film_phase_sample_Ainv[0], "k_film_phase_sample_Ainv"),
        complex(modes.kz_air_Ainv[0]),
        complex(modes.kz_film_Ainv[0]),
        complex(modes.entrance_amplitude[0]),
        int(modes.propagation_direction[0]),
        ValidityCode(modes.status[0]),
    )


def solve_exit_mode(
    k_film_phase_sample_Ainv: ArrayLike,
    wavelength_A: float,
    material: MaterialOptics,
) -> ExitMode:
    """Solve one film-to-air mode, rejecting an evanescent ambient channel."""

    k_film_phase = _vector3(k_film_phase_sample_Ainv, "k_film_phase_sample_Ainv")
    modes = _solve_exit_mode_arrays(
        k_film_phase[None, :],
        np.array([wavelength_A], dtype=np.float64),
        material,
    )
    return ExitMode(
        _vector3(modes.k_film_phase_sample_Ainv[0], "k_film_phase_sample_Ainv"),
        _vector3(modes.k_parallel_sample_Ainv[0], "k_parallel_sample_Ainv"),
        _vector3(modes.k_air_phase_sample_Ainv[0], "k_air_phase_sample_Ainv"),
        complex(modes.kz_film_Ainv[0]),
        complex(modes.kz_air_Ainv[0]),
        complex(modes.exit_amplitude[0]),
        int(modes.propagation_direction[0]),
        ValidityCode(modes.status[0]),
    )
