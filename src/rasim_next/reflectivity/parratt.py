"""General scalar Parratt recursion using shared complex-wave primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.interfaces import scalar_interface_amplitude
from rasim_next.core.wave_modes import select_normal_wavevector


@dataclass(frozen=True, slots=True)
class ParrattResult:
    """Pure dimensionless reflectivity and its dimensional complex stages."""

    qz_Ainv: NDArray[np.float64]
    kz_Ainv: NDArray[np.complex128]
    interface_amplitude: NDArray[np.complex128]
    amplitude: NDArray[np.complex128]
    reflectivity: NDArray[np.float64]
    normalization: str = "dimensionless pure Parratt reflectivity"

    def __post_init__(self) -> None:
        qz = np.array(self.qz_Ainv, dtype=np.float64, copy=True, order="C")
        kz = np.array(self.kz_Ainv, dtype=np.complex128, copy=True, order="C")
        interface = np.array(self.interface_amplitude, dtype=np.complex128, copy=True, order="C")
        amplitude = np.array(self.amplitude, dtype=np.complex128, copy=True, order="C")
        reflectivity = np.array(self.reflectivity, dtype=np.float64, copy=True, order="C")
        if (
            kz.ndim != qz.ndim + 1
            or kz.shape[:-1] != qz.shape
            or kz.shape[-1] < 2
            or interface.shape != (*qz.shape, kz.shape[-1] - 1)
            or amplitude.shape != qz.shape
            or reflectivity.shape != qz.shape
        ):
            raise ValueError("Parratt stages must share one batch and ordered layer axes")
        if not self.normalization or not all(
            np.all(np.isfinite(array)) for array in (qz, kz, interface, amplitude, reflectivity)
        ):
            raise ValueError("Parratt stages and normalization must be finite")
        for array in (qz, kz, interface, amplitude, reflectivity):
            array.setflags(write=False)
        object.__setattr__(self, "qz_Ainv", qz)
        object.__setattr__(self, "kz_Ainv", kz)
        object.__setattr__(self, "interface_amplitude", interface)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "reflectivity", reflectivity)


def _thicknesses(thickness_A: Sequence[float | None], layer_count: int) -> tuple[float | None, ...]:
    try:
        values = tuple(thickness_A)
    except TypeError as error:
        raise ValueError("thickness_A must contain one entry per ordered medium") from error
    if len(values) != layer_count or values[0] is not None or values[-1] is not None:
        raise ValueError("first and last Parratt media must have None thickness")
    result: list[float | None] = [None]
    for value in values[1:-1]:
        if value is None or not np.isfinite(value) or value < 0.0:
            raise ValueError("interior layer thickness must be finite and nonnegative")
        result.append(float(value))
    result.append(None)
    return tuple(result)


def parratt_reflectivity(
    qz_Ainv: ArrayLike,
    wavelength_A: ArrayLike,
    *,
    refractive_index: ArrayLike,
    thickness_A: Sequence[float | None],
    roughness_A: ArrayLike,
) -> ParrattResult:
    """Evaluate a lossless-ambient, passive multilayer bottom-up."""

    qz_input = np.asarray(qz_Ainv, dtype=np.float64)
    wavelength_input = np.asarray(wavelength_A, dtype=np.float64)
    try:
        batch_shape = np.broadcast_shapes(qz_input.shape, wavelength_input.shape)
        qz = np.broadcast_to(qz_input, batch_shape)
        wavelength = np.broadcast_to(wavelength_input, batch_shape)
    except ValueError as error:
        raise ValueError("qz_Ainv and wavelength_A must broadcast to one batch") from error
    if not np.all(np.isfinite(qz)) or np.any(qz < 0.0):
        raise ValueError("qz_Ainv must be finite and nonnegative")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_A must be finite and positive")

    supplied_index = np.asarray(refractive_index, dtype=np.complex128)
    if supplied_index.ndim < 1 or supplied_index.shape[-1] < 2:
        raise ValueError("refractive_index must end with at least two ordered media")
    layer_count = supplied_index.shape[-1]
    thickness = _thicknesses(thickness_A, layer_count)
    try:
        media = np.broadcast_to(supplied_index, (*batch_shape, layer_count))
        roughness = np.broadcast_to(
            np.asarray(roughness_A, dtype=np.float64), (*batch_shape, layer_count - 1)
        )
    except ValueError as error:
        raise ValueError("media and roughness must broadcast to the Parratt batch") from error
    if not np.all(np.isfinite(media)) or np.any(media.real <= 0.0):
        raise ValueError("refractive indices must be finite with positive real part")
    if np.any(media.imag < 0.0):
        raise ValueError("active-gain refractive indices are not supported")
    ambient = media[..., 0]
    if np.any(ambient.imag != 0.0):
        raise ValueError("real qz_Ainv requires a lossless incident ambient")
    if not np.all(np.isfinite(roughness)) or np.any(roughness < 0.0):
        raise ValueError("interface roughness must be finite and nonnegative")

    k0 = 2.0 * np.pi / wavelength
    if np.any(qz > 2.0 * ambient.real * k0):
        raise ValueError("qz_Ainv exceeds the incident-ambient specular range")
    ambient_normal = 0.5 * qz
    squared_normal = (media * media - ambient[..., None] ** 2) * k0[
        ..., None
    ] ** 2 + ambient_normal[..., None] ** 2
    kz = np.asarray(
        select_normal_wavevector(squared_normal, propagation_direction=1),
        dtype=np.complex128,
    )
    upper_kz = kz[..., :-1]
    lower_kz = kz[..., 1:]
    equal_zero_limit = (upper_kz == 0.0) & (lower_kz == 0.0)
    interface = (
        np.asarray(
            scalar_interface_amplitude(
                np.where(equal_zero_limit, 1.0 + 0.0j, upper_kz),
                np.where(equal_zero_limit, 1.0 + 0.0j, lower_kz),
            ),
            dtype=np.complex128,
        )
        - 1.0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        interface *= np.exp(-2.0 * upper_kz * lower_kz * roughness**2)
    if not np.all(np.isfinite(interface)):
        raise FloatingPointError("nonfinite roughened Parratt interface amplitude")

    recursion = np.array(interface[..., -1], dtype=np.complex128, copy=True)
    for interface_index in range(layer_count - 3, -1, -1):
        lower_layer = interface_index + 1
        phase = np.exp(2.0j * kz[..., lower_layer] * thickness[lower_layer])
        numerator = interface[..., interface_index] + recursion * phase
        denominator = 1.0 + interface[..., interface_index] * recursion * phase
        if np.any(denominator == 0.0):
            raise ZeroDivisionError("Parratt recursion denominator is zero")
        recursion = numerator / denominator
    reflectivity = np.abs(recursion) ** 2
    if not np.all(np.isfinite(recursion)) or not np.all(np.isfinite(reflectivity)):
        raise FloatingPointError("nonfinite Parratt recursion result")
    return ParrattResult(
        qz_Ainv=qz,
        kz_Ainv=kz,
        interface_amplitude=interface,
        amplitude=recursion,
        reflectivity=reflectivity,
    )
