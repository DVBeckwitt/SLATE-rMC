"""Compiled point evaluator for the accepted finite parent-2H Bi2Se3 fixture."""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np
import xraydb
from numpy.typing import NDArray

from rasim_next.core.contracts import EventIntensityNormalization
from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
from rasim_next.materials.optics import HC_EV_A, _f0_species
from rasim_next.ordered.motifs import _bi2se3_quintuple_layers
from rasim_next.pipeline.bragg_space import Bi2Se3TwoHStrength

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class CompiledPixelIntegral(NamedTuple):
    """Private fused pixel result retained at pixel rather than node scale."""

    per_rod_mass_A2: FloatArray
    center_per_rod_density_A2_per_px2: FloatArray
    per_rod_inverse_count_min: IntArray
    per_rod_inverse_count_max: IntArray
    per_rod_caustic: BoolArray
    valid_any: BoolArray
    valid_all: BoolArray
    center_valid: BoolArray


@dataclass(frozen=True, slots=True)
class CompiledDetectorState:
    """Immutable numeric state consumed by the no-GIL point kernel."""

    detector_zero_lab_m: FloatArray
    detector_column_step_lab_m: FloatArray
    detector_row_step_lab_m: FloatArray
    detector_pixel_area_vector_lab_m2: FloatArray
    ray_origin_lab_m: FloatArray
    sample_from_lab: FloatArray
    ki_film_sample_Ainv: FloatArray
    internal_k_Ainv: float
    air_k0_Ainv: float
    refractive_index: complex
    entrance_amplitude: complex
    incident_decay_Ainv: float
    film_thickness_A: float
    source_phase_weight: float
    sample_from_local: FloatArray
    rod_hk_population: FloatArray
    rod_parallel_local_Ainv: FloatArray
    rod_u_bounds_Ainv: FloatArray
    rod_inverse_constants: FloatArray
    b3_norm_Ainv: float
    gaussian_sigma_rad: float
    lorentzian_hwhm_rad: float
    lorentzian_probability: float
    atom_fractional_offset: FloatArray
    atom_occupancy_u_iso_element: FloatArray
    rod_atom_inplane_factor: NDArray[np.complex128]
    common_u_iso_A2: float
    f0_parameters: FloatArray
    anomalous_factor_e: NDArray[np.complex128]
    layers: int
    shared_disorder_epsilon: float
    normalization_divisor: float

    def __post_init__(self) -> None:
        rod_count = np.asarray(self.rod_hk_population).shape[0]
        atom_count = np.asarray(self.atom_fractional_offset).shape[0]
        shapes = {
            "detector_zero_lab_m": (3,),
            "detector_column_step_lab_m": (3,),
            "detector_row_step_lab_m": (3,),
            "detector_pixel_area_vector_lab_m2": (3,),
            "ray_origin_lab_m": (3,),
            "sample_from_lab": (3, 3),
            "ki_film_sample_Ainv": (3,),
            "sample_from_local": (3, 3),
            "rod_hk_population": (rod_count, 3),
            "rod_parallel_local_Ainv": (rod_count, 3),
            "rod_u_bounds_Ainv": (rod_count, 2),
            "rod_inverse_constants": (rod_count, 4),
            "atom_fractional_offset": (atom_count, 3),
            "atom_occupancy_u_iso_element": (atom_count, 3),
            "f0_parameters": (2, 11),
        }
        if rod_count == 0 or atom_count == 0:
            raise ValueError("compiled detector state requires rods and motif atoms")
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        anomalous = np.array(self.anomalous_factor_e, dtype=np.complex128, copy=True, order="C")
        if anomalous.shape != (2,) or not np.all(np.isfinite(anomalous)):
            raise ValueError("anomalous_factor_e must be finite with shape (2,)")
        anomalous.setflags(write=False)
        object.__setattr__(self, "anomalous_factor_e", anomalous)
        inplane_factor = np.array(
            self.rod_atom_inplane_factor,
            dtype=np.complex128,
            copy=True,
            order="C",
        )
        if inplane_factor.shape != (rod_count, atom_count) or not np.all(
            np.isfinite(inplane_factor)
        ):
            raise ValueError(
                "rod_atom_inplane_factor must be finite with shape (rod_count, atom_count)"
            )
        inplane_factor.setflags(write=False)
        object.__setattr__(self, "rod_atom_inplane_factor", inplane_factor)
        scalar_names = (
            "internal_k_Ainv",
            "air_k0_Ainv",
            "incident_decay_Ainv",
            "film_thickness_A",
            "source_phase_weight",
            "b3_norm_Ainv",
            "gaussian_sigma_rad",
            "lorentzian_hwhm_rad",
            "lorentzian_probability",
            "common_u_iso_A2",
            "shared_disorder_epsilon",
            "normalization_divisor",
        )
        for name in scalar_names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.internal_k_Ainv == 0.0 or self.air_k0_Ainv == 0.0 or self.b3_norm_Ainv == 0.0:
            raise ValueError("compiled wavevector and reciprocal scales must be positive")
        if self.normalization_divisor == 0.0:
            raise ValueError("normalization_divisor must be positive")
        if not 0.0 <= self.lorentzian_probability <= 1.0:
            raise ValueError("lorentzian_probability must lie in [0, 1]")
        if not 0.0 <= self.shared_disorder_epsilon <= 1.0:
            raise ValueError("shared_disorder_epsilon must lie in [0, 1]")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        for name in ("refractive_index", "entrance_amplitude"):
            value = complex(getattr(self, name))
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


def pack_bi2se3_two_h_structure(
    strength: Bi2Se3TwoHStrength,
    *,
    wavelength_A: float,
) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.complex128], int, float]:
    """Pack the existing CIF/XrayDB authorities once for exact compiled evaluation."""

    if not isinstance(strength, Bi2Se3TwoHStrength):
        raise TypeError("compiled detector integration requires Bi2Se3TwoHStrength")
    atoms = _bi2se3_quintuple_layers(strength.crystal)[0]
    if any(atom.u_iso_A2 is None for atom in atoms):
        raise ValueError("compiled Bi2Se3 integration requires declared isotropic displacement")
    elements = ("Bi", "Se")
    element_index = {element: position for position, element in enumerate(elements)}
    offsets = np.asarray([atom.fractional_offset for atom in atoms], dtype=np.float64)
    properties = np.asarray(
        [(atom.occupancy, float(atom.u_iso_A2), element_index[atom.element]) for atom in atoms],
        dtype=np.float64,
    )

    waasmaier = xraydb.get_xraydb().get_cache("Waasmaier")
    parameters = np.empty((len(elements), 11), dtype=np.float64)
    anomalous = np.empty(len(elements), dtype=np.complex128)
    energy_eV = HC_EV_A / wavelength_A
    for position, element in enumerate(elements):
        charges = {atom.charge for atom in atoms if atom.element == element}
        if len(charges) != 1:
            raise ValueError(f"compiled Bi2Se3 integration requires one charge for {element}")
        species = _f0_species(element, charges.pop())
        row = next((candidate for candidate in waasmaier if candidate.ion == species), None)
        if row is None:
            raise ValueError(f"XrayDB has no Waasmaier coefficients for {species}")
        parameters[position, 0] = float(row.offset)
        parameters[position, 1:6] = json.loads(row.scale)
        parameters[position, 6:11] = json.loads(row.exponents)
        anomalous[position] = complex(
            xraydb.f1_chantler(element, energy_eV),
            xraydb.f2_chantler(element, energy_eV),
        )

    divisor = (
        float(strength.layers)
        if strength.normalization is EventIntensityNormalization.FINITE_PER_LAYER
        else 1.0
    )
    for value in (offsets, properties, parameters, anomalous):
        value.setflags(write=False)
    return offsets, properties, parameters, anomalous, strength.layers, divisor


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _positive_normal_root(radicand: complex) -> complex:
    root = cmath.sqrt(radicand)
    if root.imag != 0.0:
        if root.imag < 0.0:
            root = -root
    elif root.real < 0.0:
        root = -root
    return root


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _wrapped_mosaic_density(
    alpha: float,
    gaussian_sigma: float,
    gaussian_probability: float,
    gaussian_normalization: float,
    lorentzian_probability: float,
    lorentzian_rho: float,
    lorentzian_one_minus_rho: float,
    lorentzian_numerator: float,
) -> float:
    two_pi = 2.0 * math.pi
    wrapped = (alpha + math.pi) % two_pi - math.pi
    density = 0.0
    if gaussian_probability > 0.0 and gaussian_sigma > 0.0:
        if gaussian_sigma >= 1.0:
            gaussian = 1.0
            harmonic = 1
            while harmonic <= 100_000:
                amplitude = 2.0 * math.exp(-0.5 * (harmonic * gaussian_sigma) ** 2)
                gaussian += amplitude * math.cos(harmonic * wrapped)
                next_amplitude = 2.0 * math.exp(-0.5 * ((harmonic + 1) * gaussian_sigma) ** 2)
                if next_amplitude <= 1.0e-15 * gaussian:
                    break
                harmonic += 1
            gaussian /= two_pi
        else:
            scaled = math.exp(-0.5 * (wrapped / gaussian_sigma) ** 2)
            # For sigma < 1 rad and wrapped in [-pi, pi], every |n| >= 2
            # image is below 1e-15 of the retained n=0,+/-1 sum.  Keeping the
            # first pair also handles the exactly wrapped +/-pi seam.
            scaled += math.exp(-0.5 * ((wrapped + two_pi) / gaussian_sigma) ** 2)
            scaled += math.exp(-0.5 * ((wrapped - two_pi) / gaussian_sigma) ** 2)
            gaussian = scaled / gaussian_normalization
        density += gaussian_probability * gaussian
    if lorentzian_probability > 0.0 and lorentzian_numerator > 0.0:
        denominator = two_pi * (
            lorentzian_one_minus_rho * lorentzian_one_minus_rho
            + 4.0 * lorentzian_rho * math.sin(0.5 * wrapped) ** 2
        )
        density += lorentzian_probability * lorentzian_numerator / denominator
    return density / math.pi


@numba.njit(nogil=True, fastmath=False, cache=False, inline="always")
def _two_h_strength_A2(
    rod_index: int,
    ell: float,
    common_damping: float,
    element_factor_0: complex,
    element_factor_1: complex,
    rod_atom_inplane_factor: NDArray[np.complex128],
    atom_fractional_offset: FloatArray,
    atom_occupancy_u_iso_element: FloatArray,
    layers: int,
    shared_disorder_epsilon: float,
    rod_hk_population: FloatArray,
    normalization_divisor: float,
) -> float:
    amplitude_plus = 0.0 + 0.0j
    amplitude_minus = 0.0 + 0.0j
    for atom in range(atom_fractional_offset.shape[0]):
        occupancy = atom_occupancy_u_iso_element[atom, 0]
        element = int(atom_occupancy_u_iso_element[atom, 2])
        phase_z = 2.0 * math.pi * ell * atom_fractional_offset[atom, 2]
        inplane_factor = rod_atom_inplane_factor[rod_index, atom]
        phase_plus = inplane_factor * complex(math.cos(phase_z), math.sin(phase_z))
        phase_minus = inplane_factor * complex(math.cos(phase_z), -math.sin(phase_z))
        element_factor = element_factor_0 if element == 0 else element_factor_1
        amplitude_plus += occupancy * element_factor * phase_plus
        amplitude_minus += occupancy * element_factor * phase_minus

    vertical_phase_angle = 2.0 * math.pi * ell / 3.0
    vertical_phase = complex(math.cos(vertical_phase_angle), math.sin(vertical_phase_angle))
    if shared_disorder_epsilon == 0.0:
        phase_power = 1.0 + 0.0j
        stack_sum = 1.0 + 0.0j
        for _ in range(1, layers):
            phase_power *= vertical_phase
            stack_sum += phase_power
        total = amplitude_plus * stack_sum
        intensity_e2 = total.real * total.real + total.imag * total.imag
    else:
        h = int(rod_hk_population[rod_index, 0])
        k = int(rod_hk_population[rod_index, 1])
        registry_index = (h + 2 * k) % 3
        if registry_index == 0:
            omega = 1.0 + 0.0j
        elif registry_index == 1:
            omega = complex(-0.5, 0.5 * math.sqrt(3.0))
        else:
            omega = complex(-0.5, -0.5 * math.sqrt(3.0))
        inverse_omega = omega.conjugate()
        alternative = 0.25 * shared_disorder_epsilon
        parent = 1.0 - shared_disorder_epsilon
        same_probability = parent + 2.0 * alternative
        flip_probability = 2.0 * alternative
        same_gauge = parent + alternative * inverse_omega + alternative * omega
        plus_to_minus_gauge = alternative * inverse_omega + alternative * omega
        minus_to_plus_gauge = alternative * omega + alternative * inverse_omega

        probability_plus = 1.0
        probability_minus = 0.0
        first_moment_plus = amplitude_plus
        first_moment_minus = 0.0 + 0.0j
        second_moment_plus = (
            amplitude_plus.real * amplitude_plus.real + amplitude_plus.imag * amplitude_plus.imag
        )
        second_moment_minus = 0.0
        phase_power = 1.0 + 0.0j
        for _ in range(1, layers):
            phase_power *= vertical_phase
            contribution_plus = phase_power * amplitude_plus
            contribution_minus = phase_power * amplitude_minus
            next_probability_plus = (
                probability_plus * same_probability + probability_minus * flip_probability
            )
            next_probability_minus = (
                probability_minus * same_probability + probability_plus * flip_probability
            )
            propagated_plus = (
                same_gauge * first_moment_plus + minus_to_plus_gauge * first_moment_minus
            )
            propagated_minus = (
                same_gauge * first_moment_minus + plus_to_minus_gauge * first_moment_plus
            )
            next_first_moment_plus = propagated_plus + next_probability_plus * contribution_plus
            next_first_moment_minus = propagated_minus + next_probability_minus * contribution_minus
            contribution_plus_squared = (
                contribution_plus.real * contribution_plus.real
                + contribution_plus.imag * contribution_plus.imag
            )
            contribution_minus_squared = (
                contribution_minus.real * contribution_minus.real
                + contribution_minus.imag * contribution_minus.imag
            )
            cross_plus = contribution_plus * propagated_plus.conjugate()
            cross_minus = contribution_minus * propagated_minus.conjugate()
            next_second_moment_plus = (
                same_probability * second_moment_plus
                + flip_probability * second_moment_minus
                + next_probability_plus * contribution_plus_squared
                + 2.0 * cross_plus.real
            )
            next_second_moment_minus = (
                same_probability * second_moment_minus
                + flip_probability * second_moment_plus
                + next_probability_minus * contribution_minus_squared
                + 2.0 * cross_minus.real
            )
            probability_plus = next_probability_plus
            probability_minus = next_probability_minus
            first_moment_plus = next_first_moment_plus
            first_moment_minus = next_first_moment_minus
            second_moment_plus = next_second_moment_plus
            second_moment_minus = next_second_moment_minus
        intensity_e2 = max(second_moment_plus + second_moment_minus, 0.0)
    return (
        CLASSICAL_ELECTRON_RADIUS_A**2
        * common_damping
        * common_damping
        * intensity_e2
        / normalization_divisor
    )


@numba.njit(nogil=True, fastmath=False, cache=False)
def _evaluate_point_into(
    column: float,
    row: float,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    internal_k_squared_Ainv2: float,
    air_k0_Ainv: float,
    refractive_air_k_squared_Ainv2: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    gaussian_probability: float,
    gaussian_normalization: float,
    lorentzian_probability: float,
    lorentzian_rho: float,
    lorentzian_one_minus_rho: float,
    lorentzian_numerator: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_u_iso_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    common_u_iso_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
    entrance_power: float,
    reconstruction_tolerance: float,
    density: FloatArray,
    inverse_count: IntArray,
    caustic: BoolArray,
) -> bool:
    rod_count = rod_hk_population.shape[0]
    for rod_index in range(rod_count):
        density[rod_index] = 0.0
        inverse_count[rod_index] = 0
        caustic[rod_index] = False
    rows, columns = detector_shape_rc
    if column < -0.5 or column > columns - 0.5 or row < -0.5 or row > rows - 0.5:
        return False

    two_pi = 2.0 * math.pi
    angular_tolerance = 2048.0 * np.finfo(np.float64).eps
    displacement_x = (
        detector_zero_lab_m[0]
        + column * detector_column_step_lab_m[0]
        + row * detector_row_step_lab_m[0]
        - ray_origin_lab_m[0]
    )
    displacement_y = (
        detector_zero_lab_m[1]
        + column * detector_column_step_lab_m[1]
        + row * detector_row_step_lab_m[1]
        - ray_origin_lab_m[1]
    )
    displacement_z = (
        detector_zero_lab_m[2]
        + column * detector_column_step_lab_m[2]
        + row * detector_row_step_lab_m[2]
        - ray_origin_lab_m[2]
    )
    distance = math.sqrt(
        displacement_x * displacement_x
        + displacement_y * displacement_y
        + displacement_z * displacement_z
    )
    if distance == 0.0:
        return False
    direction_x = displacement_x / distance
    direction_y = displacement_y / distance
    direction_z = displacement_z / distance
    scaled_direction_x = air_k0_Ainv * direction_x
    scaled_direction_y = air_k0_Ainv * direction_y
    scaled_direction_z = air_k0_Ainv * direction_z
    kf_air_x = (
        sample_from_lab[0, 0] * scaled_direction_x
        + sample_from_lab[0, 1] * scaled_direction_y
        + sample_from_lab[0, 2] * scaled_direction_z
    )
    kf_air_y = (
        sample_from_lab[1, 0] * scaled_direction_x
        + sample_from_lab[1, 1] * scaled_direction_y
        + sample_from_lab[1, 2] * scaled_direction_z
    )
    kf_air_z = (
        sample_from_lab[2, 0] * scaled_direction_x
        + sample_from_lab[2, 1] * scaled_direction_y
        + sample_from_lab[2, 2] * scaled_direction_z
    )
    if kf_air_z <= 0.0:
        return False
    parallel_squared = kf_air_x * kf_air_x + kf_air_y * kf_air_y
    normal_squared = internal_k_squared_Ainv2 - parallel_squared
    if normal_squared <= 0.0:
        return False
    kf_film_x = kf_air_x
    kf_film_y = kf_air_y
    kf_film_z = math.sqrt(normal_squared)
    q_sample_x = kf_film_x - ki_film_sample_Ainv[0]
    q_sample_y = kf_film_y - ki_film_sample_Ainv[1]
    q_sample_z = kf_film_z - ki_film_sample_Ainv[2]

    pixel_solid_angle = abs(
        direction_x * detector_pixel_area_vector_lab_m2[0]
        + direction_y * detector_pixel_area_vector_lab_m2[1]
        + direction_z * detector_pixel_area_vector_lab_m2[2]
    ) / (distance * distance)
    area_jacobian = internal_k_Ainv * air_k0_Ainv * kf_air_z * pixel_solid_angle / kf_film_z

    kz_film = _positive_normal_root(refractive_air_k_squared_Ainv2 - parallel_squared)
    kz_air = complex(kf_air_z, 0.0)
    denominator = kz_film + kz_air
    if denominator == 0.0:
        return False
    exit_amplitude = 2.0 * kz_film / denominator
    exponent = 2.0 * (incident_decay_Ainv + max(kz_film.imag, 0.0)) * film_thickness_A
    attenuation = 1.0 if exponent == 0.0 else -math.expm1(-exponent) / exponent
    optical_weight = (
        entrance_power * (exit_amplitude.real**2 + exit_amplitude.imag**2) * attenuation
    )

    q_local_x = (
        q_sample_x * sample_from_local[0, 0]
        + q_sample_y * sample_from_local[1, 0]
        + q_sample_z * sample_from_local[2, 0]
    )
    q_local_y = (
        q_sample_x * sample_from_local[0, 1]
        + q_sample_y * sample_from_local[1, 1]
        + q_sample_z * sample_from_local[2, 1]
    )
    q_local_z = (
        q_sample_x * sample_from_local[0, 2]
        + q_sample_y * sample_from_local[1, 2]
        + q_sample_z * sample_from_local[2, 2]
    )
    q_norm_squared = q_local_x * q_local_x + q_local_y * q_local_y + q_local_z * q_local_z
    q_norm = math.sqrt(q_norm_squared)
    q_xraydb_squared = q_norm_squared / (16.0 * math.pi * math.pi)
    f0_0 = f0_parameters[0, 0]
    f0_1 = f0_parameters[1, 0]
    for coefficient in range(5):
        f0_0 += f0_parameters[0, 1 + coefficient] * math.exp(
            -f0_parameters[0, 6 + coefficient] * q_xraydb_squared
        )
        f0_1 += f0_parameters[1, 1 + coefficient] * math.exp(
            -f0_parameters[1, 6 + coefficient] * q_xraydb_squared
        )
    element_factor_0 = f0_0 + anomalous_factor_e[0]
    element_factor_1 = f0_1 + anomalous_factor_e[1]
    common_damping = math.exp(-0.5 * q_norm_squared * common_u_iso_A2)

    transverse_norm = math.hypot(q_local_x, q_local_y)
    azimuth_q = math.atan2(q_local_y, q_local_x)
    for rod_index in range(rod_count):
        a = rod_parallel_local_Ainv[rod_index, 0]
        b = rod_parallel_local_Ainv[rod_index, 1]
        c0 = rod_parallel_local_Ainv[rod_index, 2]
        abs_b = rod_inverse_constants[rod_index, 0]
        parallel_norm = rod_inverse_constants[rod_index, 1]
        inverse_reference = rod_inverse_constants[rod_index, 2]
        u_tolerance = rod_inverse_constants[rod_index, 3]
        x_squared = (transverse_norm - abs_b) * (transverse_norm + abs_b)
        w_squared = (q_norm - parallel_norm) * (q_norm + parallel_norm)
        inverse_scale = max(q_norm_squared, inverse_reference, 1.0)
        inverse_tolerance = 1024.0 * np.finfo(np.float64).eps * inverse_scale
        if x_squared < -inverse_tolerance or w_squared < -inverse_tolerance:
            continue
        x_magnitude = math.sqrt(max(x_squared, 0.0))
        w_magnitude = math.sqrt(max(w_squared, 0.0))
        lower_u = rod_u_bounds_Ainv[rod_index, 0]
        upper_u = rod_u_bounds_Ainv[rod_index, 1]
        for x_sign in (-1.0, 1.0):
            x_value = x_sign * x_magnitude
            beta = (azimuth_q - math.atan2(b, x_value)) % two_pi
            for w_sign in (-1.0, 1.0):
                w_value = w_sign * w_magnitude
                alpha = (math.atan2(w_value, a) - math.atan2(q_local_z, x_value)) % two_pi
                if alpha >= two_pi - angular_tolerance:
                    alpha = 0.0
                if alpha > math.pi + angular_tolerance:
                    continue
                alpha = min(alpha, math.pi)
                u_value = w_value - c0
                if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                    continue
                sin_alpha = math.sin(alpha)
                direction_local_x = sin_alpha * math.cos(beta)
                direction_local_y = sin_alpha * math.sin(beta)
                direction_local_z = math.cos(alpha)
                direction_sample_x = (
                    sample_from_local[0, 0] * direction_local_x
                    + sample_from_local[0, 1] * direction_local_y
                    + sample_from_local[0, 2] * direction_local_z
                )
                direction_sample_y = (
                    sample_from_local[1, 0] * direction_local_x
                    + sample_from_local[1, 1] * direction_local_y
                    + sample_from_local[1, 2] * direction_local_z
                )
                direction_sample_z = (
                    sample_from_local[2, 0] * direction_local_x
                    + sample_from_local[2, 1] * direction_local_y
                    + sample_from_local[2, 2] * direction_local_z
                )
                root_sign = (
                    kf_film_x * direction_sample_x
                    + kf_film_y * direction_sample_y
                    + kf_film_z * direction_sample_z
                )
                if (branch == 2 and root_sign <= 0.0) or (branch == 1 and root_sign >= 0.0):
                    continue
                ell = u_value / b3_norm_Ainv
                strength = _two_h_strength_A2(
                    rod_index,
                    ell,
                    common_damping,
                    element_factor_0,
                    element_factor_1,
                    rod_atom_inplane_factor,
                    atom_fractional_offset,
                    atom_occupancy_u_iso_element,
                    layers,
                    shared_disorder_epsilon,
                    rod_hk_population,
                    normalization_divisor,
                )
                mosaic_density = _wrapped_mosaic_density(
                    alpha,
                    gaussian_sigma_rad,
                    gaussian_probability,
                    gaussian_normalization,
                    lorentzian_probability,
                    lorentzian_rho,
                    lorentzian_one_minus_rho,
                    lorentzian_numerator,
                )
                jacobian = abs(w_value * x_value)
                if jacobian == 0.0:
                    cos_beta = math.cos(beta)
                    sin_beta = math.sin(beta)
                    reconstructed_local_x = x_value * cos_beta - b * sin_beta
                    reconstructed_local_y = x_value * sin_beta + b * cos_beta
                    reconstructed_local_z = -a * math.sin(alpha) + w_value * math.cos(alpha)
                    reconstructed_sample_x = (
                        sample_from_local[0, 0] * reconstructed_local_x
                        + sample_from_local[0, 1] * reconstructed_local_y
                        + sample_from_local[0, 2] * reconstructed_local_z
                    )
                    reconstructed_sample_y = (
                        sample_from_local[1, 0] * reconstructed_local_x
                        + sample_from_local[1, 1] * reconstructed_local_y
                        + sample_from_local[1, 2] * reconstructed_local_z
                    )
                    reconstructed_sample_z = (
                        sample_from_local[2, 0] * reconstructed_local_x
                        + sample_from_local[2, 1] * reconstructed_local_y
                        + sample_from_local[2, 2] * reconstructed_local_z
                    )
                    reconstruction_error = math.sqrt(
                        (reconstructed_sample_x - q_sample_x) ** 2
                        + (reconstructed_sample_y - q_sample_y) ** 2
                        + (reconstructed_sample_z - q_sample_z) ** 2
                    )
                    if reconstruction_error > reconstruction_tolerance:
                        continue
                    caustic[rod_index] = True
                    if (
                        source_phase_weight > 0.0
                        and rod_hk_population[rod_index, 2] > 0.0
                        and area_jacobian > 0.0
                        and optical_weight > 0.0
                        and strength > 0.0
                        and mosaic_density > 0.0
                    ):
                        density[rod_index] = np.inf
                    continue
                density[rod_index] += (
                    mosaic_density
                    * rod_hk_population[rod_index, 2]
                    * strength
                    * area_jacobian
                    * optical_weight
                    * source_phase_weight
                    / jacobian
                )
                inverse_count[rod_index] += 1
    return True


@numba.njit(nogil=True, fastmath=False, cache=False)
def _evaluate_points_kernel(
    column_px: FloatArray,
    row_px: FloatArray,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    air_k0_Ainv: float,
    refractive_index: complex,
    entrance_amplitude: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    lorentzian_hwhm_rad: float,
    lorentzian_probability: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_u_iso_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    common_u_iso_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
    size = column_px.size
    rod_count = rod_hk_population.shape[0]
    density = np.zeros((size, rod_count), dtype=np.float64)
    inverse_count = np.zeros((size, rod_count), dtype=np.int64)
    caustic = np.zeros((size, rod_count), dtype=np.bool_)
    valid = np.zeros(size, dtype=np.bool_)
    ki_norm = math.sqrt(
        ki_film_sample_Ainv[0] ** 2 + ki_film_sample_Ainv[1] ** 2 + ki_film_sample_Ainv[2] ** 2
    )
    reconstruction_tolerance = 4096.0 * np.finfo(np.float64).eps * max(ki_norm, 1.0)
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    internal_k_squared_Ainv2 = internal_k_Ainv * internal_k_Ainv
    refractive_air_k_squared_Ainv2 = (refractive_index * air_k0_Ainv) ** 2
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)
    for point in range(size):
        valid[point] = _evaluate_point_into(
            column_px[point],
            row_px[point],
            detector_shape_rc,
            detector_zero_lab_m,
            detector_column_step_lab_m,
            detector_row_step_lab_m,
            detector_pixel_area_vector_lab_m2,
            ray_origin_lab_m,
            sample_from_lab,
            ki_film_sample_Ainv,
            internal_k_Ainv,
            internal_k_squared_Ainv2,
            air_k0_Ainv,
            refractive_air_k_squared_Ainv2,
            incident_decay_Ainv,
            film_thickness_A,
            source_phase_weight,
            sample_from_local,
            rod_hk_population,
            rod_parallel_local_Ainv,
            rod_u_bounds_Ainv,
            rod_inverse_constants,
            b3_norm_Ainv,
            gaussian_sigma_rad,
            gaussian_probability,
            gaussian_normalization,
            lorentzian_probability,
            lorentzian_rho,
            lorentzian_one_minus_rho,
            lorentzian_numerator,
            atom_fractional_offset,
            atom_occupancy_u_iso_element,
            rod_atom_inplane_factor,
            common_u_iso_A2,
            f0_parameters,
            anomalous_factor_e,
            layers,
            shared_disorder_epsilon,
            normalization_divisor,
            branch,
            entrance_power,
            reconstruction_tolerance,
            density[point],
            inverse_count[point],
            caustic[point],
        )
    return density, inverse_count, caustic, valid


@numba.njit(nogil=True, fastmath=False, cache=False)
def _integrate_pixel_boxes_kernel(
    flat_pixel_index: IntArray,
    offset_px: FloatArray,
    one_dimensional_weight: FloatArray,
    include_center_diagnostics: bool,
    detector_shape_rc: tuple[int, int],
    detector_zero_lab_m: FloatArray,
    detector_column_step_lab_m: FloatArray,
    detector_row_step_lab_m: FloatArray,
    detector_pixel_area_vector_lab_m2: FloatArray,
    ray_origin_lab_m: FloatArray,
    sample_from_lab: FloatArray,
    ki_film_sample_Ainv: FloatArray,
    internal_k_Ainv: float,
    air_k0_Ainv: float,
    refractive_index: complex,
    entrance_amplitude: complex,
    incident_decay_Ainv: float,
    film_thickness_A: float,
    source_phase_weight: float,
    sample_from_local: FloatArray,
    rod_hk_population: FloatArray,
    rod_parallel_local_Ainv: FloatArray,
    rod_u_bounds_Ainv: FloatArray,
    rod_inverse_constants: FloatArray,
    b3_norm_Ainv: float,
    gaussian_sigma_rad: float,
    lorentzian_hwhm_rad: float,
    lorentzian_probability: float,
    atom_fractional_offset: FloatArray,
    atom_occupancy_u_iso_element: FloatArray,
    rod_atom_inplane_factor: NDArray[np.complex128],
    common_u_iso_A2: float,
    f0_parameters: FloatArray,
    anomalous_factor_e: NDArray[np.complex128],
    layers: int,
    shared_disorder_epsilon: float,
    normalization_divisor: float,
    branch: int,
) -> tuple[
    FloatArray,
    FloatArray,
    IntArray,
    IntArray,
    BoolArray,
    BoolArray,
    BoolArray,
    BoolArray,
]:
    pixel_count = flat_pixel_index.size
    rod_count = rod_hk_population.shape[0]
    _, columns = detector_shape_rc
    mass = np.zeros((pixel_count, rod_count), dtype=np.float64)
    diagnostic_pixel_count = pixel_count if include_center_diagnostics else 0
    center_density = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.float64)
    count_min = np.full(
        (diagnostic_pixel_count, rod_count),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )
    count_max = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.int64)
    caustic_any = np.zeros((diagnostic_pixel_count, rod_count), dtype=np.bool_)
    valid_any = np.zeros(diagnostic_pixel_count, dtype=np.bool_)
    valid_all = np.ones(diagnostic_pixel_count, dtype=np.bool_)
    center_valid_pixel = np.zeros(diagnostic_pixel_count, dtype=np.bool_)
    density = np.empty(rod_count, dtype=np.float64)
    inverse_count = np.empty(rod_count, dtype=np.int64)
    caustic = np.empty(rod_count, dtype=np.bool_)
    ki_norm = math.sqrt(
        ki_film_sample_Ainv[0] ** 2 + ki_film_sample_Ainv[1] ** 2 + ki_film_sample_Ainv[2] ** 2
    )
    reconstruction_tolerance = 4096.0 * np.finfo(np.float64).eps * max(ki_norm, 1.0)
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    internal_k_squared_Ainv2 = internal_k_Ainv * internal_k_Ainv
    refractive_air_k_squared_Ainv2 = (refractive_index * air_k0_Ainv) ** 2
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)

    for pixel in range(pixel_count):
        pixel_row = flat_pixel_index[pixel] // columns
        pixel_column = flat_pixel_index[pixel] - pixel_row * columns
        for row_node in range(offset_px.size):
            for column_node in range(offset_px.size):
                node_valid = _evaluate_point_into(
                    pixel_column + offset_px[column_node],
                    pixel_row + offset_px[row_node],
                    detector_shape_rc,
                    detector_zero_lab_m,
                    detector_column_step_lab_m,
                    detector_row_step_lab_m,
                    detector_pixel_area_vector_lab_m2,
                    ray_origin_lab_m,
                    sample_from_lab,
                    ki_film_sample_Ainv,
                    internal_k_Ainv,
                    internal_k_squared_Ainv2,
                    air_k0_Ainv,
                    refractive_air_k_squared_Ainv2,
                    incident_decay_Ainv,
                    film_thickness_A,
                    source_phase_weight,
                    sample_from_local,
                    rod_hk_population,
                    rod_parallel_local_Ainv,
                    rod_u_bounds_Ainv,
                    rod_inverse_constants,
                    b3_norm_Ainv,
                    gaussian_sigma_rad,
                    gaussian_probability,
                    gaussian_normalization,
                    lorentzian_probability,
                    lorentzian_rho,
                    lorentzian_one_minus_rho,
                    lorentzian_numerator,
                    atom_fractional_offset,
                    atom_occupancy_u_iso_element,
                    rod_atom_inplane_factor,
                    common_u_iso_A2,
                    f0_parameters,
                    anomalous_factor_e,
                    layers,
                    shared_disorder_epsilon,
                    normalization_divisor,
                    branch,
                    entrance_power,
                    reconstruction_tolerance,
                    density,
                    inverse_count,
                    caustic,
                )
                if include_center_diagnostics:
                    valid_any[pixel] = valid_any[pixel] or node_valid
                    valid_all[pixel] = valid_all[pixel] and node_valid
                node_weight = one_dimensional_weight[column_node] * one_dimensional_weight[row_node]
                for rod_index in range(rod_count):
                    value = 0.0 if caustic[rod_index] else density[rod_index]
                    mass[pixel, rod_index] += node_weight * value
                    if include_center_diagnostics:
                        count_min[pixel, rod_index] = min(
                            count_min[pixel, rod_index], inverse_count[rod_index]
                        )
                        count_max[pixel, rod_index] = max(
                            count_max[pixel, rod_index], inverse_count[rod_index]
                        )
                        caustic_any[pixel, rod_index] = (
                            caustic_any[pixel, rod_index] or caustic[rod_index]
                        )
        if include_center_diagnostics:
            center_valid = _evaluate_point_into(
                float(pixel_column),
                float(pixel_row),
                detector_shape_rc,
                detector_zero_lab_m,
                detector_column_step_lab_m,
                detector_row_step_lab_m,
                detector_pixel_area_vector_lab_m2,
                ray_origin_lab_m,
                sample_from_lab,
                ki_film_sample_Ainv,
                internal_k_Ainv,
                internal_k_squared_Ainv2,
                air_k0_Ainv,
                refractive_air_k_squared_Ainv2,
                incident_decay_Ainv,
                film_thickness_A,
                source_phase_weight,
                sample_from_local,
                rod_hk_population,
                rod_parallel_local_Ainv,
                rod_u_bounds_Ainv,
                rod_inverse_constants,
                b3_norm_Ainv,
                gaussian_sigma_rad,
                gaussian_probability,
                gaussian_normalization,
                lorentzian_probability,
                lorentzian_rho,
                lorentzian_one_minus_rho,
                lorentzian_numerator,
                atom_fractional_offset,
                atom_occupancy_u_iso_element,
                rod_atom_inplane_factor,
                common_u_iso_A2,
                f0_parameters,
                anomalous_factor_e,
                layers,
                shared_disorder_epsilon,
                normalization_divisor,
                branch,
                entrance_power,
                reconstruction_tolerance,
                density,
                inverse_count,
                caustic,
            )
            center_valid_pixel[pixel] = center_valid
            valid_any[pixel] = valid_any[pixel] or center_valid
            valid_all[pixel] = valid_all[pixel] and center_valid
            for rod_index in range(rod_count):
                center_density[pixel, rod_index] = 0.0 if caustic[rod_index] else density[rod_index]
                count_min[pixel, rod_index] = min(
                    count_min[pixel, rod_index], inverse_count[rod_index]
                )
                count_max[pixel, rod_index] = max(
                    count_max[pixel, rod_index], inverse_count[rod_index]
                )
                caustic_any[pixel, rod_index] = caustic_any[pixel, rod_index] or caustic[rod_index]
    return (
        mass,
        center_density,
        count_min,
        count_max,
        caustic_any,
        valid_any,
        valid_all,
        center_valid_pixel,
    )


class CompiledDetectorEvaluator:
    """Python owner for one packed, reusable compiled detector kernel."""

    __slots__ = ("_detector_shape_rc", "_state")

    def __init__(self, state: CompiledDetectorState, detector_shape_rc: tuple[int, int]) -> None:
        self._state = state
        self._detector_shape_rc = detector_shape_rc

    def evaluate(
        self,
        column_px: NDArray[np.float64],
        row_px: NDArray[np.float64],
        *,
        branch: int,
    ) -> tuple[FloatArray, IntArray, BoolArray, BoolArray]:
        column = np.ascontiguousarray(column_px, dtype=np.float64).reshape(-1)
        row = np.ascontiguousarray(row_px, dtype=np.float64).reshape(-1)
        if column.shape != row.shape:
            raise ValueError("compiled detector coordinates must have equal shapes")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        state = self._state
        return _evaluate_points_kernel(
            column,
            row,
            self._detector_shape_rc,
            state.detector_zero_lab_m,
            state.detector_column_step_lab_m,
            state.detector_row_step_lab_m,
            state.detector_pixel_area_vector_lab_m2,
            state.ray_origin_lab_m,
            state.sample_from_lab,
            state.ki_film_sample_Ainv,
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.refractive_index,
            state.entrance_amplitude,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.sample_from_local,
            state.rod_hk_population,
            state.rod_parallel_local_Ainv,
            state.rod_u_bounds_Ainv,
            state.rod_inverse_constants,
            state.b3_norm_Ainv,
            state.gaussian_sigma_rad,
            state.lorentzian_hwhm_rad,
            state.lorentzian_probability,
            state.atom_fractional_offset,
            state.atom_occupancy_u_iso_element,
            state.rod_atom_inplane_factor,
            state.common_u_iso_A2,
            state.f0_parameters,
            state.anomalous_factor_e,
            state.layers,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
            branch,
        )

    def integrate_pixel_boxes(
        self,
        flat_pixel_index: NDArray[np.int64],
        *,
        offset_px: NDArray[np.float64],
        one_dimensional_weight: NDArray[np.float64],
        branch: int,
        include_center_diagnostics: bool,
    ) -> CompiledPixelIntegral:
        """Integrate exact pixel boxes without materializing node-scale fields."""

        flat_index = np.ascontiguousarray(flat_pixel_index, dtype=np.int64).reshape(-1)
        offset = np.ascontiguousarray(offset_px, dtype=np.float64).reshape(-1)
        weight = np.ascontiguousarray(one_dimensional_weight, dtype=np.float64).reshape(-1)
        if offset.size == 0 or weight.shape != offset.shape:
            raise ValueError("pixel offsets and weights must have the same nonzero shape")
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(weight)):
            raise ValueError("pixel offsets and weights must be finite")
        if np.any(weight < 0.0):
            raise ValueError("pixel quadrature weights must be nonnegative")
        rows, columns = self._detector_shape_rc
        if np.any((flat_index < 0) | (flat_index >= rows * columns)):
            raise ValueError("flat_pixel_index lies outside the active detector")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        state = self._state
        values = _integrate_pixel_boxes_kernel(
            flat_index,
            offset,
            weight,
            bool(include_center_diagnostics),
            self._detector_shape_rc,
            state.detector_zero_lab_m,
            state.detector_column_step_lab_m,
            state.detector_row_step_lab_m,
            state.detector_pixel_area_vector_lab_m2,
            state.ray_origin_lab_m,
            state.sample_from_lab,
            state.ki_film_sample_Ainv,
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.refractive_index,
            state.entrance_amplitude,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.sample_from_local,
            state.rod_hk_population,
            state.rod_parallel_local_Ainv,
            state.rod_u_bounds_Ainv,
            state.rod_inverse_constants,
            state.b3_norm_Ainv,
            state.gaussian_sigma_rad,
            state.lorentzian_hwhm_rad,
            state.lorentzian_probability,
            state.atom_fractional_offset,
            state.atom_occupancy_u_iso_element,
            state.rod_atom_inplane_factor,
            state.common_u_iso_A2,
            state.f0_parameters,
            state.anomalous_factor_e,
            state.layers,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
            branch,
        )
        return CompiledPixelIntegral(*values)


__all__ = [
    "CompiledDetectorEvaluator",
    "CompiledDetectorState",
    "CompiledPixelIntegral",
    "pack_bi2se3_two_h_structure",
]
