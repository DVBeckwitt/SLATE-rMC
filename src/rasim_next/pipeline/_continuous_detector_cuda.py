"""Deterministic CUDA evaluation of the source-averaged detector field."""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import cuda
from numpy.typing import NDArray

from rasim_next.core.scattering import CLASSICAL_ELECTRON_RADIUS_A
from rasim_next.geometry.detector import _DETECTOR_INCIDENCE_COSINE_TOL
from rasim_next.pipeline._continuous_detector_kernel import (
    _coherent_finite_stack_intensity as _cpu_coherent_finite_stack_intensity,
)
from rasim_next.pipeline._continuous_detector_kernel import (
    _wrapped_mosaic_density as _cpu_wrapped_mosaic_density,
)
from rasim_next.stacking.finite_intensity import _finite_moment_intensity

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

_FLOAT_EPS = float(np.finfo(np.float64).eps)
_FLOAT_TINY = float(np.finfo(np.float64).tiny)
_ANGULAR_TOLERANCE = 2048.0 * _FLOAT_EPS
_THREADS_PER_BLOCK = 128
_DEFAULT_COORDINATE_CHUNK_SIZE = 50_000
_finite_moment_intensity_cuda = cuda.jit(device=True, inline=True)(_finite_moment_intensity)


@dataclass(frozen=True, slots=True)
class _PackedSourceAverage:
    detector_zero_lab_m: FloatArray
    detector_column_step_lab_m: FloatArray
    detector_row_step_lab_m: FloatArray
    detector_pixel_area_vector_lab_m2: FloatArray
    sample_from_lab: FloatArray
    sample_from_local: FloatArray
    ray_origin_lab_m: FloatArray
    ki_film_sample_Ainv: FloatArray
    state_real: FloatArray
    state_complex: NDArray[np.complex128]
    state_block_offset: IntArray
    active_state_rod: BoolArray
    rod_u_bounds_Ainv: FloatArray
    rod_u_tolerance_Ainv: FloatArray
    rod_hk_population: FloatArray
    rod_parallel_local_Ainv: FloatArray
    rod_inverse_constants: FloatArray
    atom_fractional_offset: FloatArray
    atom_occupancy_element: FloatArray
    rod_atom_inplane_factor: NDArray[np.complex128]
    f0_parameters: FloatArray
    layers: int


def require_cuda_available() -> str:
    """Fail closed unless an explicitly requested CUDA device is usable."""

    if not cuda.is_available():
        raise RuntimeError("CUDA detector execution was requested, but no CUDA device is available")
    supplied_name = cuda.get_current_device().name
    return supplied_name.decode("utf-8") if isinstance(supplied_name, bytes) else str(supplied_name)


def _equal_shared(name: str, candidate: Any, reference: Any) -> None:
    if not np.array_equal(np.asarray(candidate), np.asarray(reference)):
        raise ValueError(f"compiled source states disagree on shared CUDA field {name}")


def _pack_source_average(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
    *,
    detector_shape_rc: tuple[int, int],
    master_rod_count: int,
) -> _PackedSourceAverage:
    indexed_evaluators = tuple(indexed for block in evaluator_blocks for indexed in block)
    if not indexed_evaluators:
        raise ValueError("CUDA source average requires at least one compiled state")
    first_evaluator = indexed_evaluators[0].evaluator
    first = first_evaluator._state
    if tuple(first_evaluator._detector_shape_rc) != tuple(detector_shape_rc):
        raise ValueError("compiled detector shape disagrees with the CUDA detector shape")

    state_count = len(indexed_evaluators)
    ray_origin = np.empty((state_count, 3), dtype=np.float64)
    ki_film = np.empty((state_count, 3), dtype=np.float64)
    state_real = np.empty((state_count, 27), dtype=np.float64)
    state_complex = np.empty((state_count, 5), dtype=np.complex128)
    state_block_offset = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(
                np.asarray([len(block) for block in evaluator_blocks], dtype=np.int64),
                dtype=np.int64,
            ),
        )
    )
    active_state_rod = np.zeros((state_count, master_rod_count), dtype=np.bool_)
    rod_u_bounds = np.zeros((state_count, master_rod_count, 2), dtype=np.float64)
    rod_u_tolerance = np.zeros((state_count, master_rod_count), dtype=np.float64)

    atom_count = first.atom_fractional_offset.shape[0]
    rod_hk_population = np.zeros((master_rod_count, 3), dtype=np.float64)
    rod_parallel = np.zeros((master_rod_count, 3), dtype=np.float64)
    rod_inverse = np.zeros((master_rod_count, 3), dtype=np.float64)
    rod_inplane = np.zeros((master_rod_count, atom_count), dtype=np.complex128)
    master_filled = np.zeros(master_rod_count, dtype=np.bool_)

    shared_arrays = (
        ("detector_zero_lab_m", first.detector_zero_lab_m),
        ("detector_column_step_lab_m", first.detector_column_step_lab_m),
        ("detector_row_step_lab_m", first.detector_row_step_lab_m),
        ("detector_pixel_area_vector_lab_m2", first.detector_pixel_area_vector_lab_m2),
        ("sample_from_lab", first.sample_from_lab),
        ("sample_from_local", first.sample_from_local),
        ("atom_fractional_offset", first.atom_fractional_offset),
        ("atom_occupancy_element", first.atom_occupancy_element),
        ("f0_parameters", first.f0_parameters),
    )
    shared_scalars = (
        "layers",
        "stacking_parent_code",
        "u_radial_A2",
        "u_normal_A2",
        "intensity_envelope_u_radial_A2",
        "intensity_envelope_u_normal_A2",
        "shared_disorder_epsilon",
        "normalization_divisor",
        "b3_norm_Ainv",
        "gaussian_sigma_rad",
        "lorentzian_hwhm_rad",
        "lorentzian_probability",
        "film_thickness_A",
    )

    for state_index, indexed in enumerate(indexed_evaluators):
        evaluator = indexed.evaluator
        state = evaluator._state
        if tuple(evaluator._detector_shape_rc) != tuple(detector_shape_rc):
            raise ValueError("compiled source states disagree on the detector shape")
        for name, reference in shared_arrays:
            _equal_shared(name, getattr(state, name), reference)
        for name in shared_scalars:
            if getattr(state, name) != getattr(first, name):
                raise ValueError(f"compiled source states disagree on shared CUDA field {name}")

        master_index = np.asarray(indexed.master_rod_index, dtype=np.int64)
        active_count = master_index.size
        if active_count != state.rod_hk_population.shape[0]:
            raise ValueError("active master indices do not align with the compiled rods")
        if np.any(master_index >= master_rod_count):
            raise ValueError("active master rod index lies outside the CUDA catalog")
        active_state_rod[state_index, master_index] = True
        rod_u_bounds[state_index, master_index] = state.rod_u_bounds_Ainv
        rod_u_tolerance[state_index, master_index] = state.rod_inverse_constants[:, 3]

        for local_index, master_index_value in enumerate(master_index):
            master = int(master_index_value)
            if master_filled[master]:
                _equal_shared(
                    "rod_hk_population",
                    state.rod_hk_population[local_index],
                    rod_hk_population[master],
                )
                _equal_shared(
                    "rod_parallel_local_Ainv",
                    state.rod_parallel_local_Ainv[local_index],
                    rod_parallel[master],
                )
                _equal_shared(
                    "rod_inverse_constants",
                    state.rod_inverse_constants[local_index, :3],
                    rod_inverse[master],
                )
                _equal_shared(
                    "rod_atom_inplane_factor",
                    state.rod_atom_inplane_factor[local_index],
                    rod_inplane[master],
                )
            else:
                rod_hk_population[master] = state.rod_hk_population[local_index]
                rod_parallel[master] = state.rod_parallel_local_Ainv[local_index]
                rod_inverse[master] = state.rod_inverse_constants[local_index, :3]
                rod_inplane[master] = state.rod_atom_inplane_factor[local_index]
                master_filled[master] = True

        ray_origin[state_index] = state.ray_origin_lab_m
        ki_film[state_index] = state.ki_film_sample_Ainv
        ki_norm = math.sqrt(
            state.ki_film_sample_Ainv[0] ** 2
            + state.ki_film_sample_Ainv[1] ** 2
            + state.ki_film_sample_Ainv[2] ** 2
        )
        state_real[state_index] = (
            state.internal_k_Ainv,
            state.air_k0_Ainv,
            state.incident_decay_Ainv,
            state.film_thickness_A,
            state.source_phase_weight,
            state.b3_norm_Ainv,
            state.gaussian_sigma_rad,
            state.lorentzian_hwhm_rad,
            state.lorentzian_probability,
            state.u_radial_A2,
            state.u_normal_A2,
            state.shared_disorder_epsilon,
            state.normalization_divisor,
            4096.0 * _FLOAT_EPS * max(ki_norm, 1.0),
            float(state.stacking_parent_code),
            state.intensity_envelope_u_radial_A2,
            state.intensity_envelope_u_normal_A2,
            float(state.polarization_model_code),
            float(state.specular_stitch_code),
            state.specular_top_roughness_A,
            state.specular_bottom_roughness_A,
            state.specular_qc_Ainv,
            state.specular_zero_strength_A2,
            state.specular_scale_factor,
            state.specular_blend_lower_q_over_qc,
            state.specular_blend_upper_q_over_qc,
            state.detector_path_linear_attenuation_m_inv,
        )
        state_complex[state_index] = (
            state.refractive_index,
            state.entrance_amplitude,
            state.anomalous_factor_e[0],
            state.anomalous_factor_e[1],
            state.specular_substrate_refractive_index,
        )

    return _PackedSourceAverage(
        detector_zero_lab_m=np.ascontiguousarray(first.detector_zero_lab_m),
        detector_column_step_lab_m=np.ascontiguousarray(first.detector_column_step_lab_m),
        detector_row_step_lab_m=np.ascontiguousarray(first.detector_row_step_lab_m),
        detector_pixel_area_vector_lab_m2=np.ascontiguousarray(
            first.detector_pixel_area_vector_lab_m2
        ),
        sample_from_lab=np.ascontiguousarray(first.sample_from_lab),
        sample_from_local=np.ascontiguousarray(first.sample_from_local),
        ray_origin_lab_m=ray_origin,
        ki_film_sample_Ainv=ki_film,
        state_real=state_real,
        state_complex=state_complex,
        state_block_offset=state_block_offset,
        active_state_rod=active_state_rod,
        rod_u_bounds_Ainv=rod_u_bounds,
        rod_u_tolerance_Ainv=rod_u_tolerance,
        rod_hk_population=rod_hk_population,
        rod_parallel_local_Ainv=rod_parallel,
        rod_inverse_constants=rod_inverse,
        atom_fractional_offset=np.ascontiguousarray(first.atom_fractional_offset),
        atom_occupancy_element=np.ascontiguousarray(first.atom_occupancy_element),
        rod_atom_inplane_factor=rod_inplane,
        f0_parameters=np.ascontiguousarray(first.f0_parameters),
        layers=int(first.layers),
    )


@cuda.jit(device=True, inline=True)
def _positive_normal_root(radicand: complex) -> complex:
    root = cmath.sqrt(radicand)
    if root.imag != 0.0:
        if root.imag < 0.0:
            root = -root
    elif root.real < 0.0:
        root = -root
    return root


@cuda.jit(device=True, inline=True)
def _complex_exponential(value: complex) -> complex:
    magnitude = math.exp(value.real)
    return complex(magnitude * math.cos(value.imag), magnitude * math.sin(value.imag))


@cuda.jit(device=True, inline=True)
def _empirical_parratt_strength_A2(
    phase_strength_A2: float,
    external_qz_Ainv: float,
    air_k0_Ainv: float,
    film_refractive_index: complex,
    substrate_refractive_index: complex,
    film_thickness_A: float,
    top_roughness_A: float,
    bottom_roughness_A: float,
    qc_Ainv: float,
    zero_strength_A2: float,
    dimensionless_scale_factor: float,
    blend_lower_q_over_qc: float,
    blend_upper_q_over_qc: float,
) -> float:
    film_offset = (film_refractive_index * air_k0_Ainv) ** 2 - air_k0_Ainv**2
    external_qz = abs(external_qz_Ainv)
    external_half_squared = 0.25 * external_qz * external_qz
    q_over_qc = external_qz / qc_Ainv
    if q_over_qc >= blend_upper_q_over_qc:
        return phase_strength_A2
    external_half = 0.5 * external_qz
    film_kz = _positive_normal_root(film_offset + external_half_squared)
    substrate_kz = _positive_normal_root(
        (substrate_refractive_index * air_k0_Ainv) ** 2 - air_k0_Ainv**2 + external_half_squared
    )
    top_denominator = complex(external_half, 0.0) + film_kz
    bottom_denominator = film_kz + substrate_kz
    if external_half == 0.0 and film_kz == 0.0:
        top = 0.0j
    elif top_denominator == 0.0:
        return math.nan
    else:
        top = (complex(external_half, 0.0) - film_kz) / top_denominator
    if film_kz == 0.0 and substrate_kz == 0.0:
        bottom = 0.0j
    elif bottom_denominator == 0.0:
        return math.nan
    else:
        bottom = (film_kz - substrate_kz) / bottom_denominator
    if top_roughness_A != 0.0:
        top *= _complex_exponential(
            -2.0 * complex(external_half, 0.0) * film_kz * top_roughness_A**2
        )
    if bottom_roughness_A != 0.0:
        bottom *= _complex_exponential(-2.0 * film_kz * substrate_kz * bottom_roughness_A**2)
    propagated = bottom * _complex_exponential(2.0j * film_kz * film_thickness_A)
    recursion_denominator = 1.0 + top * propagated
    if recursion_denominator == 0.0:
        return math.nan
    amplitude = (top + propagated) / recursion_denominator
    reflectivity = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag
    low_strength = (
        external_qz * external_qz * reflectivity * zero_strength_A2 / dimensionless_scale_factor
    )
    if not math.isfinite(reflectivity) or not math.isfinite(low_strength):
        return math.nan
    if q_over_qc <= blend_lower_q_over_qc:
        return low_strength
    coordinate = (q_over_qc - blend_lower_q_over_qc) / (
        blend_upper_q_over_qc - blend_lower_q_over_qc
    )
    weight = 6.0 * coordinate**5 - 15.0 * coordinate**4 + 10.0 * coordinate**3
    low_for_log = low_strength if low_strength > _FLOAT_TINY else _FLOAT_TINY
    phase_for_log = phase_strength_A2 if phase_strength_A2 > _FLOAT_TINY else _FLOAT_TINY
    return math.exp((1.0 - weight) * math.log(low_for_log) + weight * math.log(phase_for_log))


# Compile the same undecorated arithmetic for CUDA, not the CPU dispatcher.
_wrapped_mosaic_density = cuda.jit(device=True, inline=True)(_cpu_wrapped_mosaic_density.py_func)

_coherent_finite_stack_intensity = cuda.jit(device=True, inline=True)(
    _cpu_coherent_finite_stack_intensity.py_func
)


@cuda.jit(device=True, inline=True)
def _finite_stack_strength_A2(
    rod_index: int,
    ell: float,
    common_damping: float,
    parallel_norm_Ainv: float,
    w_value_Ainv: float,
    element_factor_0: complex,
    element_factor_1: complex,
    rod_atom_inplane_factor: Any,
    atom_fractional_offset: Any,
    atom_occupancy_element: Any,
    layers: int,
    stacking_parent_code: int,
    shared_disorder_epsilon: float,
    rod_hk_population: Any,
    normalization_divisor: float,
) -> float:
    amplitude_plus = 0.0 + 0.0j
    amplitude_minus = 0.0 + 0.0j
    for atom in range(atom_fractional_offset.shape[0]):
        occupancy = atom_occupancy_element[atom, 0]
        element = int(atom_occupancy_element[atom, 1])
        site_damping = math.exp(
            -0.5
            * (
                atom_occupancy_element[atom, 2] * parallel_norm_Ainv * parallel_norm_Ainv
                + atom_occupancy_element[atom, 3] * w_value_Ainv * w_value_Ainv
            )
        )
        phase_z = 2.0 * math.pi * ell * atom_fractional_offset[atom, 2]
        inplane_factor = rod_atom_inplane_factor[rod_index, atom]
        phase_plus = inplane_factor * complex(math.cos(phase_z), math.sin(phase_z))
        element_factor = element_factor_0 if element == 0 else element_factor_1
        amplitude_plus += occupancy * site_damping * element_factor * phase_plus
        if shared_disorder_epsilon != 0.0:
            phase_minus = inplane_factor * complex(math.cos(phase_z), -math.sin(phase_z))
            amplitude_minus += occupancy * site_damping * element_factor * phase_minus

    if shared_disorder_epsilon == 0.0:
        registry_index = 0
        if stacking_parent_code == 1:
            h = int(rod_hk_population[rod_index, 0])
            k = int(rod_hk_population[rod_index, 1])
            registry_index = (h + 2 * k) % 3
        amplitude_intensity = (
            amplitude_plus.real * amplitude_plus.real + amplitude_plus.imag * amplitude_plus.imag
        )
        intensity_e2 = amplitude_intensity * _coherent_finite_stack_intensity(
            layers,
            ell,
            registry_index,
        )
    else:
        vertical_phase_angle = 2.0 * math.pi * ell / 3.0
        vertical_phase = complex(math.cos(vertical_phase_angle), math.sin(vertical_phase_angle))
        h = int(rod_hk_population[rod_index, 0])
        k = int(rod_hk_population[rod_index, 1])
        registry_index = (h + 2 * k) % 3
        if registry_index == 0:
            omega = 1.0 + 0.0j
        elif registry_index == 1:
            omega = complex(-0.5, 0.5 * math.sqrt(3.0))
        else:
            omega = complex(-0.5, -0.5 * math.sqrt(3.0))
        alternative = 0.25 * shared_disorder_epsilon
        parent = 1.0 - shared_disorder_epsilon
        # The native 3R parent is b-minus; 2H is the same-registry a event.
        a = parent if stacking_parent_code == 0 else alternative
        b_minus = alternative if stacking_parent_code == 0 else parent
        intensity_e2 = _finite_moment_intensity_cuda(
            layers,
            amplitude_plus,
            amplitude_minus,
            omega,
            vertical_phase,
            a,
            alternative,
            b_minus,
            alternative,
            alternative,
            1.0,
            0.0,
        )
    return (
        CLASSICAL_ELECTRON_RADIUS_A**2
        * common_damping
        * common_damping
        * intensity_e2
        / normalization_divisor
    )


@cuda.jit(fastmath=False)
def _prepare_state_block_geometry_kernel(
    state_start: int,
    state_stop: int,
    detector_rows: int,
    detector_columns: int,
    column_px: Any,
    row_px: Any,
    detector_zero_lab_m: Any,
    detector_column_step_lab_m: Any,
    detector_row_step_lab_m: Any,
    detector_pixel_area_vector_lab_m2: Any,
    sample_from_lab: Any,
    sample_from_local: Any,
    ray_origin_lab_m: Any,
    ki_film_sample_Ainv: Any,
    state_real: Any,
    state_complex: Any,
    f0_parameters: Any,
    q_geometry: Any,
    point_factor: Any,
    valid: Any,
) -> None:
    linear_index = cuda.grid(1)
    point_count = column_px.size
    local_state_count = state_stop - state_start
    if linear_index >= local_state_count * point_count:
        return
    local_state = linear_index // point_count
    point = linear_index - local_state * point_count
    state_index = state_start + local_state
    valid[local_state, point] = False
    column = column_px[point]
    row = row_px[point]
    if column < -0.5 or column > detector_columns - 0.5 or row < -0.5 or row > detector_rows - 0.5:
        return

    displacement_x = (
        detector_zero_lab_m[0]
        + column * detector_column_step_lab_m[0]
        + row * detector_row_step_lab_m[0]
        - ray_origin_lab_m[state_index, 0]
    )
    displacement_y = (
        detector_zero_lab_m[1]
        + column * detector_column_step_lab_m[1]
        + row * detector_row_step_lab_m[1]
        - ray_origin_lab_m[state_index, 1]
    )
    displacement_z = (
        detector_zero_lab_m[2]
        + column * detector_column_step_lab_m[2]
        + row * detector_row_step_lab_m[2]
        - ray_origin_lab_m[state_index, 2]
    )
    distance = math.sqrt(
        displacement_x * displacement_x
        + displacement_y * displacement_y
        + displacement_z * displacement_z
    )
    if distance == 0.0:
        return
    direction_x = displacement_x / distance
    direction_y = displacement_y / distance
    direction_z = displacement_z / distance
    signed_pixel_area_projection = (
        direction_x * detector_pixel_area_vector_lab_m2[0]
        + direction_y * detector_pixel_area_vector_lab_m2[1]
        + direction_z * detector_pixel_area_vector_lab_m2[2]
    )
    pixel_area = math.sqrt(
        detector_pixel_area_vector_lab_m2[0] ** 2
        + detector_pixel_area_vector_lab_m2[1] ** 2
        + detector_pixel_area_vector_lab_m2[2] ** 2
    )
    if signed_pixel_area_projection <= _DETECTOR_INCIDENCE_COSINE_TOL * pixel_area:
        return
    air_k0_Ainv = state_real[state_index, 1]
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
        return
    parallel_squared = kf_air_x * kf_air_x + kf_air_y * kf_air_y
    internal_k_Ainv = state_real[state_index, 0]
    normal_squared = internal_k_Ainv * internal_k_Ainv - parallel_squared
    if normal_squared <= 0.0:
        return
    kf_film_z = math.sqrt(normal_squared)
    q_sample_x = kf_air_x - ki_film_sample_Ainv[state_index, 0]
    q_sample_y = kf_air_y - ki_film_sample_Ainv[state_index, 1]
    q_sample_z = kf_film_z - ki_film_sample_Ainv[state_index, 2]

    pixel_solid_angle = signed_pixel_area_projection / (distance * distance)
    area_jacobian = internal_k_Ainv * air_k0_Ainv * kf_air_z * pixel_solid_angle / kf_film_z

    refractive_index = state_complex[state_index, 0]
    refractive_air_k_squared_Ainv2 = (refractive_index * air_k0_Ainv) ** 2
    kz_film = _positive_normal_root(refractive_air_k_squared_Ainv2 - parallel_squared)
    denominator = kz_film + complex(kf_air_z, 0.0)
    if denominator == 0.0:
        return
    exit_amplitude = 2.0 * kz_film / denominator
    exit_decay_Ainv = kz_film.imag if kz_film.imag > 0.0 else 0.0
    exponent = 2.0 * (state_real[state_index, 2] + exit_decay_Ainv) * state_real[state_index, 3]
    attenuation = 1.0 if exponent == 0.0 else -math.expm1(-exponent) / exponent
    entrance_amplitude = state_complex[state_index, 1]
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    optical_weight = (
        entrance_power * (exit_amplitude.real**2 + exit_amplitude.imag**2) * attenuation
    )
    if state_real[state_index, 26] != 0.0:
        optical_weight *= math.exp(-state_real[state_index, 26] * distance)
    incident_normal_squared = (
        air_k0_Ainv * air_k0_Ainv
        - ki_film_sample_Ainv[state_index, 0] ** 2
        - ki_film_sample_Ainv[state_index, 1] ** 2
    )
    if incident_normal_squared < 0.0:
        incident_normal_squared = 0.0
    incident_normal = math.sqrt(incident_normal_squared)
    if ki_film_sample_Ainv[state_index, 2] < 0.0:
        incident_normal = -incident_normal
    if int(state_real[state_index, 17]) == 1:
        cosine = (
            ki_film_sample_Ainv[state_index, 0] * kf_air_x
            + ki_film_sample_Ainv[state_index, 1] * kf_air_y
            + incident_normal * kf_air_z
        ) / (air_k0_Ainv * air_k0_Ainv)
        if cosine < -1.0:
            cosine = -1.0
        elif cosine > 1.0:
            cosine = 1.0
        optical_weight *= 0.5 * (1.0 + cosine * cosine)

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

    q_geometry[local_state, point, 0] = q_sample_x
    q_geometry[local_state, point, 1] = q_sample_y
    q_geometry[local_state, point, 2] = q_sample_z
    q_geometry[local_state, point, 3] = q_local_z
    q_geometry[local_state, point, 4] = q_norm_squared
    q_geometry[local_state, point, 5] = q_norm
    q_geometry[local_state, point, 6] = math.hypot(q_local_x, q_local_y)
    q_geometry[local_state, point, 7] = math.atan2(q_local_y, q_local_x)
    q_geometry[local_state, point, 8] = (
        0.0
        if q_norm == 0.0
        else abs(
            (
                q_sample_x * q_sample_x
                + q_sample_y * q_sample_y
                + q_sample_z * (kf_air_z - incident_normal)
            )
            / q_norm
        )
    )
    point_factor[local_state, point, 0] = area_jacobian
    point_factor[local_state, point, 1] = optical_weight
    point_factor[local_state, point, 2] = f0_0
    point_factor[local_state, point, 3] = f0_1
    point_factor[local_state, point, 4] = math.exp(
        -state_real[state_index, 15] * (q_sample_x * q_sample_x + q_sample_y * q_sample_y)
        - state_real[state_index, 16] * q_sample_z * q_sample_z
    )
    valid[local_state, point] = True


@cuda.jit(fastmath=False)
def _count_valid_state_block_kernel(
    local_state_count: int,
    valid: Any,
    block_valid_source_count: Any,
    valid_source_count: Any,
) -> None:
    point = cuda.grid(1)
    if point >= valid.shape[1]:
        return
    count = 0
    for local_state in range(local_state_count):
        if valid[local_state, point]:
            count += 1
    block_valid_source_count[point] = count
    valid_source_count[point] += count


@cuda.jit(fastmath=False)
def _accumulate_state_block_kernel(
    state_start: int,
    state_stop: int,
    point_count: int,
    sample_from_local: Any,
    state_real: Any,
    state_complex: Any,
    active_state_rod: Any,
    rod_u_bounds_Ainv: Any,
    rod_u_tolerance_Ainv: Any,
    rod_hk_population: Any,
    rod_parallel_local_Ainv: Any,
    rod_inverse_constants: Any,
    atom_fractional_offset: Any,
    atom_occupancy_element: Any,
    rod_atom_inplane_factor: Any,
    layers: int,
    gaussian_sigma_rad: float,
    gaussian_probability: float,
    gaussian_normalization: float,
    lorentzian_probability: float,
    lorentzian_rho: float,
    lorentzian_one_minus_rho: float,
    lorentzian_numerator: float,
    q_geometry: Any,
    point_factor: Any,
    valid: Any,
    block_valid_source_count: Any,
    density: Any,
    caustic: Any,
) -> None:
    linear_index = cuda.grid(1)
    rod_count = rod_hk_population.shape[0]
    if linear_index >= rod_count * point_count:
        return
    rod_index = linear_index // point_count
    point = linear_index - rod_index * point_count
    if block_valid_source_count[point] == 0:
        return

    a = rod_parallel_local_Ainv[rod_index, 0]
    b = rod_parallel_local_Ainv[rod_index, 1]
    c0 = rod_parallel_local_Ainv[rod_index, 2]
    abs_b = rod_inverse_constants[rod_index, 0]
    parallel_norm = rod_inverse_constants[rod_index, 1]
    inverse_reference = rod_inverse_constants[rod_index, 2]
    block_rod_density = 0.0
    block_rod_caustic = False
    two_pi = 2.0 * math.pi
    local_state_count = state_stop - state_start
    for local_state in range(local_state_count):
        state_index = state_start + local_state
        if not valid[local_state, point] or not active_state_rod[state_index, rod_index]:
            continue
        q_sample_x = q_geometry[local_state, point, 0]
        q_sample_y = q_geometry[local_state, point, 1]
        q_sample_z = q_geometry[local_state, point, 2]
        q_local_z = q_geometry[local_state, point, 3]
        q_norm_squared = q_geometry[local_state, point, 4]
        q_norm = q_geometry[local_state, point, 5]
        transverse_norm = q_geometry[local_state, point, 6]
        azimuth_q = q_geometry[local_state, point, 7]
        x_squared = (transverse_norm - abs_b) * (transverse_norm + abs_b)
        w_squared = (q_norm - parallel_norm) * (q_norm + parallel_norm)
        inverse_scale = q_norm_squared if q_norm_squared > inverse_reference else inverse_reference
        if inverse_scale < 1.0:
            inverse_scale = 1.0
        inverse_tolerance = 1024.0 * _FLOAT_EPS * inverse_scale
        if x_squared < -inverse_tolerance or w_squared < -inverse_tolerance:
            continue
        x_magnitude = math.sqrt(x_squared if x_squared > 0.0 else 0.0)
        w_magnitude = math.sqrt(w_squared if w_squared > 0.0 else 0.0)
        lower_u = rod_u_bounds_Ainv[state_index, rod_index, 0]
        upper_u = rod_u_bounds_Ainv[state_index, rod_index, 1]
        u_tolerance = rod_u_tolerance_Ainv[state_index, rod_index]
        b3_norm_Ainv = state_real[state_index, 5]
        u_radial_A2 = state_real[state_index, 9]
        u_normal_A2 = state_real[state_index, 10]
        shared_disorder_epsilon = state_real[state_index, 11]
        normalization_divisor = state_real[state_index, 12]
        reconstruction_tolerance = state_real[state_index, 13]
        element_factor_0 = point_factor[local_state, point, 2] + state_complex[state_index, 2]
        element_factor_1 = point_factor[local_state, point, 3] + state_complex[state_index, 3]
        area_jacobian = point_factor[local_state, point, 0]
        optical_weight = point_factor[local_state, point, 1]
        event_intensity_envelope = point_factor[local_state, point, 4]
        source_phase_weight = state_real[state_index, 4]
        state_rod_density = 0.0
        state_rod_caustic = False

        for x_branch in range(2):
            x_value = (-1.0 if x_branch == 0 else 1.0) * x_magnitude
            beta = (azimuth_q - math.atan2(b, x_value)) % two_pi
            if beta >= two_pi:
                beta = 0.0
            for w_branch in range(2):
                w_value = (-1.0 if w_branch == 0 else 1.0) * w_magnitude
                alpha = (math.atan2(w_value, a) - math.atan2(q_local_z, x_value)) % two_pi
                if alpha >= two_pi - _ANGULAR_TOLERANCE:
                    alpha = 0.0
                if alpha > math.pi + _ANGULAR_TOLERANCE:
                    continue
                if alpha > math.pi:
                    alpha = math.pi
                u_value = w_value - c0
                if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                    continue
                if u_radial_A2 == u_normal_A2:
                    common_damping = math.exp(-0.5 * q_norm_squared * u_radial_A2)
                else:
                    common_damping = math.exp(
                        -0.5
                        * (
                            u_radial_A2 * parallel_norm * parallel_norm
                            + u_normal_A2 * w_value * w_value
                        )
                    )
                strength = _finite_stack_strength_A2(
                    rod_index,
                    u_value / b3_norm_Ainv,
                    common_damping,
                    parallel_norm,
                    w_value,
                    element_factor_0,
                    element_factor_1,
                    rod_atom_inplane_factor,
                    atom_fractional_offset,
                    atom_occupancy_element,
                    layers,
                    int(state_real[state_index, 14]),
                    shared_disorder_epsilon,
                    rod_hk_population,
                    normalization_divisor,
                )
                if (
                    int(state_real[state_index, 18]) == 2
                    and rod_hk_population[rod_index, 0] == 0.0
                    and rod_hk_population[rod_index, 1] == 0.0
                ):
                    strength = _empirical_parratt_strength_A2(
                        strength,
                        q_geometry[local_state, point, 8],
                        state_real[state_index, 1],
                        state_complex[state_index, 0],
                        state_complex[state_index, 4],
                        state_real[state_index, 3],
                        state_real[state_index, 19],
                        state_real[state_index, 20],
                        state_real[state_index, 21],
                        state_real[state_index, 22],
                        state_real[state_index, 23],
                        state_real[state_index, 24],
                        state_real[state_index, 25],
                    )
                if not math.isfinite(strength):
                    density[rod_index, point] = math.nan
                    return
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
                    state_rod_caustic = True
                    if (
                        source_phase_weight > 0.0
                        and rod_hk_population[rod_index, 2] > 0.0
                        and area_jacobian > 0.0
                        and optical_weight > 0.0
                        and strength > 0.0
                        and mosaic_density > 0.0
                        and event_intensity_envelope > 0.0
                    ):
                        state_rod_density = math.inf
                    continue
                state_rod_density += (
                    mosaic_density
                    * rod_hk_population[rod_index, 2]
                    * strength
                    * area_jacobian
                    * optical_weight
                    * source_phase_weight
                    * event_intensity_envelope
                    / jacobian
                )
        block_rod_density += state_rod_density
        if state_rod_caustic:
            block_rod_caustic = True
    density[rod_index, point] += block_rod_density
    if block_rod_caustic:
        caustic[rod_index, point] = True


@cuda.jit(fastmath=False)
def _reduce_physical_rods_kernel(
    per_rod_density: Any,
    per_rod_caustic: Any,
    density: Any,
    caustic: Any,
) -> None:
    """Reduce rods in stable master-catalog order at each continuous coordinate."""

    point = cuda.grid(1)
    if point >= density.size:
        return
    total = 0.0
    singular = False
    for rod_index in range(per_rod_density.shape[0]):
        total += per_rod_density[rod_index, point]
        if per_rod_caustic[rod_index, point]:
            singular = True
    density[point] = total
    caustic[point] = singular


@cuda.jit(fastmath=False)
def _reduce_selected_rod_group_kernel(
    per_rod_density: Any,
    per_rod_caustic: Any,
    rod_group_index: Any,
    group_master_rod_mask: Any,
    density: Any,
    caustic: Any,
) -> None:
    """Reduce the requested master-rod group independently at every point."""

    point = cuda.grid(1)
    if point >= density.size:
        return
    group = rod_group_index[point]
    total = 0.0
    singular = False
    for rod_index in range(per_rod_density.shape[0]):
        if group_master_rod_mask[group, rod_index]:
            total += per_rod_density[rod_index, point]
            if per_rod_caustic[rod_index, point]:
                singular = True
    density[point] = total
    caustic[point] = singular


def _evaluate_source_averaged_all_roots_cuda(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
    column_px: FloatArray,
    row_px: FloatArray,
    *,
    detector_shape_rc: tuple[int, int],
    master_rod_count: int,
    return_per_rod: bool,
    coordinate_chunk_size: int,
) -> tuple[FloatArray, BoolArray, IntArray, str]:
    """Evaluate one packed source average and stream only the requested coordinate result."""

    if (
        isinstance(coordinate_chunk_size, bool)
        or not isinstance(coordinate_chunk_size, (int, np.integer))
        or int(coordinate_chunk_size) < 1
    ):
        raise ValueError("coordinate_chunk_size must be a positive integer")
    chunk_size = int(coordinate_chunk_size)
    device_name = require_cuda_available()
    column = np.ascontiguousarray(column_px, dtype=np.float64).reshape(-1)
    row = np.ascontiguousarray(row_px, dtype=np.float64).reshape(-1)
    if column.shape != row.shape:
        raise ValueError("CUDA detector coordinates must have equal shapes")
    if column.size == 0:
        output_shape = (0, master_rod_count) if return_per_rod else (0,)
        return (
            np.zeros(output_shape, dtype=np.float64),
            np.zeros(output_shape, dtype=np.bool_),
            np.zeros(0, dtype=np.int64),
            device_name,
        )
    packed = _pack_source_average(
        evaluator_blocks,
        detector_shape_rc=detector_shape_rc,
        master_rod_count=master_rod_count,
    )
    device_detector_zero = cuda.to_device(packed.detector_zero_lab_m)
    device_detector_column_step = cuda.to_device(packed.detector_column_step_lab_m)
    device_detector_row_step = cuda.to_device(packed.detector_row_step_lab_m)
    device_detector_pixel_area = cuda.to_device(packed.detector_pixel_area_vector_lab_m2)
    device_sample_from_lab = cuda.to_device(packed.sample_from_lab)
    device_sample_from_local = cuda.to_device(packed.sample_from_local)
    device_ray_origin = cuda.to_device(packed.ray_origin_lab_m)
    device_ki_film = cuda.to_device(packed.ki_film_sample_Ainv)
    device_state_real = cuda.to_device(packed.state_real)
    device_state_complex = cuda.to_device(packed.state_complex)
    device_active_state_rod = cuda.to_device(packed.active_state_rod)
    device_rod_u_bounds = cuda.to_device(packed.rod_u_bounds_Ainv)
    device_rod_u_tolerance = cuda.to_device(packed.rod_u_tolerance_Ainv)
    device_rod_hk_population = cuda.to_device(packed.rod_hk_population)
    device_rod_parallel = cuda.to_device(packed.rod_parallel_local_Ainv)
    device_rod_inverse = cuda.to_device(packed.rod_inverse_constants)
    device_atom_offset = cuda.to_device(packed.atom_fractional_offset)
    device_atom_properties = cuda.to_device(packed.atom_occupancy_element)
    device_rod_inplane = cuda.to_device(packed.rod_atom_inplane_factor)
    device_f0_parameters = cuda.to_device(packed.f0_parameters)

    rows, columns = detector_shape_rc
    block_sizes = np.diff(packed.state_block_offset)
    maximum_block_size = int(np.max(block_sizes))
    gaussian_sigma_rad = float(packed.state_real[0, 6])
    lorentzian_hwhm_rad = float(packed.state_real[0, 7])
    lorentzian_probability = float(packed.state_real[0, 8])
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)

    if return_per_rod:
        density = np.empty((column.size, master_rod_count), dtype=np.float64)
        caustic = np.empty((column.size, master_rod_count), dtype=np.bool_)
    else:
        density = np.empty(column.size, dtype=np.float64)
        caustic = np.empty(column.size, dtype=np.bool_)
    valid_source_count = np.empty(column.size, dtype=np.int64)

    for start in range(0, column.size, chunk_size):
        stop = min(start + chunk_size, column.size)
        chunk_column = column[start:stop]
        chunk_row = row[start:stop]
        point_count = chunk_column.size
        device_column = cuda.to_device(chunk_column)
        device_row = cuda.to_device(chunk_row)
        device_per_rod_density = cuda.to_device(
            np.zeros((master_rod_count, point_count), dtype=np.float64)
        )
        device_per_rod_caustic = cuda.to_device(
            np.zeros((master_rod_count, point_count), dtype=np.bool_)
        )
        device_valid_count = cuda.to_device(np.zeros(point_count, dtype=np.int64))
        device_q_geometry = cuda.device_array(
            (maximum_block_size, point_count, 9),
            dtype=np.float64,
        )
        device_point_factor = cuda.device_array(
            (maximum_block_size, point_count, 5),
            dtype=np.float64,
        )
        device_valid = cuda.device_array(
            (maximum_block_size, point_count),
            dtype=np.bool_,
        )
        device_block_valid_count = cuda.device_array(point_count, dtype=np.int64)
        point_blocks = (point_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        rod_point_work = master_rod_count * point_count
        rod_point_blocks = (rod_point_work + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        for block_index in range(packed.state_block_offset.size - 1):
            state_start = int(packed.state_block_offset[block_index])
            state_stop = int(packed.state_block_offset[block_index + 1])
            local_state_count = state_stop - state_start
            geometry_work = local_state_count * point_count
            geometry_blocks = (geometry_work + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            _prepare_state_block_geometry_kernel[geometry_blocks, _THREADS_PER_BLOCK](
                state_start,
                state_stop,
                rows,
                columns,
                device_column,
                device_row,
                device_detector_zero,
                device_detector_column_step,
                device_detector_row_step,
                device_detector_pixel_area,
                device_sample_from_lab,
                device_sample_from_local,
                device_ray_origin,
                device_ki_film,
                device_state_real,
                device_state_complex,
                device_f0_parameters,
                device_q_geometry,
                device_point_factor,
                device_valid,
            )
            _count_valid_state_block_kernel[point_blocks, _THREADS_PER_BLOCK](
                local_state_count,
                device_valid,
                device_block_valid_count,
                device_valid_count,
            )
            _accumulate_state_block_kernel[rod_point_blocks, _THREADS_PER_BLOCK](
                state_start,
                state_stop,
                point_count,
                device_sample_from_local,
                device_state_real,
                device_state_complex,
                device_active_state_rod,
                device_rod_u_bounds,
                device_rod_u_tolerance,
                device_rod_hk_population,
                device_rod_parallel,
                device_rod_inverse,
                device_atom_offset,
                device_atom_properties,
                device_rod_inplane,
                packed.layers,
                gaussian_sigma_rad,
                gaussian_probability,
                gaussian_normalization,
                lorentzian_probability,
                lorentzian_rho,
                lorentzian_one_minus_rho,
                lorentzian_numerator,
                device_q_geometry,
                device_point_factor,
                device_valid,
                device_block_valid_count,
                device_per_rod_density,
                device_per_rod_caustic,
            )
        if return_per_rod:
            density[start:stop] = device_per_rod_density.copy_to_host().T
            caustic[start:stop] = device_per_rod_caustic.copy_to_host().T
        else:
            device_density = cuda.device_array(point_count, dtype=np.float64)
            device_caustic = cuda.device_array(point_count, dtype=np.bool_)
            _reduce_physical_rods_kernel[point_blocks, _THREADS_PER_BLOCK](
                device_per_rod_density,
                device_per_rod_caustic,
                device_density,
                device_caustic,
            )
            density[start:stop] = device_density.copy_to_host()
            caustic[start:stop] = device_caustic.copy_to_host()
        valid_source_count[start:stop] = device_valid_count.copy_to_host()
    if np.any(np.isnan(density) | np.isneginf(density)):
        raise FloatingPointError("CUDA detector evaluation produced undefined physical intensity")
    return density, caustic, valid_source_count, device_name


def evaluate_selected_source_rod_groups_all_roots_cuda(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
    column_px: FloatArray,
    row_px: FloatArray,
    evaluator_index_by_coordinate: IntArray,
    rod_group_index_by_coordinate: IntArray,
    group_master_rod_mask: BoolArray,
    *,
    detector_shape_rc: tuple[int, int],
    master_rod_count: int,
    coordinate_chunk_size: int = _DEFAULT_COORDINATE_CHUNK_SIZE,
) -> tuple[FloatArray, BoolArray, BoolArray, str]:
    """Evaluate one selected source state and master-rod group per coordinate."""

    if (
        isinstance(coordinate_chunk_size, bool)
        or not isinstance(coordinate_chunk_size, (int, np.integer))
        or int(coordinate_chunk_size) < 1
    ):
        raise ValueError("coordinate_chunk_size must be a positive integer")
    chunk_size = int(coordinate_chunk_size)
    device_name = require_cuda_available()
    column = np.ascontiguousarray(column_px, dtype=np.float64).reshape(-1)
    row = np.ascontiguousarray(row_px, dtype=np.float64).reshape(-1)
    evaluator_index = np.ascontiguousarray(
        evaluator_index_by_coordinate,
        dtype=np.int64,
    ).reshape(-1)
    group_index = np.ascontiguousarray(
        rod_group_index_by_coordinate,
        dtype=np.int64,
    ).reshape(-1)
    group_mask = np.ascontiguousarray(group_master_rod_mask, dtype=np.bool_)
    indexed_evaluators = tuple(indexed for block in evaluator_blocks for indexed in block)
    if (
        column.shape != row.shape
        or evaluator_index.shape != column.shape
        or group_index.shape != column.shape
        or group_mask.ndim != 2
        or group_mask.shape[1] != master_rod_count
        or not group_mask.shape[0]
        or np.any((evaluator_index < 0) | (evaluator_index >= len(indexed_evaluators)))
        or np.any((group_index < 0) | (group_index >= group_mask.shape[0]))
        or np.any(~np.any(group_mask, axis=1))
    ):
        raise ValueError("selected CUDA source/rod-group coordinates are inconsistent")
    if column.size == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.bool_),
            np.zeros(0, dtype=np.bool_),
            device_name,
        )

    packed = _pack_source_average(
        evaluator_blocks,
        detector_shape_rc=detector_shape_rc,
        master_rod_count=master_rod_count,
    )
    device_detector_zero = cuda.to_device(packed.detector_zero_lab_m)
    device_detector_column_step = cuda.to_device(packed.detector_column_step_lab_m)
    device_detector_row_step = cuda.to_device(packed.detector_row_step_lab_m)
    device_detector_pixel_area = cuda.to_device(packed.detector_pixel_area_vector_lab_m2)
    device_sample_from_lab = cuda.to_device(packed.sample_from_lab)
    device_sample_from_local = cuda.to_device(packed.sample_from_local)
    device_ray_origin = cuda.to_device(packed.ray_origin_lab_m)
    device_ki_film = cuda.to_device(packed.ki_film_sample_Ainv)
    device_state_real = cuda.to_device(packed.state_real)
    device_state_complex = cuda.to_device(packed.state_complex)
    device_active_state_rod = cuda.to_device(packed.active_state_rod)
    device_rod_u_bounds = cuda.to_device(packed.rod_u_bounds_Ainv)
    device_rod_u_tolerance = cuda.to_device(packed.rod_u_tolerance_Ainv)
    device_rod_hk_population = cuda.to_device(packed.rod_hk_population)
    device_rod_parallel = cuda.to_device(packed.rod_parallel_local_Ainv)
    device_rod_inverse = cuda.to_device(packed.rod_inverse_constants)
    device_atom_offset = cuda.to_device(packed.atom_fractional_offset)
    device_atom_properties = cuda.to_device(packed.atom_occupancy_element)
    device_rod_inplane = cuda.to_device(packed.rod_atom_inplane_factor)
    device_f0_parameters = cuda.to_device(packed.f0_parameters)
    device_group_mask = cuda.to_device(group_mask)

    rows, columns = detector_shape_rc
    gaussian_sigma_rad = float(packed.state_real[0, 6])
    lorentzian_hwhm_rad = float(packed.state_real[0, 7])
    lorentzian_probability = float(packed.state_real[0, 8])
    gaussian_probability = 1.0 - lorentzian_probability
    gaussian_normalization = math.sqrt(2.0 * math.pi) * gaussian_sigma_rad
    lorentzian_rho = math.exp(-lorentzian_hwhm_rad)
    lorentzian_one_minus_rho = -math.expm1(-lorentzian_hwhm_rad)
    lorentzian_numerator = -math.expm1(-2.0 * lorentzian_hwhm_rad)

    already_sorted = bool(np.all(evaluator_index[1:] >= evaluator_index[:-1]))
    order = None if already_sorted else np.argsort(evaluator_index, kind="stable")
    sorted_column = column if order is None else column[order]
    sorted_row = row if order is None else row[order]
    sorted_evaluator = evaluator_index if order is None else evaluator_index[order]
    sorted_group = group_index if order is None else group_index[order]
    sorted_density = np.empty(column.size, dtype=np.float64)
    sorted_caustic = np.empty(column.size, dtype=np.bool_)
    sorted_valid = np.empty(column.size, dtype=np.bool_)
    state_edges = np.flatnonzero(np.r_[True, sorted_evaluator[1:] != sorted_evaluator[:-1], True])
    for state_segment in range(state_edges.size - 1):
        segment_start = int(state_edges[state_segment])
        segment_stop = int(state_edges[state_segment + 1])
        state_index = int(sorted_evaluator[segment_start])
        for start in range(segment_start, segment_stop, chunk_size):
            stop = min(start + chunk_size, segment_stop)
            chunk_column = sorted_column[start:stop]
            chunk_row = sorted_row[start:stop]
            chunk_group = sorted_group[start:stop]
            point_count = chunk_column.size
            device_column = cuda.to_device(chunk_column)
            device_row = cuda.to_device(chunk_row)
            device_group_index = cuda.to_device(chunk_group)
            device_per_rod_density = cuda.to_device(
                np.zeros((master_rod_count, point_count), dtype=np.float64)
            )
            device_per_rod_caustic = cuda.to_device(
                np.zeros((master_rod_count, point_count), dtype=np.bool_)
            )
            device_valid_count = cuda.to_device(np.zeros(point_count, dtype=np.int64))
            device_q_geometry = cuda.device_array((1, point_count, 9), dtype=np.float64)
            device_point_factor = cuda.device_array((1, point_count, 5), dtype=np.float64)
            device_valid = cuda.device_array((1, point_count), dtype=np.bool_)
            device_block_valid_count = cuda.device_array(point_count, dtype=np.int64)
            point_blocks = (point_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            rod_point_work = master_rod_count * point_count
            rod_point_blocks = (rod_point_work + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            _prepare_state_block_geometry_kernel[point_blocks, _THREADS_PER_BLOCK](
                state_index,
                state_index + 1,
                rows,
                columns,
                device_column,
                device_row,
                device_detector_zero,
                device_detector_column_step,
                device_detector_row_step,
                device_detector_pixel_area,
                device_sample_from_lab,
                device_sample_from_local,
                device_ray_origin,
                device_ki_film,
                device_state_real,
                device_state_complex,
                device_f0_parameters,
                device_q_geometry,
                device_point_factor,
                device_valid,
            )
            _count_valid_state_block_kernel[point_blocks, _THREADS_PER_BLOCK](
                1,
                device_valid,
                device_block_valid_count,
                device_valid_count,
            )
            _accumulate_state_block_kernel[rod_point_blocks, _THREADS_PER_BLOCK](
                state_index,
                state_index + 1,
                point_count,
                device_sample_from_local,
                device_state_real,
                device_state_complex,
                device_active_state_rod,
                device_rod_u_bounds,
                device_rod_u_tolerance,
                device_rod_hk_population,
                device_rod_parallel,
                device_rod_inverse,
                device_atom_offset,
                device_atom_properties,
                device_rod_inplane,
                packed.layers,
                gaussian_sigma_rad,
                gaussian_probability,
                gaussian_normalization,
                lorentzian_probability,
                lorentzian_rho,
                lorentzian_one_minus_rho,
                lorentzian_numerator,
                device_q_geometry,
                device_point_factor,
                device_valid,
                device_block_valid_count,
                device_per_rod_density,
                device_per_rod_caustic,
            )
            device_density = cuda.device_array(point_count, dtype=np.float64)
            device_caustic = cuda.device_array(point_count, dtype=np.bool_)
            _reduce_selected_rod_group_kernel[point_blocks, _THREADS_PER_BLOCK](
                device_per_rod_density,
                device_per_rod_caustic,
                device_group_index,
                device_group_mask,
                device_density,
                device_caustic,
            )
            sorted_density[start:stop] = device_density.copy_to_host()
            sorted_caustic[start:stop] = device_caustic.copy_to_host()
            sorted_valid[start:stop] = device_valid_count.copy_to_host() == 1
    if np.any(np.isnan(sorted_density) | np.isneginf(sorted_density)):
        raise FloatingPointError("CUDA detector evaluation produced undefined physical intensity")
    if order is None:
        return sorted_density, sorted_caustic, sorted_valid, device_name
    inverse_order = np.empty(order.shape, dtype=np.int64)
    inverse_order[order] = np.arange(order.size, dtype=np.int64)
    return (
        sorted_density[inverse_order],
        sorted_caustic[inverse_order],
        sorted_valid[inverse_order],
        device_name,
    )


def evaluate_source_averaged_all_roots_cuda(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
    column_px: FloatArray,
    row_px: FloatArray,
    *,
    detector_shape_rc: tuple[int, int],
    master_rod_count: int,
    coordinate_chunk_size: int = _DEFAULT_COORDINATE_CHUNK_SIZE,
) -> tuple[FloatArray, BoolArray, IntArray, str]:
    """Return the detailed physical-rod field for proof and diagnostics."""

    return _evaluate_source_averaged_all_roots_cuda(
        evaluator_blocks,
        column_px,
        row_px,
        detector_shape_rc=detector_shape_rc,
        master_rod_count=master_rod_count,
        return_per_rod=True,
        coordinate_chunk_size=coordinate_chunk_size,
    )


def evaluate_source_averaged_density_all_roots_cuda(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
    column_px: FloatArray,
    row_px: FloatArray,
    *,
    detector_shape_rc: tuple[int, int],
    master_rod_count: int,
    coordinate_chunk_size: int = _DEFAULT_COORDINATE_CHUNK_SIZE,
) -> tuple[FloatArray, BoolArray, IntArray, str]:
    """Return the rod-reduced continuous density before any pixel integration."""

    return _evaluate_source_averaged_all_roots_cuda(
        evaluator_blocks,
        column_px,
        row_px,
        detector_shape_rc=detector_shape_rc,
        master_rod_count=master_rod_count,
        return_per_rod=False,
        coordinate_chunk_size=coordinate_chunk_size,
    )


__all__ = [
    "evaluate_selected_source_rod_groups_all_roots_cuda",
    "evaluate_source_averaged_all_roots_cuda",
    "evaluate_source_averaged_density_all_roots_cuda",
    "require_cuda_available",
]
