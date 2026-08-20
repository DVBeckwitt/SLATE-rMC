"""Persistent CUDA execution for forward Monte Carlo detector-pixel mass."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import cuda
from numpy.typing import NDArray

from rasim_next.geometry.detector import _DETECTOR_INCIDENCE_COSINE_TOL
from rasim_next.pipeline._continuous_detector_cuda import (
    _empirical_parratt_strength_A2,
    _finite_stack_strength_A2,
    _pack_source_average,
    _positive_normal_root,
    require_cuda_available,
)
from rasim_next.pipeline._continuous_detector_kernel import _DetectorProjection

FloatArray = NDArray[np.float64]

_FLOAT_EPS = float(np.finfo(np.float64).eps)
_THREADS_PER_BLOCK = 128
_MAXIMUM_STATE_CHUNK_SIZE = 1_000
_TARGET_STATE_DRAW_WORK = 4_096


def _flatten_evaluators(evaluator_blocks: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    evaluators = tuple(indexed for block in evaluator_blocks for indexed in block)
    if not evaluators:
        raise ValueError("forward CUDA Monte Carlo requires at least one compiled source state")
    return evaluators


def _pack_forward_projection(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
) -> _DetectorProjection:
    indexed_evaluators = _flatten_evaluators(evaluator_blocks)
    first = indexed_evaluators[0].evaluator
    column_row_covectors = np.ascontiguousarray(
        first._detector_column_row_covectors_sample_per_m,
        dtype=np.float64,
    )
    detector_normal = np.ascontiguousarray(first._detector_normal_sample, dtype=np.float64)
    ray_origin_column_row = np.empty((len(indexed_evaluators), 2), dtype=np.float64)
    ray_origin_normal = np.empty(len(indexed_evaluators), dtype=np.float64)
    for state_index, indexed in enumerate(indexed_evaluators):
        evaluator = indexed.evaluator
        if not np.array_equal(
            evaluator._detector_column_row_covectors_sample_per_m,
            column_row_covectors,
        ):
            raise ValueError("compiled source states disagree on detector projection covectors")
        if not np.array_equal(evaluator._detector_normal_sample, detector_normal):
            raise ValueError("compiled source states disagree on the detector normal")
        ray_origin_column_row[state_index] = evaluator._ray_origin_detector_column_row_px
        ray_origin_normal[state_index] = evaluator._ray_origin_detector_normal_m
    return _DetectorProjection(
        column_row_covectors,
        detector_normal,
        ray_origin_column_row,
        ray_origin_normal,
    )


def _pack_forward_geometry(
    evaluator_blocks: tuple[tuple[Any, ...], ...],
) -> tuple[FloatArray, ...]:
    """Pack only the fields allowed to change during a validated geometry rebind."""

    indexed_evaluators = _flatten_evaluators(evaluator_blocks)
    evaluators = tuple(indexed.evaluator for indexed in indexed_evaluators)
    first_evaluator = evaluators[0]
    column_row_covectors = first_evaluator._detector_column_row_covectors_sample_per_m
    detector_normal = first_evaluator._detector_normal_sample
    if any(
        evaluator._detector_column_row_covectors_sample_per_m is not column_row_covectors
        and not np.array_equal(
            evaluator._detector_column_row_covectors_sample_per_m,
            column_row_covectors,
        )
        for evaluator in evaluators[1:]
    ) or any(
        evaluator._detector_normal_sample is not detector_normal
        and not np.array_equal(evaluator._detector_normal_sample, detector_normal)
        for evaluator in evaluators[1:]
    ):
        raise ValueError("compiled source states disagree on detector projection")
    projection = (
        np.ascontiguousarray(column_row_covectors),
        np.ascontiguousarray(detector_normal),
        np.asarray(
            [evaluator._ray_origin_detector_column_row_px for evaluator in evaluators],
            dtype=np.float64,
        ),
        np.fromiter(
            (evaluator._ray_origin_detector_normal_m for evaluator in evaluators),
            dtype=np.float64,
            count=len(evaluators),
        ),
    )
    states = tuple(evaluator._state for evaluator in evaluators)
    first = states[0]
    sample_from_local = np.ascontiguousarray(first.sample_from_local, dtype=np.float64)
    if any(
        state.sample_from_local is not first.sample_from_local
        and not np.array_equal(state.sample_from_local, sample_from_local)
        for state in states[1:]
    ):
        raise ValueError("compiled source states disagree on sample/local rotation")
    ki_film = np.asarray(
        [state.ki_film_sample_Ainv for state in states],
        dtype=np.float64,
    )
    state_real = np.asarray(
        [
            (
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
                4096.0
                * _FLOAT_EPS
                * max(
                    math.sqrt(float(state.ki_film_sample_Ainv @ state.ki_film_sample_Ainv)),
                    1.0,
                ),
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
            for state in states
        ],
        dtype=np.float64,
    )
    state_complex = np.asarray(
        [
            (
                state.refractive_index,
                state.entrance_amplitude,
                state.anomalous_factor_e[0],
                state.anomalous_factor_e[1],
                state.specular_substrate_refractive_index,
            )
            for state in states
        ],
        dtype=np.complex128,
    )
    return (*projection, sample_from_local, ki_film, state_real, state_complex)


@cuda.jit(device=True, inline=True)
def _forward_root_pixel(
    state_index: int,
    rod_index: int,
    detector_rows: int,
    detector_columns: int,
    u_Ainv: float,
    kf_film_x: float,
    kf_film_y: float,
    kf_film_z: float,
    coarea_jacobian: float,
    detector_column_row_covectors_sample_per_m: Any,
    detector_normal_sample: Any,
    ray_origin_detector_column_row_px: Any,
    ray_origin_detector_normal_m: Any,
    ki_film_sample_Ainv: Any,
    state_real: Any,
    state_complex: Any,
    rod_hk_population: Any,
    rod_parallel_local_Ainv: Any,
    rod_inverse_constants: Any,
    atom_fractional_offset: Any,
    atom_occupancy_element: Any,
    rod_atom_inplane_factor: Any,
    f0_parameters: Any,
    layers: int,
) -> tuple[int, float, bool]:
    if kf_film_z <= 0.0 or coarea_jacobian <= 0.0:
        return -1, 0.0, False
    internal_k_Ainv = state_real[state_index, 0]
    kf_norm = math.sqrt(kf_film_x * kf_film_x + kf_film_y * kf_film_y + kf_film_z * kf_film_z)
    residual_scale = internal_k_Ainv if internal_k_Ainv > 1.0 else 1.0
    residual_limit = 512.0 * _FLOAT_EPS * residual_scale
    if abs(kf_norm - internal_k_Ainv) > residual_limit:
        return -1, 0.0, True

    air_k0_Ainv = state_real[state_index, 1]
    parallel_squared = kf_film_x * kf_film_x + kf_film_y * kf_film_y
    air_normal_squared = air_k0_Ainv * air_k0_Ainv - parallel_squared
    critical_scale = air_k0_Ainv * air_k0_Ainv
    if parallel_squared > critical_scale:
        critical_scale = parallel_squared
    if critical_scale < 1.0:
        critical_scale = 1.0
    critical_tolerance = 16.0 * _FLOAT_EPS * critical_scale
    if air_normal_squared < -critical_tolerance:
        return -1, 0.0, False
    kf_air_z = math.sqrt(air_normal_squared if air_normal_squared > 0.0 else 0.0)

    refractive_index = state_complex[state_index, 0]
    kz_film = _positive_normal_root((refractive_index * air_k0_Ainv) ** 2 - parallel_squared)
    denominator = kz_film + complex(kf_air_z, 0.0)
    if denominator == 0.0:
        return -1, 0.0, False
    exit_amplitude = 2.0 * kz_film / denominator
    exponent = (
        2.0
        * (state_real[state_index, 2] + (kz_film.imag if kz_film.imag > 0.0 else 0.0))
        * state_real[state_index, 3]
    )
    attenuation = 1.0 if exponent == 0.0 else -math.expm1(-exponent) / exponent
    entrance_amplitude = state_complex[state_index, 1]
    entrance_power = entrance_amplitude.real**2 + entrance_amplitude.imag**2
    optical_weight = (
        entrance_power * (exit_amplitude.real**2 + exit_amplitude.imag**2) * attenuation
    )
    source_phase_weight = state_real[state_index, 4]
    if optical_weight <= 0.0 or source_phase_weight <= 0.0:
        return -1, 0.0, False

    direction_sample_x = kf_film_x / air_k0_Ainv
    direction_sample_y = kf_film_y / air_k0_Ainv
    direction_sample_z = kf_air_z / air_k0_Ainv
    scattering_polarization = 1.0
    if int(state_real[state_index, 17]) == 1:
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
        cosine = (
            ki_film_sample_Ainv[state_index, 0] * direction_sample_x
            + ki_film_sample_Ainv[state_index, 1] * direction_sample_y
            + incident_normal * direction_sample_z
        ) / air_k0_Ainv
        if cosine < -1.0:
            cosine = -1.0
        elif cosine > 1.0:
            cosine = 1.0
        scattering_polarization = 0.5 * (1.0 + cosine * cosine)
    direction_column_per_m = (
        detector_column_row_covectors_sample_per_m[0, 0] * direction_sample_x
        + detector_column_row_covectors_sample_per_m[0, 1] * direction_sample_y
        + detector_column_row_covectors_sample_per_m[0, 2] * direction_sample_z
    )
    direction_row_per_m = (
        detector_column_row_covectors_sample_per_m[1, 0] * direction_sample_x
        + detector_column_row_covectors_sample_per_m[1, 1] * direction_sample_y
        + detector_column_row_covectors_sample_per_m[1, 2] * direction_sample_z
    )
    direction_normal = (
        detector_normal_sample[0] * direction_sample_x
        + detector_normal_sample[1] * direction_sample_y
        + detector_normal_sample[2] * direction_sample_z
    )
    if direction_normal <= _DETECTOR_INCIDENCE_COSINE_TOL:
        return -1, 0.0, False
    ray_distance = -ray_origin_detector_normal_m[state_index] / direction_normal
    if ray_distance <= 0.0:
        return -1, 0.0, False
    if state_real[state_index, 26] != 0.0:
        source_phase_weight *= math.exp(-state_real[state_index, 26] * ray_distance)
    column = (
        ray_origin_detector_column_row_px[state_index, 0] + ray_distance * direction_column_per_m
    )
    row = ray_origin_detector_column_row_px[state_index, 1] + ray_distance * direction_row_per_m
    if column < -0.5 or column > detector_columns - 0.5 or row < -0.5 or row > detector_rows - 0.5:
        return -1, 0.0, False

    q_x = kf_film_x - ki_film_sample_Ainv[state_index, 0]
    q_y = kf_film_y - ki_film_sample_Ainv[state_index, 1]
    q_z = kf_film_z - ki_film_sample_Ainv[state_index, 2]
    q_norm_squared = q_x * q_x + q_y * q_y + q_z * q_z
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
    element_factor_0 = f0_0 + state_complex[state_index, 2]
    element_factor_1 = f0_1 + state_complex[state_index, 3]

    parallel_norm = rod_inverse_constants[rod_index, 1]
    w_value = u_Ainv + rod_parallel_local_Ainv[rod_index, 2]
    u_radial_A2 = state_real[state_index, 9]
    u_normal_A2 = state_real[state_index, 10]
    if u_radial_A2 == u_normal_A2:
        common_damping = math.exp(-0.5 * q_norm_squared * u_radial_A2)
    else:
        common_damping = math.exp(
            -0.5 * (u_radial_A2 * parallel_norm * parallel_norm + u_normal_A2 * w_value * w_value)
        )
    strength = _finite_stack_strength_A2(
        rod_index,
        u_Ainv / state_real[state_index, 5],
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
        state_real[state_index, 11],
        rod_hk_population,
        state_real[state_index, 12],
    )
    if (
        int(state_real[state_index, 18]) == 2
        and rod_hk_population[rod_index, 0] == 0.0
        and rod_hk_population[rod_index, 1] == 0.0
    ):
        q_norm = math.sqrt(q_norm_squared)
        incident_air_normal_squared = (
            state_real[state_index, 1] ** 2
            - ki_film_sample_Ainv[state_index, 0] ** 2
            - ki_film_sample_Ainv[state_index, 1] ** 2
        )
        if incident_air_normal_squared < 0.0:
            incident_air_normal_squared = 0.0
        incident_air_normal = math.sqrt(incident_air_normal_squared)
        if ki_film_sample_Ainv[state_index, 2] < 0.0:
            incident_air_normal = -incident_air_normal
        external_qz = 0.0
        if q_norm > 0.0:
            external_qz = abs(
                (q_x * q_x + q_y * q_y + q_z * (kf_air_z - incident_air_normal)) / q_norm
            )
        strength = _empirical_parratt_strength_A2(
            strength,
            external_qz,
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
    event_intensity_envelope = math.exp(
        -state_real[state_index, 15] * (q_x * q_x + q_y * q_y)
        - state_real[state_index, 16] * q_z * q_z
    )
    importance_weight = (
        source_phase_weight
        * rod_hk_population[rod_index, 2]
        * strength
        * coarea_jacobian
        * optical_weight
        * scattering_polarization
        * event_intensity_envelope
    )
    if not math.isfinite(importance_weight):
        return -1, 0.0, True
    if importance_weight <= 0.0:
        return -1, 0.0, False
    pixel_column = math.floor(column + 0.5)
    pixel_row = math.floor(row + 0.5)
    if pixel_column >= detector_columns:
        pixel_column = detector_columns - 1
    if pixel_row >= detector_rows:
        pixel_row = detector_rows - 1
    return pixel_row * detector_columns + pixel_column, importance_weight, False


@cuda.jit(fastmath=False)
def _accumulate_forward_monte_carlo_kernel(
    state_start: int,
    state_stop: int,
    draw_offset: int,
    detector_rows: int,
    detector_columns: int,
    alpha_rad: Any,
    beta_rad: Any,
    detector_column_row_covectors_sample_per_m: Any,
    detector_normal_sample: Any,
    ray_origin_detector_column_row_px: Any,
    ray_origin_detector_normal_m: Any,
    sample_from_local: Any,
    ki_film_sample_Ainv: Any,
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
    f0_parameters: Any,
    layers: int,
    raw_image_A2: Any,
    replicate_total_mass_A2: Any,
    visible_hit_count: Any,
    maximum_root_weight_A2: Any,
    numeric_failure: Any,
) -> None:
    linear_index = cuda.grid(1)
    draw_count = alpha_rad.shape[1]
    local_state_count = state_stop - state_start
    if linear_index >= local_state_count * draw_count:
        return
    local_state = linear_index // draw_count
    local_draw = linear_index - local_state * draw_count
    state_index = state_start + local_state
    alpha = alpha_rad[state_index, local_draw]
    beta = beta_rad[state_index, local_draw]
    sin_alpha = math.sin(alpha)
    cos_alpha = math.cos(alpha)
    sin_beta = math.sin(beta)
    cos_beta = math.cos(beta)
    direction_local_x = sin_alpha * cos_beta
    direction_local_y = sin_alpha * sin_beta
    direction_local_z = cos_alpha
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
    direction_norm_squared = (
        direction_sample_x * direction_sample_x
        + direction_sample_y * direction_sample_y
        + direction_sample_z * direction_sample_z
    )
    direction_norm = math.sqrt(direction_norm_squared)
    incident_dot_direction = (
        ki_film_sample_Ainv[state_index, 0] * direction_sample_x
        + ki_film_sample_Ainv[state_index, 1] * direction_sample_y
        + ki_film_sample_Ainv[state_index, 2] * direction_sample_z
    )
    incident_parallel = incident_dot_direction / direction_norm_squared
    local_replicate_total = 0.0
    local_visible_hit_count = 0
    local_maximum_root_weight = 0.0

    for rod_index in range(rod_hk_population.shape[0]):
        if not active_state_rod[state_index, rod_index]:
            continue
        a = rod_parallel_local_Ainv[rod_index, 0]
        b = rod_parallel_local_Ainv[rod_index, 1]
        c0 = rod_parallel_local_Ainv[rod_index, 2]
        x0 = a * cos_alpha + c0 * sin_alpha
        q0_local_x = x0 * cos_beta - b * sin_beta
        q0_local_y = x0 * sin_beta + b * cos_beta
        q0_local_z = -a * sin_alpha + c0 * cos_alpha
        q0_sample_x = (
            sample_from_local[0, 0] * q0_local_x
            + sample_from_local[0, 1] * q0_local_y
            + sample_from_local[0, 2] * q0_local_z
        )
        q0_sample_y = (
            sample_from_local[1, 0] * q0_local_x
            + sample_from_local[1, 1] * q0_local_y
            + sample_from_local[1, 2] * q0_local_z
        )
        q0_sample_z = (
            sample_from_local[2, 0] * q0_local_x
            + sample_from_local[2, 1] * q0_local_y
            + sample_from_local[2, 2] * q0_local_z
        )
        q0_dot_direction = (
            q0_sample_x * direction_sample_x
            + q0_sample_y * direction_sample_y
            + q0_sample_z * direction_sample_z
        )
        q0_parallel = q0_dot_direction / direction_norm_squared
        lower_u = rod_u_bounds_Ainv[state_index, rod_index, 0]
        upper_u = rod_u_bounds_Ainv[state_index, rod_index, 1]
        u_tolerance = rod_u_tolerance_Ainv[state_index, rod_index]
        is_m0 = (
            int(rod_hk_population[rod_index, 0]) == 0 and int(rod_hk_population[rod_index, 1]) == 0
        )
        if is_m0:
            if incident_dot_direction == 0.0:
                continue
            u_value = -2.0 * incident_parallel - q0_parallel
            if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                continue
            kf_film_x = ki_film_sample_Ainv[state_index, 0] + u_value * direction_sample_x
            kf_film_y = ki_film_sample_Ainv[state_index, 1] + u_value * direction_sample_y
            kf_film_z = ki_film_sample_Ainv[state_index, 2] + u_value * direction_sample_z
            coarea = state_real[state_index, 0] / abs(incident_dot_direction)
            pixel, weight, failed = _forward_root_pixel(
                state_index,
                rod_index,
                detector_rows,
                detector_columns,
                u_value,
                kf_film_x,
                kf_film_y,
                kf_film_z,
                coarea,
                detector_column_row_covectors_sample_per_m,
                detector_normal_sample,
                ray_origin_detector_column_row_px,
                ray_origin_detector_normal_m,
                ki_film_sample_Ainv,
                state_real,
                state_complex,
                rod_hk_population,
                rod_parallel_local_Ainv,
                rod_inverse_constants,
                atom_fractional_offset,
                atom_occupancy_element,
                rod_atom_inplane_factor,
                f0_parameters,
                layers,
            )
            if failed:
                cuda.atomic.max(numeric_failure, 0, 1)
            if pixel >= 0:
                cuda.atomic.add(raw_image_A2, pixel, weight)
                local_replicate_total += weight
                local_visible_hit_count += 1
                if weight > local_maximum_root_weight:
                    local_maximum_root_weight = weight
            continue

        q0_perpendicular_x = q0_sample_x - q0_parallel * direction_sample_x
        q0_perpendicular_y = q0_sample_y - q0_parallel * direction_sample_y
        q0_perpendicular_z = q0_sample_z - q0_parallel * direction_sample_z
        sphere_perpendicular_x = (
            ki_film_sample_Ainv[state_index, 0]
            - incident_parallel * direction_sample_x
            + q0_perpendicular_x
        )
        sphere_perpendicular_y = (
            ki_film_sample_Ainv[state_index, 1]
            - incident_parallel * direction_sample_y
            + q0_perpendicular_y
        )
        sphere_perpendicular_z = (
            ki_film_sample_Ainv[state_index, 2]
            - incident_parallel * direction_sample_z
            + q0_perpendicular_z
        )
        perpendicular_squared = (
            sphere_perpendicular_x * sphere_perpendicular_x
            + sphere_perpendicular_y * sphere_perpendicular_y
            + sphere_perpendicular_z * sphere_perpendicular_z
        )
        discriminant = (
            state_real[state_index, 0] * state_real[state_index, 0] - perpendicular_squared
        )
        if discriminant <= 0.0:
            continue
        sqrt_discriminant = math.sqrt(discriminant)
        root_coordinate_magnitude = sqrt_discriminant / direction_norm
        parallel_offset = incident_parallel + q0_parallel
        coarea = state_real[state_index, 0] / (sqrt_discriminant * direction_norm)
        for root_slot in range(2):
            signed_root = (
                -root_coordinate_magnitude if root_slot == 0 else root_coordinate_magnitude
            )
            u_value = -parallel_offset + signed_root
            if u_value < lower_u - u_tolerance or u_value > upper_u + u_tolerance:
                continue
            kf_film_x = sphere_perpendicular_x + signed_root * direction_sample_x
            kf_film_y = sphere_perpendicular_y + signed_root * direction_sample_y
            kf_film_z = sphere_perpendicular_z + signed_root * direction_sample_z
            pixel, weight, failed = _forward_root_pixel(
                state_index,
                rod_index,
                detector_rows,
                detector_columns,
                u_value,
                kf_film_x,
                kf_film_y,
                kf_film_z,
                coarea,
                detector_column_row_covectors_sample_per_m,
                detector_normal_sample,
                ray_origin_detector_column_row_px,
                ray_origin_detector_normal_m,
                ki_film_sample_Ainv,
                state_real,
                state_complex,
                rod_hk_population,
                rod_parallel_local_Ainv,
                rod_inverse_constants,
                atom_fractional_offset,
                atom_occupancy_element,
                rod_atom_inplane_factor,
                f0_parameters,
                layers,
            )
            if failed:
                cuda.atomic.max(numeric_failure, 0, 1)
            if pixel >= 0:
                cuda.atomic.add(raw_image_A2, pixel, weight)
                local_replicate_total += weight
                local_visible_hit_count += 1
                if weight > local_maximum_root_weight:
                    local_maximum_root_weight = weight

    if local_replicate_total > 0.0:
        cuda.atomic.add(
            replicate_total_mass_A2,
            draw_offset + local_draw,
            local_replicate_total,
        )
    if local_visible_hit_count:
        cuda.atomic.add(visible_hit_count, 0, local_visible_hit_count)
    if local_maximum_root_weight > 0.0:
        cuda.atomic.max(maximum_root_weight_A2, 0, local_maximum_root_weight)


@cuda.jit(fastmath=False)
def _zero_forward_image_kernel(raw_image_A2: Any) -> None:
    index = cuda.grid(1)
    if index < raw_image_A2.size:
        raw_image_A2[index] = 0.0


@cuda.jit(fastmath=False)
def _normalize_forward_image_kernel(
    raw_image_A2: Any,
    inverse_draw_count: float,
    presentation_image_A2: Any,
) -> None:
    index = cuda.grid(1)
    if index < raw_image_A2.size:
        presentation_image_A2[index] = raw_image_A2[index] * inverse_draw_count


class CudaForwardMonteCarloWorkspace:
    """Long-lived device arrays and scratch for one compiled detector topology."""

    def __init__(
        self,
        evaluator_blocks: tuple[tuple[Any, ...], ...],
        *,
        detector_shape_rc: tuple[int, int],
        master_rod_count: int,
    ) -> None:
        self.device_name = require_cuda_available()
        packed = _pack_source_average(
            evaluator_blocks,
            detector_shape_rc=detector_shape_rc,
            master_rod_count=master_rod_count,
        )
        projection = _pack_forward_projection(evaluator_blocks)
        self._detector_shape_rc = tuple(detector_shape_rc)
        self._master_rod_count = int(master_rod_count)
        self._state_count = packed.state_real.shape[0]
        indexed_evaluators = _flatten_evaluators(evaluator_blocks)
        self._state_order = tuple(
            int(indexed.incident_state_index) for indexed in indexed_evaluators
        )
        self._master_rod_topology = tuple(
            np.asarray(indexed.master_rod_index, dtype=np.int64) for indexed in indexed_evaluators
        )
        projection_host_arrays = tuple(projection)
        transport_host_arrays = (
            packed.sample_from_local,
            packed.ki_film_sample_Ainv,
            packed.state_real,
            packed.state_complex,
        )
        active_projection = tuple(cuda.to_device(array) for array in projection_host_arrays)
        shadow_projection = tuple(
            cuda.device_array(array.shape, dtype=array.dtype) for array in projection_host_arrays
        )
        active_transport = tuple(cuda.to_device(array) for array in transport_host_arrays)
        shadow_transport = tuple(
            cuda.device_array(array.shape, dtype=array.dtype) for array in transport_host_arrays
        )
        self._projection_device_buffers = (active_projection, shadow_projection)
        self._transport_device_buffers = (active_transport, shadow_transport)
        self._active_projection_buffer = 0
        self._active_transport_buffer = 0
        (
            self._device_detector_covectors,
            self._device_detector_normal,
            self._device_ray_origin_column_row,
            self._device_ray_origin_normal,
        ) = active_projection
        (
            self._device_sample_from_local,
            self._device_ki_film,
            self._device_state_real,
            self._device_state_complex,
        ) = active_transport
        self._device_active_state_rod = cuda.to_device(packed.active_state_rod)
        self._device_rod_u_bounds = cuda.to_device(packed.rod_u_bounds_Ainv)
        self._device_rod_u_tolerance = cuda.to_device(packed.rod_u_tolerance_Ainv)
        self._device_rod_hk_population = cuda.to_device(packed.rod_hk_population)
        self._device_rod_parallel = cuda.to_device(packed.rod_parallel_local_Ainv)
        self._device_rod_inverse = cuda.to_device(packed.rod_inverse_constants)
        self._device_atom_offset = cuda.to_device(packed.atom_fractional_offset)
        self._device_atom_properties = cuda.to_device(packed.atom_occupancy_element)
        self._device_rod_inplane = cuda.to_device(packed.rod_atom_inplane_factor)
        self._device_f0_parameters = cuda.to_device(packed.f0_parameters)
        self._layers = packed.layers
        pixel_count = detector_shape_rc[0] * detector_shape_rc[1]
        self._device_raw_image = cuda.to_device(np.zeros(pixel_count, dtype=np.float64))
        self._device_presentation_image = cuda.device_array(pixel_count, dtype=np.float32)
        self._host_presentation_image = cuda.pinned_array(pixel_count, dtype=np.float32)
        self._device_replicate_total = cuda.to_device(np.empty(0, dtype=np.float64))
        self._device_visible_hit_count = cuda.to_device(np.zeros(1, dtype=np.int64))
        self._device_maximum_root_weight = cuda.to_device(np.zeros(1, dtype=np.float64))
        self._device_numeric_failure = cuda.to_device(np.zeros(1, dtype=np.int32))
        self._draw_count = 0

    @property
    def state_count(self) -> int:
        return self._state_count

    def _enqueue_reset(self) -> None:
        blocks = (self._device_raw_image.size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _zero_forward_image_kernel[blocks, _THREADS_PER_BLOCK](self._device_raw_image)
        self._device_replicate_total = cuda.to_device(np.empty(0, dtype=np.float64))
        self._device_visible_hit_count.copy_to_device(np.zeros(1, dtype=np.int64))
        self._device_maximum_root_weight.copy_to_device(np.zeros(1, dtype=np.float64))
        self._device_numeric_failure.copy_to_device(np.zeros(1, dtype=np.int32))

    def reset(self) -> None:
        self._enqueue_reset()
        cuda.synchronize()
        self._draw_count = 0

    def rebind_geometry(self, evaluator_blocks: tuple[tuple[Any, ...], ...]) -> None:
        """Update only geometry-dependent arrays for an unchanged source/rod topology."""

        indexed_evaluators = _flatten_evaluators(evaluator_blocks)
        if len(indexed_evaluators) != self._state_count:
            raise ValueError("CUDA geometry rebind requires unchanged active source-state count")
        if tuple(int(indexed.incident_state_index) for indexed in indexed_evaluators) != (
            self._state_order
        ) or any(
            not np.array_equal(indexed.master_rod_index, expected)
            for indexed, expected in zip(
                indexed_evaluators,
                self._master_rod_topology,
                strict=True,
            )
        ):
            raise ValueError("CUDA geometry rebind changed source/rod topology")
        geometry_host_arrays = _pack_forward_geometry(evaluator_blocks)
        projection_host_arrays = geometry_host_arrays[:4]
        transport_host_arrays = geometry_host_arrays[4:]
        next_projection_index = 1 - self._active_projection_buffer
        next_transport_index = 1 - self._active_transport_buffer
        staged_projection = self._projection_device_buffers[next_projection_index]
        staged_transport = self._transport_device_buffers[next_transport_index]
        if any(
            device_array.shape != host_array.shape or device_array.dtype != host_array.dtype
            for device_array, host_array in zip(
                (*staged_projection, *staged_transport),
                (*projection_host_arrays, *transport_host_arrays),
                strict=True,
            )
        ):
            raise ValueError("CUDA geometry rebind changed a dynamic array layout")
        for device_array, host_array in zip(
            (*staged_projection, *staged_transport),
            (*projection_host_arrays, *transport_host_arrays),
            strict=True,
        ):
            device_array.copy_to_device(host_array)
        self._enqueue_reset()
        cuda.synchronize()
        (
            self._device_detector_covectors,
            self._device_detector_normal,
            self._device_ray_origin_column_row,
            self._device_ray_origin_normal,
        ) = staged_projection
        (
            self._device_sample_from_local,
            self._device_ki_film,
            self._device_state_real,
            self._device_state_complex,
        ) = staged_transport
        self._active_projection_buffer = next_projection_index
        self._active_transport_buffer = next_transport_index
        self._draw_count = 0

    def rebind_detector_projection(self, projection: _DetectorProjection) -> None:
        """Update only ray-to-pixel projection arrays and reset the accumulator."""

        if not isinstance(projection, _DetectorProjection):
            raise TypeError("projection must be a _DetectorProjection")
        host_arrays = tuple(projection)
        next_buffer_index = 1 - self._active_projection_buffer
        staged_projection = self._projection_device_buffers[next_buffer_index]
        if any(
            device_array.shape != host_array.shape or device_array.dtype != host_array.dtype
            for device_array, host_array in zip(
                staged_projection,
                host_arrays,
                strict=True,
            )
        ):
            raise ValueError("CUDA detector projection changed an array layout")
        for device_array, host_array in zip(
            staged_projection,
            host_arrays,
            strict=True,
        ):
            device_array.copy_to_device(host_array)
        self._enqueue_reset()
        cuda.synchronize()
        (
            self._device_detector_covectors,
            self._device_detector_normal,
            self._device_ray_origin_column_row,
            self._device_ray_origin_normal,
        ) = staged_projection
        self._active_projection_buffer = next_buffer_index
        self._draw_count = 0

    def advance(
        self,
        alpha_rad: FloatArray,
        beta_rad: FloatArray,
        *,
        draw_start: int,
        draw_stop: int,
        cancel_requested: Any = None,
        state_chunk_size: int | None = None,
    ) -> bool:
        alpha = np.ascontiguousarray(alpha_rad, dtype=np.float64)
        beta = np.ascontiguousarray(beta_rad, dtype=np.float64)
        expected_shape = (self._state_count, draw_stop - draw_start)
        if alpha.shape != expected_shape or beta.shape != expected_shape:
            raise ValueError("CUDA latent arrays do not match state and draw ranges")
        if draw_start != self._draw_count or draw_stop <= draw_start:
            raise ValueError("CUDA progressive draw range is not contiguous")
        draw_count = draw_stop - draw_start
        if state_chunk_size is None:
            state_chunk_size = min(
                self._state_count,
                _MAXIMUM_STATE_CHUNK_SIZE,
                max(1, _TARGET_STATE_DRAW_WORK // draw_count),
            )
        elif state_chunk_size < 1:
            raise ValueError("state_chunk_size must be positive")
        if cancel_requested is not None and cancel_requested():
            return False

        previous_replicate = self._device_replicate_total.copy_to_host()
        replicate = np.zeros(draw_stop, dtype=np.float64)
        replicate[:draw_start] = previous_replicate
        self._device_replicate_total = cuda.to_device(replicate)
        self._device_numeric_failure.copy_to_device(np.zeros(1, dtype=np.int32))
        device_alpha = cuda.to_device(alpha)
        device_beta = cuda.to_device(beta)
        if cancel_requested is not None and cancel_requested():
            return False
        rows, columns = self._detector_shape_rc
        for state_start in range(0, self._state_count, state_chunk_size):
            state_stop = min(state_start + state_chunk_size, self._state_count)
            work_count = (state_stop - state_start) * draw_count
            blocks = (work_count + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
            _accumulate_forward_monte_carlo_kernel[blocks, _THREADS_PER_BLOCK](
                state_start,
                state_stop,
                draw_start,
                rows,
                columns,
                device_alpha,
                device_beta,
                self._device_detector_covectors,
                self._device_detector_normal,
                self._device_ray_origin_column_row,
                self._device_ray_origin_normal,
                self._device_sample_from_local,
                self._device_ki_film,
                self._device_state_real,
                self._device_state_complex,
                self._device_active_state_rod,
                self._device_rod_u_bounds,
                self._device_rod_u_tolerance,
                self._device_rod_hk_population,
                self._device_rod_parallel,
                self._device_rod_inverse,
                self._device_atom_offset,
                self._device_atom_properties,
                self._device_rod_inplane,
                self._device_f0_parameters,
                self._layers,
                self._device_raw_image,
                self._device_replicate_total,
                self._device_visible_hit_count,
                self._device_maximum_root_weight,
                self._device_numeric_failure,
            )
            cuda.synchronize()
            if cancel_requested is not None and cancel_requested():
                return False
        if int(self._device_numeric_failure.copy_to_host()[0]):
            raise FloatingPointError("invalid CUDA Monte Carlo root numerics")
        self._draw_count = draw_stop
        return True

    def snapshot(self) -> tuple[FloatArray, FloatArray, int, float]:
        if self._draw_count < 1:
            raise RuntimeError("CUDA snapshot requires at least one completed draw")
        return (
            self._device_raw_image.copy_to_host(),
            self._device_replicate_total.copy_to_host(),
            int(self._device_visible_hit_count.copy_to_host()[0]),
            float(self._device_maximum_root_weight.copy_to_host()[0]),
        )

    def presentation_snapshot(
        self,
    ) -> tuple[NDArray[np.float32], FloatArray, int, float]:
        """Return a transient full-native float32 view for display only."""

        if self._draw_count < 1:
            raise RuntimeError("CUDA presentation requires at least one completed draw")
        blocks = (self._device_raw_image.size + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        _normalize_forward_image_kernel[blocks, _THREADS_PER_BLOCK](
            self._device_raw_image,
            1.0 / self._draw_count,
            self._device_presentation_image,
        )
        self._device_presentation_image.copy_to_host(self._host_presentation_image)
        return (
            self._host_presentation_image.reshape(self._detector_shape_rc),
            self._device_replicate_total.copy_to_host(),
            int(self._device_visible_hit_count.copy_to_host()[0]),
            float(self._device_maximum_root_weight.copy_to_host()[0]),
        )


__all__ = ["CudaForwardMonteCarloWorkspace"]
