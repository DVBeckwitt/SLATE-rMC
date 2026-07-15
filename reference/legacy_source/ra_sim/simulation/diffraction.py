"""Core diffraction routines used by the simulator."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit, types
from numba.typed import List
from ra_sim.config.loader import get_instrument_config
from math import sin, cos, sqrt, pi, exp, acos
from ra_sim.simulation.intersection_cache_schema import (
    CACHE_COL_BEST_SAMPLE_INDEX,
    CACHE_COL_SOURCE_ROW_INDEX,
    CACHE_COL_SOURCE_TABLE_INDEX,
    CURRENT_DETECTOR_CACHE_WIDTH,
    HIT_ROW_COL_BEST_SAMPLE_INDEX,
    HIT_ROW_COL_BEAM_X_OFFSET,
    HIT_ROW_COL_BEAM_Y_OFFSET,
    HIT_ROW_COL_DETECTOR_COL,
    HIT_ROW_COL_DETECTOR_ROW,
    HIT_ROW_COL_H,
    HIT_ROW_COL_INTENSITY,
    HIT_ROW_COL_K,
    HIT_ROW_COL_L,
    HIT_ROW_COL_PHI,
    HIT_ROW_COL_PHI_OFFSET,
    HIT_ROW_COL_SOURCE_ROW_INDEX,
    HIT_ROW_COL_SOURCE_TABLE_INDEX,
    HIT_ROW_COL_THETA_OFFSET,
    HIT_ROW_COL_WAVELENGTH_OFFSET,
    HIT_ROW_WITH_CONTEXT_WIDTH,
    HIT_ROW_WITH_PROVENANCE_WIDTH,
    cache_table_to_hit_table,
)

from ra_sim.utils.calculations import (
    SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD,
    complex_sqrt,
)
from ra_sim.utils.parallel import resolve_weighted_event_worker_count



# solve_q hot-loop constants
DEFAULT_SOLVE_Q_STEPS = 1000
MIN_SOLVE_Q_STEPS = 32
MAX_SOLVE_Q_STEPS = 8192
_DEFAULT_SOLVE_Q_DTHETA = (2.0 * np.pi) / DEFAULT_SOLVE_Q_STEPS
_DEFAULT_SOLVE_Q_COS = np.cos(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_DEFAULT_SOLVE_Q_SIN = np.sin(
    _DEFAULT_SOLVE_Q_DTHETA * np.arange(DEFAULT_SOLVE_Q_STEPS, dtype=np.float64)
)
_DEFAULT_SOLVE_Q_COS.setflags(write=False)
_DEFAULT_SOLVE_Q_SIN.setflags(write=False)
DEFAULT_SOLVE_Q_BASE_INTERVALS = 48
MIN_SOLVE_Q_BASE_INTERVALS = 8
DEFAULT_SOLVE_Q_REL_TOL = 5.0e-4
MIN_SOLVE_Q_REL_TOL = 1.0e-6
MAX_SOLVE_Q_REL_TOL = 5.0e-2
SOLVE_Q_MODE_UNIFORM = 0
SOLVE_Q_MODE_ADAPTIVE = 1
DEFAULT_SOLVE_Q_MODE = SOLVE_Q_MODE_UNIFORM
_INTENSITY_CUTOFF = float(np.exp(-100.0))
_SOLVE_Q_ABS_ERR_TOL = 1.0e-20
_LOCAL_ARC_MAX_ROOTS = 4
_LOCAL_ARC_MAX_WINDOWS = 8
_LOCAL_ARC_MIN_SEARCH_STEPS = 64
_LOCAL_ARC_MAX_SEARCH_STEPS = 256
_LOCAL_ARC_MIN_STEPS_PER_WINDOW = 8
_LOCAL_ARC_GAUSS_SIGMAS = 10.0
_LOCAL_ARC_LORENTZ_GAMMAS = 24.0
_LOCAL_ARC_MIN_DTHETA = 5.0e-4
_LOCAL_ARC_FULL_CIRCLE_THETA_WINDOW = 0.75 * np.pi
_LOCAL_ARC_ROOT_TOL = 1.0e-10
_LOCAL_ARC_BOUNDARY_TOL = 1.0e-7
_LOCAL_PIXEL_CACHE_MIN_CAPACITY = 1024
_LOCAL_PIXEL_CACHE_MAX_CAPACITY = 32768
_LOCAL_PIXEL_CACHE_SCALE = 32
_LOCAL_PIXEL_CACHE_LOAD_NUM = 1
_LOCAL_PIXEL_CACHE_LOAD_DEN = 2
# Per-sample precompute table columns (reflection-invariant terms).
_SAMPLE_COL_VALID = 0
_SAMPLE_COL_I_PLANE_X = 1
_SAMPLE_COL_I_PLANE_Y = 2
_SAMPLE_COL_I_PLANE_Z = 3
_SAMPLE_COL_KX_SCAT = 4
_SAMPLE_COL_KY_SCAT = 5
_SAMPLE_COL_RE_KZ = 6
_SAMPLE_COL_IM_KZ = 7
_SAMPLE_COL_K_SCAT = 8
_SAMPLE_COL_K0 = 9
_SAMPLE_COL_TI2 = 10
_SAMPLE_COL_L_IN = 11
_SAMPLE_COL_N2_REAL = 12
_SAMPLE_COL_SOLVE_Q_REP = 13
_SAMPLE_COL_SOLVE_Q_NEXT = 14
_SAMPLE_COLS = 15


_EMPTY_PROCESS_PEAKS_WEIGHTED_EVENT_STATS = {
    "n_solve_q_calls": 0,
    "n_project_candidate_calls": 0,
    "n_valid_candidates": 0,
    "n_selected_events": 0,
    "n_stored_projected_candidates": 0,
    "candidate_buffer_capacity_max": 0,
    "candidate_buffer_requested_per_worker_bytes": 0,
    "candidate_buffer_requested_total_bytes": 0,
    "candidate_buffer_effective_max_bytes": 0,
    "n_qsets_precomputed": 0,
    "n_qset_lookup_entries": 0,
    "n_qset_reuse_hits": 0,
    "time_qset_index": 0.0,
    "pass2_mass_mismatch_count": 0,
    "pass2_mass_mismatch_max_abs": 0.0,
    "tail_fill_events": 0,
    "time_precompute": 0.0,
    "time_solve_q": 0.0,
    "time_chunk_compute": 0.0,
    "time_project": 0.0,
    "time_select": 0.0,
    "time_emit_cache": 0.0,
    "pass1_total_mass": 0.0,
    "pass2_total_mass": 0.0,
    "n_raw_beam_phases": 0,
    "n_effective_beam_phases": 0,
    "n_exact_solve_q_phase_groups": 0,
    "phase_weight_sum": 0.0,
    "phase_event_count_total": 0,
    "n_hit_table_rows": 0,
    "n_nonempty_hit_tables": 0,
    "n_representative_hit_tables": 0,
    "parallel_backend": "fast_serial",
    "parallel_worker_count": 1,
    "parallel_requested_worker_count": None,
    "parallel_effective_worker_count": 1,
    "parallel_worker_count_source": "auto",
    "hit_table_collection_mode": "quantile_events",
}
_WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES = 5 * 8
_WEIGHTED_EVENT_CANDIDATE_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_Q_DEBUG_MAX_SOLUTIONS_PER_PEAK = 8192


def _weighted_event_candidate_buffer_memory_policy(
    *,
    candidate_capacity,
    worker_count,
    max_bytes,
):
    if not isinstance(candidate_capacity, (int, np.integer)) or candidate_capacity < 0:
        raise ValueError("candidate_capacity must be a nonnegative integer.")
    if not isinstance(worker_count, (int, np.integer)) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer.")
    if not isinstance(max_bytes, (int, np.integer)) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer.")

    candidate_capacity_i = int(candidate_capacity)
    worker_count_i = int(worker_count)
    max_bytes_i = int(max_bytes)
    requested_per_worker_bytes = candidate_capacity_i * _WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES
    requested_total_bytes = requested_per_worker_bytes * worker_count_i

    fits = candidate_capacity_i == 0 or requested_total_bytes <= max_bytes_i
    return {
        "fits": bool(fits),
        "candidate_capacity": int(candidate_capacity_i),
        "worker_count": int(worker_count_i),
        "requested_per_worker_bytes": int(requested_per_worker_bytes),
        "requested_total_bytes": int(requested_total_bytes),
        "max_bytes": int(max_bytes_i),
    }


def _allocate_q_debug_buffers(num_peaks, save_flag):
    if save_flag == 1:
        q_data = np.full(
            (int(num_peaks), DEFAULT_Q_DEBUG_MAX_SOLUTIONS_PER_PEAK, 5),
            np.nan,
            dtype=np.float64,
        )
        q_count = np.zeros(int(num_peaks), dtype=np.int64)
    else:
        q_data = np.zeros((1, 1, 5), dtype=np.float64)
        q_count = np.zeros(1, dtype=np.int64)
    return q_data, q_count


def _process_peaks_weighted_event_stats(**updates):
    stats = dict(_EMPTY_PROCESS_PEAKS_WEIGHTED_EVENT_STATS)
    stats.update(updates)
    return stats


@njit(cache=True)
def wrap_to_pi(x):
    while x <= -pi:
        x += 2.0 * pi
    while x > pi:
        x -= 2.0 * pi
    return x


@njit(cache=True)
def compute_intensity_array_serial(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv):
    """
    Compute the mosaic surface density sigma(theta) on the Bragg sphere for
    each (Qx, Qy, Qz). Uses a pseudo-Voigt in the grazing-angle offset.

    Parameters
    ----------
    Qx, Qy, Qz : array-like
        Coordinates of Q vectors.
    G_vec : length-3 array
        The reciprocal-space vector for the reflection.
    sigma : float
        Gaussian width (rad).
    gamma_pv : float
        Lorentzian half-width at half-maximum (rad).
    eta_pv : float
        Mixing parameter (0=Gaussian, 1=Lorentzian).

    Returns
    -------
    intensities : array-like
        Surface density sigma(theta), same shape as Qx.
    """
    # Unpack G and compute magnitudes
    Gx, Gy, Gz = G_vec[0], G_vec[1], G_vec[2]
    G_mag = np.sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
    if G_mag < 1e-14:
        return np.zeros_like(Qx)

    Qr = np.sqrt(Qx * Qx + Qy * Qy)

    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    # Amplitude factors for normalized 1D profiles
    A_gauss = 1.0 / (sigma_eff * np.sqrt(2.0 * np.pi))
    A_lor = 1.0 / (np.pi * gamma_eff)

    # Reference grazing angle for the reflection
    Gr = np.sqrt(Gx * Gx + Gy * Gy)
    theta0 = np.arctan2(Gz, Gr)

    denom_base = 2.0 * np.pi * G_mag * G_mag

    intensities = np.empty_like(Qx)
    Qz_flat = Qz.ravel()
    Qr_flat = Qr.ravel()
    out_flat = intensities.ravel()

    for i in range(out_flat.size):
        theta = np.arctan2(Qz_flat[i], Qr_flat[i])
        dtheta = wrap_to_pi(theta - theta0)

        gauss_val = A_gauss * np.exp(-0.5 * (dtheta / sigma_eff) ** 2)
        lor_val = A_lor / (1.0 + (dtheta / gamma_eff) ** 2)
        omega = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

        # Keep a geometry normalization that is stable for pseudo-Voigt tails.
        # The previous 1/cos(theta) factor caused pole amplification when eta>0
        # (Lorentzian component), which collapsed Bragg-sphere color scales.
        out_flat[i] = omega / denom_base

    return intensities


compute_intensity_array = compute_intensity_array_serial


# =============================================================================
# 3) INTERSECT_LINE_PLANE, BATCH
# =============================================================================


@njit(cache=True)
def intersect_line_plane(P0, k_vec, P_plane, n_plane):
    """
    Intersect a single ray (start=P0, direction=k_vec) with a plane
    defined by (P_plane, n_plane). Returns the intersection point (ix, iy, iz)
    and a boolean if valid.

    Physical meaning:
      - Used to find where the scattered beam intersects e.g. the sample plane
        or a detector plane in real space.
    """
    denom = k_vec[0] * n_plane[0] + k_vec[1] * n_plane[1] + k_vec[2] * n_plane[2]
    if abs(denom) < 1e-14:
        # The ray is parallel to the plane. If the starting point already lies
        # on the plane (within a tolerance) we treat it as the intersection
        # point so that grazing rays are not discarded.
        dist = (
            (P0[0] - P_plane[0]) * n_plane[0]
            + (P0[1] - P_plane[1]) * n_plane[1]
            + (P0[2] - P_plane[2]) * n_plane[2]
        )
        if abs(dist) < 1e-6:
            return (P0[0], P0[1], P0[2], True)
        return (np.nan, np.nan, np.nan, False)
    num = (
        (P_plane[0] - P0[0]) * n_plane[0]
        + (P_plane[1] - P0[1]) * n_plane[1]
        + (P_plane[2] - P0[2]) * n_plane[2]
    )
    t = num / denom
    # Numerical precision can yield tiny negative values for *t* when the ray
    # should intersect exactly on the plane.  Allow a small tolerance so these
    # near-zero cases are not discarded which previously produced missing bands
    # on the detector when the beam was almost parallel to the sample plane.
    if t < -1e-9:
        return (np.nan, np.nan, np.nan, False)
    if t < 0.0:
        t = 0.0
    ix = P0[0] + t * k_vec[0]
    iy = P0[1] + t * k_vec[1]
    iz = P0[2] + t * k_vec[2]
    return (ix, iy, iz, True)


# ---------- JIT-safe helpers ----------
@njit(cache=True)
def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _kz_branch_decay(arg):
    """Return sqrt(arg) with the physically decaying branch (Im(kz) >= 0)."""
    kz = complex_sqrt(arg)
    if kz.imag < 0.0 or (abs(kz.imag) < 1e-15 and kz.real < 0.0):
        kz = -kz
    return kz


@njit(cache=True)
def _fresnel_t_exact(kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Exact Fresnel transmission amplitude in kz/epsilon form."""
    if s_polarization:
        den = kz_i + kz_j
        if abs(den) < 1e-30:
            return 0j
        return (2.0 * kz_i) / den

    den = eps_j * kz_i + eps_i * kz_j
    if abs(den) < 1e-30:
        return 0j
    return (2.0 * eps_j * kz_i) / den


@njit(cache=True)
def _fresnel_power_t_exact(t_amp, kz_i, kz_j, eps_i, eps_j, s_polarization):
    """Convert exact transmission amplitude to power transmission."""
    abs_t2 = t_amp.real * t_amp.real + t_amp.imag * t_amp.imag
    if abs_t2 <= 0.0:
        return 0.0

    if s_polarization:
        denom = kz_i.real
        if abs(denom) < 1e-30:
            return 0.0
        ratio = kz_j.real / denom
    else:
        den = kz_i / eps_i
        if abs(den) < 1e-30:
            return 0.0
        ratio = ((kz_j / eps_j) / den).real

    out = ratio * abs_t2
    if not np.isfinite(out) or out < 0.0:
        return 0.0
    # For passive interfaces the transmitted power fraction should stay bounded.
    if out > 1.0:
        return 1.0
    return out


@njit(cache=True)
def _sanitize_transmission_power(power):
    """Clamp transmission-like power factors to a stable physical range."""
    if not np.isfinite(power) or power <= 0.0:
        return 0.0
    if power > 1.0:
        return 1.0
    return power


@njit(cache=True)
def _thickness_to_angstrom(depth):
    """Convert positive sample depth from meters to angstrom."""
    if depth <= 0.0:
        return 0.0
    return depth * 1.0e10


@njit(fastmath=True, cache=True)
def _attenuation_depth_angstrom(thickness_angstrom):
    if thickness_angstrom > 0.0:
        return thickness_angstrom
    return 0.0


@njit(fastmath=True, cache=True)
def _exact_external_air_exit_wavevector(k_tx_prime, k_ty_prime, k_tz_prime, k0):
    if (not np.isfinite(k0)) or k0 <= 0.0:
        return False, 0.0, 0.0, 0.0, np.nan

    k0sq = k0 * k0
    kpar2 = k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime
    tol = 1.0e-12 * max(k0sq, 1.0)

    if kpar2 > k0sq + tol:
        return False, 0.0, 0.0, 0.0, np.nan

    if kpar2 > k0sq:
        kpar2 = k0sq

    kr = sqrt(max(kpar2, 0.0))
    kz_abs = sqrt(max(k0sq - kpar2, 0.0))
    kz_air = kz_abs
    if k_tz_prime < 0.0:
        kz_air = -kz_air

    twotheta_t = np.arctan2(kz_air, kr)
    return True, k_tx_prime, k_ty_prime, kz_air, twotheta_t


@njit(cache=True, nogil=True)
def _choose_local_pixel_cache_capacity(n_samp):
    desired = n_samp * _LOCAL_PIXEL_CACHE_SCALE
    if desired < _LOCAL_PIXEL_CACHE_MIN_CAPACITY:
        desired = _LOCAL_PIXEL_CACHE_MIN_CAPACITY
    if desired > _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        desired = _LOCAL_PIXEL_CACHE_MAX_CAPACITY

    capacity = _LOCAL_PIXEL_CACHE_MIN_CAPACITY
    while capacity < desired and capacity < _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        capacity *= 2
    if capacity > _LOCAL_PIXEL_CACHE_MAX_CAPACITY:
        capacity = _LOCAL_PIXEL_CACHE_MAX_CAPACITY
    return capacity


@njit(cache=True, nogil=True)
def _clear_local_pixel_cache(cache_keys, cache_values):
    for i in range(cache_keys.shape[0]):
        cache_keys[i] = -1
        cache_values[i] = 0.0


@njit(cache=True, nogil=True)
def _flush_local_pixel_cache(image, image_size, cache_keys, cache_values):
    for i in range(cache_keys.shape[0]):
        flat_idx = cache_keys[i]
        if flat_idx < 0:
            continue
        row = flat_idx // image_size
        col = flat_idx - row * image_size
        image[row, col] += cache_values[i]
        cache_keys[i] = -1
        cache_values[i] = 0.0
    return 0


@njit(cache=True, nogil=True)
def _insert_local_pixel_cache(cache_keys, cache_values, flat_idx, value):
    capacity = cache_keys.shape[0]
    mask = capacity - 1
    slot = flat_idx & mask
    for _ in range(capacity):
        key = cache_keys[slot]
        if key == -1:
            cache_keys[slot] = flat_idx
            cache_values[slot] = value
            return True, 1
        if key == flat_idx:
            cache_values[slot] += value
            return True, 0
        slot = (slot + 1) & mask
    return False, 0


@njit(cache=True, nogil=True)
def _accumulate_bilinear_cached(
    image_size,
    row_f,
    col_f,
    value,
    cache_keys,
    cache_values,
    entry_count,
    flush_limit,
):
    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)
    contrib_count = 0

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            contrib_count += 1

    if contrib_count == 0:
        return False, False, entry_count
    if entry_count + contrib_count > flush_limit:
        return True, True, entry_count

    new_count = entry_count
    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            ok, added = _insert_local_pixel_cache(
                cache_keys,
                cache_values,
                rr * image_size + cc,
                value * w_row * w_col,
            )
            if not ok:
                return True, True, entry_count
            new_count += added
    return True, False, new_count


# =============================================================================
# 4) solve_q
# =============================================================================


@njit(fastmath=True, cache=True)
def _mosaic_density_scalar(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv):
    Gx = G_vec[0]
    Gy = G_vec[1]
    Gz = G_vec[2]
    G_mag = sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
    if G_mag < 1e-14:
        return 0.0

    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    Qr = sqrt(Qx * Qx + Qy * Qy)
    Gr = sqrt(Gx * Gx + Gy * Gy)
    theta0 = np.arctan2(Gz, Gr)
    theta = np.arctan2(Qz, Qr)
    dtheta = wrap_to_pi(theta - theta0)

    A_gauss = 1.0 / (sigma_eff * sqrt(2.0 * pi))
    A_lor = 1.0 / (pi * gamma_eff)
    gauss_val = A_gauss * exp(-0.5 * (dtheta / sigma_eff) * (dtheta / sigma_eff))
    lor_val = A_lor / (1.0 + (dtheta / gamma_eff) * (dtheta / gamma_eff))
    omega = (1.0 - eta_pv) * gauss_val + eta_pv * lor_val

    denom_base = 2.0 * pi * G_mag * G_mag
    return omega / denom_base


@njit(fastmath=True, cache=True)
def _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z):
    cphi = cos(phi)
    sphi = sin(phi)
    Qx = Ox + circle_r * (cphi * e1x + sphi * e2x)
    Qy = Oy + circle_r * (cphi * e1y + sphi * e2y)
    Qz = Oz + circle_r * (cphi * e1z + sphi * e2z)
    return Qx, Qy, Qz


@njit(fastmath=True, cache=True)
def _circle_density(
    phi,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
):
    Qx, Qy, Qz = _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
    return _mosaic_density_scalar(Qx, Qy, Qz, G_vec, sigma, gamma_pv, eta_pv)


@njit(fastmath=True, cache=True)
def _interval_mass_error(phi_a, phi_b, f_a, f_m, f_b, circle_r):
    dphi = phi_b - phi_a
    mass = circle_r * dphi * (f_a + 4.0 * f_m + f_b) / 6.0
    trap = circle_r * dphi * (f_a + f_b) * 0.5
    err = abs(mass - trap)
    return mass, err


@njit(fastmath=True, cache=True)
def _circle_theta_offset(
    phi,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
):
    Qx, Qy, Qz = _circle_point(phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
    Qr = sqrt(Qx * Qx + Qy * Qy)
    theta = np.arctan2(Qz, Qr)
    return wrap_to_pi(theta - theta0)


@njit(fastmath=True, cache=True)
def _phi_periodic_distance(phi_a, phi_b):
    delta = abs(phi_a - phi_b)
    two_pi = 2.0 * pi
    while delta >= two_pi:
        delta -= two_pi
    if delta > pi:
        delta = two_pi - delta
    return delta


@njit(fastmath=True, cache=True)
def _store_local_arc_root(roots, root_count, phi_root, min_separation):
    if not np.isfinite(phi_root):
        return root_count
    for i in range(root_count):
        if _phi_periodic_distance(phi_root, roots[i]) <= min_separation:
            return root_count
    if root_count < roots.shape[0]:
        roots[root_count] = phi_root
        return root_count + 1
    return root_count


@njit(fastmath=True, cache=True)
def _refine_theta_root(
    phi_a,
    phi_b,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
):
    fa = _circle_theta_offset(phi_a, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)
    fb = _circle_theta_offset(phi_b, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)

    if abs(fa) <= _LOCAL_ARC_ROOT_TOL:
        return phi_a, True
    if abs(fb) <= _LOCAL_ARC_ROOT_TOL:
        return phi_b, True
    if fa * fb > 0.0:
        return 0.5 * (phi_a + phi_b), False

    left = phi_a
    right = phi_b
    for _ in range(48):
        mid = 0.5 * (left + right)
        fm = _circle_theta_offset(mid, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0)
        if abs(fm) <= _LOCAL_ARC_ROOT_TOL:
            return mid, True
        if fa * fm <= 0.0:
            right = mid
            fb = fm
        else:
            left = mid
            fa = fm
    return 0.5 * (left + right), True


@njit(fastmath=True, cache=True)
def _refine_theta_boundary(
    phi_inside,
    phi_outside,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    theta0,
    theta_limit,
):
    left = phi_inside
    right = phi_outside
    f_left = (
        abs(_circle_theta_offset(left, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0))
        - theta_limit
    )
    f_right = (
        abs(_circle_theta_offset(right, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0))
        - theta_limit
    )

    if f_left > 0.0:
        return left
    if f_right < 0.0:
        return right

    for _ in range(48):
        mid = 0.5 * (left + right)
        f_mid = (
            abs(
                _circle_theta_offset(
                    mid, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
                )
            )
            - theta_limit
        )
        if abs(f_mid) <= _LOCAL_ARC_BOUNDARY_TOL:
            return mid
        if f_mid <= 0.0:
            left = mid
        else:
            right = mid
    return 0.5 * (left + right)


@njit(fastmath=True, cache=True)
def _local_arc_theta_window(sigma, gamma_pv, eta_pv):
    sigma_eff = sigma
    if sigma_eff < 1e-12:
        sigma_eff = 1e-12
    gamma_eff = gamma_pv
    if gamma_eff < 1e-12:
        gamma_eff = 1e-12

    gauss_window = 0.0
    if (1.0 - eta_pv) > 1e-8:
        gauss_window = _LOCAL_ARC_GAUSS_SIGMAS * sigma_eff

    lor_window = 0.0
    if eta_pv > 1e-8:
        lor_window = _LOCAL_ARC_LORENTZ_GAMMAS * gamma_eff

    theta_window = max(gauss_window, lor_window, _LOCAL_ARC_MIN_DTHETA)
    if theta_window > pi:
        theta_window = pi
    return theta_window


@njit(fastmath=True, cache=True)
def _append_local_arc_window(starts, ends, count, start, end):
    two_pi = 2.0 * pi
    span = end - start
    if span >= two_pi - 1.0e-9:
        starts[0] = 0.0
        ends[0] = two_pi
        return 1, True

    while start < 0.0:
        start += two_pi
        end += two_pi
    while start >= two_pi:
        start -= two_pi
        end -= two_pi

    if end <= two_pi:
        if count >= starts.shape[0]:
            return 1, True
        starts[count] = start
        ends[count] = end
        return count + 1, False

    if count + 1 >= starts.shape[0]:
        return 1, True
    starts[count] = start
    ends[count] = two_pi
    starts[count + 1] = 0.0
    ends[count + 1] = end - two_pi
    return count + 2, False


@njit(fastmath=True, cache=True)
def _build_local_arc_windows(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
):
    starts = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    ends = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    two_pi = 2.0 * pi

    theta_window = _local_arc_theta_window(sigma, gamma_pv, eta_pv)
    if theta_window >= _LOCAL_ARC_FULL_CIRCLE_THETA_WINDOW:
        starts[0] = 0.0
        ends[0] = two_pi
        return starts, ends, 1, True

    Gr = sqrt(G_vec[0] * G_vec[0] + G_vec[1] * G_vec[1])
    theta0 = np.arctan2(G_vec[2], Gr)

    search_steps = int(n_steps // 2)
    if search_steps < _LOCAL_ARC_MIN_SEARCH_STEPS:
        search_steps = _LOCAL_ARC_MIN_SEARCH_STEPS
    elif search_steps > _LOCAL_ARC_MAX_SEARCH_STEPS:
        search_steps = _LOCAL_ARC_MAX_SEARCH_STEPS
    dphi = two_pi / float(search_steps)

    roots = np.empty(_LOCAL_ARC_MAX_ROOTS, dtype=np.float64)
    root_count = 0
    best_abs_0 = np.inf
    best_abs_1 = np.inf
    best_phi_0 = 0.0
    best_phi_1 = 0.0

    prev_phi = 0.0
    prev_val = _circle_theta_offset(
        prev_phi, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
    )
    prev_abs = abs(prev_val)
    best_abs_0 = prev_abs
    best_phi_0 = prev_phi

    for i in range(1, search_steps + 1):
        phi_val = float(i) * dphi
        cur_val = _circle_theta_offset(
            phi_val, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, theta0
        )
        cur_abs = abs(cur_val)

        if cur_abs < best_abs_0:
            if _phi_periodic_distance(phi_val, best_phi_0) > (2.0 * dphi):
                best_abs_1 = best_abs_0
                best_phi_1 = best_phi_0
            best_abs_0 = cur_abs
            best_phi_0 = phi_val
        elif cur_abs < best_abs_1 and _phi_periodic_distance(phi_val, best_phi_0) > (2.0 * dphi):
            best_abs_1 = cur_abs
            best_phi_1 = phi_val

        if prev_abs <= _LOCAL_ARC_ROOT_TOL:
            root_count = _store_local_arc_root(roots, root_count, prev_phi, 2.0 * dphi)
        elif cur_abs <= _LOCAL_ARC_ROOT_TOL or (prev_val * cur_val < 0.0):
            phi_root, ok_root = _refine_theta_root(
                prev_phi,
                phi_val,
                Ox,
                Oy,
                Oz,
                circle_r,
                e1x,
                e1y,
                e1z,
                e2x,
                e2y,
                e2z,
                theta0,
            )
            if ok_root:
                root_count = _store_local_arc_root(roots, root_count, phi_root, 2.0 * dphi)

        prev_phi = phi_val
        prev_val = cur_val
        prev_abs = cur_abs

    if root_count == 0:
        if best_abs_0 <= theta_window:
            root_count = _store_local_arc_root(roots, root_count, best_phi_0, 2.0 * dphi)
        if best_abs_1 <= theta_window:
            root_count = _store_local_arc_root(roots, root_count, best_phi_1, 2.0 * dphi)

    if root_count <= 0:
        starts[0] = 0.0
        ends[0] = two_pi
        return starts, ends, 1, True

    window_count = 0
    for i_root in range(root_count):
        phi_root = roots[i_root]
        left_inside = phi_root
        left_outside = phi_root - dphi
        left_found = False
        full_circle = False
        for _ in range(search_steps):
            abs_val = abs(
                _circle_theta_offset(
                    left_outside,
                    Ox,
                    Oy,
                    Oz,
                    circle_r,
                    e1x,
                    e1y,
                    e1z,
                    e2x,
                    e2y,
                    e2z,
                    theta0,
                )
            )
            if abs_val >= theta_window:
                left_found = True
                break
            left_inside = left_outside
            left_outside -= dphi
        if not left_found:
            full_circle = True

        right_inside = phi_root
        right_outside = phi_root + dphi
        right_found = False
        if not full_circle:
            for _ in range(search_steps):
                abs_val = abs(
                    _circle_theta_offset(
                        right_outside,
                        Ox,
                        Oy,
                        Oz,
                        circle_r,
                        e1x,
                        e1y,
                        e1z,
                        e2x,
                        e2y,
                        e2z,
                        theta0,
                    )
                )
                if abs_val >= theta_window:
                    right_found = True
                    break
                right_inside = right_outside
                right_outside += dphi
        if not right_found:
            full_circle = True

        if full_circle:
            starts[0] = 0.0
            ends[0] = two_pi
            return starts, ends, 1, True

        left_bound = _refine_theta_boundary(
            left_inside,
            left_outside,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            theta0,
            theta_window,
        )
        right_bound = _refine_theta_boundary(
            right_inside,
            right_outside,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            theta0,
            theta_window,
        )

        window_count, full_circle = _append_local_arc_window(
            starts, ends, window_count, left_bound, right_bound
        )
        if full_circle:
            starts[0] = 0.0
            ends[0] = two_pi
            return starts, ends, 1, True
    if window_count <= 1:
        return starts, ends, window_count, False

    for i in range(1, window_count):
        start_val = starts[i]
        end_val = ends[i]
        j = i - 1
        while j >= 0 and starts[j] > start_val:
            starts[j + 1] = starts[j]
            ends[j + 1] = ends[j]
            j -= 1
        starts[j + 1] = start_val
        ends[j + 1] = end_val

    merged_starts = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    merged_ends = np.zeros(_LOCAL_ARC_MAX_WINDOWS, dtype=np.float64)
    merged_count = 0
    for i in range(window_count):
        if merged_count == 0:
            merged_starts[0] = starts[i]
            merged_ends[0] = ends[i]
            merged_count = 1
            continue
        if starts[i] <= merged_ends[merged_count - 1] + 1.0e-9:
            if ends[i] > merged_ends[merged_count - 1]:
                merged_ends[merged_count - 1] = ends[i]
        else:
            merged_starts[merged_count] = starts[i]
            merged_ends[merged_count] = ends[i]
            merged_count += 1

    for i in range(merged_count):
        starts[i] = merged_starts[i]
        ends[i] = merged_ends[i]
    return starts, ends, merged_count, False


@njit(fastmath=True, cache=True)
def _solve_q_adaptive_domain(
    phi_start,
    phi_stop,
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    max_intervals,
    base_intervals,
    rel_err_tol,
):
    if max_intervals <= 0 or phi_stop <= phi_start:
        return np.zeros((0, 4), dtype=np.float64)

    n_base = base_intervals
    if n_base < MIN_SOLVE_Q_BASE_INTERVALS:
        n_base = MIN_SOLVE_Q_BASE_INTERVALS
    if n_base > max_intervals:
        n_base = max_intervals
    if n_base < 1:
        n_base = 1

    phi_a = np.empty(max_intervals, dtype=np.float64)
    phi_b = np.empty(max_intervals, dtype=np.float64)
    f_a = np.empty(max_intervals, dtype=np.float64)
    f_m = np.empty(max_intervals, dtype=np.float64)
    f_b = np.empty(max_intervals, dtype=np.float64)
    mass_arr = np.empty(max_intervals, dtype=np.float64)
    err_arr = np.empty(max_intervals, dtype=np.float64)

    n_intervals = n_base
    total_mass = 0.0
    total_err = 0.0
    dphi0 = (phi_stop - phi_start) / n_base
    for i in range(n_base):
        a = phi_start + i * dphi0
        b = phi_start + (i + 1) * dphi0
        m = 0.5 * (a + b)

        fa = _circle_density(
            a, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fm = _circle_density(
            m, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fb = _circle_density(
            b, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )

        mass_i, err_i = _interval_mass_error(a, b, fa, fm, fb, circle_r)
        phi_a[i] = a
        phi_b[i] = b
        f_a[i] = fa
        f_m[i] = fm
        f_b[i] = fb
        mass_arr[i] = mass_i
        err_arr[i] = err_i
        total_mass += mass_i
        total_err += err_i

    err_tol = _SOLVE_Q_ABS_ERR_TOL + rel_err_tol * abs(total_mass)

    while n_intervals < max_intervals and total_err > err_tol:
        split_idx = 0
        max_err = err_arr[0]
        for i in range(1, n_intervals):
            if err_arr[i] > max_err:
                max_err = err_arr[i]
                split_idx = i
        if max_err <= 0.0:
            break

        a = phi_a[split_idx]
        b = phi_b[split_idx]
        m = 0.5 * (a + b)
        q1 = 0.5 * (a + m)
        q3 = 0.5 * (m + b)

        fa = f_a[split_idx]
        fm = f_m[split_idx]
        fb = f_b[split_idx]
        fq1 = _circle_density(
            q1, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )
        fq3 = _circle_density(
            q3, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z, G_vec, sigma, gamma_pv, eta_pv
        )

        old_mass = mass_arr[split_idx]
        old_err = err_arr[split_idx]

        left_mass, left_err = _interval_mass_error(a, m, fa, fq1, fm, circle_r)
        right_mass, right_err = _interval_mass_error(m, b, fm, fq3, fb, circle_r)

        phi_a[split_idx] = a
        phi_b[split_idx] = m
        f_a[split_idx] = fa
        f_m[split_idx] = fq1
        f_b[split_idx] = fm
        mass_arr[split_idx] = left_mass
        err_arr[split_idx] = left_err

        phi_a[n_intervals] = m
        phi_b[n_intervals] = b
        f_a[n_intervals] = fm
        f_m[n_intervals] = fq3
        f_b[n_intervals] = fb
        mass_arr[n_intervals] = right_mass
        err_arr[n_intervals] = right_err

        total_mass += left_mass + right_mass - old_mass
        total_err += left_err + right_err - old_err
        n_intervals += 1
        err_tol = _SOLVE_Q_ABS_ERR_TOL + rel_err_tol * abs(total_mass)

    n_valid = 0
    for i in range(n_intervals):
        if mass_arr[i] > _INTENSITY_CUTOFF:
            n_valid += 1

    out = np.zeros((n_valid, 4), dtype=np.float64)
    out_idx = 0
    for i in range(n_intervals):
        mass_i = mass_arr[i]
        if mass_i <= _INTENSITY_CUTOFF:
            continue
        phi_m = 0.5 * (phi_a[i] + phi_b[i])
        Qx, Qy, Qz = _circle_point(phi_m, Ox, Oy, Oz, circle_r, e1x, e1y, e1z, e2x, e2y, e2z)
        out[out_idx, 0] = Qx
        out[out_idx, 1] = Qy
        out[out_idx, 2] = Qz
        out[out_idx, 3] = mass_i
        out_idx += 1

    return out


@njit(fastmath=True, cache=True)
def _solve_q_uniform_full_circle(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
):
    if n_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    if n_steps == DEFAULT_SOLVE_Q_STEPS:
        dtheta = float(default_solve_q_dtheta)
        cth = default_solve_q_cos
        sth = default_solve_q_sin
    else:
        dtheta = 2.0 * np.pi / n_steps
        theta_arr = dtheta * np.arange(n_steps)
        cth = np.cos(theta_arr)
        sth = np.sin(theta_arr)

    Qx_arr = Ox + circle_r * (cth * e1x + sth * e2x)
    Qy_arr = Oy + circle_r * (cth * e1y + sth * e2y)
    Qz_arr = Oz + circle_r * (cth * e1z + sth * e2z)

    sigma_arr = compute_intensity_array_serial(
        Qx_arr,
        Qy_arr,
        Qz_arr,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
    )
    ds = circle_r * dtheta
    all_int = sigma_arr * ds

    valid_idx = np.nonzero(all_int > _INTENSITY_CUTOFF)[0]
    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]

    return out


@njit(fastmath=True, cache=True)
def _solve_q_uniform(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    n_steps,
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
):
    if n_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64)

    starts, ends, window_count, use_full_circle = _build_local_arc_windows(
        Ox,
        Oy,
        Oz,
        circle_r,
        e1x,
        e1y,
        e1z,
        e2x,
        e2y,
        e2z,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
        n_steps,
    )
    if use_full_circle or window_count <= 0:
        return _solve_q_uniform_full_circle(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            n_steps,
            default_solve_q_dtheta,
            default_solve_q_cos,
            default_solve_q_sin,
        )

    target_dphi = (2.0 * pi) / float(n_steps)
    window_steps = np.empty(window_count, dtype=np.int64)
    total_samples = 0
    for i_win in range(window_count):
        span = ends[i_win] - starts[i_win]
        n_window = int(np.ceil(span / max(target_dphi, 1.0e-12)))
        if n_window < _LOCAL_ARC_MIN_STEPS_PER_WINDOW:
            n_window = _LOCAL_ARC_MIN_STEPS_PER_WINDOW
        window_steps[i_win] = n_window
        total_samples += n_window

    Qx_arr = np.empty(total_samples, dtype=np.float64)
    Qy_arr = np.empty(total_samples, dtype=np.float64)
    Qz_arr = np.empty(total_samples, dtype=np.float64)
    ds_arr = np.empty(total_samples, dtype=np.float64)
    offset = 0
    for i_win in range(window_count):
        n_window = int(window_steps[i_win])
        span = ends[i_win] - starts[i_win]
        dphi_local = span / float(n_window)
        ds = circle_r * dphi_local
        step_idx = np.arange(n_window, dtype=np.float64)
        phi_mid = starts[i_win] + (step_idx + 0.5) * dphi_local
        cos_phi = np.cos(phi_mid)
        sin_phi = np.sin(phi_mid)
        next_offset = offset + n_window

        Qx_arr[offset:next_offset] = Ox + circle_r * (cos_phi * e1x + sin_phi * e2x)
        Qy_arr[offset:next_offset] = Oy + circle_r * (cos_phi * e1y + sin_phi * e2y)
        Qz_arr[offset:next_offset] = Oz + circle_r * (cos_phi * e1z + sin_phi * e2z)
        ds_arr[offset:next_offset] = ds
        offset = next_offset

    sigma_arr = compute_intensity_array_serial(
        Qx_arr,
        Qy_arr,
        Qz_arr,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
    )
    all_int = sigma_arr * ds_arr
    valid_idx = np.nonzero(all_int > _INTENSITY_CUTOFF)[0]
    out = np.zeros((valid_idx.size, 4), dtype=np.float64)
    for i in range(valid_idx.size):
        idx = valid_idx[i]
        out[i, 0] = Qx_arr[idx]
        out[i, 1] = Qy_arr[idx]
        out[i, 2] = Qz_arr[idx]
        out[i, 3] = all_int[idx]
    return out


@njit(fastmath=True, cache=True)
def _solve_q_adaptive(
    Ox,
    Oy,
    Oz,
    circle_r,
    e1x,
    e1y,
    e1z,
    e2x,
    e2y,
    e2z,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    max_intervals,
    base_intervals,
    rel_err_tol,
):
    starts, ends, window_count, use_full_circle = _build_local_arc_windows(
        Ox,
        Oy,
        Oz,
        circle_r,
        e1x,
        e1y,
        e1z,
        e2x,
        e2y,
        e2z,
        G_vec,
        sigma,
        gamma_pv,
        eta_pv,
        max_intervals,
    )
    if use_full_circle or window_count <= 0:
        return _solve_q_adaptive_domain(
            0.0,
            2.0 * pi,
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            max_intervals,
            base_intervals,
            rel_err_tol,
        )

    total_span = 0.0
    for i_win in range(window_count):
        total_span += ends[i_win] - starts[i_win]
    if total_span <= 0.0:
        return np.zeros((0, 4), dtype=np.float64)

    chunks = List.empty_list(types.float64[:, ::1])
    total_rows = 0
    for i_win in range(window_count):
        span = ends[i_win] - starts[i_win]
        frac = span / total_span
        max_intervals_i = int(round(float(max_intervals) * frac))
        if max_intervals_i < MIN_SOLVE_Q_BASE_INTERVALS:
            max_intervals_i = MIN_SOLVE_Q_BASE_INTERVALS
        base_intervals_i = int(round(float(base_intervals) * frac))
        if base_intervals_i < MIN_SOLVE_Q_BASE_INTERVALS:
            base_intervals_i = MIN_SOLVE_Q_BASE_INTERVALS
        if max_intervals_i < base_intervals_i:
            max_intervals_i = base_intervals_i

        chunk = _solve_q_adaptive_domain(
            starts[i_win],
            ends[i_win],
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            max_intervals_i,
            base_intervals_i,
            rel_err_tol,
        )
        chunks.append(chunk)
        total_rows += chunk.shape[0]
    out = np.zeros((total_rows, 4), dtype=np.float64)
    out_idx = 0
    for i_chunk in range(len(chunks)):
        chunk = chunks[i_chunk]
        for i_row in range(chunk.shape[0]):
            out[out_idx, 0] = chunk[i_row, 0]
            out[out_idx, 1] = chunk[i_row, 1]
            out[out_idx, 2] = chunk[i_row, 2]
            out[out_idx, 3] = chunk[i_row, 3]
            out_idx += 1
    return out


@njit(fastmath=True, cache=True, nogil=True)
def solve_q(
    k_in_crystal,
    k_scat,
    G_vec,
    sigma,
    gamma_pv,
    eta_pv,
    N_steps,
    base_intervals,
    rel_err_tol,
    solve_q_mode,
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
):
    """
    Build a 'circle' in reciprocal space for the reflection G_vec, i.e. the
    set of Q that satisfies |Q|=|G| or an intersection with Ewald sphere, then
    filter by mosaic surface density compute_intensity_array.

    Physically:
      - In uniform mode, sample the full circle at fixed angular steps.
      - In adaptive mode, refine intervals deterministically where the
        pseudo-Voigt profile varies most.
      - Adaptive mode uses Simpson-weighted interval masses to preserve long
        Lorentzian tails without stochastic noise.

    Returns
    -------
    out : ndarray of shape (M,4)
        For the valid points, columns = (Qx, Qy, Qz, mosaic_intensity).
    status : int
        0 for success or a negative code indicating the failure reason.
    """
    status = 0
    if N_steps <= 0:
        return np.zeros((0, 4), dtype=np.float64), status
    if base_intervals <= 0:
        return np.zeros((0, 4), dtype=np.float64), status
    if rel_err_tol < 0.0:
        rel_err_tol = 0.0

    G_sq = G_vec[0] * G_vec[0] + G_vec[1] * G_vec[1] + G_vec[2] * G_vec[2]
    if G_sq < 1e-14:
        status = -1
        return np.zeros((0, 4), dtype=np.float64), status

    Ax = -k_in_crystal[0]
    Ay = -k_in_crystal[1]
    Az = -k_in_crystal[2]
    rA = k_scat
    A_sq = Ax * Ax + Ay * Ay + Az * Az
    if A_sq < 1e-14:
        status = -2
        return np.zeros((0, 4), dtype=np.float64), status
    A_len = sqrt(A_sq)

    c = (G_sq + A_sq - rA * rA) / (2.0 * A_len)
    # Compute circle parameters
    circle_r_sq = G_sq - c * c
    if circle_r_sq < 0.0:
        status = -3
        return np.zeros((0, 4), dtype=np.float64), status
    circle_r = np.sqrt(circle_r_sq)

    Ax_hat = Ax / A_len
    Ay_hat = Ay / A_len
    Az_hat = Az / A_len

    Ox = c * Ax_hat
    Oy = c * Ay_hat
    Oz = c * Az_hat

    # Build two orthonormal vectors (e1, e2) in the plane perpendicular to Ax_hat.
    ax, ay, az = 1.0, 0.0, 0.0
    dot_aA = ax * Ax_hat + ay * Ay_hat + az * Az_hat
    if abs(dot_aA) > 0.9999:
        ax, ay, az = 0.0, 1.0, 0.0
        dot_aA = ax * Ax_hat + ay * Ay_hat + az * Az_hat
    aox = ax - dot_aA * Ax_hat
    aoy = ay - dot_aA * Ay_hat
    aoz = az - dot_aA * Az_hat
    ao_len = np.sqrt(aox * aox + aoy * aoy + aoz * aoz)
    if ao_len < 1e-14:
        status = -4
        return np.zeros((0, 4), dtype=np.float64), status
    e1x = aox / ao_len
    e1y = aoy / ao_len
    e1z = aoz / ao_len

    e2x = Az_hat * e1y - Ay_hat * e1z
    e2y = Ax_hat * e1z - Az_hat * e1x
    e2z = Ay_hat * e1x - Ax_hat * e1y
    e2_len = np.sqrt(e2x * e2x + e2y * e2y + e2z * e2z)
    if e2_len < 1e-14:
        status = -5
        return np.zeros((0, 4), dtype=np.float64), status
    e2x /= e2_len
    e2y /= e2_len
    e2z /= e2_len

    mode_i = int(solve_q_mode)
    if mode_i == SOLVE_Q_MODE_UNIFORM:
        out = _solve_q_uniform(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            int(N_steps),
            default_solve_q_dtheta,
            default_solve_q_cos,
            default_solve_q_sin,
        )
    else:
        out = _solve_q_adaptive(
            Ox,
            Oy,
            Oz,
            circle_r,
            e1x,
            e1y,
            e1z,
            e2x,
            e2y,
            e2z,
            G_vec,
            sigma,
            gamma_pv,
            eta_pv,
            int(N_steps),
            int(base_intervals),
            float(rel_err_tol),
        )

    return out, status












@njit(fastmath=True, cache=True)
def _build_sample_rotation(
    theta_initial_deg,
    cor_angle_deg,
    psi_z_deg,
    R_z_R_y,
    R_ZY_n,
    P0,
):
    """Build reflection-invariant sample frame for the current geometry."""
    rad_theta_i = theta_initial_deg * (pi / 180.0)
    cor_axis_rad = cor_angle_deg * (pi / 180.0)
    cor_axis_yaw_rad = psi_z_deg * (pi / 180.0)

    # Pitch the CoR axis in its local x-z plane, then yaw that axis about
    # laboratory z by psi_z.
    ax = cos(cor_axis_rad)
    ay = 0.0
    az = sin(cor_axis_rad)
    c_axis_yaw = cos(cor_axis_yaw_rad)
    s_axis_yaw = sin(cor_axis_yaw_rad)
    ax_yawed = c_axis_yaw * ax + s_axis_yaw * ay
    ay_yawed = -s_axis_yaw * ax + c_axis_yaw * ay
    ax = ax_yawed
    ay = ay_yawed
    axis_norm = sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-12:
        axis_norm = 1.0
    ax /= axis_norm
    ay /= axis_norm
    az /= axis_norm

    ct = cos(rad_theta_i)
    st = sin(rad_theta_i)
    one_ct = 1.0 - ct
    R_cor = np.array(
        [
            [
                ct + ax * ax * one_ct,
                ax * ay * one_ct - az * st,
                ax * az * one_ct + ay * st,
            ],
            [
                ay * ax * one_ct + az * st,
                ct + ay * ay * one_ct,
                ay * az * one_ct - ax * st,
            ],
            [
                az * ax * one_ct - ay * st,
                az * ay * one_ct + ax * st,
                ct + az * az * one_ct,
            ],
        ]
    )
    R_sample = R_cor @ R_z_R_y

    n_surf = R_cor @ R_ZY_n
    n_surf /= sqrt(n_surf[0] * n_surf[0] + n_surf[1] * n_surf[1] + n_surf[2] * n_surf[2])

    P0_rot = R_sample @ P0
    P0_rot[0] = 0.0
    return R_sample, n_surf, P0_rot


@njit(fastmath=True, cache=True)
def _precompute_sample_terms(
    wavelength_array,
    n2,
    n2_array,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    zb,
    thickness,
    sample_width_m,
    sample_length_m,
    theta_initial_deg,
    cor_angle_deg,
    psi_z_deg,
    R_z_R_y,
    R_ZY_n,
    P0,
):
    """Precompute sample- and beam-dependent terms shared by all reflections."""
    n_samp = beam_x_array.size
    sample_terms = np.zeros((n_samp, _SAMPLE_COLS), dtype=np.float64)
    sample_terms[:, _SAMPLE_COL_SOLVE_Q_REP] = -1.0
    sample_terms[:, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
    n2_samp_out = np.empty(n_samp, dtype=np.complex128)
    eps2_out = np.empty(n_samp, dtype=np.complex128)
    thickness_angstrom = _thickness_to_angstrom(thickness)

    R_sample, n_surf, P0_rot = _build_sample_rotation(
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    )

    best_idx = 0
    if n_samp > 0:
        best_angle = theta_array[0] * theta_array[0] + phi_array[0] * phi_array[0]
        best_beam = beam_x_array[0] * beam_x_array[0] + beam_y_array[0] * beam_y_array[0]
        for ii in range(1, n_samp):
            metric = theta_array[ii] * theta_array[ii] + phi_array[ii] * phi_array[ii]
            beam_metric = beam_x_array[ii] * beam_x_array[ii] + beam_y_array[ii] * beam_y_array[ii]
            if metric < best_angle:
                best_angle = metric
                best_beam = beam_metric
                best_idx = ii
            elif abs(metric - best_angle) <= 1e-18 and beam_metric < best_beam:
                best_beam = beam_metric
                best_idx = ii

    # Build local incidence basis around the sample normal.
    sample_axis_x = np.array(
        [R_sample[0, 0], R_sample[1, 0], R_sample[2, 0]],
        dtype=np.float64,
    )
    sample_axis_y = np.array(
        [R_sample[0, 1], R_sample[1, 1], R_sample[2, 1]],
        dtype=np.float64,
    )
    half_width = 0.5 * sample_width_m if sample_width_m > 0.0 else 0.0
    half_length = 0.5 * sample_length_m if sample_length_m > 0.0 else 0.0
    u_ref = np.array([0.0, 0.0, -1.0])
    e1_temp = np.cross(n_surf, u_ref)
    e1_norm = sqrt(e1_temp[0] * e1_temp[0] + e1_temp[1] * e1_temp[1] + e1_temp[2] * e1_temp[2])
    if e1_norm < 1e-12:
        alt_refs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for ar in alt_refs:
            cross_tmp = np.cross(n_surf, ar)
            cross_norm_tmp = sqrt(
                cross_tmp[0] * cross_tmp[0]
                + cross_tmp[1] * cross_tmp[1]
                + cross_tmp[2] * cross_tmp[2]
            )
            if cross_norm_tmp > 1e-12:
                e1_temp = cross_tmp / cross_norm_tmp
                break
    else:
        e1_temp /= e1_norm
    e2_temp = np.cross(n_surf, e1_temp)

    eps1 = 1.0 + 0.0j

    for i_samp in range(n_samp):
        lam_samp = wavelength_array[i_samp]
        k0 = 2.0 * pi / lam_samp

        n2_samp = n2
        if i_samp < n2_array.size:
            n2_samp = n2_array[i_samp]
        eps2 = n2_samp * n2_samp

        n2_samp_out[i_samp] = n2_samp
        eps2_out[i_samp] = eps2
        sample_terms[i_samp, _SAMPLE_COL_K0] = k0
        sample_terms[i_samp, _SAMPLE_COL_N2_REAL] = np.real(n2_samp)

        dtheta = theta_array[i_samp]
        dphi = phi_array[i_samp]
        k_in_x = cos(dtheta) * sin(dphi)
        k_in_y = cos(dtheta) * cos(dphi)
        k_in_z = sin(dtheta)

        beam_start = np.array(
            [beam_x_array[i_samp], -20e-3, beam_y_array[i_samp] - zb],
            dtype=np.float64,
        )
        k_in = np.array([k_in_x, k_in_y, k_in_z], dtype=np.float64)

        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, P0_rot, n_surf)
        if not valid_int:
            continue

        rel_x = ix - P0_rot[0]
        rel_y = iy - P0_rot[1]
        rel_z = iz - P0_rot[2]
        if half_width > 0.0:
            x_local = rel_x * sample_axis_x[0] + rel_y * sample_axis_x[1] + rel_z * sample_axis_x[2]
            if np.abs(x_local) > half_width:
                continue
        if half_length > 0.0:
            y_local = rel_x * sample_axis_y[0] + rel_y * sample_axis_y[1] + rel_z * sample_axis_y[2]
            if np.abs(y_local) > half_length:
                continue

        sample_terms[i_samp, _SAMPLE_COL_VALID] = 1.0
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_X] = ix
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_Y] = iy
        sample_terms[i_samp, _SAMPLE_COL_I_PLANE_Z] = iz

        kn_dot = k_in_x * n_surf[0] + k_in_y * n_surf[1] + k_in_z * n_surf[2]
        if kn_dot > 1.0:
            kn_dot = 1.0
        elif kn_dot < -1.0:
            kn_dot = -1.0
        th_i_prime = (pi / 2.0) - acos(kn_dot)

        proj_incident_x = k_in_x - kn_dot * n_surf[0]
        proj_incident_y = k_in_y - kn_dot * n_surf[1]
        proj_incident_z = k_in_z - kn_dot * n_surf[2]
        pln = sqrt(
            proj_incident_x * proj_incident_x
            + proj_incident_y * proj_incident_y
            + proj_incident_z * proj_incident_z
        )
        if pln > 1e-12:
            proj_incident_x /= pln
            proj_incident_y /= pln
            proj_incident_z /= pln
        else:
            proj_incident_x = 0.0
            proj_incident_y = 0.0
            proj_incident_z = 0.0

        p1 = (
            proj_incident_x * e1_temp[0]
            + proj_incident_y * e1_temp[1]
            + proj_incident_z * e1_temp[2]
        )
        p2 = (
            proj_incident_x * e2_temp[0]
            + proj_incident_y * e2_temp[1]
            + proj_incident_z * e2_temp[2]
        )
        phi_i_prime = (pi / 2.0) - np.arctan2(p2, p1)

        k0_sq = k0 * k0
        k_par_i = k0 * np.abs(np.cos(th_i_prime))
        k_par_i_sq = k_par_i * k_par_i

        kz1_i = _kz_branch_decay((k0_sq - k_par_i_sq) + 0.0j)
        kz2_i = _kz_branch_decay((eps2 * k0_sq) - k_par_i_sq)

        k_x_scat = k_par_i * np.sin(phi_i_prime)
        k_y_scat = k_par_i * np.cos(phi_i_prime)
        re_k_z = -np.abs(kz2_i.real)
        im_k_z = np.abs(kz2_i.imag)
        k_scat = np.sqrt(np.maximum(k_par_i_sq + kz2_i.real * kz2_i.real, 0.0))

        Ti_s = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, True)
        Ti_p = _fresnel_t_exact(kz1_i, kz2_i, eps1, eps2, False)
        Ti2 = 0.5 * (
            _fresnel_power_t_exact(Ti_s, kz1_i, kz2_i, eps1, eps2, True)
            + _fresnel_power_t_exact(Ti_p, kz1_i, kz2_i, eps1, eps2, False)
        )
        Ti2 = _sanitize_transmission_power(Ti2)

        L_in = _attenuation_depth_angstrom(thickness_angstrom)

        sample_terms[i_samp, _SAMPLE_COL_KX_SCAT] = k_x_scat
        sample_terms[i_samp, _SAMPLE_COL_KY_SCAT] = k_y_scat
        sample_terms[i_samp, _SAMPLE_COL_RE_KZ] = re_k_z
        sample_terms[i_samp, _SAMPLE_COL_IM_KZ] = im_k_z
        sample_terms[i_samp, _SAMPLE_COL_K_SCAT] = k_scat
        sample_terms[i_samp, _SAMPLE_COL_TI2] = Ti2
        sample_terms[i_samp, _SAMPLE_COL_L_IN] = L_in

    _annotate_solve_q_sample_reuse(sample_terms)
    return R_sample, sample_terms, n2_samp_out, eps2_out, best_idx








@njit(fastmath=True, cache=True)
def _solve_q_reuse_terms_match(sample_terms, idx_a, idx_b):
    cols = (
        _SAMPLE_COL_KX_SCAT,
        _SAMPLE_COL_KY_SCAT,
        _SAMPLE_COL_RE_KZ,
        _SAMPLE_COL_K_SCAT,
    )
    for col in cols:
        aval = sample_terms[idx_a, col]
        bval = sample_terms[idx_b, col]
        scale = np.abs(aval)
        if np.abs(bval) > scale:
            scale = np.abs(bval)
        if np.abs(aval - bval) > (1.0e-12 * (1.0 + scale)):
            return False
    return True


@njit(fastmath=True, cache=True)
def _annotate_solve_q_sample_reuse(sample_terms):
    """Link samples whose `solve_q` inputs are numerically identical."""

    n_samp = sample_terms.shape[0]
    if n_samp <= 1:
        return

    group_reps = np.empty(n_samp, dtype=np.int64)
    group_tails = np.empty(n_samp, dtype=np.int64)
    group_count = 0

    for i_samp in range(n_samp):
        if sample_terms[i_samp, _SAMPLE_COL_VALID] <= 0.5:
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = -1.0
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
            continue

        matched_group = -1
        for i_grp in range(group_count):
            rep_idx = group_reps[i_grp]
            if _solve_q_reuse_terms_match(sample_terms, i_samp, rep_idx):
                matched_group = i_grp
                break

        if matched_group < 0:
            group_reps[group_count] = i_samp
            group_tails[group_count] = i_samp
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = float(i_samp)
            sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
            group_count += 1
            continue

        rep_idx = group_reps[matched_group]
        tail_idx = group_tails[matched_group]
        sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_REP] = float(rep_idx)
        sample_terms[i_samp, _SAMPLE_COL_SOLVE_Q_NEXT] = -1.0
        sample_terms[tail_idx, _SAMPLE_COL_SOLVE_Q_NEXT] = float(i_samp)
        group_tails[matched_group] = i_samp


@njit(fastmath=True, cache=True, nogil=True)
def _accumulate_bilinear_hit(image, image_size, row_f, col_f, value):
    """Deposit ``value`` into the four neighboring pixels around a float hit."""

    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)
    deposited = False

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col <= 0.0:
                continue
            image[rr, cc] += value * w_row * w_col
            deposited = True

    return deposited


@njit(fastmath=True, cache=True, nogil=True)
def _accumulate_and_compact_weighted_event_rows(
    image,
    image_size,
    event_rows,
    event_peak_indices,
    event_count,
):
    """Accumulate compact threaded events and retain detector-supported rows."""

    write_index = 0
    row_width = int(event_rows.shape[1])
    for read_index in range(int(event_count)):
        deposited = _accumulate_bilinear_hit(
            image,
            int(image_size),
            float(event_rows[read_index, HIT_ROW_COL_DETECTOR_ROW]),
            float(event_rows[read_index, HIT_ROW_COL_DETECTOR_COL]),
            float(event_rows[read_index, HIT_ROW_COL_INTENSITY]),
        )
        if not deposited:
            continue
        if write_index != read_index:
            for column_index in range(row_width):
                event_rows[write_index, column_index] = event_rows[read_index, column_index]
            event_peak_indices[write_index] = event_peak_indices[read_index]
        write_index += 1
    return int(write_index)


@njit(fastmath=True, cache=True)
def _candidate_has_detector_support(image_size, row_f, col_f):
    """Return whether a floating detector hit reaches at least one image pixel."""

    if not np.isfinite(row_f) or not np.isfinite(col_f):
        return False

    row0 = int(np.floor(row_f))
    col0 = int(np.floor(col_f))
    d_row = row_f - float(row0)
    d_col = col_f - float(col0)

    for row_offset in range(2):
        rr = row0 + row_offset
        if rr < 0 or rr >= image_size:
            continue
        w_row = 1.0 - d_row if row_offset == 0 else d_row
        if w_row <= 0.0:
            continue
        for col_offset in range(2):
            cc = col0 + col_offset
            if cc < 0 or cc >= image_size:
                continue
            w_col = 1.0 - d_col if col_offset == 0 else d_col
            if w_col > 0.0:
                return True
    return False














@njit(cache=True, nogil=True)
def _event_sample_unit_interval(sample_idx: int, event_idx: int, stream: int = 0) -> float:
    x = (
        0.5
        + (sample_idx + 1) * 0.6180339887498949
        + (event_idx + 1) * 0.41421356237309503
        + (stream + 1) * 0.7320508075688772
    )
    return x - np.floor(x)


@njit(cache=True, nogil=True)
def _weighted_event_targets(total_mass: float, event_count: int, sample_idx: int) -> np.ndarray:
    total_mass_f = float(total_mass)
    event_count_i = int(event_count)
    if not np.isfinite(total_mass_f) or total_mass_f <= 0.0 or event_count_i <= 0:
        return np.empty((0,), dtype=np.float64)
    targets = np.empty(event_count_i, dtype=np.float64)
    for event_idx in range(event_count_i):
        targets[event_idx] = (
            _event_sample_unit_interval(int(sample_idx), int(event_idx), stream=0) * total_mass_f
        )
    targets.sort()
    return targets


@njit(cache=True, nogil=True)
def _weighted_event_deposit(total_mass: float, event_count: int) -> float:
    total_mass_f = float(total_mass)
    event_count_i = int(event_count)
    if not np.isfinite(total_mass_f) or total_mass_f <= 0.0 or event_count_i <= 0:
        return 0.0
    return total_mass_f / float(event_count_i)


def _candidate_valid_mass(value: float) -> bool:
    return bool(np.isfinite(value) and float(value) > 0.0)


@njit(fastmath=True, cache=True, nogil=True)
def _store_weighted_event_q_debug_row(
    save_flag,
    q_data,
    q_count,
    q_debug_truncated_count,
    peak_idx,
    sample_idx,
    Qx,
    Qy,
    Qz,
    mass,
):
    if save_flag != 1 or (not np.isfinite(mass)) or mass <= 0.0:
        return
    if q_count[peak_idx] < q_data.shape[1]:
        q_store_idx = q_count[peak_idx]
        q_data[peak_idx, q_store_idx, 0] = Qx
        q_data[peak_idx, q_store_idx, 1] = Qy
        q_data[peak_idx, q_store_idx, 2] = Qz
        q_data[peak_idx, q_store_idx, 3] = mass
        q_data[peak_idx, q_store_idx, 4] = float(sample_idx)
        q_count[peak_idx] += 1
    else:
        q_debug_truncated_count[0] += 1






@njit(fastmath=True, cache=True, nogil=True)
def _project_weighted_candidate_fast(
    Qx,
    Qy,
    Qz,
    I_Q,
    reflection_intensity,
    sample_weight,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    I_plane_x,
    I_plane_y,
    I_plane_z,
    k_x_scat,
    k_y_scat,
    re_k_z,
    im_k_z,
    k0,
    Ti2,
    L_in,
    eps2,
    thickness_angstrom,
    pixel_scale,
    image_size,
):
    eps3 = 1.0 + 0.0j

    k_tx_prime = Qx + k_x_scat
    k_ty_prime = Qy + k_y_scat
    k_tz_prime = Qz + re_k_z
    kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)

    valid_exit, kf_x, kf_y, kf_z, _twotheta_t = _exact_external_air_exit_wavevector(
        k_tx_prime,
        k_ty_prime,
        k_tz_prime,
        k0,
    )
    if not valid_exit:
        return False, np.nan, np.nan, np.nan, 0.0

    k0_sq = k0 * k0
    k_par_f = kr
    k_par_f_sq = k_par_f * k_par_f
    kz2_f = _kz_branch_decay((eps2 * k0_sq) - k_par_f_sq)
    kz3_f = _kz_branch_decay((k0_sq - k_par_f_sq) + 0.0j)
    Tf_s = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, True)
    Tf_p = _fresnel_t_exact(kz2_f, kz3_f, eps2, eps3, False)
    Tf2 = 0.5 * (
        _fresnel_power_t_exact(Tf_s, kz2_f, kz3_f, eps2, eps3, True)
        + _fresnel_power_t_exact(Tf_p, kz2_f, kz3_f, eps2, eps3, False)
    )
    Tf2 = _sanitize_transmission_power(Tf2)
    im_k_z_f = np.abs(kz2_f.imag)
    L_out = _attenuation_depth_angstrom(thickness_angstrom)

    prop_att = np.exp(-2.0 * im_k_z * L_in) * np.exp(-2.0 * im_k_z_f * L_out)
    if (not np.isfinite(prop_att)) or prop_att <= 0.0:
        return False, np.nan, np.nan, np.nan, 0.0

    prop_fac = Ti2 * Tf2 * prop_att
    if (not np.isfinite(prop_fac)) or prop_fac <= 0.0:
        return False, np.nan, np.nan, np.nan, 0.0

    phi_f = np.arctan2(k_tx_prime, k_ty_prime)

    kf_prime_x = R_sample[0, 0] * kf_x + R_sample[0, 1] * kf_y + R_sample[0, 2] * kf_z
    kf_prime_y = R_sample[1, 0] * kf_x + R_sample[1, 1] * kf_y + R_sample[1, 2] * kf_z
    kf_prime_z = R_sample[2, 0] * kf_x + R_sample[2, 1] * kf_y + R_sample[2, 2] * kf_z

    denom = kf_prime_x * n_det_rot[0] + kf_prime_y * n_det_rot[1] + kf_prime_z * n_det_rot[2]
    if abs(denom) < 1e-14:
        dist = (
            (I_plane_x - Detector_Pos[0]) * n_det_rot[0]
            + (I_plane_y - Detector_Pos[1]) * n_det_rot[1]
            + (I_plane_z - Detector_Pos[2]) * n_det_rot[2]
        )
        if abs(dist) < 1e-6:
            dx = I_plane_x
            dy = I_plane_y
            dz = I_plane_z
        else:
            return False, np.nan, np.nan, np.nan, 0.0
    else:
        num = (
            (Detector_Pos[0] - I_plane_x) * n_det_rot[0]
            + (Detector_Pos[1] - I_plane_y) * n_det_rot[1]
            + (Detector_Pos[2] - I_plane_z) * n_det_rot[2]
        )
        t = num / denom
        if t < -1e-9:
            return False, np.nan, np.nan, np.nan, 0.0
        if t < 0.0:
            t = 0.0
        dx = I_plane_x + t * kf_prime_x
        dy = I_plane_y + t * kf_prime_y
        dz = I_plane_z + t * kf_prime_z

    plane_to_det_x = dx - Detector_Pos[0]
    plane_to_det_y = dy - Detector_Pos[1]
    plane_to_det_z = dz - Detector_Pos[2]
    x_det = plane_to_det_x * e1_det[0] + plane_to_det_y * e1_det[1] + plane_to_det_z * e1_det[2]
    y_det = plane_to_det_x * e2_det[0] + plane_to_det_y * e2_det[1] + plane_to_det_z * e2_det[2]
    if (not np.isfinite(x_det)) or (not np.isfinite(y_det)):
        return False, np.nan, np.nan, np.nan, 0.0

    row_f = center_row - y_det * pixel_scale
    col_f = center_col + x_det * pixel_scale
    if (not np.isfinite(row_f)) or (not np.isfinite(col_f)):
        return False, np.nan, np.nan, np.nan, 0.0
    if not _candidate_has_detector_support(image_size, row_f, col_f):
        return False, row_f, col_f, np.nan, 0.0

    mass = (
        reflection_intensity
        * sample_weight
        * I_Q
        * prop_fac
        * exp(-Qz * Qz * debye_x_sq)
        * exp(-(Qx * Qx + Qy * Qy) * debye_y_sq)
    )
    if (not np.isfinite(mass)) or mass <= 0.0:
        return False, row_f, col_f, phi_f, 0.0
    return True, row_f, col_f, phi_f, mass


@njit(fastmath=True, cache=True, nogil=True)
def _weighted_event_project_store_for_qset(
    all_q,
    peak_idx,
    sample_idx,
    reflection_intensity,
    sample_weight,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    sample_eps2_array,
    thickness,
    pixel_scale,
    image_size,
    save_flag,
    q_data,
    q_count,
    q_debug_truncated_count,
    candidate_mass,
    candidate_row,
    candidate_col,
    candidate_phi,
    candidate_peak_idx,
    candidate_count,
):
    total_mass = 0.0
    valid_count = 0
    project_calls = 0

    I_plane_x = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_X]
    I_plane_y = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Y]
    I_plane_z = sample_terms[sample_idx, _SAMPLE_COL_I_PLANE_Z]
    k_x_scat = sample_terms[sample_idx, _SAMPLE_COL_KX_SCAT]
    k_y_scat = sample_terms[sample_idx, _SAMPLE_COL_KY_SCAT]
    re_k_z = sample_terms[sample_idx, _SAMPLE_COL_RE_KZ]
    im_k_z = sample_terms[sample_idx, _SAMPLE_COL_IM_KZ]
    k0 = sample_terms[sample_idx, _SAMPLE_COL_K0]
    Ti2 = sample_terms[sample_idx, _SAMPLE_COL_TI2]
    L_in = sample_terms[sample_idx, _SAMPLE_COL_L_IN]
    eps2 = sample_eps2_array[sample_idx]
    thickness_angstrom = _thickness_to_angstrom(thickness)

    for q_idx in range(all_q.shape[0]):
        Qx = all_q[q_idx, 0]
        Qy = all_q[q_idx, 1]
        Qz = all_q[q_idx, 2]
        I_Q = all_q[q_idx, 3]
        project_calls += 1
        candidate_valid, row_f, col_f, phi_f, mass = _project_weighted_candidate_fast(
            Qx,
            Qy,
            Qz,
            I_Q,
            reflection_intensity,
            sample_weight,
            debye_x_sq,
            debye_y_sq,
            center_row,
            center_col,
            R_sample,
            n_det_rot,
            Detector_Pos,
            e1_det,
            e2_det,
            I_plane_x,
            I_plane_y,
            I_plane_z,
            k_x_scat,
            k_y_scat,
            re_k_z,
            im_k_z,
            k0,
            Ti2,
            L_in,
            eps2,
            thickness_angstrom,
            pixel_scale,
            image_size,
        )
        _store_weighted_event_q_debug_row(
            save_flag,
            q_data,
            q_count,
            q_debug_truncated_count,
            peak_idx,
            sample_idx,
            Qx,
            Qy,
            Qz,
            mass,
        )
        if not candidate_valid:
            continue

        total_mass += mass
        valid_count += 1
        if candidate_count >= candidate_mass.shape[0]:
            continue

        write_idx = candidate_count
        candidate_mass[write_idx] = mass
        candidate_row[write_idx] = row_f
        candidate_col[write_idx] = col_f
        candidate_phi[write_idx] = phi_f
        candidate_peak_idx[write_idx] = peak_idx
        candidate_count += 1

    return total_mass, valid_count, project_calls, candidate_count


@njit(fastmath=True, cache=True, nogil=True)
def _weighted_event_emit_from_stored_candidates(
    candidate_count,
    candidate_mass,
    candidate_row,
    candidate_col,
    candidate_phi,
    candidate_peak_idx,
    peak_h,
    peak_k,
    peak_l,
    sample_idx,
    targets,
    target_idx,
    cumulative_mass,
    deposit,
    collect_tables,
    flat_event_rows,
    flat_event_peak_indices,
    flat_event_count,
    event_counts,
    accumulate_image,
    image,
    image_size,
    cache_keys,
    cache_values,
    cache_entry_count,
    cache_flush_limit,
):
    pass2_total_mass = 0.0
    selected_events = 0
    have_last_valid = False
    last_row_f = np.nan
    last_col_f = np.nan
    last_phi_f = np.nan
    last_peak_idx = -1
    last_H = np.nan
    last_K = np.nan
    last_L = np.nan

    for cand_idx in range(int(candidate_count)):
        mass = candidate_mass[cand_idx]
        row_f = candidate_row[cand_idx]
        col_f = candidate_col[cand_idx]
        phi_f = candidate_phi[cand_idx]
        peak_idx = int(candidate_peak_idx[cand_idx])
        H = peak_h[peak_idx]
        K = peak_k[peak_idx]
        L = peak_l[peak_idx]

        cumulative_mass += mass
        pass2_total_mass += mass
        have_last_valid = True
        last_row_f = row_f
        last_col_f = col_f
        last_phi_f = phi_f
        last_peak_idx = peak_idx
        last_H = H
        last_K = K
        last_L = L

        hit_count = 0
        while target_idx < targets.shape[0] and cumulative_mass > targets[target_idx]:
            hit_count += 1
            target_idx += 1

        if hit_count <= 0:
            continue

        selected_events += hit_count
        event_counts[peak_idx, sample_idx] += hit_count

        deposited = True
        if accumulate_image:
            deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                image_size,
                row_f,
                col_f,
                float(hit_count) * deposit,
                cache_keys,
                cache_values,
                cache_entry_count,
                cache_flush_limit,
            )
            if needs_flush:
                cache_entry_count = _flush_local_pixel_cache(
                    image,
                    image_size,
                    cache_keys,
                    cache_values,
                )
                deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                    image_size,
                    row_f,
                    col_f,
                    float(hit_count) * deposit,
                    cache_keys,
                    cache_values,
                    cache_entry_count,
                    cache_flush_limit,
                )
                if needs_flush:
                    deposited = _accumulate_bilinear_hit(
                        image,
                        image_size,
                        row_f,
                        col_f,
                        float(hit_count) * deposit,
                    )
                    cache_entry_count = 0
            if not deposited:
                continue

        if collect_tables:
            for event_idx in range(hit_count):
                if flat_event_count >= flat_event_rows.shape[0]:
                    break
                flat_event_peak_indices[flat_event_count] = peak_idx
                flat_event_rows[flat_event_count, 0] = deposit
                flat_event_rows[flat_event_count, 1] = col_f
                flat_event_rows[flat_event_count, 2] = row_f
                flat_event_rows[flat_event_count, 3] = phi_f
                flat_event_rows[flat_event_count, 4] = H
                flat_event_rows[flat_event_count, 5] = K
                flat_event_rows[flat_event_count, 6] = L
                flat_event_rows[flat_event_count, 7] = np.nan
                flat_event_rows[flat_event_count, 8] = np.nan
                flat_event_rows[flat_event_count, 9] = float(sample_idx)
                flat_event_count += 1

    return (
        target_idx,
        cumulative_mass,
        flat_event_count,
        cache_entry_count,
        pass2_total_mass,
        selected_events,
        have_last_valid,
        last_row_f,
        last_col_f,
        last_phi_f,
        last_peak_idx,
        last_H,
        last_K,
        last_L,
    )


_WEIGHTED_EVENT_CHUNK_STAT_PROJECT_CALLS = 0
_WEIGHTED_EVENT_CHUNK_STAT_VALID_CANDIDATES = 1
_WEIGHTED_EVENT_CHUNK_STAT_SELECTED_EVENTS = 2
_WEIGHTED_EVENT_CHUNK_STAT_PASS1_TOTAL_MASS = 3
_WEIGHTED_EVENT_CHUNK_STAT_PASS2_TOTAL_MASS = 4
_WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_COUNT = 5
_WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_MAX_ABS = 6
_WEIGHTED_EVENT_CHUNK_STAT_TAIL_FILL_EVENTS = 7
_WEIGHTED_EVENT_CHUNK_STAT_STORED_CANDIDATES = 8
_WEIGHTED_EVENT_CHUNK_STAT_CANDIDATE_CAPACITY_MAX = 9
_WEIGHTED_EVENT_CHUNK_STAT_COLS = 10


@njit(fastmath=True, nogil=True, cache=True)
def _weighted_event_sample_chunk_kernel(
    worker_slot,
    sample_start,
    sample_stop,
    num_peaks,
    n_samp,
    image_size,
    phase_event_counts,
    collect_tables,
    accumulate_image_flag,
    q_values,
    qset_offsets,
    qset_lengths,
    qset_status,
    qset_id_by_sample_peak,
    all_status,
    peak_valid,
    peak_h,
    peak_k,
    peak_l,
    peak_reflection_intensity,
    sample_weight_array,
    has_sample_weight_array,
    debye_x_sq,
    debye_y_sq,
    center_row,
    center_col,
    R_sample_precomputed,
    n_det_rot,
    Detector_Pos,
    e1_det,
    e2_det,
    sample_terms,
    sample_eps2_array,
    thickness,
    pixel_scale,
    q_data,
    q_count,
    q_debug_truncated_count_parts,
    image_parts,
    cache_keys_parts,
    cache_values_parts,
    cache_flush_limit,
    candidate_mass_parts,
    candidate_row_parts,
    candidate_col_parts,
    candidate_phi_parts,
    candidate_peak_idx_parts,
    candidate_capacity,
    collect_representatives,
    representative_peak_mask,
    representative_rows_parts,
    representative_ranks_parts,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    wavelength_array,
    wavelength_center,
    flat_event_rows_parts,
    flat_event_peak_indices_parts,
    flat_event_count_parts,
    event_counts,
    stats_parts,
):
    worker_slot_i = int(worker_slot)
    sample_start_i = max(0, int(sample_start))
    sample_stop_i = min(int(sample_stop), int(n_samp))
    flat_event_count = 0
    cache_keys = cache_keys_parts[worker_slot_i]
    cache_values = cache_values_parts[worker_slot_i]
    image_part = image_parts[worker_slot_i]
    candidate_mass = candidate_mass_parts[worker_slot_i]
    candidate_row = candidate_row_parts[worker_slot_i]
    candidate_col = candidate_col_parts[worker_slot_i]
    candidate_phi = candidate_phi_parts[worker_slot_i]
    candidate_peak_idx = candidate_peak_idx_parts[worker_slot_i]
    representative_rows = representative_rows_parts[worker_slot_i]
    representative_ranks = representative_ranks_parts[worker_slot_i]
    q_debug_truncated_count = q_debug_truncated_count_parts[worker_slot_i]

    n_project_candidate_calls = 0
    n_valid_candidates = 0
    n_selected_events = 0
    pass1_total_mass = 0.0
    pass2_total_mass = 0.0
    pass2_mass_mismatch_count = 0
    pass2_mass_mismatch_max_abs = 0.0
    tail_fill_events = 0
    n_stored_projected_candidates = 0
    candidate_buffer_capacity_max = 0

    for sample_idx in range(sample_start_i, sample_stop_i):
        if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
            for peak_idx in range(num_peaks):
                all_status[peak_idx, sample_idx] = -10
            continue

        sample_weight = 1.0
        if has_sample_weight_array:
            sample_weight = float(sample_weight_array[sample_idx])
            if (not np.isfinite(sample_weight)) or sample_weight <= 0.0:
                for peak_idx in range(num_peaks):
                    all_status[peak_idx, sample_idx] = -12
                continue

        sample_total_mass = 0.0
        if accumulate_image_flag:
            _clear_local_pixel_cache(cache_keys, cache_values)
        cache_entry_count = 0
        sample_candidate_capacity = 0
        for peak_idx in range(num_peaks):
            if int(peak_valid[peak_idx]) == 0:
                continue
            qset_id = int(qset_id_by_sample_peak[sample_idx, peak_idx])
            if qset_id < 0:
                continue
            all_status[peak_idx, sample_idx] = int(qset_status[qset_id])
            sample_candidate_capacity += int(qset_lengths[qset_id])
        if sample_candidate_capacity > candidate_buffer_capacity_max:
            candidate_buffer_capacity_max = int(sample_candidate_capacity)
        if sample_candidate_capacity > candidate_capacity:
            raise RuntimeError("weighted-event candidate buffer capacity was underestimated")
        candidate_count = 0

        for peak_idx in range(num_peaks):
            if int(peak_valid[peak_idx]) == 0:
                continue

            reflection_intensity = float(peak_reflection_intensity[peak_idx])
            qset_id = int(qset_id_by_sample_peak[sample_idx, peak_idx])
            if qset_id < 0:
                continue
            offset = int(qset_offsets[qset_id])
            q_rows = int(qset_lengths[qset_id])
            all_q = q_values[offset : offset + q_rows, :]
            peak_mass, valid_count, project_calls, candidate_count = (
                _weighted_event_project_store_for_qset(
                    all_q,
                    peak_idx,
                    sample_idx,
                    reflection_intensity,
                    sample_weight,
                    debye_x_sq,
                    debye_y_sq,
                    center_row,
                    center_col,
                    R_sample_precomputed,
                    n_det_rot,
                    Detector_Pos,
                    e1_det,
                    e2_det,
                    sample_terms,
                    sample_eps2_array,
                    thickness,
                    pixel_scale,
                    int(image_size),
                    0,
                    q_data,
                    q_count,
                    q_debug_truncated_count,
                    candidate_mass,
                    candidate_row,
                    candidate_col,
                    candidate_phi,
                    candidate_peak_idx,
                    int(candidate_count),
                )
            )
            sample_total_mass += float(peak_mass)
            pass1_total_mass += float(peak_mass)
            n_valid_candidates += int(valid_count)
            n_project_candidate_calls += int(project_calls)
        n_stored_projected_candidates += int(candidate_count)

        if collect_representatives:
            for candidate_index in range(int(candidate_count)):
                representative_peak_idx = int(candidate_peak_idx[candidate_index])
                if representative_peak_idx < 0 or representative_peak_idx >= num_peaks:
                    raise RuntimeError(
                        "Projected candidate contains an out-of-range peak index."
                    )
                if not bool(representative_peak_mask[representative_peak_idx]):
                    continue
                _weighted_event_update_representative(
                    representative_rows,
                    representative_ranks,
                    representative_peak_idx,
                    sample_idx,
                    float(peak_h[representative_peak_idx]),
                    float(peak_k[representative_peak_idx]),
                    float(peak_l[representative_peak_idx]),
                    float(candidate_row[candidate_index]),
                    float(candidate_col[candidate_index]),
                    float(candidate_phi[candidate_index]),
                    float(candidate_mass[candidate_index]),
                    beam_x_array,
                    beam_y_array,
                    theta_array,
                    phi_array,
                    wavelength_array,
                    wavelength_center,
                )

        if (not np.isfinite(sample_total_mass)) or sample_total_mass <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                _flush_local_pixel_cache(image_part, int(image_size), cache_keys, cache_values)
            continue

        event_count = int(phase_event_counts[sample_idx])
        targets = _weighted_event_targets(sample_total_mass, event_count, sample_idx)
        deposit = _weighted_event_deposit(sample_total_mass, event_count)
        if targets.size <= 0 or (not np.isfinite(deposit)) or deposit <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                _flush_local_pixel_cache(image_part, int(image_size), cache_keys, cache_values)
            continue

        target_idx = 0
        cumulative_mass = 0.0
        phase_have_last_valid = False
        phase_last_row_f = np.nan
        phase_last_col_f = np.nan
        phase_last_phi_f = np.nan
        phase_last_peak_idx = -1
        phase_last_H = np.nan
        phase_last_K = np.nan
        phase_last_L = np.nan
        (
            target_idx,
            cumulative_mass,
            flat_event_count,
            cache_entry_count,
            sample_pass2_mass,
            selected_events,
            phase_have_last_valid,
            phase_last_row_f,
            phase_last_col_f,
            phase_last_phi_f,
            phase_last_peak_idx,
            phase_last_H,
            phase_last_K,
            phase_last_L,
        ) = _weighted_event_emit_from_stored_candidates(
            int(candidate_count),
            candidate_mass,
            candidate_row,
            candidate_col,
            candidate_phi,
            candidate_peak_idx,
            peak_h,
            peak_k,
            peak_l,
            sample_idx,
            targets,
            int(target_idx),
            float(cumulative_mass),
            float(deposit),
            collect_tables,
            flat_event_rows_parts[worker_slot_i],
            flat_event_peak_indices_parts[worker_slot_i],
            int(flat_event_count),
            event_counts,
            accumulate_image_flag,
            image_part,
            int(image_size),
            cache_keys,
            cache_values,
            int(cache_entry_count),
            int(cache_flush_limit),
        )
        pass2_total_mass += float(sample_pass2_mass)
        n_selected_events += int(selected_events)

        sample_pass2_mass_delta = abs(float(cumulative_mass) - float(sample_total_mass))
        sample_pass2_mass_tol = max(
            1.0e-10,
            1.0e-9 * max(1.0, abs(float(sample_total_mass))),
        )
        if (
            not np.isfinite(sample_pass2_mass_delta)
        ) or sample_pass2_mass_delta > sample_pass2_mass_tol:
            pass2_mass_mismatch_count += 1
            if np.isfinite(sample_pass2_mass_delta):
                pass2_mass_mismatch_max_abs = max(
                    float(pass2_mass_mismatch_max_abs),
                    float(sample_pass2_mass_delta),
                )
            else:
                pass2_mass_mismatch_max_abs = float("inf")
        elif target_idx < targets.shape[0] and phase_have_last_valid:
            hit_count = int(targets.shape[0] - target_idx)
            n_selected_events += int(hit_count)
            tail_fill_events += int(hit_count)
            if 0 <= phase_last_peak_idx < event_counts.shape[0]:
                event_counts[phase_last_peak_idx, sample_idx] += int(hit_count)
            target_idx = targets.shape[0]

            deposited = True
            if accumulate_image_flag:
                deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                    int(image_size),
                    float(phase_last_row_f),
                    float(phase_last_col_f),
                    float(hit_count) * float(deposit),
                    cache_keys,
                    cache_values,
                    int(cache_entry_count),
                    int(cache_flush_limit),
                )
                if needs_flush:
                    cache_entry_count = _flush_local_pixel_cache(
                        image_part,
                        int(image_size),
                        cache_keys,
                        cache_values,
                    )
                    deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                        int(image_size),
                        float(phase_last_row_f),
                        float(phase_last_col_f),
                        float(hit_count) * float(deposit),
                        cache_keys,
                        cache_values,
                        int(cache_entry_count),
                        int(cache_flush_limit),
                    )
                    if needs_flush:
                        deposited = _accumulate_bilinear_hit(
                            image_part,
                            int(image_size),
                            float(phase_last_row_f),
                            float(phase_last_col_f),
                            float(hit_count) * float(deposit),
                        )
                        cache_entry_count = 0

            if collect_tables and deposited:
                tail_start = int(flat_event_count)
                tail_stop = tail_start + int(hit_count)
                if tail_stop > flat_event_rows_parts.shape[1]:
                    tail_stop = flat_event_rows_parts.shape[1]
                for write_idx in range(tail_start, tail_stop):
                    flat_event_peak_indices_parts[worker_slot_i, write_idx] = (
                        phase_last_peak_idx
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 0] = float(deposit)
                    flat_event_rows_parts[worker_slot_i, write_idx, 1] = float(
                        phase_last_col_f
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 2] = float(
                        phase_last_row_f
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 3] = float(
                        phase_last_phi_f
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 4] = float(
                        phase_last_H
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 5] = float(
                        phase_last_K
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 6] = float(
                        phase_last_L
                    )
                    flat_event_rows_parts[worker_slot_i, write_idx, 7] = np.nan
                    flat_event_rows_parts[worker_slot_i, write_idx, 8] = np.nan
                    flat_event_rows_parts[worker_slot_i, write_idx, 9] = float(sample_idx)
                flat_event_count = tail_stop

        if accumulate_image_flag and cache_entry_count > 0:
            _flush_local_pixel_cache(image_part, int(image_size), cache_keys, cache_values)

    flat_event_count_parts[worker_slot_i] = int(flat_event_count)
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_PROJECT_CALLS] = float(
        n_project_candidate_calls
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_VALID_CANDIDATES] = float(
        n_valid_candidates
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_SELECTED_EVENTS] = float(
        n_selected_events
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_PASS1_TOTAL_MASS] = float(
        pass1_total_mass
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_PASS2_TOTAL_MASS] = float(
        pass2_total_mass
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_COUNT] = float(
        pass2_mass_mismatch_count
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_MAX_ABS] = float(
        pass2_mass_mismatch_max_abs
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_TAIL_FILL_EVENTS] = float(
        tail_fill_events
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_STORED_CANDIDATES] = float(
        n_stored_projected_candidates
    )
    stats_parts[worker_slot_i, _WEIGHTED_EVENT_CHUNK_STAT_CANDIDATE_CAPACITY_MAX] = float(
        candidate_buffer_capacity_max
    )


def _weighted_event_chunk_bounds(n_samp: int, worker_count: int):
    chunks = []
    n_samp_i = max(int(n_samp), 0)
    worker_count_i = max(int(worker_count), 1)
    for worker_slot in range(worker_count_i):
        start = worker_slot * n_samp_i // worker_count_i
        stop = (worker_slot + 1) * n_samp_i // worker_count_i
        if start < stop:
            chunks.append((int(worker_slot), int(start), int(stop)))
    return chunks


def _run_weighted_event_sample_chunks(kernel_args, *, n_samp: int, worker_count: int):
    chunks = _weighted_event_chunk_bounds(n_samp, worker_count)
    if not chunks:
        return chunks

    _weighted_event_sample_chunk_kernel(0, 0, 0, *kernel_args)
    if len(chunks) == 1:
        worker_slot, start, stop = chunks[0]
        _weighted_event_sample_chunk_kernel(worker_slot, start, stop, *kernel_args)
        return chunks

    with ThreadPoolExecutor(max_workers=max(int(worker_count), 1)) as pool:
        futures = [
            pool.submit(
                _weighted_event_sample_chunk_kernel,
                worker_slot,
                start,
                stop,
                *kernel_args,
            )
            for worker_slot, start, stop in chunks
        ]
        for future in futures:
            future.result()
    return chunks


def _weighted_event_requested_worker_count_for_stats(requested):
    if requested is None:
        return None
    try:
        parsed = int(requested)
    except Exception:
        return None
    return int(parsed) if parsed > 0 else None


def _weighted_event_config_worker_count():
    try:
        config = get_instrument_config()
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    instrument_config = config.get("instrument", config)
    if not isinstance(instrument_config, dict):
        return None
    for section_name in ("simulation", "beam"):
        section = instrument_config.get(section_name)
        if isinstance(section, dict) and "weighted_event_worker_count" in section:
            return section.get("weighted_event_worker_count")
    return None


def _resolve_weighted_event_fast_worker_count(
    *,
    requested_threads,
    n_samp,
    image_size,
    accumulate_image,
    save_flag,
):
    if int(save_flag) == 1 or int(n_samp) <= 1:
        source = "explicit" if requested_threads is not None else "auto"
        return 1, source

    resolved_requested_threads = requested_threads
    config_worker_count = None
    used_config_worker_count = False
    if requested_threads is None:
        config_worker_count = _weighted_event_config_worker_count()
        if _weighted_event_requested_worker_count_for_stats(config_worker_count) is not None:
            resolved_requested_threads = config_worker_count
            used_config_worker_count = True

    worker_count, source = resolve_weighted_event_worker_count(
        resolved_requested_threads,
        n_samp=int(n_samp),
        outer_workers=1,
    )
    if used_config_worker_count and source == "explicit":
        source = "config"
    worker_count = max(int(worker_count), 1)
    return int(worker_count), str(source)


def _weighted_event_parallel_eligible(
    *,
    worker_count,
    save_flag,
    n_samp,
):
    return int(worker_count) > 1 and int(n_samp) > 1 and int(save_flag) != 1


def _precompute_weighted_event_qsets(
    *,
    num_peaks,
    n_samp,
    sample_terms,
    sample_weight_array,
    peak_valid,
    peak_gr0,
    peak_gz0,
    sigma_rad,
    gamma_rad_m,
    eta_pv,
    solve_q_steps_i,
    solve_q_rel_tol_i,
    solve_q_mode_i,
    default_solve_q_dtheta,
    default_solve_q_cos,
    default_solve_q_sin,
    worker_count=1,
):
    index_start = time.perf_counter()
    qset_id_by_sample_peak = np.full(
        (int(n_samp), int(num_peaks)),
        -1,
        dtype=np.int64,
    )
    has_sample_weight_array = sample_weight_array is not None
    valid_peak_indices = np.flatnonzero(np.asarray(peak_valid) != 0).astype(
        np.int64,
        copy=False,
    )
    valid_peak_count = int(valid_peak_indices.size)
    rep_idx_to_block: dict[int, int] = {}
    unique_rep_indices: list[int] = []

    for sample_idx in range(int(n_samp)):
        if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
            continue
        if has_sample_weight_array:
            sample_weight = float(sample_weight_array[sample_idx])
            if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                continue

        rep_idx = int(sample_terms[sample_idx, _SAMPLE_COL_SOLVE_Q_REP])
        if rep_idx < 0:
            rep_idx = sample_idx
        if rep_idx < 0 or rep_idx >= int(n_samp):
            rep_idx = sample_idx

        block_idx = rep_idx_to_block.get(int(rep_idx))
        if block_idx is None:
            block_idx = len(unique_rep_indices)
            rep_idx_to_block[int(rep_idx)] = int(block_idx)
            unique_rep_indices.append(int(rep_idx))
        if valid_peak_count > 0:
            block_start = int(block_idx) * valid_peak_count
            qset_id_by_sample_peak[sample_idx, valid_peak_indices] = (
                block_start + np.arange(valid_peak_count, dtype=np.int64)
            )

    n_qsets = int(len(unique_rep_indices) * valid_peak_count)
    time_qset_index = time.perf_counter() - index_start
    q_arrays: list[np.ndarray | None] = [None] * n_qsets
    qset_status = np.empty(n_qsets, dtype=np.int64)

    def _solve_qset_range(start: int, stop: int) -> None:
        for qset_id in range(int(start), int(stop)):
            block_idx, peak_position = divmod(int(qset_id), valid_peak_count)
            rep_idx = int(unique_rep_indices[block_idx])
            peak_idx = int(valid_peak_indices[peak_position])
            k_in_crystal = np.asarray(
                [
                    sample_terms[rep_idx, _SAMPLE_COL_KX_SCAT],
                    sample_terms[rep_idx, _SAMPLE_COL_KY_SCAT],
                    sample_terms[rep_idx, _SAMPLE_COL_RE_KZ],
                ],
                dtype=np.float64,
            )
            G_vec = np.asarray(
                [0.0, peak_gr0[peak_idx], peak_gz0[peak_idx]],
                dtype=np.float64,
            )
            all_q_raw, stat = solve_q(
                k_in_crystal,
                float(sample_terms[rep_idx, _SAMPLE_COL_K_SCAT]),
                G_vec,
                sigma_rad,
                gamma_rad_m,
                eta_pv,
                solve_q_steps_i,
                DEFAULT_SOLVE_Q_BASE_INTERVALS,
                solve_q_rel_tol_i,
                solve_q_mode_i,
                default_solve_q_dtheta,
                default_solve_q_cos,
                default_solve_q_sin,
            )
            q_arrays[qset_id] = np.asarray(all_q_raw, dtype=np.float64).reshape(-1, 4)
            qset_status[qset_id] = int(stat)

    solve_start = time.perf_counter()
    if n_qsets > 0:
        # Compile/load the Numba specialization once before worker threads enter it.
        _solve_qset_range(0, 1)
        remaining_qsets = n_qsets - 1
        solve_worker_count = min(max(int(worker_count), 1), remaining_qsets)
        if solve_worker_count > 1:
            chunks = _weighted_event_chunk_bounds(remaining_qsets, solve_worker_count)
            with ThreadPoolExecutor(max_workers=solve_worker_count) as pool:
                futures = [
                    pool.submit(
                        _solve_qset_range,
                        int(start) + 1,
                        int(stop) + 1,
                    )
                    for _worker_slot, start, stop in chunks
                ]
                for future in futures:
                    future.result()
        elif remaining_qsets > 0:
            _solve_qset_range(1, n_qsets)
    time_solve_q = time.perf_counter() - solve_start

    pack_start = time.perf_counter()
    qset_offsets = np.empty(n_qsets, dtype=np.int64)
    qset_lengths = np.empty(n_qsets, dtype=np.int64)
    total_q = 0
    for qset_id, all_q in enumerate(q_arrays):
        if all_q is None:
            raise RuntimeError("Weighted-event Q-set solve did not produce a result.")
        qset_offsets[qset_id] = int(total_q)
        q_rows = int(all_q.shape[0])
        qset_lengths[qset_id] = q_rows
        total_q += q_rows

    q_values = np.empty((int(total_q), 4), dtype=np.float64)
    for qset_id, all_q in enumerate(q_arrays):
        if all_q is None:
            raise RuntimeError("Weighted-event Q-set solve did not produce a result.")
        q_rows = int(all_q.shape[0])
        if q_rows <= 0:
            continue
        offset = int(qset_offsets[qset_id])
        q_values[offset : offset + q_rows, :] = all_q

    time_qset_index += time.perf_counter() - pack_start

    return (
        q_values,
        qset_offsets,
        qset_lengths,
        qset_status,
        qset_id_by_sample_peak,
        int(n_qsets),
        float(time_solve_q),
        float(time_qset_index),
    )


# =============================================================================
# 5) CALCULATE_PHI
# =============================================================================














# =============================================================================
# 6) PROCESS_PEAKS_PARALLEL
# =============================================================================


def _build_weighted_event_hit_tables(
    flat_event_rows,
    flat_event_peak_indices,
    flat_event_count,
    num_peaks,
):
    if int(num_peaks) <= 0:
        return []
    event_rows = np.asarray(flat_event_rows)
    event_peak_indices = np.asarray(flat_event_peak_indices)
    event_count = int(flat_event_count)
    if event_rows.ndim != 2 or event_peak_indices.ndim != 1:
        raise ValueError("Weighted-event rows and peak indices have invalid dimensions.")
    if event_count < 0 or event_count > event_rows.shape[0] or event_count > event_peak_indices.size:
        raise ValueError("flat_event_count exceeds weighted-event buffer capacity.")
    active_peak_indices = event_peak_indices[:event_count]
    if np.any(active_peak_indices < 0) or np.any(active_peak_indices >= int(num_peaks)):
        raise ValueError("Weighted-event output contains an out-of-range peak index.")
    row_width = (
        int(event_rows.shape[1])
        if int(event_rows.shape[1]) > 0
        else HIT_ROW_WITH_PROVENANCE_WIDTH
    )
    if row_width not in {HIT_ROW_WITH_PROVENANCE_WIDTH, HIT_ROW_WITH_CONTEXT_WIDTH}:
        raise ValueError("Current weighted-event rows require the 10- or 15-column layout.")
    counts = np.zeros(int(num_peaks), dtype=np.int64)
    for row_idx in range(event_count):
        peak_idx = int(event_peak_indices[row_idx])
        counts[peak_idx] += 1
    hit_tables = [
        np.empty((int(counts[peak_idx]), row_width), dtype=np.float64)
        if counts[peak_idx] > 0
        else np.empty((0, row_width), dtype=np.float64)
        for peak_idx in range(int(num_peaks))
    ]
    offsets = np.zeros(int(num_peaks), dtype=np.int64)
    for row_idx in range(event_count):
        peak_idx = int(event_peak_indices[row_idx])
        out_idx = int(offsets[peak_idx])
        hit_tables[peak_idx][out_idx, :] = event_rows[row_idx, :]
        hit_tables[peak_idx][out_idx, HIT_ROW_COL_SOURCE_TABLE_INDEX] = float(peak_idx)
        hit_tables[peak_idx][out_idx, HIT_ROW_COL_SOURCE_ROW_INDEX] = float(out_idx)
        offsets[peak_idx] += 1
    return hit_tables


def _weighted_event_phi_branch_index(H: float, K: float, phi_f: float) -> int | None:
    if abs(float(H)) < 1.0e-12 and abs(float(K)) < 1.0e-12:
        return None
    phi_value = float(phi_f)
    if phi_value <= -pi + 1.0e-12:
        phi_value = pi
    if phi_value > SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD:
        return 1
    if phi_value < -SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD:
        return 0
    return None


def _weighted_event_representative_state(
    num_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.full(
        (int(num_peaks), 2, HIT_ROW_WITH_CONTEXT_WIDTH),
        np.nan,
        dtype=np.float64,
    )
    ranks = np.full((int(num_peaks), 2, 5), np.inf, dtype=np.float64)
    return rows, ranks


def _weighted_event_representative_peak_mask(
    num_peaks: int,
    peak_indices,
) -> np.ndarray:
    """Return the local peak mask to rank for requested representatives."""

    count = int(num_peaks)
    if count < 0:
        raise ValueError("num_peaks must be nonnegative.")
    if peak_indices is None:
        return np.ones(count, dtype=bool)
    mask = np.zeros(count, dtype=bool)
    try:
        values = np.asarray(peak_indices, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("representative_peak_indices must contain integers.") from exc
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError("representative_peak_indices must contain finite integers.")
    if np.any(values < 0) or np.any(values >= count):
        raise ValueError("representative_peak_indices contains an out-of-range index.")
    indices = values.astype(np.int64)
    mask[indices] = True
    return mask


@njit(cache=True, nogil=True)
def _weighted_event_update_representative(
    rows: np.ndarray | None,
    ranks: np.ndarray | None,
    peak_idx: int,
    sample_idx: int,
    H: float,
    K: float,
    L: float,
    row_f: float,
    col_f: float,
    phi_f: float,
    mass: float,
    beam_x_array: np.ndarray,
    beam_y_array: np.ndarray,
    theta_array: np.ndarray,
    phi_array: np.ndarray,
    wavelength_array: np.ndarray,
    wavelength_center: float,
) -> None:
    if rows is None or ranks is None:
        return
    if rows.ndim != 3 or ranks.ndim != 3:
        return
    if rows.shape[1] < 2 or ranks.shape[1] < 2:
        return
    if rows.shape[2] <= HIT_ROW_COL_WAVELENGTH_OFFSET or ranks.shape[2] < 5:
        return
    peak_i = int(peak_idx)
    sample_i = int(sample_idx)
    if peak_i < 0 or peak_i >= rows.shape[0] or peak_i >= ranks.shape[0]:
        return
    if sample_i < 0 or sample_i >= theta_array.size:
        return
    if not (
        np.isfinite(row_f)
        and np.isfinite(col_f)
        and np.isfinite(phi_f)
        and np.isfinite(float(mass))
        and float(mass) > 0.0
    ):
        return
    if not (np.isfinite(H) and np.isfinite(K) and np.isfinite(L)):
        raise ValueError("Representative Miller indices must be finite.")
    h_int = int(np.rint(float(H)))
    k_int = int(np.rint(float(K)))
    if h_int == 0 and k_int == 0:
        branch_i = 0
    else:
        phi_value = float(phi_f)
        if phi_value <= -pi + 1.0e-12:
            phi_value = pi
        if phi_value > SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD:
            branch_i = 1
        elif phi_value < -SOURCE_BRANCH_PHI_ZERO_DEADBAND_RAD:
            branch_i = 0
        else:
            return
    if branch_i < 0 or branch_i >= rows.shape[1] or branch_i >= ranks.shape[1]:
        return
    theta_offset = float(theta_array[sample_i]) if sample_i < theta_array.size else np.nan
    phi_offset = float(phi_array[sample_i]) if sample_i < phi_array.size else np.nan
    beam_x_offset = float(beam_x_array[sample_i]) if sample_i < beam_x_array.size else np.nan
    beam_y_offset = float(beam_y_array[sample_i]) if sample_i < beam_y_array.size else np.nan
    wavelength_offset = (
        float(wavelength_array[sample_i]) - float(wavelength_center)
        if sample_i < wavelength_array.size
        else np.nan
    )
    angular_rank = (
        theta_offset * theta_offset + phi_offset * phi_offset
        if np.isfinite(theta_offset) and np.isfinite(phi_offset)
        else np.inf
    )
    beam_rank = (
        beam_x_offset * beam_x_offset + beam_y_offset * beam_y_offset
        if np.isfinite(beam_x_offset) and np.isfinite(beam_y_offset)
        else np.inf
    )
    wavelength_rank = abs(float(wavelength_offset)) if np.isfinite(wavelength_offset) else np.inf
    rank = np.asarray(
        [
            float(angular_rank),
            float(beam_rank),
            float(wavelength_rank),
            -float(mass),
            float(sample_i),
        ],
        dtype=np.float64,
    )
    current = ranks[peak_i, branch_i, :]
    keep_current = True
    for rank_idx in range(rank.shape[0]):
        proposed_value = float(rank[rank_idx])
        current_value = float(current[rank_idx])
        if proposed_value < current_value:
            keep_current = False
            break
        if proposed_value > current_value:
            break
    if keep_current:
        return
    ranks[peak_i, branch_i, :] = rank
    row = rows[peak_i, branch_i, :]
    row[:] = np.nan
    row[HIT_ROW_COL_INTENSITY] = float(mass)
    row[HIT_ROW_COL_DETECTOR_COL] = float(col_f)
    row[HIT_ROW_COL_DETECTOR_ROW] = float(row_f)
    row[HIT_ROW_COL_PHI] = float(phi_f)
    row[HIT_ROW_COL_H] = float(H)
    row[HIT_ROW_COL_K] = float(K)
    row[HIT_ROW_COL_L] = float(L)
    row[HIT_ROW_COL_SOURCE_TABLE_INDEX] = float(peak_i)
    row[HIT_ROW_COL_BEST_SAMPLE_INDEX] = float(sample_i)
    row[HIT_ROW_COL_BEAM_X_OFFSET] = float(beam_x_offset)
    row[HIT_ROW_COL_BEAM_Y_OFFSET] = float(beam_y_offset)
    row[HIT_ROW_COL_THETA_OFFSET] = float(theta_offset)
    row[HIT_ROW_COL_PHI_OFFSET] = float(phi_offset)
    row[HIT_ROW_COL_WAVELENGTH_OFFSET] = float(wavelength_offset)


def _weighted_event_representative_hit_tables(
    rows: np.ndarray | None,
) -> list[np.ndarray]:
    if rows is None:
        return []
    tables: list[np.ndarray] = []
    for peak_idx in range(int(rows.shape[0])):
        peak_rows = np.asarray(rows[peak_idx], dtype=np.float64)
        valid = np.isfinite(peak_rows[:, HIT_ROW_COL_INTENSITY]) & (
            peak_rows[:, HIT_ROW_COL_INTENSITY] > 0.0
        )
        if np.any(valid):
            table = np.asarray(peak_rows[valid], dtype=np.float64).copy()
            table[:, HIT_ROW_COL_SOURCE_ROW_INDEX] = np.arange(
                int(table.shape[0]),
                dtype=np.float64,
            )
            tables.append(table)
        else:
            tables.append(np.empty((0, HIT_ROW_WITH_CONTEXT_WIDTH), dtype=np.float64))
    return tables


def _merge_weighted_event_representative_parts(
    rows_parts: np.ndarray,
    ranks_parts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge independent worker rankings with the serial lexicographic rule."""

    worker_count, num_peaks = int(ranks_parts.shape[0]), int(ranks_parts.shape[1])
    rows, ranks = _weighted_event_representative_state(num_peaks)
    for worker_idx in range(worker_count):
        for peak_idx in range(num_peaks):
            for branch_idx in range(2):
                proposed_rank = ranks_parts[worker_idx, peak_idx, branch_idx]
                current_rank = ranks[peak_idx, branch_idx]
                for rank_idx in range(int(proposed_rank.shape[0])):
                    proposed_value = float(proposed_rank[rank_idx])
                    current_value = float(current_rank[rank_idx])
                    if proposed_value < current_value:
                        ranks[peak_idx, branch_idx, :] = proposed_rank
                        rows[peak_idx, branch_idx, :] = rows_parts[
                            worker_idx, peak_idx, branch_idx, :
                        ]
                        break
                    if proposed_value > current_value:
                        break
    return rows, ranks


def _process_peaks_parallel_weighted_events_fast_serial(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    thickness=50e-9,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    numba_thread_count=None,
    events_per_beam_phase=50,
    collect_representative_hit_tables=False,
    representative_peak_indices=None,
    hit_table_collection_mode="quantile_events",
    *,
    weighted_event_candidate_buffer_max_bytes=_WEIGHTED_EVENT_CANDIDATE_DEFAULT_MAX_BYTES,
):
    if not isinstance(events_per_beam_phase, (int, np.integer)) or not (
        1 <= events_per_beam_phase <= 1000
    ):
        raise ValueError("events_per_beam_phase must be an integer from 1 through 1000.")
    if numba_thread_count is not None and (
        not isinstance(numba_thread_count, (int, np.integer)) or numba_thread_count < 1
    ):
        raise ValueError("numba_thread_count must be a positive integer or None.")
    if not isinstance(weighted_event_candidate_buffer_max_bytes, (int, np.integer)) or (
        weighted_event_candidate_buffer_max_bytes < 1
    ):
        raise ValueError("weighted_event_candidate_buffer_max_bytes must be a positive integer.")
    requested_worker_count = _weighted_event_requested_worker_count_for_stats(numba_thread_count)

    miller = np.asarray(miller, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    image = np.asarray(image, dtype=np.float64)
    beam_x_array = np.asarray(beam_x_array, dtype=np.float64).reshape(-1)
    beam_y_array = np.asarray(beam_y_array, dtype=np.float64).reshape(-1)
    theta_array = np.asarray(theta_array, dtype=np.float64).reshape(-1)
    phi_array = np.asarray(phi_array, dtype=np.float64).reshape(-1)
    wavelength_array = np.asarray(wavelength_array, dtype=np.float64).reshape(-1)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    unit_x = np.asarray(unit_x, dtype=np.float64).reshape(-1)
    n_detector = np.asarray(n_detector, dtype=np.float64).reshape(-1)
    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError("miller must have shape (n, 3).")
    if intensities.size != miller.shape[0]:
        raise ValueError("intensities length must match the number of Miller rows.")
    if not np.all(np.isfinite(miller)):
        raise ValueError("miller must contain finite values.")
    if not np.all(np.isfinite(intensities)):
        raise ValueError("intensities must contain finite values.")
    sample_count = beam_x_array.size
    if not all(
        values.size == sample_count
        for values in (beam_y_array, theta_array, phi_array, wavelength_array)
    ):
        raise ValueError("Beam coordinate and wavelength arrays must have equal lengths.")
    if center.size < 2:
        raise ValueError("center must contain detector row and column coordinates.")
    if unit_x.size != 3 or n_detector.size != 3:
        raise ValueError("unit_x and n_detector must each contain three components.")
    collect_representatives = bool(collect_representative_hit_tables)
    normalized_hit_table_collection_mode = str(hit_table_collection_mode)
    if normalized_hit_table_collection_mode not in {
        "quantile_events",
        "all_weighted_candidates",
    }:
        raise ValueError(
            "hit_table_collection_mode must be 'quantile_events' or "
            "'all_weighted_candidates'."
        )
    collect_all_weighted_candidates = bool(
        normalized_hit_table_collection_mode == "all_weighted_candidates"
    )
    if collect_all_weighted_candidates and not bool(collect_hit_tables):
        raise ValueError("All-weighted-candidate mode requires hit-table collection.")
    gamma_rad = float(gamma_deg) * (pi / 180.0)
    Gamma_rad = float(Gamma_deg) * (pi / 180.0)
    chi_rad = float(chi_deg) * (pi / 180.0)
    psi_rad = float(psi_deg) * (pi / 180.0)
    sigma_rad = float(sigma_pv_deg) * (pi / 180.0)
    gamma_rad_m = float(gamma_pv_deg) * (pi / 180.0)
    solve_q_steps_i = int(np.clip(int(solve_q_steps), MIN_SOLVE_Q_STEPS, MAX_SOLVE_Q_STEPS))
    solve_q_rel_tol_i = float(
        np.clip(float(solve_q_rel_tol), MIN_SOLVE_Q_REL_TOL, MAX_SOLVE_Q_REL_TOL)
    )
    solve_q_mode_i = (
        SOLVE_Q_MODE_UNIFORM if int(solve_q_mode) == SOLVE_Q_MODE_UNIFORM else SOLVE_Q_MODE_ADAPTIVE
    )

    cg = cos(gamma_rad)
    sg = sin(gamma_rad)
    cG = cos(Gamma_rad)
    sG = sin(Gamma_rad)
    R_x_det = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]], dtype=np.float64)
    R_z_det = np.array([[cG, sG, 0.0], [-sG, cG, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    nd_temp = R_x_det @ n_detector
    n_det_rot = R_z_det @ nd_temp
    nd_len = sqrt(float(np.dot(n_det_rot, n_det_rot)))
    if nd_len > 1.0e-30:
        n_det_rot = n_det_rot / nd_len

    Detector_Pos = np.array([0.0, float(Distance_CoR_to_Detector), 0.0], dtype=np.float64)

    dot_e1 = float(np.dot(unit_x, n_det_rot))
    e1_det = unit_x - dot_e1 * n_det_rot
    e1_len = sqrt(float(np.dot(e1_det, e1_det)))
    if e1_len < 1.0e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det = e1_det / e1_len

    e2_det = -np.cross(n_det_rot, e1_det)
    e2_len = sqrt(float(np.dot(e2_det, e2_det)))
    if e2_len < 1.0e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det = e2_det / e2_len

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    R_y = np.array([[c_chi, 0.0, s_chi], [0.0, 1.0, 0.0], [-s_chi, 0.0, c_chi]], dtype=np.float64)
    c_psi = cos(psi_rad)
    s_psi = sin(psi_rad)
    R_z = np.array([[c_psi, s_psi, 0.0], [-s_psi, c_psi, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_z_R_y = R_z @ R_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_ZY_n = R_z_R_y @ n1
    nzy_len = sqrt(float(np.dot(R_ZY_n, R_ZY_n)))
    if nzy_len > 1.0e-30:
        R_ZY_n = R_ZY_n / nzy_len
    P0 = np.array([0.0, 0.0, -float(zs)], dtype=np.float64)

    num_peaks = int(miller.shape[0])
    n_samp = int(beam_x_array.size)
    return_hit_tables = bool(collect_hit_tables)
    collect_tables = bool(return_hit_tables and not collect_all_weighted_candidates)
    accumulate_image_flag = bool(accumulate_image)
    finite_wavelengths = wavelength_array[np.isfinite(wavelength_array)]
    wavelength_center = (
        float(np.mean(finite_wavelengths)) if finite_wavelengths.size else float("nan")
    )
    worker_count, worker_count_source = _resolve_weighted_event_fast_worker_count(
        requested_threads=numba_thread_count,
        n_samp=n_samp,
        image_size=image_size,
        accumulate_image=accumulate_image_flag,
        save_flag=save_flag,
    )
    if collect_all_weighted_candidates and int(worker_count) != 1:
        worker_count = 1
        worker_count_source = f"{worker_count_source}:all_weighted_candidates_serial"
    all_status = np.zeros((num_peaks, n_samp), dtype=np.int64)
    event_counts = np.zeros((num_peaks, n_samp), dtype=np.int64)
    phase_event_counts = np.full(n_samp, int(events_per_beam_phase), dtype=np.int64)
    flat_capacity_total = int(np.sum(phase_event_counts)) if n_samp > 0 else 0

    q_data, q_count = _allocate_q_debug_buffers(num_peaks, save_flag)
    q_debug_truncated_count = np.zeros(1, dtype=np.int64)
    if return_hit_tables:
        miss_tables = [np.empty((0, 3), dtype=np.float64) for _ in range(num_peaks)]
    else:
        miss_tables = []
    if collect_tables:
        flat_capacity = max(int(flat_capacity_total), 0)
        flat_event_rows = np.empty(
            (flat_capacity, HIT_ROW_WITH_PROVENANCE_WIDTH),
            dtype=np.float64,
        )
        flat_event_rows.fill(np.nan)
        flat_event_peak_indices = np.empty(flat_capacity, dtype=np.int64)
    else:
        flat_event_rows = np.empty((0, HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)
        flat_event_peak_indices = np.empty(0, dtype=np.int64)
    flat_event_count = 0
    weighted_candidate_tables = []
    weighted_candidate_write_offsets = np.zeros(num_peaks, dtype=np.int64)

    sample_weight_array = sample_weights
    if sample_weight_array is not None:
        sample_weight_array = np.asarray(sample_weight_array, dtype=np.float64).reshape(-1)
        if sample_weight_array.shape[0] != n_samp:
            raise ValueError("sample_weights length does not match beam samples.")

    if n2_sample_array_override is None:
        raise ValueError("n2_sample_array_override is required for exact optics.")
    n2_sample_array = np.ascontiguousarray(
        np.asarray(n2_sample_array_override, dtype=np.complex128).reshape(-1),
        dtype=np.complex128,
    )
    if n2_sample_array.size != n_samp:
        raise ValueError("n2_sample_array_override length does not match beam samples.")
    if not np.all(np.isfinite(n2_sample_array.real) & np.isfinite(n2_sample_array.imag)):
        raise ValueError("n2_sample_array_override must contain finite values.")

    time_precompute = 0.0
    time_solve_q = 0.0
    time_chunk_compute = 0.0
    time_project = 0.0
    time_select = 0.0
    time_emit_cache = 0.0
    n_solve_q_calls = 0
    n_project_candidate_calls = 0
    n_valid_candidates = 0
    n_selected_events = 0
    pass1_total_mass = 0.0
    pass2_total_mass = 0.0
    pass2_mass_mismatch_count = 0
    pass2_mass_mismatch_max_abs = 0.0
    tail_fill_events = 0
    n_stored_projected_candidates = 0
    candidate_buffer_capacity_max = 0
    candidate_buffer_requested_per_worker_bytes = 0
    candidate_buffer_requested_total_bytes = 0
    candidate_buffer_effective_max_bytes = 0
    candidate_mass = np.empty(0, dtype=np.float64)
    candidate_row = np.empty(0, dtype=np.float64)
    candidate_col = np.empty(0, dtype=np.float64)
    candidate_phi = np.empty(0, dtype=np.float64)
    candidate_peak_idx = np.empty(0, dtype=np.int64)
    n_qsets_precomputed = 0
    n_qset_lookup_entries = 0
    n_qset_reuse_hits = 0
    time_qset_index = 0.0
    n_raw_beam_phases = int(n_samp)
    n_effective_beam_phases = 0
    n_exact_solve_q_phase_groups = 0
    phase_weight_sum = 0.0
    phase_event_count_total = 0
    weighted_event_candidate_buffer_max_bytes = int(weighted_event_candidate_buffer_max_bytes)
    candidate_buffer_effective_max_bytes = int(weighted_event_candidate_buffer_max_bytes)
    representative_rows = None
    representative_ranks = None
    representative_peak_mask = None
    if collect_representatives:
        representative_peak_mask = _weighted_event_representative_peak_mask(
            num_peaks,
            representative_peak_indices,
        )
        representative_rows, representative_ranks = _weighted_event_representative_state(num_peaks)

    precompute_start = time.perf_counter()
    (
        R_sample_precomputed,
        sample_terms,
        sample_n2_array,
        sample_eps2_array,
        _best_idx_precomputed,
    ) = _precompute_sample_terms(
        wavelength_array,
        n2,
        n2_sample_array,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        zb,
        thickness,
        sample_width_m,
        sample_length_m,
        theta_initial_deg,
        cor_angle_deg,
        psi_z_deg,
        R_z_R_y,
        R_ZY_n,
        P0,
    )
    solve_q_phase_reps = set()
    for phase_idx in range(n_samp):
        if sample_terms[phase_idx, _SAMPLE_COL_VALID] <= 0.5:
            continue
        phase_weight = 1.0
        if sample_weight_array is not None:
            phase_weight = float(sample_weight_array[phase_idx])
            if (not np.isfinite(phase_weight)) or phase_weight <= 0.0:
                continue
        rep_idx = int(sample_terms[phase_idx, _SAMPLE_COL_SOLVE_Q_REP])
        if rep_idx < 0 or rep_idx >= n_samp:
            rep_idx = phase_idx
        solve_q_phase_reps.add(int(rep_idx))
        n_effective_beam_phases += 1
        phase_weight_sum += float(phase_weight)
        phase_event_count_total += int(phase_event_counts[phase_idx])
    n_exact_solve_q_phase_groups = int(len(solve_q_phase_reps))
    time_precompute += time.perf_counter() - precompute_start

    debye_x_sq = float(debye_x) * float(debye_x)
    debye_y_sq = float(debye_y) * float(debye_y)
    pixel_size_eff = float(pixel_size_m)
    if (not np.isfinite(pixel_size_eff)) or pixel_size_eff <= 0.0:
        pixel_size_eff = 100e-6
    pixel_scale = 1.0 / pixel_size_eff
    center_row = float(center[0]) if center.size > 0 else 0.0
    center_col = float(center[1]) if center.size > 1 else center_row

    peak_h = (
        miller[:, 0].astype(np.float64, copy=False)
        if num_peaks > 0
        else np.empty(0, dtype=np.float64)
    )
    peak_k = (
        miller[:, 1].astype(np.float64, copy=False)
        if num_peaks > 0
        else np.empty(0, dtype=np.float64)
    )
    peak_l = (
        miller[:, 2].astype(np.float64, copy=False)
        if num_peaks > 0
        else np.empty(0, dtype=np.float64)
    )
    peak_reflection_intensity = np.zeros(num_peaks, dtype=np.float64)
    peak_gr0 = np.zeros(num_peaks, dtype=np.float64)
    peak_gz0 = np.zeros(num_peaks, dtype=np.float64)
    peak_valid = np.zeros(num_peaks, dtype=np.uint8)
    for peak_idx in range(num_peaks):
        reflection_intensity = (
            float(intensities[peak_idx]) if peak_idx < intensities.shape[0] else 0.0
        )
        peak_reflection_intensity[peak_idx] = reflection_intensity
        H = float(peak_h[peak_idx])
        K = float(peak_k[peak_idx])
        L = float(peak_l[peak_idx])
        if np.isfinite(reflection_intensity) and reflection_intensity > 0.0 and L >= 0.0:
            peak_valid[peak_idx] = 1
            peak_gz0[peak_idx] = 2.0 * pi * (L / float(cv))
            peak_gr0[peak_idx] = (
                4.0 * pi / float(av) * sqrt(max((H * H + H * K + K * K) / 3.0, 0.0))
            )

    if accumulate_image_flag:
        cache_capacity = int(_choose_local_pixel_cache_capacity(n_samp))
        cache_keys = np.empty(cache_capacity, dtype=np.int64)
        cache_values = np.empty(cache_capacity, dtype=np.float64)
        cache_flush_limit = (
            cache_capacity * _LOCAL_PIXEL_CACHE_LOAD_NUM
        ) // _LOCAL_PIXEL_CACHE_LOAD_DEN
        if cache_flush_limit < 4:
            cache_flush_limit = 4
    else:
        cache_keys = np.empty(1, dtype=np.int64)
        cache_values = np.empty(1, dtype=np.float64)
        cache_flush_limit = 0

    if _weighted_event_parallel_eligible(
        worker_count=worker_count,
        save_flag=save_flag,
        n_samp=n_samp,
    ):
        q_precompute_start = time.perf_counter()
        (
            q_values,
            qset_offsets,
            qset_lengths,
            qset_status,
            qset_id_by_sample_peak,
            n_solve_q_calls,
            precomputed_solve_time,
            _precomputed_qset_index_time,
        ) = _precompute_weighted_event_qsets(
            num_peaks=num_peaks,
            n_samp=n_samp,
            sample_terms=sample_terms,
            sample_weight_array=sample_weight_array,
            peak_valid=peak_valid,
            peak_gr0=peak_gr0,
            peak_gz0=peak_gz0,
            sigma_rad=sigma_rad,
            gamma_rad_m=gamma_rad_m,
            eta_pv=eta_pv,
            solve_q_steps_i=solve_q_steps_i,
            solve_q_rel_tol_i=solve_q_rel_tol_i,
            solve_q_mode_i=solve_q_mode_i,
            default_solve_q_dtheta=_DEFAULT_SOLVE_Q_DTHETA,
            default_solve_q_cos=_DEFAULT_SOLVE_Q_COS,
            default_solve_q_sin=_DEFAULT_SOLVE_Q_SIN,
            worker_count=worker_count,
        )
        q_precompute_elapsed = time.perf_counter() - q_precompute_start
        time_solve_q += float(precomputed_solve_time)
        time_qset_index += max(
            0.0,
            float(q_precompute_elapsed) - float(precomputed_solve_time),
        )

        chunks = _weighted_event_chunk_bounds(n_samp, worker_count)
        active_worker_count = max(int(worker_count), 1)
        n_qsets_precomputed = int(qset_offsets.shape[0])
        n_qset_lookup_entries = int(np.count_nonzero(qset_id_by_sample_peak >= 0))
        n_qset_reuse_hits = max(0, int(n_qset_lookup_entries) - int(n_qsets_precomputed))
        max_sample_candidate_capacity = 0
        for candidate_sample_idx in range(n_samp):
            if sample_terms[candidate_sample_idx, _SAMPLE_COL_VALID] <= 0.5:
                continue
            if sample_weight_array is not None:
                candidate_sample_weight = float(sample_weight_array[candidate_sample_idx])
                if (not np.isfinite(candidate_sample_weight)) or candidate_sample_weight <= 0.0:
                    continue
            sample_candidate_capacity = 0
            for candidate_peak_idx in range(num_peaks):
                if int(peak_valid[candidate_peak_idx]) == 0:
                    continue
                qset_id = int(qset_id_by_sample_peak[candidate_sample_idx, candidate_peak_idx])
                if qset_id < 0:
                    continue
                sample_candidate_capacity += int(qset_lengths[qset_id])
            max_sample_candidate_capacity = max(
                int(max_sample_candidate_capacity),
                int(sample_candidate_capacity),
            )
        candidate_buffer_policy = _weighted_event_candidate_buffer_memory_policy(
            candidate_capacity=max_sample_candidate_capacity,
            worker_count=active_worker_count,
            max_bytes=weighted_event_candidate_buffer_max_bytes,
        )
        candidate_buffer_requested_per_worker_bytes = int(
            candidate_buffer_policy["requested_per_worker_bytes"]
        )
        candidate_buffer_requested_total_bytes = int(
            candidate_buffer_policy["requested_total_bytes"]
        )
        candidate_buffer_effective_max_bytes = int(candidate_buffer_policy["max_bytes"])
        if not candidate_buffer_policy["fits"]:
            raise MemoryError(
                "Weighted-event projected-candidate buffer requires "
                f"{candidate_buffer_policy['requested_total_bytes']} bytes; limit is "
                f"{candidate_buffer_policy['max_bytes']} bytes."
            )
        chunk_candidate_capacity = int(max_sample_candidate_capacity)
        collect_chunk_event_rows = bool(collect_tables or accumulate_image_flag)
        if collect_chunk_event_rows:
            max_worker_flat_capacity = max(
                int(np.sum(phase_event_counts[start:stop])) for _worker_slot, start, stop in chunks
            )
            flat_event_rows_parts = np.empty(
                (
                    active_worker_count,
                    max_worker_flat_capacity,
                    HIT_ROW_WITH_PROVENANCE_WIDTH,
                ),
                dtype=np.float64,
            )
            flat_event_peak_indices_parts = np.empty(
                (active_worker_count, max_worker_flat_capacity),
                dtype=np.int64,
            )
        else:
            flat_event_rows_parts = np.empty(
                (active_worker_count, 0, HIT_ROW_WITH_PROVENANCE_WIDTH),
                dtype=np.float64,
            )
            flat_event_peak_indices_parts = np.empty((active_worker_count, 0), dtype=np.int64)
        flat_event_count_parts = np.zeros(active_worker_count, dtype=np.int64)

        image_parts = np.zeros((active_worker_count, 1, 1), dtype=np.float64)
        cache_keys_parts = np.empty((active_worker_count, 1), dtype=np.int64)
        cache_values_parts = np.empty((active_worker_count, 1), dtype=np.float64)
        cache_keys_parts.fill(-1)
        cache_values_parts.fill(0.0)
        chunk_cache_flush_limit = 0

        candidate_mass_parts = np.empty(
            (active_worker_count, chunk_candidate_capacity),
            dtype=np.float64,
        )
        candidate_row_parts = np.empty(
            (active_worker_count, chunk_candidate_capacity),
            dtype=np.float64,
        )
        candidate_col_parts = np.empty(
            (active_worker_count, chunk_candidate_capacity),
            dtype=np.float64,
        )
        candidate_phi_parts = np.empty(
            (active_worker_count, chunk_candidate_capacity),
            dtype=np.float64,
        )
        candidate_peak_idx_parts = np.empty(
            (active_worker_count, chunk_candidate_capacity),
            dtype=np.int64,
        )
        representative_peak_mask_arg = (
            np.asarray(representative_peak_mask, dtype=np.bool_)
            if representative_peak_mask is not None
            else np.zeros(num_peaks, dtype=np.bool_)
        )
        if collect_representatives:
            representative_rows_parts = np.full(
                (
                    active_worker_count,
                    num_peaks,
                    2,
                    HIT_ROW_WITH_CONTEXT_WIDTH,
                ),
                np.nan,
                dtype=np.float64,
            )
            representative_ranks_parts = np.full(
                (active_worker_count, num_peaks, 2, 5),
                np.inf,
                dtype=np.float64,
            )
        else:
            representative_rows_parts = np.empty(
                (active_worker_count, 0, 2, HIT_ROW_WITH_CONTEXT_WIDTH),
                dtype=np.float64,
            )
            representative_ranks_parts = np.empty(
                (active_worker_count, 0, 2, 5),
                dtype=np.float64,
            )

        stats_parts = np.zeros(
            (active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_COLS),
            dtype=np.float64,
        )
        q_data_chunk = np.zeros((1, 1, 5), dtype=np.float64)
        q_count_chunk = np.zeros(1, dtype=np.int64)
        # Threaded Q-debug capture remains disabled: save_flag == 1 resolves to one worker.
        # Keep a distinct truncation counter to avoid aliasing if this path changes later.
        q_debug_truncated_count_parts = np.zeros((active_worker_count, 1), dtype=np.int64)
        sample_weight_array_arg = (
            sample_weight_array
            if sample_weight_array is not None
            else np.empty(0, dtype=np.float64)
        )

        chunk_start = time.perf_counter()
        chunks = _run_weighted_event_sample_chunks(
            (
                int(num_peaks),
                int(n_samp),
                int(image_size),
                phase_event_counts,
                bool(collect_chunk_event_rows),
                False,
                q_values,
                qset_offsets,
                qset_lengths,
                qset_status,
                qset_id_by_sample_peak,
                all_status,
                peak_valid,
                peak_h,
                peak_k,
                peak_l,
                peak_reflection_intensity,
                sample_weight_array_arg,
                bool(sample_weight_array is not None),
                float(debye_x_sq),
                float(debye_y_sq),
                float(center_row),
                float(center_col),
                R_sample_precomputed,
                n_det_rot,
                Detector_Pos,
                e1_det,
                e2_det,
                sample_terms,
                sample_eps2_array,
                float(thickness),
                float(pixel_scale),
                q_data_chunk,
                q_count_chunk,
                q_debug_truncated_count_parts,
                image_parts,
                cache_keys_parts,
                cache_values_parts,
                int(chunk_cache_flush_limit),
                candidate_mass_parts,
                candidate_row_parts,
                candidate_col_parts,
                candidate_phi_parts,
                candidate_peak_idx_parts,
                int(chunk_candidate_capacity),
                bool(collect_representatives),
                representative_peak_mask_arg,
                representative_rows_parts,
                representative_ranks_parts,
                beam_x_array,
                beam_y_array,
                theta_array,
                phi_array,
                wavelength_array,
                float(wavelength_center),
                flat_event_rows_parts,
                flat_event_peak_indices_parts,
                flat_event_count_parts,
                event_counts,
                stats_parts,
            ),
            n_samp=n_samp,
            worker_count=active_worker_count,
        )
        time_chunk_compute += time.perf_counter() - chunk_start

        if collect_representatives:
            representative_rows, representative_ranks = (
                _merge_weighted_event_representative_parts(
                    representative_rows_parts,
                    representative_ranks_parts,
                )
            )

        flat_event_count = int(np.sum(flat_event_count_parts[:active_worker_count]))
        if collect_chunk_event_rows and flat_event_count > 0:
            flat_event_rows = np.empty(
                (flat_event_count, HIT_ROW_WITH_PROVENANCE_WIDTH),
                dtype=np.float64,
            )
            flat_event_peak_indices = np.empty(flat_event_count, dtype=np.int64)
            out_offset = 0
            for worker_slot in range(active_worker_count):
                count = int(flat_event_count_parts[worker_slot])
                if count <= 0:
                    continue
                flat_event_rows[out_offset : out_offset + count, :] = flat_event_rows_parts[
                    worker_slot, :count, :
                ]
                flat_event_peak_indices[out_offset : out_offset + count] = (
                    flat_event_peak_indices_parts[worker_slot, :count]
                )
                out_offset += count
        elif collect_chunk_event_rows:
            flat_event_rows = np.empty(
                (0, HIT_ROW_WITH_PROVENANCE_WIDTH),
                dtype=np.float64,
            )
            flat_event_peak_indices = np.empty(0, dtype=np.int64)

        if accumulate_image_flag and flat_event_count > 0:
            flat_event_count = _accumulate_and_compact_weighted_event_rows(
                image,
                int(image_size),
                flat_event_rows,
                flat_event_peak_indices,
                int(flat_event_count),
            )

        n_project_candidate_calls = int(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_PROJECT_CALLS])
        )
        n_valid_candidates = int(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_VALID_CANDIDATES])
        )
        n_selected_events = int(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_SELECTED_EVENTS])
        )
        pass1_total_mass = float(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_PASS1_TOTAL_MASS])
        )
        pass2_total_mass = float(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_PASS2_TOTAL_MASS])
        )
        pass2_mass_mismatch_count = int(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_COUNT])
        )
        pass2_mass_mismatch_max_abs = float(
            np.max(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_MISMATCH_MAX_ABS])
            if len(chunks) > 0
            else 0.0
        )
        tail_fill_events = int(
            np.sum(stats_parts[:active_worker_count, _WEIGHTED_EVENT_CHUNK_STAT_TAIL_FILL_EVENTS])
        )
        n_stored_projected_candidates = int(
            np.sum(
                stats_parts[
                    :active_worker_count,
                    _WEIGHTED_EVENT_CHUNK_STAT_STORED_CANDIDATES,
                ]
            )
        )
        candidate_buffer_capacity_max = int(
            np.max(
                stats_parts[
                    :active_worker_count,
                    _WEIGHTED_EVENT_CHUNK_STAT_CANDIDATE_CAPACITY_MAX,
                ]
            )
            if len(chunks) > 0
            else 0
        )
        q_debug_truncated_solution_count = int(
            np.sum(q_debug_truncated_count_parts[:active_worker_count, 0])
        )

        emit_start = time.perf_counter()
        if best_sample_indices_out is not None:
            best_sample_indices_out[:] = -1
            for peak_idx in range(num_peaks):
                best_count = -1
                best_sample = -1
                for sample_idx in range(n_samp):
                    count = int(event_counts[peak_idx, sample_idx])
                    if count > best_count:
                        best_count = count
                        best_sample = sample_idx
                if best_count > 0 and peak_idx < best_sample_indices_out.shape[0]:
                    best_sample_indices_out[peak_idx] = int(best_sample)
                elif peak_idx < best_sample_indices_out.shape[0]:
                    best_sample_indices_out[peak_idx] = -1

        if collect_tables:
            hit_tables = _build_weighted_event_hit_tables(
                flat_event_rows,
                flat_event_peak_indices,
                flat_event_count,
                num_peaks,
            )
        else:
            hit_tables = []
        representative_hit_tables = (
            _weighted_event_representative_hit_tables(representative_rows)
            if collect_representatives
            else []
        )
        time_emit_cache += time.perf_counter() - emit_start

        weighted_event_stats = _process_peaks_weighted_event_stats(
            n_solve_q_calls=int(n_solve_q_calls),
            n_project_candidate_calls=int(n_project_candidate_calls),
            n_valid_candidates=int(n_valid_candidates),
            n_selected_events=int(n_selected_events),
            n_stored_projected_candidates=int(n_stored_projected_candidates),
            candidate_buffer_capacity_max=int(candidate_buffer_capacity_max),
            candidate_buffer_requested_per_worker_bytes=int(
                candidate_buffer_requested_per_worker_bytes
            ),
            candidate_buffer_requested_total_bytes=int(candidate_buffer_requested_total_bytes),
            candidate_buffer_effective_max_bytes=int(candidate_buffer_effective_max_bytes),
            n_qsets_precomputed=int(n_qsets_precomputed),
            n_qset_lookup_entries=int(n_qset_lookup_entries),
            n_qset_reuse_hits=int(n_qset_reuse_hits),
            time_qset_index=float(time_qset_index),
            pass2_mass_mismatch_count=int(pass2_mass_mismatch_count),
            pass2_mass_mismatch_max_abs=float(pass2_mass_mismatch_max_abs),
            tail_fill_events=int(tail_fill_events),
            time_precompute=float(time_precompute),
            time_solve_q=float(time_solve_q),
            time_chunk_compute=float(time_chunk_compute),
            time_project=float(time_project),
            time_select=float(time_select),
            time_emit_cache=float(time_emit_cache),
            pass1_total_mass=float(pass1_total_mass),
            pass2_total_mass=float(pass2_total_mass),
            n_raw_beam_phases=int(n_raw_beam_phases),
            n_effective_beam_phases=int(n_effective_beam_phases),
            n_exact_solve_q_phase_groups=int(n_exact_solve_q_phase_groups),
            phase_weight_sum=float(phase_weight_sum),
            phase_event_count_total=int(phase_event_count_total),
            parallel_backend="threaded_njit_chunks",
            parallel_worker_count=int(active_worker_count),
            parallel_requested_worker_count=requested_worker_count,
            parallel_effective_worker_count=int(active_worker_count),
            parallel_worker_count_source=str(worker_count_source),
            hit_table_collection_mode="quantile_events",
        )
        if q_debug_truncated_solution_count:
            raise RuntimeError(
                "Q-space export exceeded the per-reflection event capacity; "
                "reduce the active beam/mosaic sampling before exporting."
            )
        return (
            image,
            hit_tables,
            q_data,
            q_count,
            all_status,
            miss_tables,
            representative_hit_tables,
            weighted_event_stats,
        )

    (
        q_values,
        qset_offsets,
        qset_lengths,
        qset_status,
        qset_id_by_sample_peak,
        n_solve_q_calls,
        qset_solve_time,
        qset_index_time,
    ) = _precompute_weighted_event_qsets(
        num_peaks=num_peaks,
        n_samp=n_samp,
        sample_terms=sample_terms,
        sample_weight_array=sample_weight_array,
        peak_valid=peak_valid,
        peak_gr0=peak_gr0,
        peak_gz0=peak_gz0,
        sigma_rad=sigma_rad,
        gamma_rad_m=gamma_rad_m,
        eta_pv=eta_pv,
        solve_q_steps_i=solve_q_steps_i,
        solve_q_rel_tol_i=solve_q_rel_tol_i,
        solve_q_mode_i=solve_q_mode_i,
        default_solve_q_dtheta=_DEFAULT_SOLVE_Q_DTHETA,
        default_solve_q_cos=_DEFAULT_SOLVE_Q_COS,
        default_solve_q_sin=_DEFAULT_SOLVE_Q_SIN,
    )
    time_solve_q += float(qset_solve_time)
    time_qset_index += float(qset_index_time)
    n_qsets_precomputed = int(qset_offsets.shape[0])
    n_qset_lookup_entries = int(np.count_nonzero(qset_id_by_sample_peak >= 0))
    n_qset_reuse_hits = int(n_qset_lookup_entries - n_qsets_precomputed)

    if collect_all_weighted_candidates:
        weighted_candidate_capacities = np.zeros(num_peaks, dtype=np.int64)
        max_weighted_candidate_sample_capacity = 0
        for sample_idx in range(n_samp):
            if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
                continue
            if sample_weight_array is not None:
                sample_weight = float(sample_weight_array[sample_idx])
                if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                    continue
            sample_candidate_capacity = 0
            for peak_idx in range(num_peaks):
                qset_id = int(qset_id_by_sample_peak[sample_idx, peak_idx])
                if qset_id >= 0:
                    candidate_capacity = int(qset_lengths[qset_id])
                    sample_candidate_capacity += candidate_capacity
                    weighted_candidate_capacities[peak_idx] += candidate_capacity
            max_weighted_candidate_sample_capacity = max(
                max_weighted_candidate_sample_capacity,
                sample_candidate_capacity,
            )
        weighted_candidate_capacity = int(np.sum(weighted_candidate_capacities))
        weighted_candidate_output_bytes = int(weighted_candidate_capacity) * (
            HIT_ROW_WITH_PROVENANCE_WIDTH * 8
        )
        weighted_candidate_working_bytes = (
            int(max_weighted_candidate_sample_capacity)
            * _WEIGHTED_EVENT_CANDIDATE_RECORD_BYTES
        )
        weighted_candidate_total_bytes = (
            weighted_candidate_output_bytes + weighted_candidate_working_bytes
        )
        if weighted_candidate_total_bytes > weighted_event_candidate_buffer_max_bytes:
            raise MemoryError(
                "All-weighted-candidate output and working buffers require "
                f"{weighted_candidate_total_bytes} bytes; limit is "
                f"{weighted_event_candidate_buffer_max_bytes} bytes."
            )
        weighted_candidate_tables = [
            np.full(
                (int(capacity), HIT_ROW_WITH_PROVENANCE_WIDTH),
                np.nan,
                dtype=np.float64,
            )
            for capacity in weighted_candidate_capacities
        ]

    for sample_idx in range(n_samp):
        if sample_terms[sample_idx, _SAMPLE_COL_VALID] <= 0.5:
            all_status[:, sample_idx] = -10
            continue

        sample_weight = 1.0
        if sample_weight_array is not None:
            sample_weight = float(sample_weight_array[sample_idx])
            if not np.isfinite(sample_weight) or sample_weight <= 0.0:
                all_status[:, sample_idx] = -12
                continue

        sample_total_mass = 0.0
        if accumulate_image_flag:
            _clear_local_pixel_cache(cache_keys, cache_values)
        cache_entry_count = 0
        sample_qsets = []
        sample_candidate_capacity = 0

        for peak_idx in range(num_peaks):
            if int(peak_valid[peak_idx]) == 0:
                continue

            qset_id = int(qset_id_by_sample_peak[sample_idx, peak_idx])
            if qset_id < 0:
                continue
            all_status[peak_idx, sample_idx] = int(qset_status[qset_id])
            offset = int(qset_offsets[qset_id])
            length = int(qset_lengths[qset_id])
            all_q = q_values[offset : offset + length, :]
            sample_qsets.append((int(peak_idx), all_q))
            sample_candidate_capacity += int(length)

        sample_candidate_buffer_policy = _weighted_event_candidate_buffer_memory_policy(
            candidate_capacity=sample_candidate_capacity,
            worker_count=1,
            max_bytes=weighted_event_candidate_buffer_max_bytes,
        )
        candidate_buffer_requested_per_worker_bytes = max(
            int(candidate_buffer_requested_per_worker_bytes),
            int(sample_candidate_buffer_policy["requested_per_worker_bytes"]),
        )
        candidate_buffer_requested_total_bytes = max(
            int(candidate_buffer_requested_total_bytes),
            int(sample_candidate_buffer_policy["requested_total_bytes"]),
        )
        candidate_buffer_effective_max_bytes = int(sample_candidate_buffer_policy["max_bytes"])
        if not sample_candidate_buffer_policy["fits"]:
            raise MemoryError(
                "Weighted-event projected-candidate buffer requires "
                f"{sample_candidate_buffer_policy['requested_total_bytes']} bytes; limit is "
                f"{sample_candidate_buffer_policy['max_bytes']} bytes."
            )
        if sample_candidate_capacity > candidate_buffer_capacity_max:
            candidate_buffer_capacity_max = int(sample_candidate_capacity)
        if sample_candidate_capacity > candidate_mass.shape[0]:
            candidate_mass = np.empty(int(sample_candidate_capacity), dtype=np.float64)
            candidate_row = np.empty(int(sample_candidate_capacity), dtype=np.float64)
            candidate_col = np.empty(int(sample_candidate_capacity), dtype=np.float64)
            candidate_phi = np.empty(int(sample_candidate_capacity), dtype=np.float64)
            candidate_peak_idx = np.empty(int(sample_candidate_capacity), dtype=np.int64)

        candidate_count = 0
        for peak_idx, all_q in sample_qsets:
            reflection_intensity = float(peak_reflection_intensity[peak_idx])

            project_start = time.perf_counter()
            peak_mass, valid_count, project_calls, candidate_count = (
                _weighted_event_project_store_for_qset(
                    all_q,
                    peak_idx,
                    sample_idx,
                    reflection_intensity,
                    sample_weight,
                    debye_x_sq,
                    debye_y_sq,
                    center_row,
                    center_col,
                    R_sample_precomputed,
                    n_det_rot,
                    Detector_Pos,
                    e1_det,
                    e2_det,
                    sample_terms,
                    sample_eps2_array,
                    thickness,
                    pixel_scale,
                    int(image_size),
                    int(save_flag),
                    q_data,
                    q_count,
                    q_debug_truncated_count,
                    candidate_mass,
                    candidate_row,
                    candidate_col,
                    candidate_phi,
                    candidate_peak_idx,
                    int(candidate_count),
                )
            )
            time_project += time.perf_counter() - project_start
            sample_total_mass += float(peak_mass)
            pass1_total_mass += float(peak_mass)
            n_valid_candidates += int(valid_count)
            n_project_candidate_calls += int(project_calls)
        n_stored_projected_candidates += int(candidate_count)
        if collect_representatives:
            for cand_idx in range(int(candidate_count)):
                candidate_peak_i = int(candidate_peak_idx[cand_idx])
                if candidate_peak_i < 0 or candidate_peak_i >= num_peaks:
                    raise RuntimeError("Projected candidate contains an out-of-range peak index.")
                if representative_peak_mask is not None and not bool(
                    representative_peak_mask[candidate_peak_i]
                ):
                    continue
                _weighted_event_update_representative(
                    rows=representative_rows,
                    ranks=representative_ranks,
                    peak_idx=int(candidate_peak_i),
                    sample_idx=int(sample_idx),
                    H=float(peak_h[candidate_peak_i]),
                    K=float(peak_k[candidate_peak_i]),
                    L=float(peak_l[candidate_peak_i]),
                    row_f=float(candidate_row[cand_idx]),
                    col_f=float(candidate_col[cand_idx]),
                    phi_f=float(candidate_phi[cand_idx]),
                    mass=float(candidate_mass[cand_idx]),
                    beam_x_array=beam_x_array,
                    beam_y_array=beam_y_array,
                    theta_array=theta_array,
                    phi_array=phi_array,
                    wavelength_array=wavelength_array,
                    wavelength_center=float(wavelength_center),
                )

        if not np.isfinite(sample_total_mass) or sample_total_mass <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                flush_start = time.perf_counter()
                _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
                time_emit_cache += time.perf_counter() - flush_start
            continue

        if collect_all_weighted_candidates:
            for candidate_index in range(int(candidate_count)):
                candidate_peak_i = int(candidate_peak_idx[candidate_index])
                if candidate_peak_i < 0 or candidate_peak_i >= num_peaks:
                    raise RuntimeError("Projected candidate contains an out-of-range peak index.")
                if accumulate_image_flag and not _accumulate_bilinear_hit(
                    image,
                    int(image_size),
                    float(candidate_row[candidate_index]),
                    float(candidate_col[candidate_index]),
                    float(candidate_mass[candidate_index]),
                ):
                    continue
                output_index = int(weighted_candidate_write_offsets[candidate_peak_i])
                output_table = weighted_candidate_tables[candidate_peak_i]
                if output_index >= output_table.shape[0]:
                    raise RuntimeError("All-weighted-candidate output exceeded its capacity.")
                output_table[output_index, 0] = float(candidate_mass[candidate_index])
                output_table[output_index, 1] = float(candidate_col[candidate_index])
                output_table[output_index, 2] = float(candidate_row[candidate_index])
                output_table[output_index, 3] = float(candidate_phi[candidate_index])
                output_table[output_index, 4] = float(peak_h[candidate_peak_i])
                output_table[output_index, 5] = float(peak_k[candidate_peak_i])
                output_table[output_index, 6] = float(peak_l[candidate_peak_i])
                output_table[output_index, HIT_ROW_COL_SOURCE_TABLE_INDEX] = float(
                    candidate_peak_i
                )
                output_table[output_index, HIT_ROW_COL_SOURCE_ROW_INDEX] = float(output_index)
                output_table[output_index, 9] = float(sample_idx)
                weighted_candidate_write_offsets[candidate_peak_i] = output_index + 1
                event_counts[candidate_peak_i, sample_idx] += 1
                n_selected_events += 1
            pass2_total_mass += float(sample_total_mass)
            continue

        select_start = time.perf_counter()
        event_count = int(phase_event_counts[sample_idx])
        targets = _weighted_event_targets(sample_total_mass, event_count, sample_idx)
        deposit = _weighted_event_deposit(sample_total_mass, event_count)
        time_select += time.perf_counter() - select_start
        if targets.size <= 0 or not np.isfinite(deposit) or deposit <= 0.0:
            if accumulate_image_flag and cache_entry_count > 0:
                flush_start = time.perf_counter()
                _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
                time_emit_cache += time.perf_counter() - flush_start
            continue

        target_idx = 0
        cumulative_mass = 0.0
        phase_have_last_valid = False
        phase_last_row_f = np.nan
        phase_last_col_f = np.nan
        phase_last_phi_f = np.nan
        phase_last_peak_idx = -1
        phase_last_H = np.nan
        phase_last_K = np.nan
        phase_last_L = np.nan
        emit_start = time.perf_counter()
        (
            target_idx,
            cumulative_mass,
            flat_event_count,
            cache_entry_count,
            sample_pass2_mass,
            selected_events,
            phase_have_last_valid,
            phase_last_row_f,
            phase_last_col_f,
            phase_last_phi_f,
            phase_last_peak_idx,
            phase_last_H,
            phase_last_K,
            phase_last_L,
        ) = _weighted_event_emit_from_stored_candidates(
            int(candidate_count),
            candidate_mass,
            candidate_row,
            candidate_col,
            candidate_phi,
            candidate_peak_idx,
            peak_h,
            peak_k,
            peak_l,
            sample_idx,
            targets,
            int(target_idx),
            float(cumulative_mass),
            float(deposit),
            collect_tables,
            flat_event_rows,
            flat_event_peak_indices,
            int(flat_event_count),
            event_counts,
            accumulate_image_flag,
            image,
            int(image_size),
            cache_keys,
            cache_values,
            int(cache_entry_count),
            int(cache_flush_limit),
        )
        time_emit_cache += time.perf_counter() - emit_start
        pass2_total_mass += float(sample_pass2_mass)
        n_selected_events += int(selected_events)

        sample_pass2_mass_delta = abs(float(cumulative_mass) - float(sample_total_mass))
        sample_pass2_mass_tol = max(
            1.0e-10,
            1.0e-9 * max(1.0, abs(float(sample_total_mass))),
        )
        if (
            not np.isfinite(sample_pass2_mass_delta)
        ) or sample_pass2_mass_delta > sample_pass2_mass_tol:
            pass2_mass_mismatch_count += 1
            if np.isfinite(sample_pass2_mass_delta):
                pass2_mass_mismatch_max_abs = max(
                    float(pass2_mass_mismatch_max_abs),
                    float(sample_pass2_mass_delta),
                )
            else:
                pass2_mass_mismatch_max_abs = float("inf")
        elif target_idx < targets.shape[0] and phase_have_last_valid:
            hit_count = int(targets.shape[0] - target_idx)
            n_selected_events += int(hit_count)
            tail_fill_events += int(hit_count)
            if 0 <= phase_last_peak_idx < event_counts.shape[0]:
                event_counts[phase_last_peak_idx, sample_idx] += int(hit_count)
            target_idx = targets.shape[0]

            deposited = True
            if accumulate_image_flag:
                deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                    int(image_size),
                    float(phase_last_row_f),
                    float(phase_last_col_f),
                    float(hit_count) * float(deposit),
                    cache_keys,
                    cache_values,
                    int(cache_entry_count),
                    int(cache_flush_limit),
                )
                if needs_flush:
                    cache_entry_count = _flush_local_pixel_cache(
                        image,
                        int(image_size),
                        cache_keys,
                        cache_values,
                    )
                    deposited, needs_flush, cache_entry_count = _accumulate_bilinear_cached(
                        int(image_size),
                        float(phase_last_row_f),
                        float(phase_last_col_f),
                        float(hit_count) * float(deposit),
                        cache_keys,
                        cache_values,
                        int(cache_entry_count),
                        int(cache_flush_limit),
                    )
                    if needs_flush:
                        deposited = _accumulate_bilinear_hit(
                            image,
                            int(image_size),
                            float(phase_last_row_f),
                            float(phase_last_col_f),
                            float(hit_count) * float(deposit),
                        )
                        cache_entry_count = 0

            if collect_tables and deposited:
                for _event_idx in range(hit_count):
                    if flat_event_count >= flat_event_rows.shape[0]:
                        break
                    flat_event_peak_indices[flat_event_count] = phase_last_peak_idx
                    flat_event_rows[flat_event_count, 0] = float(deposit)
                    flat_event_rows[flat_event_count, 1] = float(phase_last_col_f)
                    flat_event_rows[flat_event_count, 2] = float(phase_last_row_f)
                    flat_event_rows[flat_event_count, 3] = float(phase_last_phi_f)
                    flat_event_rows[flat_event_count, 4] = float(phase_last_H)
                    flat_event_rows[flat_event_count, 5] = float(phase_last_K)
                    flat_event_rows[flat_event_count, 6] = float(phase_last_L)
                    flat_event_rows[flat_event_count, 7] = np.nan
                    flat_event_rows[flat_event_count, 8] = np.nan
                    flat_event_rows[flat_event_count, 9] = float(sample_idx)
                    flat_event_count += 1

        if accumulate_image_flag and cache_entry_count > 0:
            flush_start = time.perf_counter()
            _flush_local_pixel_cache(image, int(image_size), cache_keys, cache_values)
            time_emit_cache += time.perf_counter() - flush_start

    emit_start = time.perf_counter()
    if best_sample_indices_out is not None:
        best_sample_indices_out[:] = -1
        for peak_idx in range(num_peaks):
            best_count = -1
            best_sample = -1
            for sample_idx in range(n_samp):
                count = int(event_counts[peak_idx, sample_idx])
                if count > best_count:
                    best_count = count
                    best_sample = sample_idx
            if best_count > 0 and peak_idx < best_sample_indices_out.shape[0]:
                best_sample_indices_out[peak_idx] = int(best_sample)
            elif peak_idx < best_sample_indices_out.shape[0]:
                best_sample_indices_out[peak_idx] = -1

    if collect_all_weighted_candidates:
        for peak_idx, table in enumerate(weighted_candidate_tables):
            table.resize(
                (int(weighted_candidate_write_offsets[peak_idx]), HIT_ROW_WITH_PROVENANCE_WIDTH),
                refcheck=False,
            )
        hit_tables = weighted_candidate_tables
    elif collect_tables:
        hit_tables = _build_weighted_event_hit_tables(
            flat_event_rows,
            flat_event_peak_indices,
            flat_event_count,
            num_peaks,
        )
    else:
        hit_tables = []
    representative_hit_tables = (
        _weighted_event_representative_hit_tables(representative_rows)
        if collect_representatives
        else []
    )
    time_emit_cache += time.perf_counter() - emit_start

    weighted_event_stats = _process_peaks_weighted_event_stats(
        n_solve_q_calls=int(n_solve_q_calls),
        n_project_candidate_calls=int(n_project_candidate_calls),
        n_valid_candidates=int(n_valid_candidates),
        n_selected_events=int(n_selected_events),
        n_stored_projected_candidates=int(n_stored_projected_candidates),
        candidate_buffer_capacity_max=int(candidate_buffer_capacity_max),
        candidate_buffer_requested_per_worker_bytes=int(
            candidate_buffer_requested_per_worker_bytes
        ),
        candidate_buffer_requested_total_bytes=int(candidate_buffer_requested_total_bytes),
        candidate_buffer_effective_max_bytes=int(candidate_buffer_effective_max_bytes),
        n_qsets_precomputed=int(n_qsets_precomputed),
        n_qset_lookup_entries=int(n_qset_lookup_entries),
        n_qset_reuse_hits=int(n_qset_reuse_hits),
        time_qset_index=float(time_qset_index),
        pass2_mass_mismatch_count=int(pass2_mass_mismatch_count),
        pass2_mass_mismatch_max_abs=float(pass2_mass_mismatch_max_abs),
        tail_fill_events=int(tail_fill_events),
        time_precompute=float(time_precompute),
        time_solve_q=float(time_solve_q),
        time_chunk_compute=float(time_chunk_compute),
        time_project=float(time_project),
        time_select=float(time_select),
        time_emit_cache=float(time_emit_cache),
        pass1_total_mass=float(pass1_total_mass),
        pass2_total_mass=float(pass2_total_mass),
        n_raw_beam_phases=int(n_raw_beam_phases),
        n_effective_beam_phases=int(n_effective_beam_phases),
        n_exact_solve_q_phase_groups=int(n_exact_solve_q_phase_groups),
        phase_weight_sum=float(phase_weight_sum),
        phase_event_count_total=int(
            n_selected_events if collect_all_weighted_candidates else phase_event_count_total
        ),
        n_hit_table_rows=int(sum(table.shape[0] for table in hit_tables)),
        n_nonempty_hit_tables=int(sum(table.shape[0] > 0 for table in hit_tables)),
        n_representative_hit_tables=int(len(representative_hit_tables)),
        parallel_backend="fast_serial",
        parallel_worker_count=1,
        parallel_requested_worker_count=requested_worker_count,
        parallel_effective_worker_count=1,
        parallel_worker_count_source=str(worker_count_source),
        hit_table_collection_mode=normalized_hit_table_collection_mode,
    )
    if int(q_debug_truncated_count[0]):
        raise RuntimeError(
            "Q-space export exceeded the per-reflection event capacity; "
            "reduce the active beam/mosaic sampling before exporting."
        )
    return (
        image,
        hit_tables,
        q_data,
        q_count,
        all_status,
        miss_tables,
        representative_hit_tables,
        weighted_event_stats,
    )


def process_peaks_parallel(
    miller,
    intensities,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    thickness=50e-9,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    sample_weights=None,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    numba_thread_count=None,
    events_per_beam_phase=50,
    collect_representative_hit_tables=False,
    representative_peak_indices=None,
    hit_table_collection_mode="quantile_events",
):
    """Inject runtime defaults before entering cached compiled kernel."""

    (
        image_out,
        hit_tables,
        q_data,
        q_count,
        all_status,
        miss_tables,
        representative_hit_tables,
        weighted_event_stats,
    ) = _process_peaks_parallel_weighted_events_fast_serial(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        thickness,
        solve_q_steps,
        solve_q_rel_tol,
        solve_q_mode,
        sample_weights,
        best_sample_indices_out,
        collect_hit_tables,
        pixel_size_m,
        sample_width_m,
        sample_length_m,
        n2_sample_array_override,
        accumulate_image,
        numba_thread_count=numba_thread_count,
        events_per_beam_phase=events_per_beam_phase,
        collect_representative_hit_tables=collect_representative_hit_tables,
        representative_peak_indices=representative_peak_indices,
        hit_table_collection_mode=hit_table_collection_mode,
    )
    return (
        image_out,
        hit_tables,
        q_data,
        q_count,
        all_status,
        miss_tables,
        representative_hit_tables,
        weighted_event_stats,
    )






def _cluster_hit_positions(hits_arr, *, merge_radius_px=1.5):
    """Merge nearby hit-table rows into subpixel centroids."""

    merge_radius_sq = float(merge_radius_px) * float(merge_radius_px)
    clusters = []

    for hit in hits_arr[np.argsort(hits_arr[:, 0])[::-1]]:
        intensity = float(hit[0])
        col = float(hit[1])
        row = float(hit[2])
        if not (
            np.isfinite(intensity) and np.isfinite(col) and np.isfinite(row) and intensity > 0.0
        ):
            continue

        best_cluster_idx = None
        best_dist_sq = float("inf")
        for idx, cluster in enumerate(clusters):
            center_col = cluster["weighted_col_sum"] / cluster["total_intensity"]
            center_row = cluster["weighted_row_sum"] / cluster["total_intensity"]
            dist_sq = (col - center_col) ** 2 + (row - center_row) ** 2
            if dist_sq <= merge_radius_sq and dist_sq < best_dist_sq:
                best_cluster_idx = idx
                best_dist_sq = dist_sq

        if best_cluster_idx is None:
            clusters.append(
                {
                    "total_intensity": intensity,
                    "peak_intensity": intensity,
                    "weighted_col_sum": intensity * col,
                    "weighted_row_sum": intensity * row,
                }
            )
            continue

        cluster = clusters[best_cluster_idx]
        cluster["total_intensity"] += intensity
        cluster["weighted_col_sum"] += intensity * col
        cluster["weighted_row_sum"] += intensity * row
        if intensity > cluster["peak_intensity"]:
            cluster["peak_intensity"] = intensity

    clusters.sort(
        key=lambda cluster: (
            float(cluster["total_intensity"]),
            float(cluster["peak_intensity"]),
        ),
        reverse=True,
    )

    out = []
    for cluster in clusters:
        total_intensity = float(cluster["total_intensity"])
        if total_intensity <= 0.0:
            continue
        out.append(
            (
                total_intensity,
                float(cluster["weighted_col_sum"]) / total_intensity,
                float(cluster["weighted_row_sum"]) / total_intensity,
            )
        )
    return out


def hit_tables_to_max_positions(hit_tables):
    """Extract up to two subpixel peak centers per reflection from ``hit_tables``.

    ``process_peaks_parallel`` returns a list of pixel-hit tables whose first
    seven columns are ``[intensity, col, row, phi, H, K, L]``. Current kernel
    rows may also carry ``[source_table_index, source_row_index,
    best_sample_index]`` so the intersection cache can recover the detector ray
    closest to the mosaic top. The ``col``/``row`` coordinates are stored in
    floating detector-pixel units. The returned ``max_positions`` array has
    shape ``(N, 6)`` and contains the two strongest centers per reflection:
    ``(I0, x0, y0, I1, x1, y1)``.  Nearby hit-table rows are merged into
    intensity-weighted centroids so small parameter changes remain visible to
    the optimizer.
    """

    num_peaks = len(hit_tables)
    max_positions = np.zeros((num_peaks, 6), dtype=np.float64)

    for i, hits in enumerate(hit_tables):
        hits_arr = np.asarray(hits)
        if hits_arr.size == 0:
            continue

        clustered_hits = _cluster_hit_positions(hits_arr)
        if not clustered_hits:
            continue

        primary = clustered_hits[0]
        max_positions[i, 0:3] = primary

        if len(clustered_hits) > 1:
            secondary = clustered_hits[1]
            max_positions[i, 3:6] = secondary

    return max_positions


def intersection_cache_to_hit_tables(intersection_cache):
    """Convert width-17/19 intersection caches into provenance hit rows."""

    hit_tables = []
    if intersection_cache is None:
        return hit_tables

    for table in intersection_cache:
        hit_tables.append(cache_table_to_hit_table(table))

    return hit_tables


def _intersection_cache_nominal_integer(values: np.ndarray) -> int | None:
    """Return the nearest integer Bragg index represented by one cache column."""

    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return None
    return int(np.rint(float(np.mean(finite))))


def _intersection_cache_group_key(cache_table: np.ndarray) -> tuple[object, ...] | None:
    """Return the nominal Bragg-family key for one cache table."""

    table = np.asarray(cache_table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] <= 0 or table.shape[1] < 9:
        return None

    l_key = _intersection_cache_nominal_integer(table[:, 8])
    if l_key is None:
        return None

    h_val = float(table[0, 6])
    k_val = float(table[0, 7])
    if (
        np.isfinite(h_val)
        and np.isfinite(k_val)
        and abs(h_val) < 1.0e-12
        and abs(k_val) < 1.0e-12
    ):
        return ("specular", l_key)

    qr_vals = np.asarray(table[:, 0], dtype=np.float64).reshape(-1)
    finite_qr = qr_vals[np.isfinite(qr_vals)]
    if finite_qr.size > 0:
        return ("non_specular", round(float(np.mean(finite_qr)), 8), l_key)

    h_key = _intersection_cache_nominal_integer(table[:, 6])
    k_key = _intersection_cache_nominal_integer(table[:, 7])
    return ("non_specular_hkl", h_key, k_key, l_key)


def _sample_context_index_valid(
    sample_idx: int,
    beam_x_array: np.ndarray | None,
    beam_y_array: np.ndarray | None,
    theta_array: np.ndarray | None,
    phi_array: np.ndarray | None,
    wavelength_array: np.ndarray | None,
) -> bool:
    """Return whether all per-sample context arrays contain ``sample_idx``."""

    if sample_idx < 0:
        return False
    arrays = (beam_x_array, beam_y_array, theta_array, phi_array, wavelength_array)
    for arr in arrays:
        if arr is None or sample_idx >= arr.shape[0]:
            return False
    return True


def _hit_table_row_sample_indices(hits_arr: np.ndarray) -> np.ndarray:
    """Return per-hit beam sample indices carried by provenance-width hit tables."""

    n_rows = int(hits_arr.shape[0]) if hits_arr.ndim == 2 else 0
    row_sample_indices = np.full(n_rows, -1, dtype=np.int64)
    if hits_arr.ndim != 2 or hits_arr.shape[1] <= HIT_ROW_COL_BEST_SAMPLE_INDEX:
        return row_sample_indices
    raw_values = np.asarray(hits_arr[:, HIT_ROW_COL_BEST_SAMPLE_INDEX], dtype=np.float64)
    finite_mask = np.isfinite(raw_values)
    row_sample_indices[finite_mask] = np.rint(raw_values[finite_mask]).astype(np.int64)
    row_sample_indices[row_sample_indices < 0] = -1
    return row_sample_indices


def _hit_table_row_context_offsets(hits_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return explicit per-row beam-context offsets carried by extended hit rows."""

    n_rows = int(hits_arr.shape[0]) if hits_arr.ndim == 2 else 0
    offsets = np.full((n_rows, 5), np.nan, dtype=np.float64)
    valid = np.zeros(n_rows, dtype=bool)
    if hits_arr.ndim != 2 or hits_arr.shape[1] < HIT_ROW_WITH_CONTEXT_WIDTH:
        return valid, offsets
    offsets[:, 0] = hits_arr[:, HIT_ROW_COL_BEAM_X_OFFSET]
    offsets[:, 1] = hits_arr[:, HIT_ROW_COL_BEAM_Y_OFFSET]
    offsets[:, 2] = hits_arr[:, HIT_ROW_COL_THETA_OFFSET]
    offsets[:, 3] = hits_arr[:, HIT_ROW_COL_PHI_OFFSET]
    offsets[:, 4] = hits_arr[:, HIT_ROW_COL_WAVELENGTH_OFFSET]
    valid = np.all(np.isfinite(offsets), axis=1)
    return valid, offsets


def build_intersection_cache(
    hit_tables,
    av,
    cv,
    beam_x_array=None,
    beam_y_array=None,
    theta_array=None,
    phi_array=None,
    wavelength_array=None,
    best_sample_indices_out=None,
):
    """Convert hit tables into a per-Bragg-peak detector intersection cache.

    Current detector-cache rows use 17 columns:
    ``[Qr, Qz, detector_col, detector_row, intensity, phi, H, K, L,
    beam_x_offset, beam_z_offset, divergence_x_offset, divergence_z_offset,
    wavelength_offset, source_table_index, source_row_index, best_sample_index]``.
    Weighted-event hit tables keep
    all sampled rows, including duplicates, so this builder preserves all rows
    after nominal family grouping and only normalizes them into one-row cache
    tables.
    """

    if hit_tables is None:
        return []

    beam_x_arr = None if beam_x_array is None else np.asarray(beam_x_array, dtype=np.float64)
    beam_y_arr = None if beam_y_array is None else np.asarray(beam_y_array, dtype=np.float64)
    theta_arr = None if theta_array is None else np.asarray(theta_array, dtype=np.float64)
    phi_arr = None if phi_array is None else np.asarray(phi_array, dtype=np.float64)
    wavelength_arr = (
        None if wavelength_array is None else np.asarray(wavelength_array, dtype=np.float64)
    )
    best_sample_arr = (
        None
        if best_sample_indices_out is None
        else np.asarray(best_sample_indices_out, dtype=np.int64).reshape(-1)
    )

    av_val = float(av)
    cv_val = float(cv)
    qr_scale = np.nan
    qz_scale = np.nan
    if np.isfinite(av_val) and abs(av_val) > 1.0e-12:
        qr_scale = 4.0 * np.pi / av_val
    if np.isfinite(cv_val) and abs(cv_val) > 1.0e-12:
        qz_scale = 2.0 * np.pi / cv_val

    grouped_cache_tables: dict[tuple[object, ...], list[np.ndarray]] = {}
    beam_x_center = float("nan")
    beam_y_center = float("nan")
    theta_center = float("nan")
    phi_center = float("nan")
    wavelength_center = float("nan")
    if beam_x_arr is not None and beam_x_arr.size > 0:
        beam_x_center = float(np.mean(beam_x_arr))
    if beam_y_arr is not None and beam_y_arr.size > 0:
        beam_y_center = float(np.mean(beam_y_arr))
    if theta_arr is not None and theta_arr.size > 0:
        theta_center = float(np.mean(theta_arr))
    if phi_arr is not None and phi_arr.size > 0:
        phi_center = float(np.mean(phi_arr))
    if wavelength_arr is not None and wavelength_arr.size > 0:
        wavelength_center = float(np.mean(wavelength_arr))
    has_beam_context_arrays = (
        beam_x_arr is not None
        and beam_y_arr is not None
        and theta_arr is not None
        and phi_arr is not None
        and wavelength_arr is not None
    )

    for table_idx, hits in enumerate(hit_tables):
        hits_arr = np.asarray(hits, dtype=np.float64)
        if hits_arr.ndim != 2 or hits_arr.shape[0] == 0:
            continue
        if hits_arr.shape[1] < HIT_ROW_WITH_PROVENANCE_WIDTH:
            raise ValueError("Current hit tables require source provenance.")
        source_table_values = hits_arr[:, HIT_ROW_COL_SOURCE_TABLE_INDEX]
        source_row_values = hits_arr[:, HIT_ROW_COL_SOURCE_ROW_INDEX]
        if not (
            np.all(np.isfinite(source_table_values))
            and np.all(source_table_values >= 0.0)
            and np.all(source_table_values <= float(2**53 - 1))
            and np.all(source_table_values == np.rint(source_table_values))
            and np.all(np.isfinite(source_row_values))
            and np.all(source_row_values >= 0.0)
            and np.all(source_row_values <= float(2**53 - 1))
            and np.all(source_row_values == np.rint(source_row_values))
        ):
            raise ValueError("Current hit-table source provenance must be nonnegative integers.")

        h_vals = hits_arr[:, 4]
        k_vals = hits_arr[:, 5]
        l_vals = hits_arr[:, 6]
        qr_vals = qr_scale * np.sqrt(
            np.clip((h_vals * h_vals + h_vals * k_vals + k_vals * k_vals) / 3.0, 0.0, None)
        )
        qz_vals = qz_scale * l_vals

        sample_idx = -1
        if best_sample_arr is not None and table_idx < best_sample_arr.shape[0]:
            sample_idx = int(best_sample_arr[table_idx])
        row_sample_indices = _hit_table_row_sample_indices(hits_arr)
        row_context_valid, row_context_offsets = _hit_table_row_context_offsets(hits_arr)

        if has_beam_context_arrays:
            has_valid_sample_idx = _sample_context_index_valid(
                sample_idx,
                beam_x_arr,
                beam_y_arr,
                theta_arr,
                phi_arr,
                wavelength_arr,
            )
            has_valid_row_sample_idx = any(
                _sample_context_index_valid(
                    int(row_sample_idx),
                    beam_x_arr,
                    beam_y_arr,
                    theta_arr,
                    phi_arr,
                    wavelength_arr,
                )
                for row_sample_idx in row_sample_indices
            )
            if not (has_valid_sample_idx or has_valid_row_sample_idx):
                if not np.any(row_context_valid):
                    continue
                keep_rows = row_context_valid
                hits_arr = np.asarray(hits_arr[keep_rows], dtype=np.float64)
                h_vals = hits_arr[:, 4]
                k_vals = hits_arr[:, 5]
                l_vals = hits_arr[:, 6]
                qr_vals = qr_vals[keep_rows]
                qz_vals = qz_vals[keep_rows]
                source_table_values = source_table_values[keep_rows]
                source_row_values = source_row_values[keep_rows]
                row_sample_indices = row_sample_indices[keep_rows]
                row_context_offsets = row_context_offsets[keep_rows]
                row_context_valid = row_context_valid[keep_rows]

        n_rows = hits_arr.shape[0]
        cache_table = np.empty((n_rows, CURRENT_DETECTOR_CACHE_WIDTH), dtype=np.float64)
        cache_table[:, 0] = qr_vals
        cache_table[:, 1] = qz_vals
        cache_table[:, 2] = hits_arr[:, 1]
        cache_table[:, 3] = hits_arr[:, 2]
        cache_table[:, 4] = hits_arr[:, 0]
        cache_table[:, 5] = hits_arr[:, 3]
        cache_table[:, 6:9] = hits_arr[:, 4:7]
        cache_table[:, 9:14] = np.nan
        cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX] = np.nan
        cache_table[:, CACHE_COL_SOURCE_TABLE_INDEX] = source_table_values
        cache_table[:, CACHE_COL_SOURCE_ROW_INDEX] = source_row_values
        if np.any(row_context_valid):
            cache_table[row_context_valid, 9:14] = row_context_offsets[row_context_valid]
        if has_beam_context_arrays:
            for row_idx in range(n_rows):
                row_sample_idx = int(row_sample_indices[row_idx])
                sample_for_row = row_sample_idx if row_sample_idx >= 0 else sample_idx
                if row_sample_idx >= 0:
                    cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX] = float(row_sample_idx)
                if row_context_valid[row_idx]:
                    continue
                if _sample_context_index_valid(
                    sample_for_row,
                    beam_x_arr,
                    beam_y_arr,
                    theta_arr,
                    phi_arr,
                    wavelength_arr,
                ):
                    cache_table[row_idx, 9] = beam_x_arr[sample_for_row] - beam_x_center
                    cache_table[row_idx, 10] = beam_y_arr[sample_for_row] - beam_y_center
                    cache_table[row_idx, 11] = theta_arr[sample_for_row] - theta_center
                    cache_table[row_idx, 12] = phi_arr[sample_for_row] - phi_center
                    cache_table[row_idx, 13] = wavelength_arr[sample_for_row] - wavelength_center
                    if not np.isfinite(cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX]):
                        cache_table[row_idx, CACHE_COL_BEST_SAMPLE_INDEX] = float(sample_for_row)
        else:
            valid_row_samples = row_sample_indices >= 0
            missing_best_sample = ~np.isfinite(cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX])
            write_row_samples = valid_row_samples & missing_best_sample
            cache_table[write_row_samples, CACHE_COL_BEST_SAMPLE_INDEX] = row_sample_indices[
                write_row_samples
            ].astype(np.float64)
            if sample_idx >= 0:
                group_sample_rows = ~valid_row_samples & ~np.isfinite(
                    cache_table[:, CACHE_COL_BEST_SAMPLE_INDEX]
                )
                group_sample_rows &= ~row_context_valid
                cache_table[group_sample_rows, CACHE_COL_BEST_SAMPLE_INDEX] = float(sample_idx)
        group_key = _intersection_cache_group_key(cache_table)
        if group_key is None:
            group_key = ("ungrouped", int(table_idx))
        grouped_cache_tables.setdefault(group_key, []).append(cache_table)

    cache = []
    for group_tables in grouped_cache_tables.values():
        valid_group_tables = []
        for cache_table in group_tables:
            table = np.asarray(cache_table, dtype=np.float64)
            if table.ndim != 2 or table.shape[0] <= 0:
                continue
            valid_group_tables.append(table)
        if not valid_group_tables:
            continue
        combined_table = (
            np.vstack(valid_group_tables)
            if len(valid_group_tables) > 1
            else np.asarray(valid_group_tables[0], dtype=np.float64)
        )
        expanded_tables = [
            np.asarray(combined_table[row_idx : row_idx + 1, :], dtype=np.float64).copy()
            for row_idx in range(combined_table.shape[0])
        ]
        cache.extend(expanded_tables)

    return cache


def process_qr_rods_parallel(
    qr_dict,
    image_size,
    av,
    cv,
    lambda_,
    image,
    Distance_CoR_to_Detector,
    gamma_deg,
    Gamma_deg,
    chi_deg,
    psi_deg,
    psi_z_deg,
    zs,
    zb,
    n2,
    beam_x_array,
    beam_y_array,
    theta_array,
    phi_array,
    sigma_pv_deg,
    gamma_pv_deg,
    eta_pv,
    wavelength_array,
    debye_x,
    debye_y,
    center,
    theta_initial_deg,
    cor_angle_deg,
    unit_x,
    n_detector,
    save_flag,
    thickness=0.0,
    solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
    solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
    solve_q_mode=DEFAULT_SOLVE_Q_MODE,
    best_sample_indices_out=None,
    collect_hit_tables=True,
    pixel_size_m=100e-6,
    sample_width_m=0.0,
    sample_length_m=0.0,
    n2_sample_array_override=None,
    accumulate_image=True,
    numba_thread_count=None,
    events_per_beam_phase=50,
    collect_representative_hit_tables=False,
    representative_peak_indices=None,
):
    """Wrapper to process Hendricks–Teller rods instead of individual reflections.

    The Hendricks–Teller preprocessing groups symmetry-related in-plane peaks
    into ``Qr`` rods and records how many peaks contributed to each rod in the
    ``deg`` field.  ``qr_dict_to_arrays`` already returns rod intensities as the
    total summed intensity over the grouped HK pairs, so we forward that array
    unchanged to avoid double-counting.  The degeneracy array is returned so
    downstream code can still track how many symmetry-equivalent HK pairs
    contributed to each rod.
    """
    from ra_sim.utils.stacking_fault import qr_dict_to_arrays

    miller, intensities, degeneracy, _ = qr_dict_to_arrays(qr_dict)

    result = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        av,
        cv,
        lambda_,
        image,
        Distance_CoR_to_Detector,
        gamma_deg,
        Gamma_deg,
        chi_deg,
        psi_deg,
        psi_z_deg,
        zs,
        zb,
        n2,
        beam_x_array,
        beam_y_array,
        theta_array,
        phi_array,
        sigma_pv_deg,
        gamma_pv_deg,
        eta_pv,
        wavelength_array,
        debye_x,
        debye_y,
        center,
        theta_initial_deg,
        cor_angle_deg,
        unit_x,
        n_detector,
        save_flag,
        thickness,
        solve_q_steps,
        solve_q_rel_tol,
        solve_q_mode,
        best_sample_indices_out=best_sample_indices_out,
        collect_hit_tables=collect_hit_tables,
        pixel_size_m=pixel_size_m,
        sample_width_m=sample_width_m,
        sample_length_m=sample_length_m,
        n2_sample_array_override=n2_sample_array_override,
        accumulate_image=accumulate_image,
        numba_thread_count=numba_thread_count,
        events_per_beam_phase=events_per_beam_phase,
        collect_representative_hit_tables=collect_representative_hit_tables,
        representative_peak_indices=representative_peak_indices,
    )

    return (*result[:6], degeneracy, result[6], result[7])
