"""Helpers to inspect Bragg-sphere and Ewald-sphere intersections."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, pi, sin

import numpy as np

from .diffraction import _kz_branch_decay, intersect_line_plane

@dataclass(frozen=True)
class IntersectionGeometry:
    image_size: int
    center_col: float
    center_row: float
    distance_cor_to_detector: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    theta_initial_deg: float
    cor_angle_deg: float
    n_detector: np.ndarray
    unit_x: np.ndarray
    pixel_size_m: float = 100e-6


@dataclass(frozen=True)
class QrCylinderDetectorTrace:
    qr_value: float
    branch_sign: int
    detector_col: np.ndarray
    detector_row: np.ndarray
    qz: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class _BeamContext:
    i_plane: np.ndarray
    r_sample: np.ndarray
    detector_pos: np.ndarray
    n_det_rot: np.ndarray
    e1_det: np.ndarray
    e2_det: np.ndarray
    k_scat: float
    k_x_scat: float
    k_y_scat: float
    re_k_z: float
    k_in_crystal: np.ndarray
    all_q: np.ndarray


@dataclass(frozen=True)
class NominalProjectionFrame:
    i_plane: np.ndarray
    r_sample: np.ndarray
    detector_pos: np.ndarray
    n_det_rot: np.ndarray
    e1_det: np.ndarray
    e2_det: np.ndarray
    u_i_lab: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        return np.asarray(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _vector_or_default(value: np.ndarray | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(default if value is None else value, dtype=np.float64)


def _build_detector_frame(geometry: IntersectionGeometry):
    gamma_rad = np.deg2rad(float(geometry.gamma_deg))
    Gamma_rad = np.deg2rad(float(geometry.Gamma_deg))

    cg = np.cos(gamma_rad)
    sg = np.sin(gamma_rad)
    cG = np.cos(Gamma_rad)
    sG = np.sin(Gamma_rad)

    r_x_det = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cg, sg],
            [0.0, -sg, cg],
        ],
        dtype=np.float64,
    )
    r_z_det = np.array(
        [
            [cG, sG, 0.0],
            [-sG, cG, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n_det_rot = _unit(r_z_det @ (r_x_det @ np.asarray(geometry.n_detector, dtype=np.float64)))
    detector_pos = np.array([0.0, float(geometry.distance_cor_to_detector), 0.0], dtype=np.float64)

    unit_x = np.asarray(geometry.unit_x, dtype=np.float64)
    e1_det = unit_x - np.dot(unit_x, n_det_rot) * n_det_rot
    if float(np.linalg.norm(e1_det)) < 1e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det = _unit(e1_det)

    e2_det = -np.cross(n_det_rot, e1_det)
    if float(np.linalg.norm(e2_det)) < 1e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det = _unit(e2_det)

    return detector_pos, n_det_rot, e1_det, e2_det


def _build_sample_frame(geometry: IntersectionGeometry):
    chi_rad = np.deg2rad(float(geometry.chi_deg))
    psi_rad = np.deg2rad(float(geometry.psi_deg))
    psi_z_rad = np.deg2rad(float(geometry.psi_z_deg))
    theta_initial_rad = np.deg2rad(float(geometry.theta_initial_deg))
    cor_angle_rad = np.deg2rad(float(geometry.cor_angle_deg))

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    r_y = np.array(
        [
            [c_chi, 0.0, s_chi],
            [0.0, 1.0, 0.0],
            [-s_chi, 0.0, c_chi],
        ],
        dtype=np.float64,
    )

    c_psi = cos(psi_rad)
    s_psi = sin(psi_rad)
    r_z = np.array(
        [
            [c_psi, s_psi, 0.0],
            [-s_psi, c_psi, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    r_z_r_y = r_z @ r_y

    ct = cos(theta_initial_rad)
    st = sin(theta_initial_rad)
    ax = cos(cor_angle_rad)
    ay = 0.0
    az = sin(cor_angle_rad)
    c_psi_z = cos(psi_z_rad)
    s_psi_z = sin(psi_z_rad)
    ax, ay = c_psi_z * ax + s_psi_z * ay, -s_psi_z * ax + c_psi_z * ay
    axis = _unit(np.array([ax, ay, az], dtype=np.float64))
    ax = float(axis[0])
    ay = float(axis[1])
    az = float(axis[2])
    one_ct = 1.0 - ct
    r_cor = np.array(
        [
            [ct + ax * ax * one_ct, ax * ay * one_ct - az * st, ax * az * one_ct + ay * st],
            [ay * ax * one_ct + az * st, ct + ay * ay * one_ct, ay * az * one_ct - ax * st],
            [az * ax * one_ct - ay * st, az * ay * one_ct + ax * st, ct + az * az * one_ct],
        ],
        dtype=np.float64,
    )
    r_sample = r_cor @ r_z_r_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_surf = _unit(r_cor @ (r_z_r_y @ n1))

    p0 = np.array([0.0, 0.0, -float(geometry.zs)], dtype=np.float64)
    p0_rot = r_sample @ p0
    p0_rot[0] = 0.0

    return r_sample, n_surf, p0_rot


def build_nominal_projection_frame(
    *,
    distance_cor_to_detector: float,
    gamma_deg: float,
    Gamma_deg: float,
    chi_deg: float,
    psi_deg: float,
    psi_z_deg: float,
    zs: float,
    zb: float,
    theta_initial_deg: float,
    cor_angle_deg: float,
    n_detector: np.ndarray | None = None,
    unit_x: np.ndarray | None = None,
    beam_x: float = 0.0,
    beam_y: float = 0.0,
    dtheta: float = 0.0,
    dphi: float = 0.0,
) -> NominalProjectionFrame:
    """Return one shared detector/sample projection frame for one nominal beam."""

    geometry = IntersectionGeometry(
        image_size=1,
        center_col=0.0,
        center_row=0.0,
        distance_cor_to_detector=float(distance_cor_to_detector),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
        chi_deg=float(chi_deg),
        psi_deg=float(psi_deg),
        psi_z_deg=float(psi_z_deg),
        zs=float(zs),
        zb=float(zb),
        theta_initial_deg=float(theta_initial_deg),
        cor_angle_deg=float(cor_angle_deg),
        n_detector=_vector_or_default(n_detector, (0.0, 1.0, 0.0)),
        unit_x=_vector_or_default(unit_x, (1.0, 0.0, 0.0)),
    )

    detector_pos, n_det_rot, e1_det, e2_det = _build_detector_frame(geometry)
    r_sample, n_surf, p0_rot = _build_sample_frame(geometry)
    beam_start = np.array(
        [float(beam_x), -20e-3, -float(geometry.zb) + float(beam_y)],
        dtype=np.float64,
    )
    u_i_lab = np.array(
        [
            cos(float(dtheta)) * sin(float(dphi)),
            cos(float(dtheta)) * cos(float(dphi)),
            sin(float(dtheta)),
        ],
        dtype=np.float64,
    )

    ix, iy, iz, valid_int = intersect_line_plane(beam_start, u_i_lab, p0_rot, n_surf)
    if not valid_int:
        raise ValueError("Beam sample does not intersect the sample plane.")

    return NominalProjectionFrame(
        i_plane=np.array([ix, iy, iz], dtype=np.float64),
        r_sample=r_sample,
        detector_pos=detector_pos,
        n_det_rot=n_det_rot,
        e1_det=e1_det,
        e2_det=e2_det,
        u_i_lab=u_i_lab,
    )


def _build_single_beam_context(
    *,
    geometry: IntersectionGeometry,
    beam_x: float,
    beam_y: float,
    dtheta: float,
    dphi: float,
    wavelength: float,
    n2: complex,
) -> _BeamContext:
    """Return the projection context for one beam sample.

    This mirrors the single-sample geometry used by the intersection analysis,
    but without solving for any specific Bragg reflection.
    """

    if wavelength <= 0.0:
        raise ValueError("Beam wavelength must be positive.")

    frame = build_nominal_projection_frame(
        distance_cor_to_detector=float(geometry.distance_cor_to_detector),
        gamma_deg=float(geometry.gamma_deg),
        Gamma_deg=float(geometry.Gamma_deg),
        chi_deg=float(geometry.chi_deg),
        psi_deg=float(geometry.psi_deg),
        psi_z_deg=float(geometry.psi_z_deg),
        zs=float(geometry.zs),
        zb=float(geometry.zb),
        theta_initial_deg=float(geometry.theta_initial_deg),
        cor_angle_deg=float(geometry.cor_angle_deg),
        n_detector=np.asarray(geometry.n_detector, dtype=np.float64),
        unit_x=np.asarray(geometry.unit_x, dtype=np.float64),
        beam_x=float(beam_x),
        beam_y=float(beam_y),
        dtheta=float(dtheta),
        dphi=float(dphi),
    )
    r_sample, n_surf, _p0_rot = _build_sample_frame(geometry)
    k_in = frame.u_i_lab
    i_plane = frame.i_plane
    kn_dot = float(np.dot(k_in, n_surf))
    kn_dot = float(np.clip(kn_dot, -1.0, 1.0))
    th_i_prime = (pi / 2.0) - acos(kn_dot)

    projected = k_in - kn_dot * n_surf
    pln = float(np.linalg.norm(projected))
    if pln > 1e-12:
        projected = projected / pln
    else:
        projected[:] = 0.0
    e1_temp = np.cross(n_surf, np.array([0.0, 0.0, -1.0], dtype=np.float64))
    if float(np.linalg.norm(e1_temp)) < 1e-12:
        e1_temp = np.cross(n_surf, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    e1_temp = _unit(e1_temp)
    e2_temp = np.cross(n_surf, e1_temp)
    p1 = float(np.dot(projected, e1_temp))
    p2 = float(np.dot(projected, e2_temp))
    phi_i_prime = (pi / 2.0) - np.arctan2(p2, p1)

    k0 = 2.0 * pi / float(wavelength)
    k_par_i = k0 * abs(np.cos(th_i_prime))
    kz2_i = _kz_branch_decay((n2 * n2 * k0 * k0) - (k_par_i * k_par_i))
    k_x_scat = float(k_par_i * np.sin(phi_i_prime))
    k_y_scat = float(k_par_i * np.cos(phi_i_prime))
    re_k_z = float(-abs(kz2_i.real))
    k_scat = float(np.sqrt(max(k_par_i * k_par_i + re_k_z * re_k_z, 0.0)))
    k_in_crystal = np.array([k_x_scat, k_y_scat, re_k_z], dtype=np.float64)

    return _BeamContext(
        i_plane=i_plane,
        r_sample=r_sample,
        detector_pos=frame.detector_pos,
        n_det_rot=frame.n_det_rot,
        e1_det=frame.e1_det,
        e2_det=frame.e2_det,
        k_scat=k_scat,
        k_x_scat=k_x_scat,
        k_y_scat=k_y_scat,
        re_k_z=re_k_z,
        k_in_crystal=k_in_crystal,
        all_q=np.zeros((0, 4), dtype=np.float64),
    )


def _project_kf_to_detector(
    kf_lab: np.ndarray,
    beam_ctx: _BeamContext,
    geometry: IntersectionGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project forward lab-frame scattering rays onto the detector plane."""

    n_det_rot = np.asarray(beam_ctx.n_det_rot, dtype=np.float64)
    detector_pos = np.asarray(beam_ctx.detector_pos, dtype=np.float64)
    i_plane = np.asarray(beam_ctx.i_plane, dtype=np.float64)
    dirs = np.asarray(kf_lab, dtype=np.float64)

    cols = np.full(dirs.shape[0], np.nan, dtype=np.float64)
    rows = np.full(dirs.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(dirs.shape[0], dtype=bool)
    if dirs.ndim != 2 or dirs.shape[1] != 3:
        return cols, rows, valid

    center_col = float(geometry.center_col)
    center_row = float(geometry.center_row)
    pixel_size = float(geometry.pixel_size_m)
    image_size = int(geometry.image_size)

    dist = float(np.dot(i_plane - detector_pos, n_det_rot))
    num = -dist
    denom = dirs @ n_det_rot
    hit_pts = np.full_like(dirs, np.nan, dtype=np.float64)

    parallel_mask = np.abs(denom) < 1e-14
    regular_mask = ~parallel_mask
    if np.any(regular_mask):
        t = num / denom[regular_mask]
        good_t = t >= -1e-9
        if np.any(good_t):
            t_good = np.where(t[good_t] < 0.0, 0.0, t[good_t])
            regular_idx = np.nonzero(regular_mask)[0][good_t]
            hit_pts[regular_idx] = i_plane[None, :] + t_good[:, None] * dirs[regular_idx]
            valid[regular_idx] = True

    if np.any(parallel_mask):
        if abs(dist) < 1e-6:
            hit_pts[parallel_mask] = i_plane[None, :]
            valid[parallel_mask] = True

    if not np.any(valid):
        return cols, rows, valid

    plane_to_det = hit_pts[valid] - detector_pos[None, :]
    x_det = plane_to_det @ np.asarray(beam_ctx.e1_det, dtype=np.float64)
    y_det = plane_to_det @ np.asarray(beam_ctx.e2_det, dtype=np.float64)

    cols_valid = center_col + x_det / pixel_size
    rows_valid = center_row - y_det / pixel_size
    in_bounds = (
        np.isfinite(cols_valid)
        & np.isfinite(rows_valid)
        & (cols_valid >= 0.0)
        & (cols_valid < image_size)
        & (rows_valid >= 0.0)
        & (rows_valid < image_size)
    )

    valid_idx = np.nonzero(valid)[0]
    cols[valid_idx[in_bounds]] = cols_valid[in_bounds]
    rows[valid_idx[in_bounds]] = rows_valid[in_bounds]

    kept = np.zeros_like(valid)
    kept[valid_idx[in_bounds]] = True
    return cols, rows, kept


def project_qr_cylinder_to_detector(
    *,
    qr_value: float,
    geometry: IntersectionGeometry,
    wavelength: float,
    n2: complex,
    beam_x: float = 0.0,
    beam_y: float = 0.0,
    dtheta: float = 0.0,
    dphi: float = 0.0,
    phi_samples: int = 721,
) -> list[QrCylinderDetectorTrace]:
    """Project an analytic Ewald/constant-Qr cylinder intersection onto the detector.

    Returns one detector trace per intersection branch. Invalid samples are kept
    as NaNs so callers can plot the arrays directly and let Matplotlib break the
    line wherever the branch leaves reciprocal space or the detector bounds.
    """

    qr = float(qr_value)
    if not np.isfinite(qr) or qr < 0.0:
        return []
    sample_count = int(max(16, phi_samples))

    beam_ctx = _build_single_beam_context(
        geometry=geometry,
        beam_x=float(beam_x),
        beam_y=float(beam_y),
        dtheta=float(dtheta),
        dphi=float(dphi),
        wavelength=float(wavelength),
        n2=n2,
    )

    phi = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=True, dtype=np.float64)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    kx = float(beam_ctx.k_x_scat)
    ky = float(beam_ctx.k_y_scat)
    kz = float(beam_ctx.re_k_z)

    radicand = kz * kz - qr * qr - (2.0 * qr) * (kx * cphi + ky * sphi)
    base_valid = np.isfinite(radicand) & (radicand >= 0.0)
    if not np.any(base_valid):
        return []

    traces: list[QrCylinderDetectorTrace] = []
    qx = qr * cphi
    qy = qr * sphi
    sqrt_term = np.zeros_like(radicand, dtype=np.float64)
    sqrt_term[base_valid] = np.sqrt(radicand[base_valid])

    for branch_sign in (-1, 1):
        k_tx_prime = kx + qx
        k_ty_prime = ky + qy
        k_tz_prime = branch_sign * sqrt_term
        qz = k_tz_prime - kz

        kr = np.sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
        twotheta_t_prime = np.zeros_like(kr, dtype=np.float64)
        nonzero_kr = kr >= 1e-12
        twotheta_t_prime[nonzero_kr] = np.arctan(k_tz_prime[nonzero_kr] / kr[nonzero_kr])

        cos_term = np.cos(twotheta_t_prime) * float(np.real(n2))
        twotheta_t = np.arccos(np.clip(cos_term, -1.0, 1.0)) * np.sign(twotheta_t_prime)
        phi_f = np.arctan2(k_tx_prime, k_ty_prime)
        kf = np.column_stack(
            [
                beam_ctx.k_scat * np.cos(twotheta_t) * np.sin(phi_f),
                beam_ctx.k_scat * np.cos(twotheta_t) * np.cos(phi_f),
                beam_ctx.k_scat * np.sin(twotheta_t),
            ]
        )
        kf_lab = kf @ np.asarray(beam_ctx.r_sample, dtype=np.float64).T

        cols, rows, det_valid = _project_kf_to_detector(kf_lab, beam_ctx, geometry)
        valid = base_valid & det_valid

        if np.any(valid):
            traces.append(
                QrCylinderDetectorTrace(
                    qr_value=qr,
                    branch_sign=int(branch_sign),
                    detector_col=np.where(valid, cols, np.nan),
                    detector_row=np.where(valid, rows, np.nan),
                    qz=np.where(valid, qz, np.nan),
                    valid_mask=valid,
                )
            )

    return traces


def detector_points_to_sample_qr_qz(
    *,
    detector_col: object,
    detector_row: object,
    geometry: IntersectionGeometry,
    wavelength: float,
    n2: complex,
    beam_x: float = 0.0,
    beam_y: float = 0.0,
    dtheta: float = 0.0,
    dphi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(qr_abs, qz, valid_mask)`` for native detector coordinates.

    This is the inverse of the single-beam detector projection used by
    :func:`project_qr_cylinder_to_detector`, including the same sample frame and
    refraction convention.
    """

    try:
        col_values, row_values = np.broadcast_arrays(
            np.asarray(detector_col, dtype=np.float64),
            np.asarray(detector_row, dtype=np.float64),
        )
    except Exception:
        col_values = np.asarray(detector_col, dtype=np.float64)
        qr_empty = np.full(col_values.shape, np.nan, dtype=np.float64)
        return qr_empty, qr_empty.copy(), np.zeros(col_values.shape, dtype=bool)

    qr_abs = np.full(col_values.shape, np.nan, dtype=np.float64)
    qz = np.full(col_values.shape, np.nan, dtype=np.float64)
    valid = np.zeros(col_values.shape, dtype=bool)
    if col_values.size <= 0:
        return qr_abs, qz, valid

    try:
        beam_ctx = _build_single_beam_context(
            geometry=geometry,
            beam_x=float(beam_x),
            beam_y=float(beam_y),
            dtheta=float(dtheta),
            dphi=float(dphi),
            wavelength=float(wavelength),
            n2=n2,
        )
    except Exception:
        return qr_abs, qz, valid

    pixel_size = float(geometry.pixel_size_m)
    n2_real = float(np.real(n2))
    if (
        not np.isfinite(pixel_size)
        or pixel_size <= 0.0
        or not np.isfinite(n2_real)
        or abs(n2_real) <= 1.0e-14
    ):
        return qr_abs, qz, valid

    flat_cols = np.asarray(col_values, dtype=np.float64).reshape(-1)
    flat_rows = np.asarray(row_values, dtype=np.float64).reshape(-1)
    finite_input = np.isfinite(flat_cols) & np.isfinite(flat_rows)
    if not np.any(finite_input):
        return qr_abs, qz, valid

    x_det = (flat_cols[finite_input] - float(geometry.center_col)) * pixel_size
    y_det = (float(geometry.center_row) - flat_rows[finite_input]) * pixel_size
    detector_pos = np.asarray(beam_ctx.detector_pos, dtype=np.float64)
    e1_det = np.asarray(beam_ctx.e1_det, dtype=np.float64)
    e2_det = np.asarray(beam_ctx.e2_det, dtype=np.float64)
    detector_points = detector_pos[None, :] + x_det[:, None] * e1_det + y_det[:, None] * e2_det
    outgoing = detector_points - np.asarray(beam_ctx.i_plane, dtype=np.float64)[None, :]
    outgoing_norm = np.linalg.norm(outgoing, axis=1)
    good_ray = np.isfinite(outgoing_norm) & (outgoing_norm > 1.0e-14)
    if not np.any(good_ray):
        return qr_abs, qz, valid

    finite_indices = np.flatnonzero(finite_input)
    good_indices = finite_indices[good_ray]
    u_f_lab = outgoing[good_ray] / outgoing_norm[good_ray, None]
    kf_sample = (u_f_lab @ np.asarray(beam_ctx.r_sample, dtype=np.float64)) * float(
        beam_ctx.k_scat
    )

    kf_x = np.asarray(kf_sample[:, 0], dtype=np.float64)
    kf_y = np.asarray(kf_sample[:, 1], dtype=np.float64)
    kf_z = np.asarray(kf_sample[:, 2], dtype=np.float64)
    theta_t = np.arctan2(kf_z, np.hypot(kf_x, kf_y))
    phi_f = np.arctan2(kf_x, kf_y)

    theta_prime = np.sign(theta_t) * np.arccos(
        np.clip(np.cos(np.abs(theta_t)) / n2_real, -1.0, 1.0)
    )
    k_tx_prime = float(beam_ctx.k_scat) * np.cos(theta_prime) * np.sin(phi_f)
    k_ty_prime = float(beam_ctx.k_scat) * np.cos(theta_prime) * np.cos(phi_f)
    k_tz_prime = float(beam_ctx.k_scat) * np.sin(theta_prime)

    qx = k_tx_prime - float(beam_ctx.k_x_scat)
    qy = k_ty_prime - float(beam_ctx.k_y_scat)
    qz_values = k_tz_prime - float(beam_ctx.re_k_z)
    qr_values = np.hypot(qx, qy)
    good_q = np.isfinite(qr_values) & np.isfinite(qz_values)
    if not np.any(good_q):
        return qr_abs, qz, valid

    qr_flat = qr_abs.reshape(-1)
    qz_flat = qz.reshape(-1)
    valid_flat = valid.reshape(-1)
    qr_flat[good_indices[good_q]] = qr_values[good_q]
    qz_flat[good_indices[good_q]] = qz_values[good_q]
    valid_flat[good_indices[good_q]] = True
    return qr_abs, qz, valid
