"""Automatic detector-native hBN powder-ring calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares

from rasim_next.geometry.instrument import compose_intrinsic_xy_rotation

FloatArray = NDArray[np.float64]

HBN_LATTICE_A_A = 2.504
HBN_LATTICE_C_A = 6.661
CU_K_ALPHA_WAVELENGTH_A = 1.5406
HBN_RING_HKL = ((0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (0, 0, 4))


def hbn_two_theta_rad(
    *,
    lattice_a_A: float = HBN_LATTICE_A_A,
    lattice_c_A: float = HBN_LATTICE_C_A,
    wavelength_A: float = CU_K_ALPHA_WAVELENGTH_A,
) -> FloatArray:
    """Return the five declared hBN powder-ring scattering angles."""

    a = float(lattice_a_A)
    c = float(lattice_c_A)
    wavelength = float(wavelength_A)
    if not all(math.isfinite(value) and value > 0.0 for value in (a, c, wavelength)):
        raise ValueError("hBN lattice constants and wavelength must be finite and positive")
    reciprocal_d2 = np.asarray(
        [
            4.0 * (h * h + h * k + k * k) / (3.0 * a * a) + layer * layer / (c * c)
            for h, k, layer in HBN_RING_HKL
        ],
        dtype=np.float64,
    )
    sine_theta = 0.5 * wavelength * np.sqrt(reciprocal_d2)
    if np.any(sine_theta >= 1.0):
        raise ValueError("an hBN ring has no physical Bragg angle")
    result = 2.0 * np.arcsin(sine_theta)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class HbnRingObservations:
    """Automatically traced detector-native hBN ring coordinates."""

    coordinates_px: FloatArray
    ring_index: NDArray[np.int64]
    two_theta_rad: FloatArray
    angular_sector: NDArray[np.int64]

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates_px, dtype=np.float64)
        rings = np.asarray(self.ring_index, dtype=np.int64)
        angles = np.asarray(self.two_theta_rad, dtype=np.float64)
        sectors = np.asarray(self.angular_sector, dtype=np.int64)
        count = rings.size
        if (
            coordinates.shape != (count, 2)
            or angles.shape != (len(HBN_RING_HKL),)
            or sectors.shape != (count,)
            or not np.all(np.isfinite(coordinates))
            or not np.all(np.isfinite(angles))
            or np.any(rings < 0)
            or np.any(rings >= angles.size)
            or np.any(sectors < 0)
        ):
            raise ValueError("invalid hBN ring observations")
        if count < 20 or len(set(int(value) for value in rings)) < 2:
            raise ValueError("hBN calibration requires at least two well-sampled rings")
        for value in (coordinates, rings, angles, sectors):
            value.setflags(write=False)
        object.__setattr__(self, "coordinates_px", coordinates)
        object.__setattr__(self, "ring_index", rings)
        object.__setattr__(self, "two_theta_rad", angles)
        object.__setattr__(self, "angular_sector", sectors)


@dataclass(frozen=True, slots=True)
class HbnDetectorCalibration:
    """Physical hBN result; distance is calibrant-private and is not a shared fit value."""

    detector_column_tilt_rad: float
    detector_row_tilt_rad: float
    beam_center_column_px: float
    beam_center_row_px: float
    calibrant_distance_m: float
    residual_rms_px: float
    residual_max_px: float
    ring_rms_px: tuple[float, ...]
    ring_point_count: tuple[int, ...]
    ring_angular_coverage_fraction: tuple[float, ...]
    standard_error: tuple[float, ...]
    jacobian_rank: int
    scaled_jacobian_condition: float
    success: bool

    @property
    def values(self) -> FloatArray:
        result = np.asarray(
            (
                self.detector_column_tilt_rad,
                self.detector_row_tilt_rad,
                self.beam_center_column_px,
                self.beam_center_row_px,
                self.calibrant_distance_m,
            ),
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result


def _unit_vector(value: ArrayLike, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _ring_residual_px(
    values: FloatArray,
    observations: HbnRingObservations,
    *,
    base_detector_rotation: FloatArray,
    beam_direction_lab: FloatArray,
    detector_column_pitch_m: float,
    detector_row_pitch_m: float,
) -> FloatArray:
    tilt_column, tilt_row, center_column, center_row, distance_m = values
    detector_rotation = compose_intrinsic_xy_rotation(
        base_detector_rotation,
        tilt_column,
        tilt_row,
    )
    beam_detector = detector_rotation.T @ beam_direction_lab
    offset_detector_m = np.column_stack(
        (
            (observations.coordinates_px[:, 0] - center_column) * detector_column_pitch_m,
            (observations.coordinates_px[:, 1] - center_row) * detector_row_pitch_m,
            np.zeros(len(observations.ring_index), dtype=np.float64),
        )
    )
    scattered = distance_m * beam_detector + offset_detector_m
    cosine = (scattered @ beam_detector) / np.linalg.norm(scattered, axis=1)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    pitch = math.sqrt(detector_column_pitch_m * detector_row_pitch_m)
    return (angle - observations.two_theta_rad[observations.ring_index]) * distance_m / pitch


def _ring_curves_px(
    values: FloatArray,
    two_theta_rad: FloatArray,
    azimuth_rad: FloatArray,
    *,
    base_detector_rotation: FloatArray,
    beam_direction_lab: FloatArray,
    detector_column_pitch_m: float,
    detector_row_pitch_m: float,
) -> tuple[FloatArray, ...]:
    tilt_column, tilt_row, center_column, center_row, distance_m = values
    detector_rotation = compose_intrinsic_xy_rotation(
        base_detector_rotation,
        tilt_column,
        tilt_row,
    )
    beam = detector_rotation.T @ beam_direction_lab
    if beam[2] <= 0.0:
        raise ValueError("beam must intersect the detector front plane")
    trial = np.asarray((1.0, 0.0, 0.0))
    first = trial - beam * float(trial @ beam)
    if np.linalg.norm(first) < 1.0e-8:
        trial = np.asarray((0.0, 1.0, 0.0))
        first = trial - beam * float(trial @ beam)
    first /= np.linalg.norm(first)
    second = np.cross(beam, first)
    curves = []
    for angle in two_theta_rad:
        direction = math.cos(float(angle)) * beam[:, None] + math.sin(float(angle)) * (
            first[:, None] * np.cos(azimuth_rad) + second[:, None] * np.sin(azimuth_rad)
        )
        distance_along_ray = distance_m * beam[2] / direction[2]
        detector_point_m = -distance_m * beam[:, None] + direction * distance_along_ray
        curves.append(
            np.column_stack(
                (
                    center_column + detector_point_m[0] / detector_column_pitch_m,
                    center_row + detector_point_m[1] / detector_row_pitch_m,
                )
            )
        )
    return tuple(curves)


def _fit_observations(
    observations: HbnRingObservations,
    initial: FloatArray,
    *,
    base_detector_rotation: FloatArray,
    beam_direction_lab: FloatArray,
    detector_column_pitch_m: float,
    detector_row_pitch_m: float,
    detector_shape_rc: tuple[int, int],
) -> HbnDetectorCalibration:
    rows, columns = detector_shape_rc
    mean_pitch = math.sqrt(detector_column_pitch_m * detector_row_pitch_m)
    lower = np.asarray((-0.15, -0.15, 0.0, 0.0, 0.04), dtype=np.float64)
    upper = np.asarray((0.15, 0.15, columns - 1.0, rows - 1.0, 0.12), dtype=np.float64)
    scale = np.asarray((0.03, 0.03, 10.0, 10.0, 100.0 * mean_pitch), dtype=np.float64)

    def residual(values: FloatArray) -> FloatArray:
        return _ring_residual_px(
            values,
            observations,
            base_detector_rotation=base_detector_rotation,
            beam_direction_lab=beam_direction_lab,
            detector_column_pitch_m=detector_column_pitch_m,
            detector_row_pitch_m=detector_row_pitch_m,
        )

    fit = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        x_scale=scale,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=1000,
    )
    raw = residual(fit.x)
    scaled_jacobian = fit.jac * scale
    singular = np.linalg.svd(scaled_jacobian, compute_uv=False)
    tolerance = singular[0] * max(scaled_jacobian.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    degrees_of_freedom = max(raw.size - fit.x.size, 1)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * float(raw @ raw) / degrees_of_freedom
    standard_error = tuple(float(value) for value in np.sqrt(np.maximum(np.diag(covariance), 0.0)))
    ring_rms = []
    ring_count = []
    ring_coverage = []
    sector_count = int(np.max(observations.angular_sector)) + 1
    for ring_index in range(len(observations.two_theta_rad)):
        selected = observations.ring_index == ring_index
        ring_count.append(int(np.count_nonzero(selected)))
        ring_rms.append(
            float(np.sqrt(np.mean(raw[selected] ** 2))) if np.any(selected) else math.inf
        )
        ring_coverage.append(
            len(set(int(value) for value in observations.angular_sector[selected])) / sector_count
            if np.any(selected)
            else 0.0
        )
    active = np.isclose(fit.x, lower, rtol=0.0, atol=1.0e-8 * (upper - lower)) | np.isclose(
        fit.x,
        upper,
        rtol=0.0,
        atol=1.0e-8 * (upper - lower),
    )
    success = bool(
        fit.success
        and rank == fit.x.size
        and condition < 1.0e8
        and not np.any(active)
        and all(count >= 8 for count in ring_count)
        and all(coverage >= 0.15 for coverage in ring_coverage)
        and max(ring_rms) <= 2.5
    )
    return HbnDetectorCalibration(
        detector_column_tilt_rad=float(fit.x[0]),
        detector_row_tilt_rad=float(fit.x[1]),
        beam_center_column_px=float(fit.x[2]),
        beam_center_row_px=float(fit.x[3]),
        calibrant_distance_m=float(fit.x[4]),
        residual_rms_px=float(np.sqrt(np.mean(raw**2))),
        residual_max_px=float(np.max(np.abs(raw))),
        ring_rms_px=tuple(ring_rms),
        ring_point_count=tuple(ring_count),
        ring_angular_coverage_fraction=tuple(ring_coverage),
        standard_error=standard_error,
        jacobian_rank=rank,
        scaled_jacobian_condition=condition,
        success=success,
    )


def _sector_balanced_observations(
    coordinates_by_ring: tuple[FloatArray, ...],
    score_by_ring: tuple[FloatArray, ...],
    azimuth_by_ring: tuple[FloatArray, ...],
    two_theta_rad: FloatArray,
    *,
    sector_count: int = 36,
) -> HbnRingObservations:
    coordinates = []
    rings = []
    sectors = []
    for ring_index, (ring_coordinates, ring_score, ring_azimuth) in enumerate(
        zip(coordinates_by_ring, score_by_ring, azimuth_by_ring, strict=True)
    ):
        sector = np.floor(np.mod(ring_azimuth, 2.0 * np.pi) * sector_count / (2.0 * np.pi)).astype(
            np.int64
        )
        for sector_index in sorted(set(int(value) for value in sector)):
            candidates = np.flatnonzero(sector == sector_index)
            selected = int(candidates[np.argmax(ring_score[candidates])])
            coordinates.append(ring_coordinates[selected])
            rings.append(ring_index)
            sectors.append(sector_index)
    return HbnRingObservations(
        coordinates_px=np.asarray(coordinates, dtype=np.float64),
        ring_index=np.asarray(rings, dtype=np.int64),
        two_theta_rad=two_theta_rad,
        angular_sector=np.asarray(sectors, dtype=np.int64),
    )


def fit_hbn_detector_calibration(
    counts: ArrayLike,
    dark_counts: ArrayLike,
    *,
    base_detector_rotation: ArrayLike,
    beam_direction_lab: ArrayLike,
    detector_column_pitch_m: float,
    detector_row_pitch_m: float,
    initial_beam_center_px: tuple[float, float],
    initial_calibrant_distance_m: float = 0.074,
) -> tuple[HbnRingObservations, HbnDetectorCalibration]:
    """Trace the five hBN rings without clicks and fit center, tilts, and private distance."""

    image = np.asarray(counts, dtype=np.float64)
    dark = np.asarray(dark_counts, dtype=np.float64)
    if image.ndim != 2 or dark.shape != image.shape or not np.all(np.isfinite(image + dark)):
        raise ValueError("counts and dark_counts must be equal finite two-dimensional arrays")
    column_pitch = float(detector_column_pitch_m)
    row_pitch = float(detector_row_pitch_m)
    if not all(math.isfinite(value) and value > 0.0 for value in (column_pitch, row_pitch)):
        raise ValueError("detector pitches must be finite and positive")
    rotation = np.asarray(base_detector_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12):
        raise ValueError("base_detector_rotation must be orthonormal")
    beam = _unit_vector(beam_direction_lab, "beam_direction_lab")
    center = np.asarray(initial_beam_center_px, dtype=np.float64)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("initial_beam_center_px must contain finite column and row")
    distance = float(initial_calibrant_distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("initial_calibrant_distance_m must be finite and positive")

    log_signal = np.log1p(np.maximum(image - dark, 0.0))
    two_theta = hbn_two_theta_rad()
    azimuth = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    mean_pitch = math.sqrt(column_pitch * row_pitch)
    radii_px = distance * np.tan(two_theta) / mean_pitch
    radial = np.arange(max(20.0, radii_px[0] - 80.0), radii_px[-1] + 100.0, 0.5)
    sampled = map_coordinates(
        log_signal,
        [
            center[1] + np.sin(azimuth)[:, None] * radial,
            center[0] + np.cos(azimuth)[:, None] * radial,
        ],
        order=1,
        mode="constant",
        cval=0.0,
    )
    radial_score = gaussian_filter1d(sampled, 2.0, axis=1) - gaussian_filter1d(
        sampled,
        24.0,
        axis=1,
    )
    coarse_coordinates = []
    coarse_scores = []
    coarse_azimuths = []
    half_windows = (45.0, 45.0, 28.0, 45.0, 55.0)
    for expected, half_window in zip(radii_px, half_windows, strict=True):
        lower_index = int(np.searchsorted(radial, expected - half_window))
        upper_index = int(np.searchsorted(radial, expected + half_window))
        block = radial_score[:, lower_index:upper_index]
        peak_index = np.argmax(block, axis=1)
        peak_score = block[np.arange(azimuth.size), peak_index]
        median = np.median(block, axis=1)
        mad = 1.4826 * np.median(np.abs(block - median[:, None]), axis=1) + 1.0e-6
        snr = (peak_score - median) / mad
        radius = radial[lower_index + peak_index]
        keep = (snr >= 3.0) & (peak_score > 0.04)
        coarse_coordinates.append(
            np.column_stack(
                (
                    center[0] + np.cos(azimuth[keep]) * radius[keep],
                    center[1] + np.sin(azimuth[keep]) * radius[keep],
                )
            )
        )
        coarse_scores.append(snr[keep])
        coarse_azimuths.append(azimuth[keep])
    coarse = _sector_balanced_observations(
        tuple(coarse_coordinates),
        tuple(coarse_scores),
        tuple(coarse_azimuths),
        two_theta,
    )
    seed_mask = coarse.ring_index < 2
    seed_observations = HbnRingObservations(
        coordinates_px=coarse.coordinates_px[seed_mask],
        ring_index=coarse.ring_index[seed_mask],
        two_theta_rad=coarse.two_theta_rad,
        angular_sector=coarse.angular_sector[seed_mask],
    )
    initial = np.asarray((0.0, 0.0, center[0], center[1], distance), dtype=np.float64)
    calibration = _fit_observations(
        seed_observations,
        initial,
        base_detector_rotation=rotation,
        beam_direction_lab=beam,
        detector_column_pitch_m=column_pitch,
        detector_row_pitch_m=row_pitch,
        detector_shape_rc=image.shape,
    )

    observations = coarse
    for _ in range(4):
        curves = _ring_curves_px(
            calibration.values,
            two_theta,
            azimuth,
            base_detector_rotation=rotation,
            beam_direction_lab=beam,
            detector_column_pitch_m=column_pitch,
            detector_row_pitch_m=row_pitch,
        )
        coordinates_by_ring = []
        score_by_ring = []
        azimuth_by_ring = []
        offsets = np.arange(-12.0, 12.01, 0.5)
        for ring_index, curve in enumerate(curves):
            tangent = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            sample_coordinates = curve[:, None, :] + normal[:, None, :] * offsets[None, :, None]
            profiles = map_coordinates(
                log_signal,
                [sample_coordinates[:, :, 1], sample_coordinates[:, :, 0]],
                order=1,
                mode="constant",
                cval=0.0,
            )
            local_score = gaussian_filter1d(profiles, 2.0, axis=1) - gaussian_filter1d(
                profiles,
                10.0,
                axis=1,
            )
            peak_index = np.argmax(local_score, axis=1)
            peak_score = local_score[np.arange(azimuth.size), peak_index]
            median = np.median(local_score, axis=1)
            mad = 1.4826 * np.median(np.abs(local_score - median[:, None]), axis=1) + 1.0e-6
            snr = (peak_score - median) / mad
            chosen = sample_coordinates[np.arange(azimuth.size), peak_index]
            keep = (
                (snr >= 3.0)
                & (peak_score > max(0.04, float(np.quantile(peak_score, 0.15))))
                & (np.abs(offsets[peak_index]) < 11.5)
            )
            if np.count_nonzero(keep) < 12:
                coordinates_by_ring.append(coarse_coordinates[ring_index])
                score_by_ring.append(coarse_scores[ring_index])
                azimuth_by_ring.append(coarse_azimuths[ring_index])
            else:
                coordinates_by_ring.append(chosen[keep])
                score_by_ring.append(snr[keep])
                azimuth_by_ring.append(azimuth[keep])
        observations = _sector_balanced_observations(
            tuple(coordinates_by_ring),
            tuple(score_by_ring),
            tuple(azimuth_by_ring),
            two_theta,
        )
        calibration = _fit_observations(
            observations,
            calibration.values,
            base_detector_rotation=rotation,
            beam_direction_lab=beam,
            detector_column_pitch_m=column_pitch,
            detector_row_pitch_m=row_pitch,
            detector_shape_rc=image.shape,
        )
    return observations, calibration


def evaluate_hbn_residual_px(
    values: ArrayLike,
    observations: HbnRingObservations,
    *,
    base_detector_rotation: ArrayLike,
    beam_direction_lab: ArrayLike,
    detector_column_pitch_m: float,
    detector_row_pitch_m: float,
) -> FloatArray:
    """Evaluate fixed automatic observations for a joint geometry fit."""

    parameters = np.asarray(values, dtype=np.float64)
    if parameters.shape != (5,) or not np.all(np.isfinite(parameters)):
        raise ValueError("hBN geometry values must contain five finite parameters")
    return _ring_residual_px(
        parameters,
        observations,
        base_detector_rotation=np.asarray(base_detector_rotation, dtype=np.float64),
        beam_direction_lab=_unit_vector(beam_direction_lab, "beam_direction_lab"),
        detector_column_pitch_m=float(detector_column_pitch_m),
        detector_row_pitch_m=float(detector_row_pitch_m),
    )
