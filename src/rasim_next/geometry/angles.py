"""Bidirectional continuous detector and scattering-angle coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vector3
from rasim_next.geometry.detector import (
    _POSITION_TOL_M,
    _detector_coordinates_to_lab_points,
    _project_detector_rays,
)
from rasim_next.geometry.instrument import CompiledInstrument

_FRAME_TOL = 1e-12
_AZIMUTH_TOL = 32.0 * np.finfo(np.float64).eps


def _readonly_array(
    value: ArrayLike,
    dtype: np.dtype[np.generic] | type[np.generic] | str,
    name: str,
) -> NDArray[np.generic]:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _coordinate_arrays(
    first: ArrayLike,
    second: ArrayLike,
    first_name: str,
    second_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        first_array, second_array = np.broadcast_arrays(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{first_name} and {second_name} must be real numeric arrays") from error
    if not np.all(np.isfinite(first_array)) or not np.all(np.isfinite(second_array)):
        raise ValueError(f"{first_name} and {second_name} must contain only finite values")
    return (
        np.array(first_array, dtype=np.float64, copy=True, order="C"),
        np.array(second_array, dtype=np.float64, copy=True, order="C"),
    )


def _wrap_pi(angle_rad: ArrayLike) -> NDArray[np.float64]:
    angle = np.asarray(angle_rad, dtype=np.float64)
    return np.asarray((angle + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def _raw_chi_to_phi(raw_chi_rad: ArrayLike) -> NDArray[np.float64]:
    return _wrap_pi(-np.pi / 2.0 - np.asarray(raw_chi_rad, dtype=np.float64))


def _status_array(shape: tuple[int, ...], status: ValidityCode) -> NDArray[np.str_]:
    return np.full(shape, status, dtype="U32")


def _validate_statuses(status: NDArray[np.str_]) -> None:
    known = np.array([item.value for item in ValidityCode], dtype="U32")
    if not np.all(np.isin(status, known)):
        raise ValueError("status contains an unknown validity code")


@dataclass(frozen=True, slots=True)
class AngleFrame:
    """Fixed lab-frame origin and image-oriented basis for scattering angles."""

    origin_lab_m: NDArray[np.float64]
    row_down_lab: NDArray[np.float64]
    column_right_lab: NDArray[np.float64]
    direct_beam_lab: NDArray[np.float64]
    revision: str

    def __post_init__(self) -> None:
        origin = finite_vector3(self.origin_lab_m, "origin_lab_m")
        row_down = finite_vector3(self.row_down_lab, "row_down_lab")
        column_right = finite_vector3(self.column_right_lab, "column_right_lab")
        direct_beam = finite_vector3(self.direct_beam_lab, "direct_beam_lab")
        axes = (row_down, column_right, direct_beam)
        norms = tuple(float(np.linalg.norm(axis)) for axis in axes)
        if any(not np.isclose(norm, 1.0, rtol=0.0, atol=_FRAME_TOL) for norm in norms):
            raise ValueError("angle-frame basis vectors must have unit length")
        row_down = row_down / norms[0]
        column_right = column_right / norms[1]
        direct_beam = direct_beam / norms[2]
        if any(
            not np.isclose(value, 0.0, rtol=0.0, atol=_FRAME_TOL)
            for value in (
                np.dot(row_down, column_right),
                np.dot(row_down, direct_beam),
                np.dot(column_right, direct_beam),
            )
        ):
            raise ValueError("angle-frame basis vectors must be mutually orthogonal")
        if not np.allclose(
            np.cross(column_right, row_down),
            direct_beam,
            rtol=0.0,
            atol=_FRAME_TOL,
        ):
            raise ValueError("angle-frame axes must form a right-handed detector-image basis")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a nonempty string")
        for axis in (row_down, column_right, direct_beam):
            axis.setflags(write=False)
        object.__setattr__(self, "origin_lab_m", origin)
        object.__setattr__(self, "row_down_lab", row_down)
        object.__setattr__(self, "column_right_lab", column_right)
        object.__setattr__(self, "direct_beam_lab", direct_beam)


@dataclass(frozen=True, slots=True)
class DetectorAngles:
    """Shape-preserving angle coordinates derived from detector coordinates."""

    two_theta_rad: NDArray[np.float64]
    chi_raw_rad: NDArray[np.float64]
    phi_rad: NDArray[np.float64]
    valid: NDArray[np.bool_]
    azimuth_valid: NDArray[np.bool_]
    status: NDArray[np.str_]

    def __post_init__(self) -> None:
        fields = (
            ("two_theta_rad", np.float64),
            ("chi_raw_rad", np.float64),
            ("phi_rad", np.float64),
            ("valid", np.bool_),
            ("azimuth_valid", np.bool_),
            ("status", "U32"),
        )
        arrays = {name: _readonly_array(getattr(self, name), dtype, name) for name, dtype in fields}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("detector-angle arrays must have equal shapes")
        _validate_statuses(arrays["status"])
        if not np.array_equal(arrays["valid"], arrays["status"] == ValidityCode.VALID):
            raise ValueError("valid must agree with status == VALID")
        if np.any(arrays["azimuth_valid"] & ~arrays["valid"]):
            raise ValueError("azimuth_valid cannot be true for an invalid coordinate")
        two_theta = arrays["two_theta_rad"]
        chi_raw = arrays["chi_raw_rad"]
        phi = arrays["phi_rad"]
        if np.any((two_theta < 0.0) | (two_theta > np.pi)):
            raise ValueError("two_theta_rad must lie in [0, pi]")
        if np.any((chi_raw < -np.pi) | (chi_raw >= np.pi)):
            raise ValueError("chi_raw_rad must lie in [-pi, pi)")
        if np.any((phi < -np.pi) | (phi >= np.pi)):
            raise ValueError("phi_rad must lie in [-pi, pi)")
        azimuth_valid = arrays["azimuth_valid"]
        if not np.allclose(
            phi[azimuth_valid],
            _raw_chi_to_phi(chi_raw[azimuth_valid]),
            rtol=0.0,
            atol=8.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("phi_rad must be the wrapped display transform of chi_raw_rad")
        polar = arrays["valid"] & ~azimuth_valid
        if np.any(polar & (two_theta > _AZIMUTH_TOL) & ((np.pi - two_theta) > _AZIMUTH_TOL)):
            raise ValueError("a valid non-polar angle must have a valid azimuth")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class DetectorCoordinates:
    """Shape-preserving detector coordinates projected from scattering angles."""

    column_px: NDArray[np.float64]
    row_px: NDArray[np.float64]
    valid: NDArray[np.bool_]
    status: NDArray[np.str_]

    def __post_init__(self) -> None:
        fields = (
            ("column_px", np.float64),
            ("row_px", np.float64),
            ("valid", np.bool_),
            ("status", "U32"),
        )
        arrays = {name: _readonly_array(getattr(self, name), dtype, name) for name, dtype in fields}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("detector-coordinate arrays must have equal shapes")
        _validate_statuses(arrays["status"])
        if not np.array_equal(arrays["valid"], arrays["status"] == ValidityCode.VALID):
            raise ValueError("valid must agree with status == VALID")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


def _validate_context(instrument: CompiledInstrument, angle_frame: AngleFrame) -> None:
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")


def _origin_is_in_detector_plane(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> bool:
    detector_from_lab = instrument.lab_from_detector.inverse()
    origin_detector_m = detector_from_lab.apply_point(angle_frame.origin_lab_m)
    return bool(abs(float(origin_detector_m[2])) <= _POSITION_TOL_M)


def _inverse_support_tolerance_px(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> float:
    origin_detector_m = instrument.lab_from_detector.inverse().apply_point(angle_frame.origin_lab_m)
    rows, columns = instrument.detector_shape_rc
    condition_scale = max(
        1.0,
        float(rows),
        float(columns),
        abs(float(origin_detector_m[0])) / instrument.detector_column_pitch_m,
        abs(float(origin_detector_m[1])) / instrument.detector_row_pitch_m,
        abs(float(origin_detector_m[2]))
        / min(instrument.detector_column_pitch_m, instrument.detector_row_pitch_m),
    )
    return 64.0 * np.finfo(np.float64).eps * condition_scale


def detector_coordinates_to_angles(
    column_px: ArrayLike,
    row_px: ArrayLike,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> DetectorAngles:
    """Convert continuous ``(column_px, row_px)`` coordinates to radians."""

    _validate_context(instrument, angle_frame)
    columns, rows = _coordinate_arrays(column_px, row_px, "column_px", "row_px")
    shape = columns.shape
    status = _status_array(shape, ValidityCode.VALID)
    detector_rows, detector_columns = instrument.detector_shape_rc
    inside = (
        (columns >= -0.5)
        & (columns <= detector_columns - 0.5)
        & (rows >= -0.5)
        & (rows <= detector_rows - 0.5)
    )
    status[~inside] = ValidityCode.OUTSIDE_SUPPORT
    if _origin_is_in_detector_plane(instrument, angle_frame):
        status[...] = ValidityCode.NO_SOLUTION

    points_lab_m = _detector_coordinates_to_lab_points(columns, rows, instrument)
    displacement_lab_m = points_lab_m - angle_frame.origin_lab_m
    distance_m = np.linalg.norm(displacement_lab_m, axis=-1)
    active = status == ValidityCode.VALID
    no_direction = active & (distance_m <= _POSITION_TOL_M)
    status[no_direction] = ValidityCode.NO_SOLUTION

    t1 = displacement_lab_m @ angle_frame.row_down_lab
    t2 = displacement_lab_m @ angle_frame.column_right_lab
    t3 = displacement_lab_m @ angle_frame.direct_beam_lab
    transverse = np.hypot(t1, t2)
    numeric_failure = (status == ValidityCode.VALID) & ~(
        np.isfinite(t1) & np.isfinite(t2) & np.isfinite(t3) & np.isfinite(transverse)
    )
    status[numeric_failure] = ValidityCode.NUMERIC_FAILURE
    valid = status == ValidityCode.VALID

    two_theta = np.zeros(shape, dtype=np.float64)
    chi_raw = np.zeros(shape, dtype=np.float64)
    phi = np.zeros(shape, dtype=np.float64)
    two_theta[valid] = np.arctan2(transverse[valid], t3[valid])
    azimuth_valid = valid & (transverse > _AZIMUTH_TOL * distance_m)
    chi_raw[azimuth_valid] = _wrap_pi(np.arctan2(t1[azimuth_valid], t2[azimuth_valid]))
    phi[azimuth_valid] = _raw_chi_to_phi(chi_raw[azimuth_valid])
    return DetectorAngles(two_theta, chi_raw, phi, valid, azimuth_valid, status)


def angles_to_detector_coordinates(
    two_theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> DetectorCoordinates:
    """Project scattering angles to continuous detector coordinates."""

    _validate_context(instrument, angle_frame)
    two_theta, phi = _coordinate_arrays(
        two_theta_rad,
        phi_rad,
        "two_theta_rad",
        "phi_rad",
    )
    if np.any((two_theta < 0.0) | (two_theta > np.pi)):
        raise ValueError("two_theta_rad must lie in the closed interval [0, pi]")
    shape = two_theta.shape
    if _origin_is_in_detector_plane(instrument, angle_frame):
        status = _status_array(shape, ValidityCode.NO_SOLUTION)
        return DetectorCoordinates(
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.bool_),
            status,
        )

    phi = _wrap_pi(phi)
    sine = np.sin(two_theta)
    cosine = np.cos(two_theta)
    at_pole = (two_theta <= _AZIMUTH_TOL) | ((np.pi - two_theta) <= _AZIMUTH_TOL)
    sine[at_pole] = 0.0
    cosine[at_pole] = np.where(two_theta[at_pole] <= np.pi / 2.0, 1.0, -1.0)
    directions_lab = (
        (-sine * np.cos(phi))[..., None] * angle_frame.row_down_lab
        + (-sine * np.sin(phi))[..., None] * angle_frame.column_right_lab
        + cosine[..., None] * angle_frame.direct_beam_lab
    )
    direction_norm = np.linalg.norm(directions_lab, axis=-1, keepdims=True)
    directions_lab = directions_lab / direction_norm
    flat_directions = np.reshape(directions_lab, (-1, 3))
    flat_origins = np.broadcast_to(angle_frame.origin_lab_m, flat_directions.shape)
    projections = _project_detector_rays(
        flat_origins,
        flat_directions,
        instrument,
        support_tolerance_px=_inverse_support_tolerance_px(instrument, angle_frame),
    )
    status = np.reshape(projections.status, shape)
    valid = status == ValidityCode.VALID
    return DetectorCoordinates(
        np.reshape(projections.column_px, shape),
        np.reshape(projections.row_px, shape),
        valid,
        status,
    )
