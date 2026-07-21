"""Continuous detector-density pullback into canonical scattering angles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.geometry.angles import (
    AngleFrame,
    DetectorCoordinateAreaMeasure,
    DetectorCoordinates,
    angles_to_detector_coordinate_area_measure,
)
from rasim_next.geometry.instrument import CompiledInstrument


class _DetectorCoordinateDensityValues(Protocol):
    density_A2_per_px2: NDArray[np.float64]
    caustic: NDArray[np.bool_]


class _DetectorCoordinateDensityFunction(Protocol):
    @property
    def instrument(self) -> CompiledInstrument: ...

    @property
    def measure_id(self) -> str: ...

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
    ) -> _DetectorCoordinateDensityValues: ...


def _broadcast_angles(
    two_theta_rad: ArrayLike,
    phi_rad: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        two_theta, phi = np.broadcast_arrays(
            np.asarray(two_theta_rad, dtype=np.float64),
            np.asarray(phi_rad, dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("two_theta_rad and phi_rad must be real numeric arrays") from error
    if not np.all(np.isfinite(two_theta)) or not np.all(np.isfinite(phi)):
        raise ValueError("two_theta_rad and phi_rad must contain only finite values")
    if np.any((two_theta < 0.0) | (two_theta > np.pi)):
        raise ValueError("two_theta_rad must lie in the closed interval [0, pi]")
    canonical_phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
    return (
        np.array(two_theta, dtype=np.float64, copy=True, order="C"),
        np.array(canonical_phi, dtype=np.float64, copy=True, order="C"),
    )


def _readonly_nonnegative(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    *,
    allow_infinite: bool,
) -> NDArray[np.float64]:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or np.any(np.isnan(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must have shape {shape}, be nonnegative, and contain no NaN")
    if not allow_infinite and np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ContinuousNormalizedAngleValues:
    """Pointwise angular signal/support densities and their normalized intensity."""

    two_theta_rad: NDArray[np.float64]
    phi_rad: NDArray[np.float64]
    detector_coordinate_area_measure: DetectorCoordinateAreaMeasure
    signal_density_A2_per_rad2: NDArray[np.float64]
    normalization_density_px2_per_rad2: NDArray[np.float64]
    intensity_A2_per_px2: NDArray[np.float64]
    caustic: NDArray[np.bool_]
    valid: NDArray[np.bool_]

    signal_measure_id: ClassVar[str] = "raw_detector_angle_signal_density_A2_per_rad2.v1"
    normalization_measure_id: ClassVar[str] = "detector_area_density_px2_per_rad2.v1"
    intensity_measure_id: ClassVar[str] = "raw_detector_area_normalized_intensity_A2_per_px2.v1"
    finite_bin_reduction: ClassVar[str] = "integrate_signal_and_normalization_then_divide.v1"
    detector_solid_angle_applied: ClassVar[bool] = False

    def __post_init__(self) -> None:
        two_theta, phi = _broadcast_angles(self.two_theta_rad, self.phi_rad)
        shape = two_theta.shape
        if not isinstance(self.detector_coordinate_area_measure, DetectorCoordinateAreaMeasure):
            raise TypeError(
                "detector_coordinate_area_measure must be DetectorCoordinateAreaMeasure"
            )
        detector_coordinates = self.detector_coordinate_area_measure.coordinates
        if detector_coordinates.column_px.shape != shape:
            raise ValueError("detector coordinates must preserve the angle-array shape")
        signal = _readonly_nonnegative(
            self.signal_density_A2_per_rad2,
            shape,
            "signal_density_A2_per_rad2",
            allow_infinite=True,
        )
        normalization = _readonly_nonnegative(
            self.normalization_density_px2_per_rad2,
            shape,
            "normalization_density_px2_per_rad2",
            allow_infinite=False,
        )
        intensity = _readonly_nonnegative(
            self.intensity_A2_per_px2,
            shape,
            "intensity_A2_per_px2",
            allow_infinite=True,
        )
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != shape or valid.shape != shape:
            raise ValueError("caustic and valid must preserve the angle-array shape")
        expected_normalization = (
            self.detector_coordinate_area_measure.detector_area_jacobian_px2_per_rad2
        )
        expected_valid = detector_coordinates.valid & (normalization > 0.0)
        if not np.array_equal(normalization, expected_normalization):
            raise ValueError("normalization must equal the detector-area coordinate Jacobian")
        if not np.array_equal(valid, expected_valid):
            raise ValueError("valid must identify positive normalized detector support")
        invalid = ~valid
        if (
            np.any(signal[invalid] != 0.0)
            or np.any(normalization[invalid] != 0.0)
            or np.any(intensity[invalid] != 0.0)
            or np.any(caustic[invalid])
        ):
            raise ValueError("invalid angle coordinates must have zero values and no caustic")
        infinite = np.isinf(signal) | np.isinf(intensity)
        if np.any(infinite & ~caustic):
            raise ValueError("infinite angle intensity must be identified as a caustic")
        finite = valid & ~infinite
        if not np.allclose(
            signal[finite],
            intensity[finite] * normalization[finite],
            rtol=8.0 * np.finfo(np.float64).eps,
            atol=0.0,
        ):
            raise ValueError("signal must equal normalized intensity times normalization density")
        for array in (two_theta, phi, caustic, valid):
            array.setflags(write=False)
        object.__setattr__(self, "two_theta_rad", two_theta)
        object.__setattr__(self, "phi_rad", phi)
        object.__setattr__(self, "signal_density_A2_per_rad2", signal)
        object.__setattr__(self, "normalization_density_px2_per_rad2", normalization)
        object.__setattr__(self, "intensity_A2_per_px2", intensity)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "valid", valid)

    @property
    def detector_coordinates(self) -> DetectorCoordinates:
        return self.detector_coordinate_area_measure.coordinates


@dataclass(frozen=True, slots=True)
class ContinuousNormalizedAngleFunction:
    """Pull back one pose-bound detector function without rasterizing it first."""

    detector_function: _DetectorCoordinateDensityFunction = field(repr=False)
    angle_frame: AngleFrame

    def __post_init__(self) -> None:
        if not isinstance(self.angle_frame, AngleFrame):
            raise TypeError("angle_frame must be an AngleFrame")
        if not isinstance(getattr(self.detector_function, "instrument", None), CompiledInstrument):
            raise TypeError("detector_function must expose its pose-bound CompiledInstrument")
        if (
            getattr(self.detector_function, "measure_id", None)
            != "raw_detector_coordinate_density_A2_per_px2.v1"
        ):
            raise ValueError("detector_function must expose the raw detector-coordinate density")
        if not callable(getattr(self.detector_function, "evaluate_detector_coordinates", None)):
            raise TypeError("detector_function must evaluate continuous detector coordinates")

    @property
    def instrument(self) -> CompiledInstrument:
        return self.detector_function.instrument

    def evaluate(
        self,
        two_theta_rad: ArrayLike,
        phi_rad: ArrayLike,
    ) -> ContinuousNormalizedAngleValues:
        """Evaluate continuous ``S``, ``N``, and ``I=S/N`` at canonical angles."""

        two_theta, phi = _broadcast_angles(two_theta_rad, phi_rad)
        coordinate_measure = angles_to_detector_coordinate_area_measure(
            two_theta,
            phi,
            instrument=self.instrument,
            angle_frame=self.angle_frame,
        )
        coordinates = coordinate_measure.coordinates
        normalization = coordinate_measure.detector_area_jacobian_px2_per_rad2
        valid = coordinates.valid & (normalization > 0.0)
        signal = np.zeros(two_theta.shape, dtype=np.float64)
        intensity = np.zeros(two_theta.shape, dtype=np.float64)
        caustic = np.zeros(two_theta.shape, dtype=np.bool_)
        flat_valid = np.flatnonzero(valid)
        if flat_valid.size:
            detector_values = self.detector_function.evaluate_detector_coordinates(
                coordinates.column_px.ravel()[flat_valid],
                coordinates.row_px.ravel()[flat_valid],
            )
            detector_density = detector_values.density_A2_per_px2.ravel()
            detector_caustic = np.any(detector_values.caustic, axis=-1).ravel()
            flat_normalization = normalization.ravel()[flat_valid]
            flat_signal = signal.ravel()
            flat_intensity = intensity.ravel()
            flat_caustic = caustic.ravel()
            flat_signal[flat_valid] = detector_density * flat_normalization
            flat_intensity[flat_valid] = detector_density
            flat_caustic[flat_valid] = detector_caustic
        return ContinuousNormalizedAngleValues(
            two_theta_rad=two_theta,
            phi_rad=phi,
            detector_coordinate_area_measure=coordinate_measure,
            signal_density_A2_per_rad2=signal,
            normalization_density_px2_per_rad2=normalization,
            intensity_A2_per_px2=intensity,
            caustic=caustic,
            valid=valid,
        )

    def __call__(
        self,
        two_theta_rad: ArrayLike,
        phi_rad: ArrayLike,
    ) -> ContinuousNormalizedAngleValues:
        return self.evaluate(two_theta_rad, phi_rad)
