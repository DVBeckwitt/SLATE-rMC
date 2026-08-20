"""Continuous-incidence acquisition provenance and deterministic quadrature."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.core.contracts import canonical_revision_sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CONTINUOUS_INCIDENCE_ACQUISITION_SCHEMA_VERSION = "rasim-continuous-incidence-acquisition-v1"


@dataclass(frozen=True, slots=True)
class ContinuousIncidenceAcquisition:
    """Physical support and nonnumerical provenance for one continuous scan."""

    traversal_start_deg: float
    traversal_stop_deg: float
    exposure_time_s: float | None = None
    one_way_traversal_count: int | None = None
    complete_cycle_count: int | None = None
    scan_image_step_deg: float | None = None
    forward_reverse_speed_difference_percent: float | None = None
    endpoint_excess_exposure_percent: float | None = None
    interior_dwell_rms_deviation_percent: float | None = None
    motor_exposure: str = "uniform_normalized"
    nominal_start_deg: float = field(init=False)
    nominal_stop_deg: float = field(init=False)
    prediction_revision: str = field(init=False)
    provenance_revision: str = field(init=False)

    def __post_init__(self) -> None:
        traversal_start = _real_float(self.traversal_start_deg, "traversal_start_deg")
        traversal_stop = _real_float(self.traversal_stop_deg, "traversal_stop_deg")
        if traversal_start == traversal_stop:
            raise ValueError("continuous incidence acquisition requires nonzero support")
        nominal_start = min(traversal_start, traversal_stop)
        nominal_stop = max(traversal_start, traversal_stop)
        exposure_time = _optional_real_float(self.exposure_time_s, "exposure_time_s")
        scan_step = _optional_real_float(self.scan_image_step_deg, "scan_image_step_deg")
        speed_difference = _optional_real_float(
            self.forward_reverse_speed_difference_percent,
            "forward_reverse_speed_difference_percent",
        )
        endpoint_excess = _optional_real_float(
            self.endpoint_excess_exposure_percent,
            "endpoint_excess_exposure_percent",
        )
        dwell_rms = _optional_real_float(
            self.interior_dwell_rms_deviation_percent,
            "interior_dwell_rms_deviation_percent",
        )
        if (exposure_time is not None and exposure_time <= 0.0) or (
            scan_step is not None and scan_step <= 0.0
        ):
            raise ValueError("scan exposure time and metadata step must be positive")
        if any(
            value is not None and value < 0.0
            for value in (speed_difference, endpoint_excess, dwell_rms)
        ):
            raise ValueError("measured motor deviations must be nonnegative")
        traversal_count = _optional_positive_integer(
            self.one_way_traversal_count,
            "one_way_traversal_count",
        )
        cycle_count = _optional_positive_integer(
            self.complete_cycle_count,
            "complete_cycle_count",
        )
        if (traversal_count is None) != (cycle_count is None):
            raise ValueError("traversal and cycle counts must be reported together")
        if traversal_count is not None and traversal_count != 2 * cycle_count:
            raise ValueError("one-way traversal count must equal twice the complete cycle count")
        if self.motor_exposure != "uniform_normalized":
            raise ValueError("only uniform_normalized motor exposure is supported")

        prediction_revision = canonical_revision_sha256(
            ("definition_id", "continuous_incidence_acquisition_prediction.v1"),
            ("nominal_start_deg", nominal_start),
            ("nominal_stop_deg", nominal_stop),
            ("motor_exposure", self.motor_exposure),
        )
        provenance_revision = canonical_revision_sha256(
            ("definition_id", "continuous_incidence_acquisition_provenance.v1"),
            ("prediction_revision", prediction_revision),
            ("traversal_start_deg", traversal_start),
            ("traversal_stop_deg", traversal_stop),
            ("exposure_time_s", _reported_value(exposure_time)),
            ("one_way_traversal_count", _reported_value(traversal_count)),
            ("complete_cycle_count", _reported_value(cycle_count)),
            ("scan_image_step_deg", _reported_value(scan_step)),
            (
                "forward_reverse_speed_difference_percent",
                _reported_value(speed_difference),
            ),
            ("endpoint_excess_exposure_percent", _reported_value(endpoint_excess)),
            ("interior_dwell_rms_deviation_percent", _reported_value(dwell_rms)),
        )
        object.__setattr__(self, "traversal_start_deg", traversal_start)
        object.__setattr__(self, "traversal_stop_deg", traversal_stop)
        object.__setattr__(self, "exposure_time_s", exposure_time)
        object.__setattr__(self, "one_way_traversal_count", traversal_count)
        object.__setattr__(self, "complete_cycle_count", cycle_count)
        object.__setattr__(self, "scan_image_step_deg", scan_step)
        object.__setattr__(
            self,
            "forward_reverse_speed_difference_percent",
            speed_difference,
        )
        object.__setattr__(self, "endpoint_excess_exposure_percent", endpoint_excess)
        object.__setattr__(self, "interior_dwell_rms_deviation_percent", dwell_rms)
        object.__setattr__(self, "nominal_start_deg", nominal_start)
        object.__setattr__(self, "nominal_stop_deg", nominal_stop)
        object.__setattr__(self, "prediction_revision", prediction_revision)
        object.__setattr__(self, "provenance_revision", provenance_revision)

    def to_mapping(self) -> dict[str, object]:
        """Return the canonical acquisition record used by tracked configurations."""

        return {
            "schema_version": CONTINUOUS_INCIDENCE_ACQUISITION_SCHEMA_VERSION,
            "traversal_start_deg": self.traversal_start_deg,
            "traversal_stop_deg": self.traversal_stop_deg,
            "motor_exposure": self.motor_exposure,
            "exposure_time_s": self.exposure_time_s,
            "one_way_traversal_count": self.one_way_traversal_count,
            "complete_cycle_count": self.complete_cycle_count,
            "scan_image_step_deg": self.scan_image_step_deg,
            "forward_reverse_speed_difference_percent": (
                self.forward_reverse_speed_difference_percent
            ),
            "endpoint_excess_exposure_percent": self.endpoint_excess_exposure_percent,
            "interior_dwell_rms_deviation_percent": (self.interior_dwell_rms_deviation_percent),
        }

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> ContinuousIncidenceAcquisition:
        """Load one strict acquisition record while keeping metadata out of prediction."""

        if not isinstance(document, Mapping):
            raise TypeError("continuous incidence acquisition must be a mapping")
        expected = {
            "schema_version",
            "traversal_start_deg",
            "traversal_stop_deg",
            "motor_exposure",
            "exposure_time_s",
            "one_way_traversal_count",
            "complete_cycle_count",
            "scan_image_step_deg",
            "forward_reverse_speed_difference_percent",
            "endpoint_excess_exposure_percent",
            "interior_dwell_rms_deviation_percent",
        }
        if set(document) != expected:
            raise ValueError("continuous incidence acquisition fields are not exact")
        if document["schema_version"] != CONTINUOUS_INCIDENCE_ACQUISITION_SCHEMA_VERSION:
            raise ValueError("unsupported continuous incidence acquisition schema")
        return cls(
            traversal_start_deg=document["traversal_start_deg"],
            traversal_stop_deg=document["traversal_stop_deg"],
            motor_exposure=document["motor_exposure"],
            exposure_time_s=document["exposure_time_s"],
            one_way_traversal_count=document["one_way_traversal_count"],
            complete_cycle_count=document["complete_cycle_count"],
            scan_image_step_deg=document["scan_image_step_deg"],
            forward_reverse_speed_difference_percent=document[
                "forward_reverse_speed_difference_percent"
            ],
            endpoint_excess_exposure_percent=document["endpoint_excess_exposure_percent"],
            interior_dwell_rms_deviation_percent=document["interior_dwell_rms_deviation_percent"],
        )

    def uniform_panel_quadrature(
        self,
        *,
        incidence_angle_calibration_revision: str,
        effective_incidence_angle_rad: ArrayLike | None = None,
        panel_edges_deg: tuple[float, ...] | None = None,
        gauss_order: int = 16,
    ) -> IncidenceAngleQuadrature:
        """Compile the normalized uniform law without using step or cycle metadata."""

        if panel_edges_deg is None:
            edges = np.asarray(
                (self.nominal_start_deg, self.nominal_stop_deg),
                dtype=np.float64,
            )
        else:
            edges = _readonly_float_vector(panel_edges_deg, "panel_edges_deg")
        if (
            edges.size < 2
            or np.any(np.diff(edges) <= 0.0)
            or edges[0] != self.nominal_start_deg
            or edges[-1] != self.nominal_stop_deg
        ):
            raise ValueError("panel edges must partition the canonical acquisition support")
        commanded_parts: list[FloatArray] = []
        mass_parts: list[FloatArray] = []
        panel_parts: list[IntArray] = []
        width_deg = self.nominal_stop_deg - self.nominal_start_deg
        for panel, (start_deg, stop_deg) in enumerate(pairwise(edges)):
            node, local_mass = compile_uniform_incidence_angle_legendre_rule(
                math.radians(float(start_deg)),
                math.radians(float(stop_deg)),
                gauss_order=gauss_order,
            )
            commanded_parts.append(node)
            mass_parts.append(local_mass * ((float(stop_deg) - float(start_deg)) / width_deg))
            panel_parts.append(np.full(node.size, panel, dtype=np.int64))
        commanded = np.ascontiguousarray(np.concatenate(commanded_parts), dtype=np.float64)
        mass = np.ascontiguousarray(np.concatenate(mass_parts), dtype=np.float64)
        panel_index = np.ascontiguousarray(np.concatenate(panel_parts), dtype=np.int64)
        effective = (
            commanded
            if effective_incidence_angle_rad is None
            else _readonly_float_vector(
                effective_incidence_angle_rad,
                "effective_incidence_angle_rad",
            )
        )
        if effective.shape != commanded.shape:
            raise ValueError("effective incidence angles must align with commanded nodes")
        delta = effective - commanded
        tolerance = (
            16.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.max(np.abs(effective))),
            )
        )
        if np.max(np.abs(delta - delta[0])) > tolerance:
            raise ValueError("scan nodes must apply one common incidence calibration exactly once")
        common_delta = float(delta[0])
        return IncidenceAngleQuadrature(
            incidence_angle_rad=effective,
            commanded_incidence_angle_rad=commanded,
            exposure_probability_mass=mass,
            panel_index=panel_index,
            exposure_density_id="uniform_normalized_incidence_angle_density.v1",
            incidence_angle_calibration_revision=incidence_angle_calibration_revision,
            support_lower_incidence_angle_rad=(math.radians(self.nominal_start_deg) + common_delta),
            support_upper_incidence_angle_rad=(math.radians(self.nominal_stop_deg) + common_delta),
            acquisition_prediction_revision=self.prediction_revision,
        )


def _readonly_float_vector(value: ArrayLike, name: str) -> FloatArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real")
    result = np.array(supplied, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite nonempty one-dimensional array")
    result.setflags(write=False)
    return result


def _readonly_integer_vector(value: ArrayLike, name: str) -> IntArray:
    supplied = np.asarray(value)
    if supplied.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    limits = np.iinfo(np.int64)
    if np.any(supplied < limits.min) or np.any(supplied > limits.max):
        raise ValueError(f"{name} values are outside the int64 range")
    result = np.array(supplied, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1 or not result.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    result.setflags(write=False)
    return result


def _versioned_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9._-]*\.v[1-9][0-9]*", value) is None
    ):
        raise ValueError(f"{name} must be a versioned identifier ending in .vN")
    return value


def _real_float(value: object, name: str) -> float:
    supplied = np.asarray(value)
    if supplied.ndim != 0 or supplied.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real scalar")
    result = float(supplied)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_real_float(value: object | None, name: str) -> float | None:
    return None if value is None else _real_float(value, name)


def _optional_positive_integer(value: object | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _reported_value(value: float | int | None) -> float | int | str:
    return "unreported" if value is None else value


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def compile_uniform_incidence_angle_legendre_rule(
    lower_incidence_angle_rad: float,
    upper_incidence_angle_rad: float,
    *,
    gauss_order: int = 8,
    subdivision_count: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Return commanded-angle nodes and normalized masses for one fixed rule."""

    lower = _real_float(lower_incidence_angle_rad, "lower_incidence_angle_rad")
    upper = _real_float(upper_incidence_angle_rad, "upper_incidence_angle_rad")
    if lower >= upper:
        raise ValueError("incidence-angle bounds must be finite and increasing")
    if (
        isinstance(gauss_order, bool)
        or not isinstance(gauss_order, int)
        or gauss_order < 2
        or gauss_order % 2 != 0
    ):
        raise ValueError("gauss_order must be an even integer of at least two")
    if (
        isinstance(subdivision_count, bool)
        or not isinstance(subdivision_count, int)
        or subdivision_count < 1
    ):
        raise ValueError("subdivision_count must be a positive integer")

    canonical_node, canonical_weight = np.polynomial.legendre.leggauss(gauss_order)
    edge_fraction = np.linspace(0.0, 1.0, subdivision_count + 1, dtype=np.float64)
    edges = (1.0 - edge_fraction) * lower + edge_fraction * upper
    node_parts: list[FloatArray] = []
    mass_parts: list[FloatArray] = []
    for start, stop in pairwise(edges):
        node_parts.append(
            0.5 * (1.0 - canonical_node) * start + 0.5 * (1.0 + canonical_node) * stop
        )
        mass_parts.append(canonical_weight / (2.0 * subdivision_count))
    nodes = np.ascontiguousarray(np.concatenate(node_parts), dtype=np.float64)
    masses = np.ascontiguousarray(np.concatenate(mass_parts), dtype=np.float64)
    if (
        not np.all(np.isfinite(nodes))
        or np.any(np.diff(nodes) <= 0.0)
        or nodes[0] <= lower
        or nodes[-1] >= upper
    ):
        raise ValueError("incidence-angle interval is not resolvable by the requested rule")
    nodes.setflags(write=False)
    masses.setflags(write=False)
    return nodes, masses


@dataclass(frozen=True, slots=True)
class IncidenceAngleQuadrature:
    """Normalized deterministic quadrature for an incidence-angle exposure."""

    incidence_angle_rad: FloatArray
    exposure_probability_mass: FloatArray
    exposure_density_id: str
    incidence_angle_calibration_revision: str
    commanded_incidence_angle_rad: FloatArray | None = None
    panel_index: IntArray | None = None
    acquisition_prediction_revision: str | None = None
    support_lower_incidence_angle_rad: float | None = None
    support_upper_incidence_angle_rad: float | None = None
    quadrature_revision: str = field(init=False)

    def __post_init__(self) -> None:
        angle = _readonly_float_vector(self.incidence_angle_rad, "incidence_angle_rad")
        mass = _readonly_float_vector(
            self.exposure_probability_mass,
            "exposure_probability_mass",
        )
        if angle.shape != mass.shape:
            raise ValueError("incidence angles and exposure masses must have equal shape")
        if np.any(np.diff(angle) <= 0.0):
            raise ValueError("incidence angles must be strictly increasing")
        commanded = (
            angle
            if self.commanded_incidence_angle_rad is None
            else _readonly_float_vector(
                self.commanded_incidence_angle_rad,
                "commanded_incidence_angle_rad",
            )
        )
        if commanded.shape != angle.shape or np.any(np.diff(commanded) <= 0.0):
            raise ValueError("commanded incidence angles must align and increase")
        panel = (
            np.zeros(angle.size, dtype=np.int64)
            if self.panel_index is None
            else _readonly_integer_vector(self.panel_index, "panel_index")
        )
        if (
            panel.shape != angle.shape
            or np.any(panel < 0)
            or np.any(np.diff(panel) < 0)
            or not np.array_equal(np.unique(panel), np.arange(int(panel[-1]) + 1))
        ):
            raise ValueError("panel_index must be aligned, ordered, and contiguous from zero")
        panel.setflags(write=False)
        if np.any(mass <= 0.0):
            raise ValueError("exposure probability masses must be positive")
        normalization_error = abs(math.fsum(mass) - 1.0)
        if normalization_error > 32.0 * np.finfo(np.float64).eps:
            raise ValueError("exposure probability masses must already sum to one")
        exposure_density_id = _versioned_id(
            self.exposure_density_id,
            "exposure_density_id",
        )
        calibration_revision = _required_text(
            self.incidence_angle_calibration_revision,
            "incidence_angle_calibration_revision",
        )
        acquisition_revision = self.acquisition_prediction_revision
        if acquisition_revision is not None:
            acquisition_revision = _required_text(
                acquisition_revision,
                "acquisition_prediction_revision",
            )

        lower = self.support_lower_incidence_angle_rad
        upper = self.support_upper_incidence_angle_rad
        has_support = lower is not None or upper is not None
        if has_support:
            if lower is None or upper is None:
                raise ValueError("incidence-angle support requires both finite bounds")
            lower = _real_float(lower, "support_lower_incidence_angle_rad")
            upper = _real_float(upper, "support_upper_incidence_angle_rad")
            if lower >= upper or angle[0] <= lower or angle[-1] >= upper:
                raise ValueError("quadrature nodes must lie strictly inside their support")

        revision = canonical_revision_sha256(
            ("definition_id", "normalized_incidence_angle_quadrature.v1"),
            ("incidence_angle_rad", angle),
            ("commanded_incidence_angle_rad", commanded),
            ("exposure_probability_mass", mass),
            ("panel_index", panel),
            ("exposure_density_id", exposure_density_id),
            ("incidence_angle_calibration_revision", calibration_revision),
            ("acquisition_prediction_revision", acquisition_revision or "none"),
            ("has_continuous_support", int(has_support)),
            ("support_lower_incidence_angle_rad", 0.0 if lower is None else lower),
            ("support_upper_incidence_angle_rad", 0.0 if upper is None else upper),
        )
        object.__setattr__(self, "incidence_angle_rad", angle)
        object.__setattr__(self, "commanded_incidence_angle_rad", commanded)
        object.__setattr__(self, "exposure_probability_mass", mass)
        object.__setattr__(self, "panel_index", panel)
        object.__setattr__(self, "exposure_density_id", exposure_density_id)
        object.__setattr__(
            self,
            "incidence_angle_calibration_revision",
            calibration_revision,
        )
        object.__setattr__(self, "support_lower_incidence_angle_rad", lower)
        object.__setattr__(self, "support_upper_incidence_angle_rad", upper)
        object.__setattr__(self, "acquisition_prediction_revision", acquisition_revision)
        object.__setattr__(self, "quadrature_revision", revision)

    def integrate(self, value: ArrayLike) -> float | FloatArray:
        """Apply the normalized angle probability masses to one node-valued array."""

        supplied = np.asarray(value)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError("incidence-angle integrand must be real")
        array = np.asarray(supplied.real, dtype=np.float64)
        if array.shape[:1] != self.incidence_angle_rad.shape or not np.all(np.isfinite(array)):
            raise ValueError("incidence-angle integrand must align with quadrature nodes")
        if array.ndim == 1:
            return math.fsum(
                float(mass) * float(item)
                for mass, item in zip(self.exposure_probability_mass, array, strict=True)
            )
        result = np.empty(array.shape[1:], dtype=np.float64)
        for index in np.ndindex(result.shape):
            result[index] = math.fsum(
                float(mass) * float(item)
                for mass, item in zip(
                    self.exposure_probability_mass,
                    array[(slice(None), *index)],
                    strict=True,
                )
            )
        result.setflags(write=False)
        return result

    def require_acquisition(self, acquisition: ContinuousIncidenceAcquisition) -> None:
        """Fail closed when a cached quadrature is requested for another support."""

        if not isinstance(acquisition, ContinuousIncidenceAcquisition):
            raise TypeError("acquisition must be ContinuousIncidenceAcquisition")
        if self.acquisition_prediction_revision != acquisition.prediction_revision:
            raise ValueError("cached quadrature has a different acquisition support")

    @classmethod
    def uniform_legendre(
        cls,
        lower_incidence_angle_rad: float,
        upper_incidence_angle_rad: float,
        *,
        gauss_order: int = 8,
        subdivision_count: int = 1,
        incidence_angle_calibration_revision: str,
    ) -> IncidenceAngleQuadrature:
        """Compile a normalized uniform-exposure composite Gauss--Legendre rule."""

        lower = _real_float(lower_incidence_angle_rad, "lower_incidence_angle_rad")
        upper = _real_float(upper_incidence_angle_rad, "upper_incidence_angle_rad")
        nodes, masses = compile_uniform_incidence_angle_legendre_rule(
            lower,
            upper,
            gauss_order=gauss_order,
            subdivision_count=subdivision_count,
        )
        return cls(
            incidence_angle_rad=nodes,
            commanded_incidence_angle_rad=nodes,
            exposure_probability_mass=masses,
            panel_index=np.zeros(nodes.size, dtype=np.int64),
            exposure_density_id="uniform_normalized_incidence_angle_density.v1",
            support_lower_incidence_angle_rad=lower,
            support_upper_incidence_angle_rad=upper,
            incidence_angle_calibration_revision=(incidence_angle_calibration_revision),
        )


__all__ = [
    "CONTINUOUS_INCIDENCE_ACQUISITION_SCHEMA_VERSION",
    "ContinuousIncidenceAcquisition",
    "IncidenceAngleQuadrature",
    "compile_uniform_incidence_angle_legendre_rule",
]
