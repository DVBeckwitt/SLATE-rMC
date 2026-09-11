"""Conditional Gaussian beam position and its native-pixel probability integral."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.contracts import IncidentStateBatch
from rasim_next.geometry.instrument import CompiledInstrument

if TYPE_CHECKING:
    from rasim_next.pipeline.configured_simulation import SourceConfiguration


@dataclass(frozen=True, slots=True)
class ConditionalBeamPosition:
    """Residual source-position factor, bound to conditional-mean source rows.

    The two columns map independent standard normals to LAB metres. Angular,
    wavelength and mosaic probabilities remain owned by the existing sampler.
    """

    factor_lab_m: NDArray[np.float64]
    source_revision: str

    def __post_init__(self) -> None:
        factor = np.array(self.factor_lab_m, dtype=np.float64, copy=True, order="C")
        if factor.shape != (3, 2) or not np.all(np.isfinite(factor)):
            raise ValueError("factor_lab_m must be finite with shape (3, 2)")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        factor.setflags(write=False)
        object.__setattr__(self, "factor_lab_m", factor)

    @classmethod
    def from_source(
        cls, source: SourceConfiguration, *, source_revision: str
    ) -> ConditionalBeamPosition:
        # A zero divergence width reveals no angular latent on that axis.
        return cls(
            _conditional_position_factor(
                source.transverse_axes_lab,
                source.spatial_sigma_m,
                source.divergence_sigma_rad,
                source.position_divergence_correlation,
            ),
            source_revision,
        )

    def compile_projection(
        self, states: IncidentStateBatch, instrument: CompiledInstrument
    ) -> NDArray[np.float64]:
        """Map source normals to detector column/row offsets and normal metres.

        This is the exact affine derivative of the two canonical plane
        intersections. The outgoing-direction part is completed per root below.
        Forward-flight Gaussian tails must each be below the explicit 8-sigma
        bound; unsupported clipping/absorption is never approximated silently.
        """
        if (
            states.source_revision != self.source_revision
            or states.source_sampling_model_id != "conditional_position_mean.v1"
        ):
            raise ValueError("beam position requires its own conditional-mean source rows")
        values = json.loads(states.source_parameter_provenance)["values"]
        declared_factor = _conditional_position_factor(
            [[float.fromhex(value) for value in axis] for axis in values["transverse_axes_lab"]],
            [float.fromhex(value) for value in values["spatial_sigma_m"]],
            [float.fromhex(value) for value in values["divergence_sigma_rad"]],
            [
                float.fromhex(value)
                for value in values.get("position_divergence_correlation", ["0x0.0p+0"] * 2)
            ],
        )
        if not np.array_equal(self.factor_lab_m, declared_factor):
            raise ValueError("residual beam position does not match the declared source law")
        if instrument.sample_support_model_id != "unbounded_plane.v1":
            raise ValueError("smooth beam position requires unbounded sample support; use sampled")
        if instrument.detector_path_linear_attenuation_m_inv != 0.0 or any(
            instrument.detector_path_linear_attenuation_m_inv_by_wavelength
        ):
            raise ValueError(
                "smooth beam position requires zero external-path absorption; use sampled"
            )
        normal = instrument.lab_from_sample.rotation[:, 2]
        directions = states.source_direction_lab
        denominator = directions @ normal
        incoming = np.abs(denominator) > 1e-14
        distance = np.zeros(len(directions))
        distance[incoming] = (
            (instrument.lab_from_sample.translation_m - states.source_origin_lab_m[incoming])
            @ normal
        ) / denominator[incoming]
        distance_sigma = np.zeros(len(directions))
        distance_sigma[incoming] = np.linalg.norm(normal @ self.factor_lab_m) / np.abs(
            denominator[incoming]
        )
        if np.any(incoming & (distance_sigma > 0.0) & (np.abs(distance) <= 8.0 * distance_sigma)):
            raise ValueError("beam crosses the forward sample-ray boundary; use sampled")
        active = states.valid
        footprint = np.zeros((len(directions), 3, 2))
        footprint[active] = self.factor_lab_m - directions[active, :, None] * (
            (normal @ self.factor_lab_m)[None, None, :] / denominator[active, None, None]
        )
        detector_factor = np.einsum(
            "ij,njk->nik", instrument.lab_from_detector.rotation.T, footprint
        )
        normal_distance = (
            instrument.lab_from_detector.translation_m - states.sample_intersection_lab_m
        ) @ instrument.lab_from_detector.rotation[:, 2]
        if np.any(
            active & (normal_distance <= 8.0 * np.linalg.norm(detector_factor[:, 2], axis=1))
        ):
            raise ValueError("beam crosses the forward detector-ray boundary; use sampled")
        detector_factor[:, 0] /= instrument.detector_column_pitch_m
        detector_factor[:, 1] /= instrument.detector_row_pitch_m
        return np.ascontiguousarray(detector_factor)


def _conditional_position_factor(axes, sigma, divergence_sigma, correlation):
    rho = np.where(np.asarray(divergence_sigma) > 0.0, correlation, 0.0)
    residual_sigma = np.asarray(sigma) * np.sqrt(1.0 - rho * rho)
    return np.asarray(axes).T * residual_sigma


def _projected_position_factor(column, row, origin_column, origin_row, origin_normal, factor):
    """Scalar geometry shared verbatim by the CPU and CUDA compilers."""
    column_scale = (column - origin_column) / origin_normal
    row_scale = (row - origin_row) / origin_normal
    return (
        factor[0, 0] + column_scale * factor[2, 0],
        factor[0, 1] + column_scale * factor[2, 1],
        factor[1, 0] + row_scale * factor[2, 0],
        factor[1, 1] + row_scale * factor[2, 1],
    )


def gaussian_pixel_probability(dc, dr, f00, f01, f10, f11):
    """Positive pixel-box quadrature of a possibly singular Gaussian.

    Arguments are pixel-center minus Gaussian mean and its 2x2 factor in px.
    Condition on the wider marginal; integrate the other axis with erf. Split
    at narrow conditional transitions so subpixel and nearly rank-one beams do
    not disappear between quadrature nodes. Four-point Gauss panels have width
    at most one standard deviation in either changing Gaussian coordinate.
    The primary marginal is truncated at six sigma (no renormalization).
    This plain scalar function is compiled unchanged for both processors.
    """
    sx = math.sqrt(f00 * f00 + f01 * f01)
    sy = math.sqrt(f10 * f10 + f11 * f11)
    if sx < sy:
        dc, dr = dr, dc
        sx, sy = sy, sx
        f00, f01, f10, f11 = f10, f11, f00, f01
    if sx == 0.0:
        return 1.0 if -0.5 < dc <= 0.5 and -0.5 < dr <= 0.5 else 0.0
    a = (dc - 0.5) / sx
    b = (dc + 0.5) / sx
    a = a if a > -6.0 else -6.0
    b = b if b < 6.0 else 6.0
    if a >= b:
        return 0.0
    root_two = math.sqrt(2.0)
    if sy == 0.0:
        if not -0.5 < dr <= 0.5:
            return 0.0
        return 0.5 * (math.erfc(a / root_two) - math.erfc(b / root_two))
    c = (dr - 0.5) / sy
    d = (dr + 0.5) / sy
    rho = (f00 / sx) * (f10 / sy) + (f01 / sx) * (f11 / sy)
    residual = abs((f00 / sx) * (f11 / sy) - (f01 / sx) * (f10 / sy))
    if rho < 0.0:
        c, d = -d, -c
        rho = -rho
    if residual == 0.0:
        a = a if a > c else c
        b = b if b < d else d
        return 0.5 * (math.erfc(a / root_two) - math.erfc(b / root_two)) if a < b else 0.0
    if rho == 0.0:
        return (
            0.25
            * (math.erfc(a / root_two) - math.erfc(b / root_two))
            * (math.erfc(c / root_two) - math.erfc(d / root_two))
        )
    result = 0.0
    left = a
    while left < b:
        # Stop at the entrance/exit of each conditional-CDF transition.
        right = b if b < left + 1.0 else left + 1.0
        for boundary in (
            c - 6.0 * residual,
            c + 6.0 * residual,
            d - 6.0 * residual,
            d + 6.0 * residual,
        ):
            crossing = boundary / rho
            if crossing > left:
                right = right if right < crossing else crossing
        midpoint = 0.5 * (left + right)
        transitioning = (
            abs(c - rho * midpoint) < 6.0 * residual or abs(d - rho * midpoint) < 6.0 * residual
        )
        if transitioning:
            step_end = left + residual / rho
            right = right if right < step_end else step_end
        if right <= left:
            break
        midpoint = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
        for node, weight in (
            (-0.8611363115940526, 0.3478548451374538),
            (-0.3399810435848563, 0.6521451548625461),
            (0.3399810435848563, 0.6521451548625461),
            (0.8611363115940526, 0.3478548451374538),
        ):
            x = midpoint + half_width * node
            low = (c - rho * x) / (root_two * residual)
            high = (d - rho * x) / (root_two * residual)
            probability = 0.5 * (math.erfc(low) - math.erfc(high))
            result += half_width * weight * math.exp(-0.5 * x * x) * probability
        left = right
    return result / math.sqrt(2.0 * math.pi)
