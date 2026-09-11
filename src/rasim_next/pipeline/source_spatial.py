"""Conditional Gaussian beam position transported to a continuous detector plane.

These kernels integrate position only. Their weights must be physical integrated
scattering masses from a separately qualified direction/latent quadrature. They
do not supply structure factors, mosaic probabilities, optics or source weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from rasim_next.core.validity import ValidityCode
from rasim_next.geometry._vectors import finite_vectors3
from rasim_next.geometry.detector import _intersect_detector_plane
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection
from rasim_next.sampling.source import ConditionalSourceSamples


@numba.njit(nogil=True)
def _normal_interval_probability(low: float, high: float) -> float:
    root2 = math.sqrt(2.0)
    if low >= 0:
        return 0.5 * (math.erfc(low / root2) - math.erfc(high / root2))
    if high <= 0:
        return 0.5 * (math.erfc(-high / root2) - math.erfc(-low / root2))
    return 1.0 - 0.5 * (math.erfc(high / root2) + math.erfc(-low / root2))


@numba.njit(nogil=True)
def _rectangle_probability(
    mx, my, sx, beta, conditional_y, xlow, xhigh, ylow, yhigh, nodes, weights, radius
):
    """Normal-x integral of the conditional normal-y CDF; no pixel-center blur."""
    lo = max((xlow - mx) / sx, -radius)
    hi = min((xhigh - mx) / sx, radius)
    if hi <= lo:
        return 0.0
    slope = beta * sx
    if slope == 0.0:
        return _normal_interval_probability(lo, hi) * _normal_interval_probability(
            (ylow - my) / conditional_y, (yhigh - my) / conditional_y
        )
    mean_a, mean_b = my + slope * lo, my + slope * hi
    mean_low, mean_high = min(mean_a, mean_b), max(mean_a, mean_b)
    if ylow <= mean_low - radius * conditional_y and yhigh >= mean_high + radius * conditional_y:
        return _normal_interval_probability(lo, hi)
    # Resolve both the marginal Gaussian and a narrow conditional-CDF transition.
    edges = [lo, hi]
    for j in range(1, math.ceil((hi - lo) / 0.5)):
        edges.append(lo + j * 0.5)
    if slope != 0:
        transition_width = conditional_y / abs(slope)
        for yedge in (ylow, yhigh):
            center = (yedge - my) / slope
            for offset in (-radius, -2.0, 0.0, 2.0, radius):
                edge = center + offset * transition_width
                if lo < edge < hi:
                    edges.append(edge)
    edges.sort()
    total = 0.0
    for j in range(len(edges) - 1):
        mean_a = my + slope * edges[j]
        mean_b = my + slope * edges[j + 1]
        mean_low, mean_high = min(mean_a, mean_b), max(mean_a, mean_b)
        if (
            ylow <= mean_low - radius * conditional_y
            and yhigh >= mean_high + radius * conditional_y
        ):
            total += _normal_interval_probability(edges[j], edges[j + 1])
            continue
        if yhigh <= mean_low - radius * conditional_y or ylow >= mean_high + radius * conditional_y:
            continue
        midpoint = 0.5 * (edges[j] + edges[j + 1])
        half = 0.5 * (edges[j + 1] - edges[j])
        for k in range(len(nodes)):
            x = midpoint + half * nodes[k]
            ym = my + slope * x
            probability = _normal_interval_probability(
                (ylow - ym) / conditional_y, (yhigh - ym) / conditional_y
            )
            total += (
                half * weights[k] * math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi) * probability
            )
    return total


def _integration_rule(order: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError("quadrature_order must be an integer of at least two")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("gaussian_tail_radius must be finite and positive")
    return np.polynomial.legendre.leggauss(order)


@numba.njit(nogil=True)
def _correlation_angle_coefficients(slope, conditional_y, nodes, weights):
    rho = slope / math.hypot(slope, conditional_y)
    if abs(rho) >= 1.0 or slope == 0.0:
        return np.empty((3, 0))
    end = math.asin(abs(rho))
    edges = [0.0]
    cosine = 0.5
    while math.acos(cosine) < end:
        edges.append(math.acos(cosine))
        cosine *= 0.5
    edges.append(end)
    sign = 1.0 if rho > 0 else -1.0
    coefficients = np.empty((3, (len(edges) - 1) * len(nodes)))
    for panel in range(len(edges) - 1):
        half = 0.5 * (edges[panel + 1] - edges[panel])
        midpoint = 0.5 * (edges[panel + 1] + edges[panel])
        for i in range(len(nodes)):
            angle = sign * (midpoint + half * nodes[i])
            j = panel * len(nodes) + i
            coefficients[0, j] = 0.5 / math.cos(angle) ** 2
            coefficients[1, j] = math.sin(angle)
            coefficients[2, j] = sign * half * weights[i] / (2.0 * math.pi)
    return coefficients


@numba.njit(nogil=True)
def _correlated_rectangle_probability(
    mx,
    my,
    sx,
    beta,
    conditional_y,
    xlow,
    xhigh,
    ylow,
    yhigh,
    nodes,
    weights,
    radius,
    angle_coefficients,
):
    """Plackett angle integral, with conditional-CDF quadrature near degeneracy.

    The rho derivative of the bivariate normal CDF becomes a smooth integral
    under rho=sin(theta). Composite panels resolve its endpoint near unit
    correlation. Rectangle differencing retains the clipped X interval of the
    conditional integral; cancellation uses the independently conditioned form.
    """
    if angle_coefficients.shape[1] == 0:
        return _rectangle_probability(
            mx,
            my,
            sx,
            beta,
            conditional_y,
            xlow,
            xhigh,
            ylow,
            yhigh,
            nodes,
            weights,
            radius,
        )
    lo, hi = max((xlow - mx) / sx, -radius), min((xhigh - mx) / sx, radius)
    if hi <= lo:
        return 0.0
    slope = beta * sx
    mean_a, mean_b = my + slope * lo, my + slope * hi
    if (
        ylow <= min(mean_a, mean_b) - radius * conditional_y
        and yhigh >= max(mean_a, mean_b) + radius * conditional_y
    ):
        return _normal_interval_probability(lo, hi)
    sy = math.hypot(slope, conditional_y)
    yl, yh = (ylow - my) / sy, (yhigh - my) / sy
    base = _normal_interval_probability(lo, hi) * _normal_interval_probability(yl, yh)
    correction = 0.0
    for i in range(angle_coefficients.shape[1]):
        a, sine, w = angle_coefficients[:, i]
        # Completing the square avoids cancellation near unit correlation.
        correction += w * (
            math.exp(-a * (hi - sine * yh) ** 2 - 0.5 * yh * yh)
            - math.exp(-a * (lo - sine * yh) ** 2 - 0.5 * yh * yh)
            - math.exp(-a * (hi - sine * yl) ** 2 - 0.5 * yl * yl)
            + math.exp(-a * (lo - sine * yl) ** 2 - 0.5 * yl * yl)
        )
    value = base + correction
    if value < 0.0 or value < 1e-12 * (base + abs(correction)):
        return _rectangle_probability(
            mx,
            my,
            sx,
            beta,
            conditional_y,
            xlow,
            xhigh,
            ylow,
            yhigh,
            nodes,
            weights,
            radius,
        )
    return value


@numba.njit(nogil=True)
def _project_gaussian_regions(
    mean,
    factor,
    ptr,
    low,
    high,
    prefix,
    membership_ptr,
    owner,
    run_weight,
    nobs,
    nodes,
    weights,
    angle_nodes,
    angle_weights,
    radius,
):
    out_ptr = [0]
    out_owner = [np.int64(0)]
    out_weight = [0.0]
    out_owner.pop()
    out_weight.pop()
    sums = np.zeros(nobs)
    seen = np.zeros(nobs, np.int64)
    for i in range(len(mean)):
        stamp = i + 1
        touched = [np.int64(0)]
        touched.pop()
        mx, my = mean[i]
        f = factor[i]
        sx = math.sqrt(f[0, 0] ** 2 + f[0, 1] ** 2)
        beta = (f[0, 0] * f[1, 0] + f[0, 1] * f[1, 1]) / (sx * sx)
        conditional_y = abs(f[0, 0] * f[1, 1] - f[0, 1] * f[1, 0]) / sx
        angle_coefficients = _correlation_angle_coefficients(
            beta * sx, conditional_y, angle_nodes, angle_weights
        )
        clow = max(0, math.ceil(mx - radius * sx - 0.5))
        chigh = min(len(ptr) - 2, math.floor(mx + radius * sx + 0.5))
        for c in range(clow, chigh + 1):
            ym = my + beta * (c - mx)
            yr = 0.5 * abs(beta) + radius * conditional_y
            a, b = ptr[c], ptr[c + 1]
            start = a + np.searchsorted(prefix[a:b], ym - yr)
            stop = a + np.searchsorted(low[a:b], ym + yr, side="right")
            for j in range(start, stop):
                if high[j] < ym - yr:
                    continue
                value = _correlated_rectangle_probability(
                    mx,
                    my,
                    sx,
                    beta,
                    conditional_y,
                    c - 0.5,
                    c + 0.5,
                    low[j],
                    high[j],
                    nodes,
                    weights,
                    radius,
                    angle_coefficients,
                )
                if value > 0:
                    for member in range(membership_ptr[j], membership_ptr[j + 1]):
                        o = owner[member]
                        if seen[o] != stamp:
                            seen[o] = stamp
                            sums[o] = 0.0
                            touched.append(o)
                        sums[o] += value * run_weight[member]
        for o in touched:
            out_owner.append(o)
            out_weight.append(sums[o])
        out_ptr.append(len(out_owner))
    return np.asarray(out_ptr), np.asarray(out_owner), np.asarray(out_weight)


@dataclass(frozen=True, slots=True)
class NativeSpatialRegionProjection:
    """Gaussian box probabilities over frozen native region weights, without a raster.

    The native projector supplies each region's piecewise constant pixel weights.
    Contiguous equal-weight row runs share one rectangle integral. Identical
    rectangles are integrated once and distributed to their observations with
    the original weights; no surviving-mass normalization occurs.
    """

    projection: NativePixelRegionProjection
    _runs: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = self.projection
        if not len(p.observation_row):
            runs = (
                np.zeros(p.detector_shape_rc[1] + 1, dtype=np.int64),
                np.empty(0),
                np.empty(0),
                np.empty(0),
                np.zeros(1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0),
            )
            for a in runs:
                a.setflags(write=False)
            object.__setattr__(self, "_runs", runs)
            return
        rows, columns = np.divmod(p.flat_pixel_index[p.pixel_column_index], p.detector_shape_rc[1])
        owner, weight = p.observation_row, p.detector_area_weight_px2
        order = np.lexsort((rows, columns, owner))
        rows, columns, owner, weight = (a[order] for a in (rows, columns, owner, weight))
        starts = np.r_[
            0,
            1
            + np.flatnonzero(
                (np.diff(owner) != 0)
                | (np.diff(columns) != 0)
                | (np.diff(rows) != 1)
                | (weight[1:] != weight[:-1])
            ),
        ]
        ends = np.r_[starts[1:] - 1, len(rows) - 1]
        low, high = rows[starts] - 0.5, rows[ends] + 0.5
        columns, owner, weight = columns[starts], owner[starts], weight[starts]
        rectangles, membership = np.unique(
            np.column_stack((columns, low, high)), axis=0, return_inverse=True
        )
        order = np.argsort(membership, kind="stable")
        owner, weight = owner[order], weight[order]
        membership_ptr = np.searchsorted(membership[order], np.arange(len(rectangles) + 1))
        columns, low, high = rectangles.T
        ptr = np.searchsorted(columns, np.arange(p.detector_shape_rc[1] + 1))
        prefix = high.copy()
        for c in range(p.detector_shape_rc[1]):
            prefix[ptr[c] : ptr[c + 1]] = np.maximum.accumulate(high[ptr[c] : ptr[c + 1]])
        runs = (ptr, low, high, prefix, membership_ptr, owner, weight)
        for a in runs:
            a.setflags(write=False)
        object.__setattr__(self, "_runs", runs)

    def probabilities(
        self, kernels: DetectorSpatialKernels, *, quadrature_order: int, gaussian_tail_radius: float
    ) -> csr_matrix:
        """Return [kernel,region] probability; Gaussian approximation bound ≤6Φ(-radius).

        This includes 4Φ projection-tail truncation and 2Φ interior-CDF shortcuts.
        Multiply that bound by the maximum summed region weight for overlapping
        or weighted regions. It excludes reference-node quadrature error and the
        separately reported backward-flight bound. Increase quadrature order to
        check the conditional-CDF integral independently of tail truncation.
        """
        nodes, weights = _integration_rule(quadrature_order, gaussian_tail_radius)
        angle_nodes, angle_weights = _integration_rule(
            max(16, quadrature_order), gaussian_tail_radius
        )
        ptr, owner, mass = _project_gaussian_regions(
            kernels.mean_px,
            kernels.factor_px,
            *self._runs,
            self.projection.observation_count,
            nodes,
            weights,
            angle_nodes,
            angle_weights,
            float(gaussian_tail_radius),
        )
        if np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise FloatingPointError("invalid integrated spatial probability")
        return csr_matrix(
            (mass, owner, ptr), shape=(len(kernels.mean_px), self.projection.observation_count)
        )


@numba.njit(nogil=True)
def _deposit_gaussian_pixels(
    mean, factor, mass, shape, nodes, weights, angle_nodes, angle_weights, radius
):
    image = np.zeros(shape)
    for i in range(len(mean)):
        if mass[i] == 0:
            continue
        mx, my = mean[i]
        f = factor[i]
        sx = math.sqrt(f[0, 0] ** 2 + f[0, 1] ** 2)
        beta = (f[0, 0] * f[1, 0] + f[0, 1] * f[1, 1]) / (sx * sx)
        conditional_y = abs(f[0, 0] * f[1, 1] - f[0, 1] * f[1, 0]) / sx
        angle_coefficients = _correlation_angle_coefficients(
            beta * sx, conditional_y, angle_nodes, angle_weights
        )
        for c in range(
            max(0, math.ceil(mx - radius * sx - 0.5)),
            min(shape[1] - 1, math.floor(mx + radius * sx + 0.5)) + 1,
        ):
            ym = my + beta * (c - mx)
            yr = 0.5 * abs(beta) + radius * conditional_y
            for r in range(
                max(0, math.ceil(ym - yr - 0.5)),
                min(shape[0] - 1, math.floor(ym + yr + 0.5)) + 1,
            ):
                image[r, c] += mass[i] * _correlated_rectangle_probability(
                    mx,
                    my,
                    sx,
                    beta,
                    conditional_y,
                    c - 0.5,
                    c + 0.5,
                    r - 0.5,
                    r + 0.5,
                    nodes,
                    weights,
                    radius,
                    angle_coefficients,
                )
    return image


@dataclass(frozen=True, slots=True)
class DetectorSpatialKernels:
    """Unit-mass Gaussian densities per native pixel area, on the unbounded plane.

    ``factor_px`` maps two independent standard normals to (column, row) offsets.
    It includes the spatial area Jacobian. No additional solid angle, determinant,
    or affine magnification factor should multiply the supplied integrated masses.
    Panel/ROI acceptance is applied to the returned density or sampled coordinates;
    off-panel kernel centers are deliberately retained. The reported probability
    bound covers omitted forward-ray truncation, not angular quadrature error.
    """

    mean_px: NDArray[np.float64]
    factor_px: NDArray[np.float64]
    backward_probability_bound: NDArray[np.float64]
    _inverse_factor: NDArray[np.float64] = field(init=False, repr=False)
    _normalization: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mean = np.array(self.mean_px, dtype=np.float64, copy=True)
        factor = np.array(self.factor_px, dtype=np.float64, copy=True)
        loss = np.array(self.backward_probability_bound, dtype=np.float64, copy=True)
        if mean.ndim != 2 or mean.shape[1] != 2 or not len(mean):
            raise ValueError("mean_px must have nonempty shape (N, 2)")
        if factor.shape != (len(mean), 2, 2) or loss.shape != (len(mean),):
            raise ValueError("factor_px and probability bounds must align with mean_px")
        if not all(np.all(np.isfinite(a)) for a in (mean, factor, loss)):
            raise ValueError("spatial kernel arrays must be finite")
        if np.any((loss < 0.0) | (loss > 1.0)):
            raise ValueError("backward_probability_bound must lie in [0, 1]")
        determinant = np.linalg.det(factor)
        scale = np.linalg.norm(factor[:, 0], axis=1) * np.linalg.norm(factor[:, 1], axis=1)
        if np.any(np.abs(determinant) <= 64.0 * np.finfo(float).eps * scale):
            raise ValueError("singular spatial kernel requires a lower-dimensional measure")
        inverse = np.linalg.inv(factor)
        normalization = 1.0 / (2.0 * np.pi * np.abs(determinant))
        for name, value in (
            ("mean_px", mean),
            ("factor_px", factor),
            ("backward_probability_bound", loss),
            ("_inverse_factor", inverse),
            ("_normalization", normalization),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def density_at(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        integrated_mass: ArrayLike,
    ) -> NDArray[np.float64]:
        """Sum kernels with raw integrated masses; output has mass units per px².

        Evaluation is blocked to bound memory; no radius cutoff or renormalization is
        used. This is a continuous function evaluation, not an image convolution.
        """
        columns, rows = np.broadcast_arrays(
            np.asarray(column_px, dtype=np.float64),
            np.asarray(row_px, dtype=np.float64),
        )
        points = np.column_stack((columns.ravel(), rows.ravel()))
        mass = np.asarray(integrated_mass, dtype=np.float64)
        if mass.shape != (len(self.mean_px),) or not np.all(np.isfinite(mass)):
            raise ValueError("integrated_mass must be a finite value per kernel")
        if np.any(mass < 0.0) or not np.all(np.isfinite(points)):
            raise ValueError("masses must be nonnegative and detector coordinates finite")
        result = np.zeros(len(points))
        for first in range(0, len(points), 256):
            selected = points[first : first + 256]
            for start in range(0, len(mass), 256):
                stop = start + 256
                delta = selected[:, None, :] - self.mean_px[None, start:stop, :]
                standardized = np.einsum(
                    "kij,pkj->pki",
                    self._inverse_factor[start:stop],
                    delta,
                )
                exponent = -0.5 * np.sum(standardized * standardized, axis=-1)
                result[first : first + len(selected)] += np.exp(exponent) @ (
                    mass[start:stop] * self._normalization[start:stop]
                )
        return result.reshape(columns.shape)

    def sample(
        self,
        count: int,
        *,
        integrated_mass: ArrayLike,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Sample the normalized mixture on the unbounded plane, in (column,row).

        The caller retains sum(integrated_mass) as the raw intensity. Do not normalize
        again after selecting detector/ROI hits: that would erase physical lost mass.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        mass = np.asarray(integrated_mass, dtype=np.float64)
        if (
            mass.shape != (len(self.mean_px),)
            or not np.all(np.isfinite(mass))
            or np.any(mass < 0.0)
            or not np.isfinite(mass.sum())
            or mass.sum() <= 0.0
        ):
            raise ValueError("sampling requires finite nonnegative masses with positive total")
        chosen = rng.choice(len(mass), size=count, p=mass / mass.sum())
        normal = rng.standard_normal((count, 2))
        return self.mean_px[chosen] + np.einsum("nij,nj->ni", self.factor_px[chosen], normal)

    def integrate_native_pixels(
        self,
        detector_shape_rc: tuple[int, int],
        *,
        integrated_mass: ArrayLike,
        quadrature_order: int,
        gaussian_tail_radius: float,
    ) -> NDArray[np.float64]:
        """Integrate the same continuous kernels into [row,column] pixel masses.

        Off-panel centers contribute through their tails. Omitted Gaussian mass
        and interior-CDF approximation error is at most sum(mass)*6*Phi(-gaussian_tail_radius), in addition to source
        backward-flight, outgoing-quadrature and conditional-CDF quadrature errors.
        Increase quadrature_order to check the last error separately. No image blur, new SF
        evaluation, pixel-center approximation or survivor renormalization occurs.
        """
        shape = tuple(detector_shape_rc)
        if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("detector_shape_rc requires two positive integers")
        mass = np.asarray(integrated_mass, dtype=np.float64)
        if mass.shape != (len(self.mean_px),) or np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError("integrated_mass must be a finite nonnegative value per kernel")
        nodes, weights = _integration_rule(quadrature_order, gaussian_tail_radius)
        angle_nodes, angle_weights = _integration_rule(
            max(16, quadrature_order), gaussian_tail_radius
        )
        result = _deposit_gaussian_pixels(
            self.mean_px,
            self.factor_px,
            mass,
            shape,
            nodes,
            weights,
            angle_nodes,
            angle_weights,
            float(gaussian_tail_radius),
        )
        if np.any(~np.isfinite(result)):
            raise FloatingPointError("integrated spatial mass is nonfinite")
        return result

    def sample_native_pixel_mass(
        self,
        count: int,
        detector_shape_rc: tuple[int, int],
        *,
        integrated_mass: ArrayLike,
        rng: np.random.Generator,
        quadrature_order: int,
        gaussian_tail_radius: float,
    ) -> NDArray[np.float64]:
        """Monte Carlo over physical kernel masses with source position integrated.

        Each chosen component contributes its pixel probabilities. The result
        estimates the same raw mass as integrate_native_pixels, normalized by
        attempted draws. It does not simulate acquisition counts or source positions.
        """
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        mass = np.asarray(integrated_mass, dtype=np.float64)
        if (
            mass.shape != (len(self.mean_px),)
            or np.any(~np.isfinite(mass))
            or np.any(mass < 0)
            or not np.isfinite(mass.sum())
            or mass.sum() <= 0
        ):
            raise ValueError("sampling requires finite nonnegative masses with positive total")
        selected = rng.choice(len(mass), size=count, p=mass / mass.sum())
        sampled_mass = np.bincount(selected, minlength=len(mass)) * (mass.sum() / count)
        return self.integrate_native_pixels(
            detector_shape_rc,
            integrated_mass=sampled_mass,
            quadrature_order=quadrature_order,
            gaussian_tail_radius=gaussian_tail_radius,
        )


def validate_conditional_spatial_support(
    source: ConditionalSourceSamples, instrument: CompiledInstrument
) -> None:
    """Check the declared spatial measure even when no outgoing event is reachable."""
    if not isinstance(source, ConditionalSourceSamples):
        raise TypeError("source must retain its conditional position law")
    if instrument.sample_support_model_id != "unbounded_plane.v1":
        raise ValueError("conditional kernels require unbounded sample support")
    if instrument.detector_path_linear_attenuation_m_inv != 0.0 or any(
        instrument.detector_path_linear_attenuation_m_inv_by_wavelength
    ):
        raise ValueError("conditional kernels require zero external-path absorption")
    if np.linalg.matrix_rank(source.conditional_origin_factor_lab_m) != 2:
        raise ValueError("singular source position requires a lower-dimensional measure")


def compile_conditional_spatial_kernels(
    *,
    instrument: CompiledInstrument,
    source: ConditionalSourceSamples,
    source_state_index: int,
    outgoing_direction_lab: ArrayLike,
    maximum_backward_probability: float,
) -> DetectorSpatialKernels:
    """Integrate Gaussian source position conditional on one angular latent draw.

    The source companion supplies the conditional mean and spatial covariance.
    Their correlation with the actual angular draw was applied once by sampling;
    this transport never reconstructs angular latents or adds the mean shift again.

    Current scope is an unbounded planar sample with zero external-path absorption.
    Gaussian tails can cross the sample/detector planes behind the ray. The caller
    declares an allowed probability error; a union bound is returned, never hidden
    or renormalized. Singular projections fail explicitly instead of inventing blur.
    """
    validate_conditional_spatial_support(source, instrument)
    tolerance = float(maximum_backward_probability)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("maximum_backward_probability must lie in [0, 1)")
    if type(source_state_index) is not int or not 0 <= source_state_index < len(
        source.mean_rays.wavelength_A
    ):
        raise ValueError("source_state_index must identify one conditional source row")
    source_mean = source.mean_rays.origin_lab_m[source_state_index]
    direction = source.mean_rays.direction_lab[source_state_index]
    source_factor = source.conditional_origin_factor_lab_m
    sample = _intersect_sample_rays(
        source_mean[None, :],
        direction[None, :],
        lab_from_sample=instrument.lab_from_sample,
        sample_from_lab=instrument.sample_from_lab,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    if sample.status[0] != ValidityCode.VALID:
        raise ValueError(f"invalid mean incident ray: {sample.status[0]}")
    sample_normal = instrument.lab_from_sample.rotation[:, 2]
    distance_factor = -(sample_normal @ source_factor) / (sample_normal @ direction)
    footprint_factor = source_factor + direction[:, None] * distance_factor
    incoming_sigma = np.linalg.norm(distance_factor)
    incoming_loss = (
        max(float(ndtr(-sample.ray_distance_m[0] / incoming_sigma)), np.finfo(float).tiny)
        if incoming_sigma > 0.0
        else 0.0
    )
    outgoing = finite_vectors3(outgoing_direction_lab, "outgoing_direction_lab")
    origins = np.broadcast_to(sample.point_lab_m, outgoing.shape)
    intersections = _intersect_detector_plane(origins, outgoing, instrument)
    if np.any(intersections.status != ValidityCode.VALID):
        raise ValueError("every mean outgoing ray must meet the forward detector plane")
    detector_factor = instrument.lab_from_detector.rotation.T @ footprint_factor
    directions = intersections.direction_detector
    outgoing_distance_factor = -detector_factor[2] / directions[:, 2, None]
    outgoing_sigma = np.linalg.norm(outgoing_distance_factor, axis=1)
    loss = np.zeros(len(outgoing))
    active = outgoing_sigma > 0.0
    # A Gaussian has nonzero tails even when its CDF underflows in float64.
    # Keep a conservative positive bound instead of silently claiming exact support.
    loss[active] = np.maximum(
        ndtr(-intersections.distance_m[active] / outgoing_sigma[active]),
        np.finfo(float).tiny,
    )
    loss = np.minimum(loss + incoming_loss, 1.0)
    if np.any(loss > tolerance):
        raise ValueError("backward Gaussian probability exceeds the declared budget")
    factor = (
        detector_factor[None, :2, :]
        + directions[:, :2, None] * outgoing_distance_factor[:, None, :]
    )
    factor /= np.array([instrument.detector_column_pitch_m, instrument.detector_row_pitch_m])[
        None, :, None
    ]
    return DetectorSpatialKernels(
        np.column_stack((intersections.column_px, intersections.row_px)),
        factor,
        loss,
    )
