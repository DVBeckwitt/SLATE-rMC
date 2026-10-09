"""Independent crystal-azimuth integration on the physical Ewald sphere.

Signed rod coordinates resolve the structure factor and remove the radial fold.
The resulting conditional position kernels are shared by fitting and rendering.
All arrays returned here are one quadrature batch, not a retained event raster.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from itertools import pairwise
from time import perf_counter

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import qmc

from painted_ewald import MosaicParameters, Rod
from painted_ewald.validation import proper_rotation, reciprocal_basis, reject_complex
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.core.scattering import polarization_model_code, scattering_polarization_weight
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.detector import (
    _detector_coordinates_to_lab_points,
    _intersect_detector_plane,
)
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.sample import _intersect_sample_rays
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.optics.attenuation import (
    incident_illuminated_path_weight,
    mode_decay_constant,
    scalar_optical_weight,
    uniform_depth_attenuation,
)
from rasim_next.optics.refraction import _solve_exit_mode_arrays
from rasim_next.pipeline._continuous_detector_kernel import local_m0_geometry, local_m0_phase_q_Ainv
from rasim_next.pipeline.continuous_detector import _incident_phase_shell_offset_Ainv2
from rasim_next.pipeline.source_spatial import (
    DetectorSpatialKernels,
    compile_conditional_spatial_kernels,
    conditional_spatial_angular_rate,
    validate_conditional_spatial_support,
)
from rasim_next.sampling.source import ConditionalSourceSamples


def _ewald_frame(ki: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = ki / np.linalg.norm(ki)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(axis @ reference) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    first = reference - (reference @ axis) * axis
    first /= np.linalg.norm(first)
    return axis, first, np.cross(axis, first)


def elastic_axial_cutoff_Ainv(*, ki_sample_Ainv: ArrayLike, radial_Ainv: float) -> float | None:
    """Return positive axial Ewald endpoint, or None when radius is unreachable."""
    ki = np.asarray(ki_sample_Ainv, dtype=np.float64)
    radial = float(radial_Ainv)
    if (
        ki.shape != (3,)
        or np.any(~np.isfinite(ki))
        or not np.isfinite(radial)
        or radial < 0
        or np.linalg.norm(ki) == 0
    ):
        raise ValueError("elastic cutoff requires a finite nonzero ki and nonnegative radius")
    diameter_squared = 4 * float(ki @ ki)
    remainder = diameter_squared - radial**2
    if remainder < 0:
        return None
    return float(np.sqrt(remainder))


def conditional_ewald_region_bounds(
    *,
    native_bounds_px: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    source_state_index: int,
    instrument: CompiledInstrument,
    material: MaterialOptics,
    source_latent_radius: float,
    local_m0: bool = False,
) -> NDArray[np.float64]:
    """Conservative [Qmin,Qmax,zeta_start,zeta_width] for native rectangles.

    Rectangles use (column_low,column_high,row_low,row_high), including pixel
    edges. Every source position in the independent latent square [-radius,radius]^2
    is enclosed. Outside probability is at most 4*Phi(-radius), before intensity
    weighting. Bounds use interval arithmetic, never SF or mosaic cutoffs.
    A reversed Q interval denotes an empty intersection of the conservative
    geometric and elastic-shell bounds. Such a channel has no supported event.
    """
    bounds = np.asarray(native_bounds_px, dtype=np.float64)
    radius = float(source_latent_radius)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 4
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[:, 2] >= bounds[:, 3])
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("finite ordered native rectangles and positive latent radius required")
    si = source_state_index
    signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    origins = (
        source.mean_rays.origin_lab_m[si]
        + radius * signs @ source.conditional_origin_factor_lab_m.T
    )
    footprint = _intersect_sample_rays(
        origins,
        np.broadcast_to(source.mean_rays.direction_lab[si], origins.shape),
        lab_from_sample=instrument.lab_from_sample,
        sample_from_lab=instrument.sample_from_lab,
        sample_support_model_id=instrument.sample_support_model_id,
        sample_width_m=instrument.sample_width_m,
        sample_length_m=instrument.sample_length_m,
    )
    if np.any(footprint.status != ValidityCode.VALID):
        raise ValueError("bounded source square must intersect the sample forward")
    corners_c = bounds[:, [0, 0, 1, 1]]
    corners_r = bounds[:, [2, 3, 2, 3]]
    detector = instrument.sample_from_lab.apply_point(
        _detector_coordinates_to_lab_points(corners_c.ravel(), corners_r.ravel(), instrument)
    ).reshape(-1, 4, 3)
    foot = instrument.sample_from_lab.apply_point(footprint.point_lab_m)
    low, high = detector.min(axis=1) - foot.max(axis=0), detector.max(axis=1) - foot.min(axis=0)
    nearest = np.where(low > 0, low, np.where(high < 0, high, 0.0))
    furthest = np.maximum(abs(low), abs(high))
    direction_low, direction_high = np.empty_like(low), np.empty_like(high)
    for i in range(3):
        other = [j for j in range(3) if j != i]
        small, large = (nearest[:, other] ** 2).sum(axis=1), (furthest[:, other] ** 2).sum(axis=1)
        dl = np.sqrt(low[:, i] ** 2 + np.where(low[:, i] >= 0, large, small))
        dh = np.sqrt(high[:, i] ** 2 + np.where(high[:, i] >= 0, small, large))
        direction_low[:, i] = np.divide(low[:, i], dl, out=np.full(len(low), -1.0), where=dl > 0)
        direction_high[:, i] = np.divide(high[:, i], dh, out=np.ones(len(low)), where=dh > 0)
    k0 = 2 * np.pi / incident.states.wavelength_A[si]
    kf_low, kf_high = k0 * direction_low, k0 * direction_high
    if local_m0:
        ki = k0 * incident.states.direction_sample[si]
    else:
        ki = incident.states.k_film_phase_sample_Ainv[si]
        offset = _incident_phase_shell_offset_Ainv2(incident, material, si)
        kf_low[:, 2] = np.sqrt(np.maximum(0, np.maximum(kf_low[:, 2], 0) ** 2 + offset))
        kf_high[:, 2] = np.sqrt(np.maximum(0, np.maximum(kf_high[:, 2], 0) ** 2 + offset))
    qlow, qhigh = kf_low - ki, kf_high - ki
    q_near = np.where(qlow > 0, qlow, np.where(qhigh < 0, qhigh, 0.0))
    q_far = np.maximum(abs(qlow), abs(qhigh))
    k2 = ki @ ki
    dotlow = np.minimum(kf_low * ki, kf_high * ki).sum(axis=1)
    dothigh = np.maximum(kf_low * ki, kf_high * ki).sum(axis=1)
    qmin = np.maximum(np.linalg.norm(q_near, axis=1), np.sqrt(np.maximum(0, 2 * k2 - 2 * dothigh)))
    qmax = np.minimum(np.linalg.norm(q_far, axis=1), np.sqrt(np.maximum(0, 2 * k2 - 2 * dotlow)))
    qmax = np.minimum(qmax, 2 * np.sqrt(k2))
    _, first, second = _ewald_frame(ki)
    axes = np.stack((first, second))
    xylo = np.minimum(kf_low[:, None, :] * axes, kf_high[:, None, :] * axes).sum(axis=2)
    xyhi = np.maximum(kf_low[:, None, :] * axes, kf_high[:, None, :] * axes).sum(axis=2)
    theta = np.mod(
        np.arctan2(
            xylo[:, 1:2] + (xyhi[:, 1:2] - xylo[:, 1:2]) * np.array([0, 1, 0, 1]),
            xylo[:, 0:1] + (xyhi[:, 0:1] - xylo[:, 0:1]) * np.array([0, 0, 1, 1]),
        ),
        2 * np.pi,
    )
    theta.sort(axis=1)
    gap = np.diff(np.column_stack((theta, theta[:, 0] + 2 * np.pi)), axis=1)
    missing = gap.argmax(axis=1)
    start = theta[np.arange(len(theta)), (missing + 1) % 4]
    width = 2 * np.pi - gap[np.arange(len(theta)), missing]
    wraps_origin = np.all((xylo <= 0) & (xyhi >= 0), axis=1)
    start[wraps_origin], width[wraps_origin] = 0, 2 * np.pi
    pad = 64 * np.finfo(float).eps
    return np.column_stack(
        (np.maximum(0, qmin - pad), qmax + pad, start - pad, np.minimum(2 * np.pi, width + 2 * pad))
    )


def native_angular_resolution_regions(
    *,
    native_bounds_px: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    source_state_index: int,
    instrument: CompiledInstrument,
    material: MaterialOptics,
    local_m0: bool,
    resolution_fraction: float,
    source_latent_radius: float,
    maximum_backward_probability: float,
) -> NDArray[np.float64]:
    """Return [Qlow,Qhigh,angle_start,angle_width,width_at_max_transverse] seeds.

    Corners and centers span every requested native rectangle. Their local rates
    set a deterministic proposal resolution, not a supremum or an error bound.
    Observable refinement remains mandatory. No grazing denominator is floored.
    """
    bounds = np.asarray(native_bounds_px, dtype=float)
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 4
        or not len(bounds)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
        or np.any(bounds[:, 2] >= bounds[:, 3])
        or not np.isfinite(resolution_fraction)
        or resolution_fraction <= 0
    ):
        raise ValueError(
            "native angular resolution requires complete rectangles and positive fraction"
        )
    # Subdivide geometry only: the observable and its integration union remain
    # unchanged. Whole-panel render requests need the same local resolution as ROIs.
    counts = np.maximum(1, np.ceil((bounds[:, [1, 3]] - bounds[:, [0, 2]]) / 64))
    if np.sum(np.prod(counts, axis=1)) > 65536:
        raise ValueError("native angular geometry exceeds the resolution-region budget")
    tiles = []
    for row_bounds, (nc, nr) in zip(bounds, counts.astype(int), strict=True):
        columns = np.linspace(row_bounds[0], row_bounds[1], nc + 1)
        rows = np.linspace(row_bounds[2], row_bounds[3], nr + 1)
        for c0, c1 in pairwise(columns):
            for r0, r1 in pairwise(rows):
                tiles.append((c0, c1, r0, r1))
    bounds = np.asarray(tiles)
    regions = conditional_ewald_region_bounds(
        native_bounds_px=bounds,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        instrument=instrument,
        material=material,
        source_latent_radius=source_latent_radius,
        local_m0=local_m0,
    )
    keep = regions[:, 0] <= regions[:, 1]
    bounds, regions = bounds[keep], regions[keep]
    if not len(bounds):
        return np.empty((0, 5))
    column = np.column_stack((bounds[:, [0, 0, 1, 1]], (bounds[:, 0] + bounds[:, 1]) / 2))
    row = np.column_stack((bounds[:, [2, 3, 2, 3]], (bounds[:, 2] + bounds[:, 3]) / 2))
    points = np.column_stack((column.ravel(), row.ravel()))
    si, states = source_state_index, incident.states
    k0 = 2 * np.pi / states.wavelength_A[si]
    ki = k0 * states.direction_sample[si] if local_m0 else states.k_film_phase_sample_Ainv[si]
    axis = ki / np.linalg.norm(ki)
    offset = 0.0 if local_m0 else _incident_phase_shell_offset_Ainv2(incident, material, si)
    rates = np.zeros(len(points))
    informative = np.zeros(len(points), dtype=bool)
    propagating = np.zeros(len(points), dtype=bool)
    for first in range(0, len(points), 4096):
        pixel = points[first : first + 4096]
        lab = _detector_coordinates_to_lab_points(pixel[:, 0], pixel[:, 1], instrument)
        direction_lab = lab - states.sample_intersection_lab_m[si]
        direction_lab /= np.linalg.norm(direction_lab, axis=1)[:, None]
        direction = instrument.sample_from_lab.apply_vector(direction_lab)
        keep = np.ones(len(direction), dtype=bool)
        if not local_m0:
            keep = (direction[:, 2] > 0) & (k0**2 * direction[:, 2] ** 2 + offset > 0)
            direction, direction_lab = direction[keep], direction_lab[keep]
        if not len(direction):
            continue
        propagating[first : first + len(pixel)][keep] = True
        wavevector = k0 * direction.copy()
        if not local_m0:
            wavevector[:, 2] = np.sqrt(wavevector[:, 2] ** 2 + offset)
        transverse = np.linalg.norm(np.cross(axis, wavevector), axis=1)
        tangent = np.cross(axis, wavevector) / k0
        if not local_m0:
            tangent[:, 2] *= wavevector[:, 2] / (k0 * direction[:, 2])
        rate = conditional_spatial_angular_rate(
            instrument=instrument,
            source=source,
            source_state_index=si,
            outgoing_direction_lab=direction_lab,
            outgoing_derivative_lab=instrument.lab_from_sample.apply_vector(tangent),
            maximum_backward_probability=maximum_backward_probability,
            source_latent_radius=source_latent_radius,
        )
        # Azimuthal motion scales linearly with the physical Ewald-circle radius.
        # Keep the sampled spatial Jacobian, then restore each axial node's radius.
        rates[first : first + len(pixel)][keep] = np.divide(
            rate, transverse, out=np.zeros_like(rate), where=transverse > 0
        )
        informative[first : first + len(pixel)][keep] = transverse > 0
    if np.any(propagating.reshape(-1, 5).any(axis=1) & ~informative.reshape(-1, 5).any(axis=1)):
        raise ValueError("native angular region has only degenerate geometry witnesses")
    k = np.linalg.norm(ki)
    qstar = np.clip(
        np.sqrt(2) * k, np.minimum(regions[:, 0], 2 * k), np.minimum(regions[:, 1], 2 * k)
    )
    ratio = qstar / (2 * k)
    maximum_transverse = qstar * np.sqrt((1 - ratio) * (1 + ratio))
    maximum_rate = rates.reshape(-1, 5).max(axis=1) * maximum_transverse
    width = np.full(len(bounds), 2 * np.pi)
    np.divide(resolution_fraction, maximum_rate, out=width, where=maximum_rate > 0)
    width = np.minimum(width, 2 * np.pi)
    if np.any(~np.isfinite(width)) or np.any(width <= 64 * np.finfo(float).eps):
        raise ValueError("native angular resolution exceeds floating-point precision")
    return np.column_stack((regions, width))


def fiber_ewald_coordinates(
    *,
    ki_sample_Ainv: ArrayLike,
    radial_Ainv: float,
    axial_offset_Ainv: float,
    b3_norm_Ainv: float,
    L: ArrayLike,
    ewald_azimuth_rad: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, kf, normal-cone angle and dL/dzeta measure coefficient.

    Inputs L and azimuth broadcast. The azimuth axis is the incident phase vector;
    zero azimuth lies towards sample +z (or +x at normal incidence). Unsupported
    |Q|>2|ki| and the direct Q=0 point have zero coefficient. Both signed L sheets
    remain explicit. Multiplying by SF and the normal-circle average gives raw
    Ewald mass per dL dzeta, before source and optical factors.
    """
    ki = np.asarray(ki_sample_Ainv, dtype=np.float64)
    r, offset, b3 = float(radial_Ainv), float(axial_offset_Ainv), float(b3_norm_Ainv)
    if ki.shape != (3,) or not np.all(np.isfinite(ki)) or np.linalg.norm(ki) == 0:
        raise ValueError("ki_sample_Ainv must be a finite nonzero vector")
    if not np.all(np.isfinite([r, offset, b3])) or r < 0 or b3 <= 0:
        raise ValueError("rod radius must be nonnegative and b3 norm positive")
    ell, azimuth = np.broadcast_arrays(
        np.asarray(L, dtype=np.float64), np.asarray(ewald_azimuth_rad, dtype=np.float64)
    )
    if not np.all(np.isfinite(ell)) or not np.all(np.isfinite(azimuth)):
        raise ValueError("rod and Ewald azimuth coordinates must be finite")
    k = np.linalg.norm(ki)
    cutoff = elastic_axial_cutoff_Ainv(ki_sample_Ainv=ki, radial_Ainv=r)
    axis, first, second = _ewald_frame(ki)
    w = offset + b3 * ell
    q = np.hypot(r, w)
    supported = (q > 0) & (False if cutoff is None else (np.abs(w) <= cutoff))
    along = -q * q / (2 * k)
    transverse = q * np.sqrt(np.maximum(0.0, 1.0 - (q / (2 * k)) ** 2))
    direction = np.cos(azimuth)[..., None] * first + np.sin(azimuth)[..., None] * second
    q_sample = along[..., None] * axis + transverse[..., None] * direction
    coefficient = np.divide(b3, q, out=np.zeros_like(q), where=supported)
    return q_sample, q_sample + ki, np.arctan2(r, w), coefficient


@dataclass(frozen=True, slots=True)
class FiberDetectorTransfer:
    """One batch of source-weighted transfer, before SF and mosaic probability."""

    quadrature_index: NDArray[np.int64]
    polar_angle_rad: NDArray[np.float64]
    cone_angle_rad: NDArray[np.float64]
    coefficient_per_L_rad: NDArray[np.float64]
    spatial: DetectorSpatialKernels
    phase_q_radial_squared_Ainv2: NDArray[np.float64]
    phase_q_normal_squared_Ainv2: NDArray[np.float64]
    attenuation_decay_sum_Ainv: NDArray[np.float64] | None = None
    reference_thickness_A: float = 0.0

    def __post_init__(self) -> None:
        size = len(self.spatial.mean_px)
        for name in (
            "quadrature_index",
            "polar_angle_rad",
            "cone_angle_rad",
            "coefficient_per_L_rad",
            "phase_q_radial_squared_Ainv2",
            "phase_q_normal_squared_Ainv2",
        ):
            dtype = np.int64 if name == "quadrature_index" else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.shape != (size,) or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative and align with spatial kernels")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.attenuation_decay_sum_Ainv is not None:
            decay = np.array(self.attenuation_decay_sum_Ainv, dtype=np.float64, copy=True)
            if decay.shape != (size,) or np.any(~np.isfinite(decay)) or np.any(decay < 0):
                raise ValueError("attenuation decay must align with transfer and be nonnegative")
            decay.setflags(write=False)
            object.__setattr__(self, "attenuation_decay_sum_Ainv", decay)
        if not np.isfinite(self.reference_thickness_A) or self.reference_thickness_A < 0:
            raise ValueError("reference thickness must be finite and nonnegative")

    def coefficients_at_thickness(self, thickness_A: float) -> NDArray[np.float64]:
        """Reweight only the shared depth average; geometry and source stay fixed."""
        if not np.isfinite(thickness_A) or thickness_A < 0:
            raise ValueError("thickness must be finite and nonnegative")
        decay = self.attenuation_decay_sum_Ainv
        if decay is None:
            return self.coefficient_per_L_rad
        return self.coefficient_per_L_rad * (
            uniform_depth_attenuation(decay, 0.0, thickness_A)
            / uniform_depth_attenuation(decay, 0.0, self.reference_thickness_A)
        )


@dataclass(frozen=True, slots=True)
class FiberScatteringTransfer:
    """Immutable optical events before detector rejection or spectral mass."""

    quadrature_index: NDArray[np.int64]
    polar_angle_rad: NDArray[np.float64]
    cone_angle_rad: NDArray[np.float64]
    coefficient_per_L_rad: NDArray[np.float64]
    outgoing_direction_lab: NDArray[np.float64]
    phase_q_radial_squared_Ainv2: NDArray[np.float64]
    phase_q_normal_squared_Ainv2: NDArray[np.float64]
    attenuation_decay_sum_Ainv: NDArray[np.float64] | None = None
    reference_thickness_A: float = 0.0
    local_phase_q_Ainv: NDArray[np.float64] | None = None
    local_external_q_Ainv: NDArray[np.float64] | None = None

    def __post_init__(self):
        size = len(self.quadrature_index)
        if np.asarray(self.quadrature_index).dtype.kind not in "iu":
            raise TypeError("quadrature indices must be integers")
        for name in (
            "quadrature_index",
            "polar_angle_rad",
            "cone_angle_rad",
            "coefficient_per_L_rad",
            "outgoing_direction_lab",
            "phase_q_radial_squared_Ainv2",
            "phase_q_normal_squared_Ainv2",
            "attenuation_decay_sum_Ainv",
            "local_phase_q_Ainv",
            "local_external_q_Ainv",
        ):
            supplied = getattr(self, name)
            if supplied is None:
                continue
            if np.iscomplexobj(supplied):
                raise ValueError("scattering arrays must be real")
            value = np.array(
                supplied, dtype=np.int64 if name == "quadrature_index" else float, copy=True
            )
            shape = (size, 3) if name == "outgoing_direction_lab" else (size,)
            if (
                value.shape != shape
                or np.any(~np.isfinite(value))
                or (name != "outgoing_direction_lab" and np.any(value < 0))
            ):
                raise ValueError("scattering arrays must be finite and aligned")
            if name.endswith("angle_rad") and np.any(value > np.pi):
                raise ValueError("scattering normal angles must lie in [0, pi]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.allclose(
            np.linalg.norm(self.outgoing_direction_lab, axis=1), 1.0, rtol=0, atol=1e-12
        ):
            raise ValueError("outgoing scattering directions must be unit vectors")
        if not np.isfinite(self.reference_thickness_A) or self.reference_thickness_A < 0:
            raise ValueError("reference thickness must be finite and nonnegative")
        if (self.local_phase_q_Ainv is None) != (self.local_external_q_Ainv is None):
            raise ValueError("local scattering requires both phase and external Q")

    @property
    def retained_bytes(self):
        return sum(
            value.nbytes
            for value in (
                self.quadrature_index,
                self.polar_angle_rad,
                self.cone_angle_rad,
                self.coefficient_per_L_rad,
                self.outgoing_direction_lab,
                self.phase_q_radial_squared_Ainv2,
                self.phase_q_normal_squared_Ainv2,
                self.attenuation_decay_sum_Ainv,
                self.local_phase_q_Ainv,
                self.local_external_q_Ainv,
            )
            if value is not None
        )


@dataclass
class FiberScatteringCache:
    """Explicit byte-bounded execution owner; no detector-pruned events retained."""

    maximum_bytes: int = 256 * 1024**2
    build_count: int = field(default=0, init=False)
    reuse_count: int = field(default=0, init=False)
    build_seconds: float = field(default=0.0, init=False)
    retained_bytes: int = field(default=0, init=False)
    peak_retained_bytes: int = field(default=0, init=False)
    entries: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)

    def __post_init__(self):
        if type(self.maximum_bytes) is not int or self.maximum_bytes < 0:
            raise ValueError("maximum_bytes must be a nonnegative integer")

    def compile(self, *, local, coordinates, context):
        si = context["source_state_index"]
        states = context["incident"].states
        instrument = context["instrument"]
        if type(si) is not int or not 0 <= si < len(states.valid) or not states.valid[si]:
            raise ValueError("source_state_index must identify a valid incident state")
        if (
            states.source_revision != context["source"].mean_rays.source_revision
            or states.material_revision != context["material"].material_revision
            or states.sample_geometry_revision != instrument.sample_geometry_revision
        ):
            raise ValueError("source, material and geometry must match incident states")
        key = canonical_revision_sha256(
            ("local", local),
            *(
                (name, (value.h, value.k, value.population) if isinstance(value, Rod) else value)
                for name, value in coordinates.items()
            ),
            ("material", context["material"].material_revision),
            ("wavelength", states.wavelength_A[si]),
            ("direction", states.direction_sample[si]),
            ("phase_wavevector", states.k_film_phase_sample_Ainv[si]),
            ("normal_wavevector", states.kz_film_Ainv[si]),
            ("entrance", states.entrance_amplitude[si]),
            ("polarization", states.polarization_state_id[si]),
            ("sample_rotation", instrument.lab_from_sample.rotation),
            ("crystal_rotation", instrument.sample_from_crystal.rotation),
            ("thickness_A", instrument.film_thickness_A),
        )
        if key in self.entries:
            self.reuse_count += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        start = perf_counter()
        result = (_compile_local_scattering if local else _compile_fiber_scattering)(
            **coordinates, **context
        )
        self.build_seconds += perf_counter() - start
        self.build_count += 1
        size = 0 if result is None else result.retained_bytes
        if size <= self.maximum_bytes:
            while self.entries and (
                self.retained_bytes + size > self.maximum_bytes or len(self.entries) >= 4096
            ):
                _, removed = self.entries.popitem(last=False)
                self.retained_bytes -= 0 if removed is None else removed.retained_bytes
            self.entries[key] = result
            self.retained_bytes += size
            self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        return result


def project_conditional_fiber_transfer(
    scattering,
    *,
    source,
    incident,
    source_state_index,
    instrument,
    maximum_backward_probability,
    include_source_mass=True,
):
    """Recompute current spatial transport, preserving newly visible events."""
    if scattering is None:
        return None, None
    si = source_state_index
    states = incident.states
    if (
        states.source_revision != source.mean_rays.source_revision
        or states.sample_geometry_revision != instrument.sample_geometry_revision
    ):
        raise ValueError("source and sample geometry must match current incident states")
    outgoing = scattering.outgoing_direction_lab
    plane = _intersect_detector_plane(
        np.broadcast_to(states.sample_intersection_lab_m[si], outgoing.shape), outgoing, instrument
    )
    selected = np.flatnonzero(plane.status == ValidityCode.VALID)
    if not len(selected):
        return None, None
    spatial = compile_conditional_spatial_kernels(
        instrument=instrument,
        source=source,
        source_state_index=si,
        outgoing_direction_lab=outgoing[selected],
        maximum_backward_probability=maximum_backward_probability,
    )
    coefficient = scattering.coefficient_per_L_rad[selected] * states.footprint_acceptance[si]
    if include_source_mass:
        coefficient *= states.source_weight[si]
    decay = scattering.attenuation_decay_sum_Ainv
    transfer = FiberDetectorTransfer(
        scattering.quadrature_index[selected],
        scattering.polar_angle_rad[selected],
        scattering.cone_angle_rad[selected],
        coefficient,
        spatial,
        scattering.phase_q_radial_squared_Ainv2[selected],
        scattering.phase_q_normal_squared_Ainv2[selected],
        None if decay is None else decay[selected],
        scattering.reference_thickness_A,
    )
    local = (
        None
        if scattering.local_phase_q_Ainv is None
        else LocalM0DetectorTransfer(
            transfer,
            scattering.local_phase_q_Ainv[selected],
            scattering.local_external_q_Ainv[selected],
        )
    )
    return transfer, local


def _compile_fiber_scattering(
    *,
    rod: Rod,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    L: ArrayLike,
    ewald_azimuth_rad: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    source_state_index: int,
    material: MaterialOptics,
    instrument: CompiledInstrument,
) -> FiberScatteringTransfer | None:
    """Physical scattering before source mass, footprint and detector transport."""
    if not isinstance(rod, Rod):
        raise TypeError("fiber transfer requires a physical Rod")
    states, si = incident.states, source_state_index
    if type(si) is not int or not 0 <= si < len(states.valid) or not states.valid[si]:
        raise ValueError("source_state_index must identify a valid incident state")
    if states.source_revision != source.mean_rays.source_revision:
        raise ValueError("incident state and conditional source revisions disagree")
    if states.material_revision != material.material_revision:
        raise ValueError("incident state and optical material revisions disagree")
    if states.sample_geometry_revision != instrument.sample_geometry_revision:
        raise ValueError("incident state and instrument sample revisions disagree")
    basis = reciprocal_basis(reciprocal_basis_Ainv)
    rotation = proper_rotation(crystal_to_sample)
    if not np.allclose(rotation, instrument.sample_from_crystal.rotation, rtol=0, atol=1e-12):
        raise ValueError("crystal_to_sample must match the instrument mounting")
    b3 = np.linalg.norm(basis[:, 2])
    normal = basis[:, 2] / b3
    anchor = rod.h * basis[:, 0] + rod.k * basis[:, 1]
    offset = anchor @ normal
    radius = np.linalg.norm(anchor - offset * normal)
    q, kf, cone, measure = fiber_ewald_coordinates(
        ki_sample_Ainv=states.k_film_phase_sample_Ainv[si],
        radial_Ainv=radius,
        axial_offset_Ainv=offset,
        b3_norm_Ainv=b3,
        L=L,
        ewald_azimuth_rad=ewald_azimuth_rad,
    )
    q, kf, cone, measure = q.reshape(-1, 3), kf.reshape(-1, 3), cone.ravel(), measure.ravel()
    selected = np.flatnonzero((measure > 0) & (kf[:, 2] > 0))
    if not len(selected):
        return None
    wavelength = states.wavelength_A[si]
    air_k = 2 * np.pi / wavelength
    modes = _solve_exit_mode_arrays(
        kf[selected],
        np.full(len(selected), wavelength),
        material,
        phase_shell_offset_Ainv2=_incident_phase_shell_offset_Ainv2(incident, material, si),
    )
    supported = modes.status == ValidityCode.VALID
    indices = np.flatnonzero(supported)
    selected = selected[indices]
    if not len(selected):
        return None
    outgoing_sample = modes.k_air_phase_sample_Ainv[indices] / air_k
    outgoing_lab = instrument.lab_from_sample.apply_vector(outgoing_sample)
    incident_sign = -1 if states.direction_sample[si, 2] < 0 else 1
    decay = mode_decay_constant(states.kz_film_Ainv[si], incident_sign) + mode_decay_constant(
        modes.kz_film_Ainv[indices], modes.propagation_direction[indices]
    )
    attenuation = uniform_depth_attenuation(decay, 0.0, instrument.film_thickness_A)
    coefficient = measure[selected] * scalar_optical_weight(
        states.entrance_amplitude[si], modes.exit_amplitude[indices], attenuation
    )
    coefficient *= rod.population * incident_illuminated_path_weight(states.direction_sample[si])
    coefficient *= scattering_polarization_weight(
        states.direction_sample[si], outgoing_sample, model_id=states.polarization_state_id[si]
    )
    q_norm = np.linalg.norm(q[selected], axis=1)
    polar = np.arccos(np.clip(q[selected] @ (rotation @ normal) / q_norm, -1.0, 1.0))
    return FiberScatteringTransfer(
        selected,
        polar,
        cone[selected],
        coefficient,
        outgoing_lab,
        np.sum(q[selected, :2] ** 2, axis=1),
        q[selected, 2] ** 2,
        decay,
        instrument.film_thickness_A,
    )


@numba.njit(nogil=True, fastmath=False, cache=False)
def _local_m0_geometry_batch(
    outgoing, solid_angle, ki, k0, refractive_index, rotation, polarization
):
    result = np.zeros((len(outgoing), 7))
    for i in range(len(outgoing)):
        g = local_m0_geometry(
            outgoing[i, 0],
            outgoing[i, 1],
            outgoing[i, 2],
            solid_angle[i],
            ki,
            k0,
            refractive_index,
            rotation,
            polarization,
        )
        result[i, 0] = g.valid
        result[i, 1] = g.alpha_rad
        result[i, 2] = g.phase_q_Ainv
        result[i, 3] = g.external_q_Ainv
        result[i, 4] = g.density_geometry_per_px2
        result[i, 5] = g.phase_q_radial_squared_Ainv2
        result[i, 6] = g.phase_q_normal_squared_Ainv2
    return result


@dataclass(frozen=True, slots=True)
class LocalM0DetectorTransfer:
    """Local-lamella geometry and both signed SF arguments, before the stitch."""

    transfer: FiberDetectorTransfer
    phase_q_Ainv: NDArray[np.float64]
    external_q_Ainv: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("phase_q_Ainv", "external_q_Ainv"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if (
                value.shape != self.transfer.quadrature_index.shape
                or np.any(~np.isfinite(value))
                or np.any(value < 0)
            ):
                raise ValueError(f"{name} must align with the local transfer and be nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _compile_local_scattering(
    *,
    external_q_Ainv: ArrayLike,
    ewald_azimuth_rad: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    source_state_index: int,
    material: MaterialOptics,
    instrument: CompiledInstrument,
) -> FiberScatteringTransfer | None:
    """Integrate local planes on the air Ewald sphere per dQexternal dzeta.

    A single positive external-Q coordinate represents each scattering ray;
    callers sum its two signed SF values with the two directed mosaic masses.
    This uses the shared local-m0 geometry and the same conditional spatial
    integration as the nonzero rods. No macroscopic exit-plane cutoff applies.
    """
    states, si = incident.states, source_state_index
    if type(si) is not int or not 0 <= si < len(states.valid) or not states.valid[si]:
        raise ValueError("source_state_index must identify a valid incident state")
    if (
        states.source_revision != source.mean_rays.source_revision
        or states.material_revision != material.material_revision
        or states.sample_geometry_revision != instrument.sample_geometry_revision
    ):
        raise ValueError("source, material and geometry must match the incident state")
    external, azimuth = np.broadcast_arrays(external_q_Ainv, ewald_azimuth_rad)
    if np.any(external < 0):
        raise ValueError("local chart requires nonnegative external Q")
    k0 = 2 * np.pi / states.wavelength_A[si]
    _, outgoing, _, measure = fiber_ewald_coordinates(
        ki_sample_Ainv=k0 * states.direction_sample[si],
        radial_Ainv=0,
        axial_offset_Ainv=0,
        b3_norm_Ainv=1,
        L=external,
        ewald_azimuth_rad=azimuth,
    )
    outgoing = outgoing.reshape(-1, 3) / k0
    indices = np.flatnonzero(measure.ravel() > 0)
    if not len(indices):
        return None
    refraction_index = np.flatnonzero(material.wavelength_A == states.wavelength_A[si])
    if len(refraction_index) != 1:
        raise ValueError("material must bind the exact source wavelength")
    g = _local_m0_geometry_batch(
        outgoing[indices],
        external.ravel()[indices] / k0**2,
        states.k_film_phase_sample_Ainv[si],
        k0,
        material.n_complex[refraction_index[0]],
        instrument.sample_from_crystal.rotation,
        polarization_model_code(states.polarization_state_id[si]),
    )
    valid = g[:, 0] == 1
    indices, g = indices[valid], g[valid]
    if not len(indices):
        return None
    coefficient = g[:, 4] * incident_illuminated_path_weight(states.direction_sample[si])
    return FiberScatteringTransfer(
        indices,
        g[:, 1],
        np.zeros(len(indices)),
        coefficient,
        instrument.lab_from_sample.apply_vector(outgoing[indices]),
        g[:, 5],
        g[:, 6],
        local_phase_q_Ainv=g[:, 2],
        local_external_q_Ainv=g[:, 3],
    )


def compile_conditional_fiber_transfer(
    *,
    rod,
    reciprocal_basis_Ainv,
    crystal_to_sample,
    L,
    ewald_azimuth_rad,
    source,
    incident,
    source_state_index,
    material,
    instrument,
    maximum_backward_probability,
):
    """Compile direct source-weighted scattering and current detector transport."""
    scattering = _compile_fiber_scattering(
        rod=rod,
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        L=L,
        ewald_azimuth_rad=ewald_azimuth_rad,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        material=material,
        instrument=instrument,
    )
    return project_conditional_fiber_transfer(
        scattering,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        instrument=instrument,
        maximum_backward_probability=maximum_backward_probability,
    )[0]


def compile_conditional_local_m0_transfer(
    *,
    external_q_Ainv,
    ewald_azimuth_rad,
    source,
    incident,
    source_state_index,
    material,
    instrument,
    maximum_backward_probability,
):
    """Compile direct local-lamella scattering and current detector transport."""
    scattering = _compile_local_scattering(
        external_q_Ainv=external_q_Ainv,
        ewald_azimuth_rad=ewald_azimuth_rad,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        material=material,
        instrument=instrument,
    )
    return project_conditional_fiber_transfer(
        scattering,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        instrument=instrument,
        maximum_backward_probability=maximum_backward_probability,
    )[1]


@numba.njit(nogil=True)
def _axial_mixture_quantiles(quantiles, lower, upper, centers, width, weights, uniform_mass=0.2):
    """Monotone inverse of the complete normalized axial mixture CDF."""
    z = np.arctan2((upper - lower) * width, width * width + (lower - centers) * (upper - centers))
    x = np.empty(len(quantiles))
    pdf = np.empty(len(quantiles))
    weights = weights / weights.sum()
    for i in range(len(quantiles)):
        lo, hi = lower, upper
        v = lo + (hi - lo) * quantiles[i]
        # Newton can stagnate near opposite bracket edges in a narrow mixture.
        # Keep its fast path, then guarantee contraction with bounded bisection.
        for _iteration in range(150):
            cdf = uniform_mass * (v - lower) / (upper - lower)
            density = uniform_mass / (upper - lower)
            for j in range(len(centers)):
                delta = v - centers[j]
                partial_mass = np.arctan2(
                    (v - lower) * width, width * width + (lower - centers[j]) * delta
                )
                cdf += (1 - uniform_mass) * weights[j] * partial_mass / z[j]
                density += (
                    (1 - uniform_mass) * weights[j] * width / (delta * delta + width * width) / z[j]
                )
            error = cdf - quantiles[i]
            if not np.isfinite(error) or not np.isfinite(density) or density <= 0:
                raise ValueError("axial inverse CDF requires a finite positive density")
            if abs(error) < 2e-15 or hi - lo < 2e-14 * max(1.0, upper - lower):
                break
            if error > 0:
                hi = v
            else:
                lo = v
            proposed = v - error / density if _iteration < 70 else 0.5 * (lo + hi)
            v = proposed if lo < proposed < hi else 0.5 * (lo + hi)
        else:
            raise ValueError("axial inverse CDF did not converge within its iteration budget")
        x[i] = v
        pdf[i] = density
    return x, pdf


@numba.njit(nogil=True)
def _wrapped_cauchy_cdf(value, width):
    # Keep the original argument for libm range reduction. Subtracting a rounded
    # 2*pi first loses precision at narrow peaks near the periodic boundary.
    turns = 2 * np.sign(value) * max(0.0, np.ceil((abs(value) - 2 * np.pi) / (4 * np.pi)))
    return (
        turns + 0.5 + np.arctan2(np.sin(value / 2), np.tanh(width / 2) * np.cos(value / 2)) / np.pi
    )


@numba.njit(nogil=True)
def _angular_cdf_density(value, centers, widths, offsets, uniform_mass):
    cdf = uniform_mass * value / (2 * np.pi)
    density = uniform_mass / (2 * np.pi)
    for j in range(len(centers)):
        delta = value - centers[j]
        width = widths[j]
        rho = np.exp(-width)
        h = -np.expm1(-width)
        cdf += (1 - uniform_mass) / len(centers) * (_wrapped_cauchy_cdf(delta, width) - offsets[j])
        density += (
            (1 - uniform_mass)
            / len(centers)
            * (-np.expm1(-2 * width))
            / (2 * np.pi * (h * h + 4 * rho * np.sin(delta / 2) ** 2))
        )
    return cdf, density


@numba.njit(nogil=True)
def _angular_intervals(q, bounds):
    lower = np.empty(2 * len(bounds))
    upper = np.empty(2 * len(bounds))
    count = 0
    for row in bounds:
        if q < row[0] or q > row[1] or row[3] == 0:
            continue
        start = row[2] % (2 * np.pi)
        width = row[3]
        if width >= 2 * np.pi:
            return np.array([0.0]), np.array([2 * np.pi])
        stop = start + width
        lower[count], upper[count] = start, min(stop, 2 * np.pi)
        count += 1
        if stop > 2 * np.pi:
            lower[count], upper[count] = 0.0, stop - 2 * np.pi
            count += 1
    if count == 0:
        return lower[:0], upper[:0]
    order = np.argsort(lower[:count])
    lo, hi = lower[:count][order], upper[:count][order]
    merged = 1
    for j in range(1, count):
        if lo[j] <= hi[merged - 1]:
            hi[merged - 1] = max(hi[merged - 1], hi[j])
        else:
            lo[merged], hi[merged] = lo[j], hi[j]
            merged += 1
    return lo[:merged], hi[:merged]


@numba.njit(nogil=True)
def _angular_inverse_cdf(
    target, left, right, centers, widths, offsets, uniform_mass, fraction, require_cdf=False
):
    value = left + (right - left) * fraction
    for iteration in range(150):
        cdf, density = _angular_cdf_density(value, centers, widths, offsets, uniform_mass)
        error = cdf - target
        if not np.isfinite(error) or not np.isfinite(density) or density <= 0:
            raise ValueError("angular inverse CDF requires a finite positive density")
        if abs(error) < 2e-15 or (not require_cdf and right - left < 2e-14):
            return value, density
        if error > 0:
            right = value
        else:
            left = value
        proposed = value - error / density if iteration < 70 else 0.5 * (left + right)
        value = proposed if left < proposed < right else 0.5 * (left + right)
    raise ValueError(
        "angular inverse CDF did not converge within its iteration budget", error, right - left
    )


@numba.njit(nogil=True, fastmath=False)
def _resolved_angular_cdf_panels(
    q, bounds, centers, widths, power, maximum_width, regions, maximum_nodes, k
):
    """Freeze all CDF panels before streaming; bound physical gaps and total work."""
    order = 8
    initial_panels = 2 ** max(0, power - 3)
    minimum_count = 0.0
    for coordinate in q:
        lo, hi = _angular_intervals(coordinate, bounds)
        for arc in range(len(lo)):
            minimum_count += (
                max(initial_panels, np.ceil((hi[arc] - lo[arc]) / maximum_width)) * order
            )
            if minimum_count > maximum_nodes:
                raise ValueError("angular panel node budget exceeded before allocation")
    indices = []
    physical_left = []
    physical_right = []
    cdf_left = []
    cdf_right = []
    for i in range(len(q)):
        local_regions = regions[(regions[:, 0] <= q[i]) & (q[i] <= regions[:, 1])]
        ratio = min(q[i] / (2 * k), 1.0)
        transverse = q[i] * np.sqrt((1 - ratio) * (1 + ratio))
        offsets = np.empty(centers.shape[1])
        for j in range(len(offsets)):
            offsets[j] = _wrapped_cauchy_cdf(-centers[i, j], widths[i, j])
        lo, hi = _angular_intervals(q[i], bounds)
        for arc in range(len(lo)):
            base = _angular_cdf_density(lo[arc], centers[i], widths[i], offsets, 0.2)[0]
            end = _angular_cdf_density(hi[arc], centers[i], widths[i], offsets, 0.2)[0]
            if not end > base:
                raise ValueError("reachable angular interval has unresolved CDF mass")
            left, pleft = lo[arc], base
            for j in range(initial_panels):
                pright = base + (end - base) * (j + 1) / initial_panels
                right = (
                    hi[arc]
                    if j + 1 == initial_panels
                    else _angular_inverse_cdf(
                        pright, left, hi[arc], centers[i], widths[i], offsets, 0.2, 0.5
                    )[0]
                )
                stack = [(left, right, pleft, pright)]
                while stack:
                    a, b, pa, pb = stack.pop()
                    cap = maximum_width
                    for region in local_regions:
                        start = region[2] % (2 * np.pi)
                        stop = start + region[3]
                        if transverse > 0 and (
                            (a < min(stop, 2 * np.pi) and b > start)
                            or (stop > 2 * np.pi and a < stop - 2 * np.pi)
                        ):
                            peak_q = min(
                                max(np.sqrt(2) * k, min(region[0], 2 * k)),
                                min(region[1], 2 * k),
                            )
                            peak_ratio = peak_q / (2 * k)
                            peak_transverse = peak_q * np.sqrt((1 - peak_ratio) * (1 + peak_ratio))
                            cap = min(cap, region[4] * peak_transverse / transverse)
                    if b - a > cap:
                        pm = pa + (pb - pa) / 2
                        middle = _angular_inverse_cdf(
                            pm, a, b, centers[i], widths[i], offsets, 0.2, 0.5
                        )[0]
                        if not a < middle < b or not pa < pm < pb:
                            raise ValueError("angular resolution exceeds floating-point precision")
                        stack.append((middle, b, pm, pb))
                        stack.append((a, middle, pa, pm))
                    else:
                        indices.append(i)
                        physical_left.append(a)
                        physical_right.append(b)
                        cdf_left.append(pa)
                        cdf_right.append(pb)
                    if (len(indices) + len(stack)) * order > maximum_nodes:
                        raise ValueError("angular panel node budget exceeded before allocation")
                left, pleft = right, pright
    return (
        np.array(indices, dtype=np.int64),
        np.array(physical_left),
        np.array(physical_right),
        np.array(cdf_left),
        np.array(cdf_right),
    )


@numba.njit(nogil=True)
def _angular_cdf_panel_blocks(panels, centers, widths, nodes, weights, batch_size):
    axial_index = np.empty(batch_size, dtype=np.int64)
    phi, mass = np.empty(batch_size), np.empty(batch_size)
    count = 0
    for panel in range(len(panels[0])):
        i = panels[0][panel]
        offsets = np.empty(centers.shape[1])
        for j in range(len(offsets)):
            offsets[j] = _wrapped_cauchy_cdf(-centers[i, j], widths[i, j])
        half = (panels[4][panel] - panels[3][panel]) / 2
        for j in range(len(nodes)):
            value, density = _angular_inverse_cdf(
                panels[3][panel] + half * (nodes[j] + 1),
                panels[1][panel],
                panels[2][panel],
                centers[i],
                widths[i],
                offsets,
                0.2,
                (nodes[j] + 1) / 2,
            )
            axial_index[count], phi[count] = i, value
            mass[count] = half * weights[j] / density
            count += 1
            if count == batch_size:
                yield axial_index, phi, mass
                axial_index = np.empty(batch_size, dtype=np.int64)
                phi, mass = np.empty(batch_size), np.empty(batch_size)
                count = 0
    if count:
        yield axial_index[:count], phi[:count], mass[:count]


def _angular_proposal_parameters(t, radius, ki, normal, sigma, gamma):
    axis, e1, e2 = _ewald_frame(ki)
    k = np.linalg.norm(ki)
    q = np.hypot(radius, t)
    cone = np.arctan2(radius, t)
    a = -q / (2 * k) * (axis @ normal)
    amplitude = np.sqrt(np.maximum(0, 1 - (q / (2 * k)) ** 2)) * np.hypot(e1 @ normal, e2 @ normal)
    phase = np.arctan2(e2 @ normal, e1 @ normal)
    centers = []
    widths = []
    for c in (cone, np.pi - cone):
        target = np.divide(np.cos(c) - a, amplitude, out=np.zeros_like(t), where=amplitude > 0)
        angle = np.arccos(np.clip(target, -1, 1))
        derivative = amplitude * abs(np.sin(angle)) / np.maximum(np.sin(c), 1e-6)
        curvature = amplitude * abs(np.cos(angle)) / np.maximum(np.sin(c), 1e-6)
        closest = np.arccos(np.clip(a + amplitude * np.cos(angle), -1, 1))
        distance = abs(closest - c)
        for base_width in (sigma, gamma):
            effective = np.hypot(base_width, distance)
            width = effective / np.maximum(derivative + np.sqrt(curvature * effective), 1e-5)
            if radius == 0:
                width = np.hypot(base_width, distance)
            width = np.clip(width, 1e-5, np.pi)
            for sign in (1, -1):
                centers.append(phase + sign * angle)
                widths.append(width)
    return np.column_stack(centers), np.column_stack(widths)


@dataclass(frozen=True, slots=True)
class FiberQuadratureNodes:
    """Shared exact axial nodes and source-specific joint integration coordinates.

    ``weight_Ainv_rad`` integrates the positive axial coordinate and Ewald azimuth.
    Alternatively strength_weighted_mass integrates W du dphi and requires
    canonical signed_strength_fractions=S/W; ordinary weights must then be None.
    Empty geometric support contributes zero while retaining the original number
    of attempted nodes. Mosaic, optical and source factors are never included.
    """

    positive_axial_Ainv: NDArray[np.float64]
    axial_index: NDArray[np.int64]
    ewald_azimuth_rad: NDArray[np.float64]
    weight_Ainv_rad: NDArray[np.float64] | None
    strength_weighted_mass: NDArray[np.float64] | None = None
    signed_strength_fractions: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        index = np.asarray(self.axial_index)
        if index.dtype.kind not in "iu":
            raise TypeError("axial_index must contain integers")
        weighted = self.strength_weighted_mass is not None
        if weighted == (self.weight_Ainv_rad is not None):
            raise ValueError("declare exactly one ordinary du or strength-weighted W du measure")
        if weighted != (self.signed_strength_fractions is not None):
            raise ValueError("strength-weighted nodes require canonical signed fractions")
        for name in (
            "positive_axial_Ainv",
            "axial_index",
            "ewald_azimuth_rad",
            "weight_Ainv_rad",
            "strength_weighted_mass",
            "signed_strength_fractions",
        ):
            if getattr(self, name) is None:
                continue
            reject_complex(getattr(self, name), name)
            dtype = np.int64 if name == "axial_index" else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            dimensions = 2 if name == "signed_strength_fractions" else 1
            if (
                value.ndim != dimensions
                or np.any(~np.isfinite(value))
                or (name != "ewald_azimuth_rad" and np.any(value < 0))
            ):
                raise ValueError("quadrature arrays must be finite; non-angle entries nonnegative")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        mass = self.strength_weighted_mass if weighted else self.weight_Ainv_rad
        if (
            self.ewald_azimuth_rad.shape != self.axial_index.shape
            or mass.shape != self.axial_index.shape
            or np.any(self.axial_index >= len(self.positive_axial_Ainv))
            or np.any(self.ewald_azimuth_rad < -2 * np.pi)
            or np.any(self.ewald_azimuth_rad > 2 * np.pi)
            or np.any(mass <= 0)
        ):
            raise ValueError(
                "retained quadrature indices, azimuths and positive weights must align"
            )
        if weighted and (
            self.signed_strength_fractions.shape != (2, len(self.positive_axial_Ainv))
            or not np.allclose(self.signed_strength_fractions.sum(axis=0), 1, rtol=0, atol=5e-15)
        ):
            raise ValueError("canonical signed fractions must align and sum to one")


@numba.njit(nogil=True, fastmath=False, cache=False)
def _importance_block(q, pdf, quantiles, first, last, bounds, centers, widths):
    """Conditional disjoint-arc importance measure; no physical factors or pruning."""
    strata = quantiles.shape[1]
    index = np.empty((last - first) * strata, dtype=np.int64)
    azimuth = np.empty(len(index))
    mass = np.empty(len(index))
    count = 0
    for i in range(first, last):
        left, right = _angular_intervals(q[i], bounds)
        if not len(left):
            continue
        offsets = np.array(
            [_wrapped_cauchy_cdf(-centers[i, n], widths[i, n]) for n in range(centers.shape[1])]
        )
        cdf_left = np.empty(len(left))
        arc_mass = np.empty(len(left))
        for k in range(len(left)):
            cdf_left[k] = _angular_cdf_density(left[k], centers[i], widths[i], offsets, 0.2)[0]
            cdf_right = _angular_cdf_density(right[k], centers[i], widths[i], offsets, 0.2)[0]
            arc_mass[k] = cdf_right - cdf_left[k]
            # Two endpoint evaluations must resolve mass above the inverse-CDF tolerance.
            if not np.isfinite(arc_mass[k]) or arc_mass[k] <= 4e-15:
                raise ValueError("nonempty local-m0 arc has unresolved proposal CDF mass")
        total = arc_mass.sum()
        if not np.isfinite(total) or total <= 0 or not np.isfinite(pdf[i]) or pdf[i] <= 0:
            raise ValueError("local-m0 importance proposal requires finite positive densities")
        for j in range(strata):
            target = quantiles[i, j] * total
            k, cumulative = 0, 0.0
            while k < len(arc_mass) and target >= cumulative + arc_mass[k]:
                cumulative += arc_mass[k]
                k += 1
            if k == len(arc_mass):
                raise ValueError("local-m0 restricted CDF target rounded outside its arcs")
            remainder = target - cumulative
            cdf_target = cdf_left[k] + remainder
            if (
                remainder < 0
                or cdf_target >= cdf_left[k] + arc_mass[k]
                or (remainder > 0 and cdf_target <= cdf_left[k])
            ):
                raise ValueError("local-m0 restricted CDF target is unresolved")
            inverse_left, inverse_right, inverse_target = left[k], right[k], cdf_target
            at_half_turn = _angular_cdf_density(np.pi, centers[i], widths[i], offsets, 0.2)[0]
            if cdf_target >= at_half_turn:
                # Exact periodic re-expression, with the low part of mathematical
                # 2*pi retained. Keep original arc masses and Sobol strata unchanged.
                inverse_left = (max(left[k], np.pi) - 2 * np.pi) - 2.4492935982947064e-16
                inverse_right = (right[k] - 2 * np.pi) - 2.4492935982947064e-16
                inverse_target -= 1.0
            else:
                inverse_right = min(inverse_right, np.pi)
            phi, density = _angular_inverse_cdf(
                inverse_target,
                inverse_left,
                inverse_right,
                centers[i],
                widths[i],
                offsets,
                0.2,
                remainder / arc_mass[k],
                True,
            )
            error = (
                _angular_cdf_density(phi, centers[i], widths[i], offsets, 0.2)[0] - inverse_target
            )
            if not inverse_left <= phi < inverse_right or abs(error) > 2e-15:
                raise ValueError(
                    "local-m0 inverse CDF failed its original arc bracket",
                    phi - inverse_left,
                    inverse_right - phi,
                    error,
                    cdf_left[k],
                    cdf_left[k] + arc_mass[k],
                )
            index[count], azimuth[count] = i, phi
            mass[count] = total / (len(q) * strata * pdf[i] * density)
            count += 1
    return index[:count], azimuth[:count], mass[:count]


def iter_local_m0_coordinates(
    *,
    axial_bounds_Ainv,
    axial_peak_centers_Ainv,
    axial_peak_half_width_Ainv,
    radial_Ainv,
    ki_sample_Ainv,
    normal_sample,
    source_region_bounds,
    reference_mosaic,
    axial_power,
    angular_power,
    axial_seed,
    angular_resolution_regions,
    maximum_angular_panel_nodes,
    batch_size,
    angular_rule="resolved_cdf_gl8.v1",
):
    """Existing local-lamella endpoint measure, confined to the zero rod.

    This channel integrates external Q, with no fabricated q=0 response. Its
    default uses full-support Sobol axial nodes and native-resolution GL8 angular
    panels. Explicit nominal importance uses at least 32 conditional CDF strata;
    both remain independent of the regular strength-weighted internal phase chart.
    """
    if radial_Ainv != 0:
        raise ValueError("the local-m0 endpoint integrator requires radius zero")
    if angular_rule not in {"resolved_cdf_gl8.v1", "cdf_stratified_importance.v1"}:
        raise ValueError("unknown local-m0 angular rule")
    importance = angular_rule == "cdf_stratified_importance.v1"
    if importance and (
        axial_power < 12
        or angular_power < 5
        or batch_size < 2**angular_power
        or batch_size > 16384
        or batch_size % 2**angular_power
    ):
        raise ValueError("local-m0 importance needs at least 4096 axial nodes and 32 whole strata")
    if importance and 2 ** (axial_power + angular_power) > maximum_angular_panel_nodes:
        raise ValueError("local-m0 importance node budget exceeded before allocation")
    lower, upper = axial_bounds_Ainv
    unit = qmc.Sobol(2, scramble=True, seed=axial_seed).random_base2(axial_power)
    axial, pdf = _axial_mixture_quantiles(
        unit[:, 0],
        lower,
        upper,
        np.asarray(axial_peak_centers_Ainv),
        axial_peak_half_width_Ainv,
        np.ones(len(axial_peak_centers_Ainv)),
    )
    mass = 1 / (len(axial) * pdf)
    centers, widths = _angular_proposal_parameters(
        axial,
        0,
        ki_sample_Ainv,
        normal_sample,
        reference_mosaic.gaussian_sigma_rad,
        reference_mosaic.lorentzian_half_width_rad,
    )
    if importance:
        strata = 2**angular_power
        quantiles = (np.arange(strata)[None, :] + unit[:, 1, None]) / strata
        for first in range(0, len(axial), batch_size // strata):
            index, phi, weight = _importance_block(
                axial,
                pdf,
                quantiles,
                first,
                min(first + batch_size // strata, len(axial)),
                np.asarray(source_region_bounds),
                centers,
                widths,
            )
            if len(index):
                yield FiberQuadratureNodes(axial, index, phi, weight)
        return
    nodes, weights = roots_legendre(8)
    used_nodes = 0
    # Bound live preparation without resetting the total work guard. Global
    # axial nodes and their probability masses are unchanged by these slices.
    for first in range(0, len(axial), 16):
        stop = min(first + 16, len(axial))
        panels = _resolved_angular_cdf_panels(
            axial[first:stop],
            np.asarray(source_region_bounds),
            centers[first:stop],
            widths[first:stop],
            angular_power,
            2 * np.pi,
            np.asarray(angular_resolution_regions),
            maximum_angular_panel_nodes - used_nodes,
            np.linalg.norm(ki_sample_Ainv),
        )
        count = len(panels[0]) * 8
        used_nodes += count
        if not count:
            continue
        for index, phi, angular_mass in _angular_cdf_panel_blocks(
            panels,
            centers[first:stop],
            widths[first:stop],
            nodes,
            weights,
            min(batch_size, count),
        ):
            global_index = first + index
            yield FiberQuadratureNodes(axial, global_index, phi, mass[global_index] * angular_mass)


def iter_fixed_fiber_coordinates(*, coordinate_parameters, rule, source_index, group_index):
    """Strength-independent nominal du dphi rule, with full proposal support.

    The versioned proposal uses equal Cauchy peaks spaced by b3, width .02*b3,
    and a .2 uniform mass. Original source/group indices fix the scramble/rotation.
    No sampled strength, actual mosaic, or surviving detector event selects nodes.
    """
    p = coordinate_parameters
    count = 2**rule.regular_angular_power
    unit = qmc.Sobol(
        2, scramble=True, seed=7919 * rule.regular_seed + 65537 * group_index + 1009
    ).random_base2(rule.regular_axial_power)
    axial, pdf = _axial_mixture_quantiles(
        unit[:, 0],
        *p["axial_bounds_Ainv"],
        p["axial_peak_centers_Ainv"],
        p["axial_peak_half_width_Ainv"],
        np.ones(len(p["axial_peak_centers_Ainv"])),
    )
    radius = p["radial_Ainv"]
    centers, widths = _angular_proposal_parameters(
        axial,
        radius,
        p["ki_sample_Ainv"],
        p["normal_sample"],
        p["reference_mosaic"].gaussian_sigma_rad,
        p["reference_mosaic"].lorentzian_half_width_rad,
    )
    rotation = np.random.default_rng(
        8191 * source_index + 7919 * rule.regular_seed + 131 * group_index + 973
    ).random()
    quantiles = np.mod(unit[:, 1, None] + rotation + (np.arange(count) + 0.5) / count, 1.0)
    q = np.hypot(radius, axial)
    step = rule.batch_size // count
    for first in range(0, len(axial), step):
        index, phi, weight = _importance_block(
            q,
            pdf,
            quantiles,
            first,
            min(first + step, len(axial)),
            np.asarray(p["source_region_bounds"]),
            centers,
            widths,
        )
        if len(index):
            yield FiberQuadratureNodes(axial, index, phi, weight)


@dataclass(frozen=True, slots=True)
class FiberIntegrationRule:
    """Explicit regular estimator and necessary local-m0 endpoint controls.

    The default uses positive W du panels and native-pixel indicators.
    Fixed importance is a separately named nominal estimator for response reuse.
    Numerical controls never remove physical peaks, tails or signed rods.
    """

    regular_rule: str = "strength_gauss_pixel_error.v1"
    regular_axial_power: int = 12
    regular_angular_power: int = 5
    regular_seed: int = 0
    strength_gauss_order: int = 4
    strength_scalar_order: int = 16
    strength_scalar_phase_step_rad: float = np.pi / 4
    angular_initial_power: int = 5
    pixel_error_rtol: float = 5e-5
    pixel_error_atol: float = 1e-15
    pixel_error_initial_width_rad: float = 0.1
    pixel_error_maximum_depth: int = 32
    pixel_error_maximum_bytes: int = 256 * 1024**2
    local_m0_axial_power: int = 12
    local_m0_angular_power: int = 5
    local_m0_seed: int = 0
    local_m0_peak_spacing_L: float = 1.0
    local_m0_peak_half_width_L: float = 0.02
    local_m0_axial_peak_coordinate: str = "external_q"
    local_m0_angular_resolution_fraction: float = 0.5
    source_latent_radius: float = 8.0
    maximum_backward_probability: float = 1e-12
    maximum_angular_panel_nodes: int = 4194304
    batch_size: int = 16384
    cone_quadrature_order: int = 16
    stitch_grid_size: int = 513
    regular_q_bounds_Ainv: tuple[float, float] | None = None
    local_m0_q_bounds_Ainv: tuple[float, float] | None = None
    frozen_ewald_bounds_Ainv_rad: tuple[float, float, float, float] | None = None
    local_m0_angular_rule: str = "resolved_cdf_gl8.v1"
    local_m0_replica: int = 0

    def __post_init__(self):
        if self.regular_rule not in {"strength_gauss_pixel_error.v1", "fixed_importance.v1"}:
            raise ValueError("unknown regular integration rule")
        for name in ("regular_axial_power", "regular_angular_power", "regular_seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.local_m0_angular_rule not in {
            "resolved_cdf_gl8.v1",
            "cdf_stratified_importance.v1",
        }:
            raise ValueError("unknown local-m0 angular rule")
        if type(self.local_m0_replica) is not int or self.local_m0_replica < 0:
            raise ValueError("local-m0 replica identity must be a nonnegative integer")
        for name in (
            "strength_gauss_order",
            "strength_scalar_order",
            "pixel_error_maximum_depth",
            "pixel_error_maximum_bytes",
            "maximum_angular_panel_nodes",
            "batch_size",
            "cone_quadrature_order",
            "stitch_grid_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.regular_rule == "fixed_importance.v1" and (
            self.regular_axial_power < 11
            or self.regular_angular_power < 5
            or self.regular_seed >= 2**32
            or self.batch_size < 2**self.regular_angular_power
            or self.batch_size > 16384
            or self.batch_size % 2**self.regular_angular_power
            or 2 ** (self.regular_axial_power + self.regular_angular_power)
            > self.maximum_angular_panel_nodes
        ):
            raise ValueError(
                "fixed regular importance requires bounded whole strata, at least 2048 by 32"
            )
        if (
            self.strength_gauss_order > 16
            or self.strength_scalar_order < max(8, self.strength_gauss_order)
            or self.cone_quadrature_order < 4
            or self.stitch_grid_size < 257
        ):
            raise ValueError("inadequate strength, cone or stitch quadrature order")
        for name in (
            "angular_initial_power",
            "local_m0_axial_power",
            "local_m0_angular_power",
            "local_m0_seed",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.local_m0_angular_rule == "cdf_stratified_importance.v1" and (
            self.local_m0_axial_power < 12
            or self.local_m0_angular_power < 5
            or self.batch_size < 2**self.local_m0_angular_power
            or self.batch_size > 16384
            or self.batch_size % 2**self.local_m0_angular_power
            or self.local_m0_seed >= 2**32
        ):
            raise ValueError(
                "local-m0 importance needs at least 4096 by 32 and bounded whole strata"
            )
        for name in (
            "strength_scalar_phase_step_rad",
            "pixel_error_rtol",
            "pixel_error_atol",
            "pixel_error_initial_width_rad",
            "local_m0_peak_spacing_L",
            "local_m0_peak_half_width_L",
            "local_m0_angular_resolution_fraction",
            "source_latent_radius",
        ):
            value = getattr(self, name)
            if (
                np.ndim(value) != 0
                or np.iscomplexobj(value)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.local_m0_axial_peak_coordinate not in {"external_q", "film_phase_q_first_source"}:
            raise ValueError("unknown local-m0 proposal coordinate")
        if not 0 <= self.maximum_backward_probability < 1:
            raise ValueError("maximum_backward_probability must lie in [0,1)")
        for name in ("regular_q_bounds_Ainv", "local_m0_q_bounds_Ainv"):
            value = getattr(self, name)
            if value is not None:
                if np.iscomplexobj(value):
                    raise ValueError("Q support must be real")
                bounds = tuple(float(v) for v in value)
                if (
                    len(bounds) != 2
                    or not np.all(np.isfinite(bounds))
                    or not 0 <= bounds[0] < bounds[1]
                ):
                    raise ValueError("Q support must be a finite increasing nonnegative pair")
                object.__setattr__(self, name, bounds)
        if self.frozen_ewald_bounds_Ainv_rad is not None:
            supplied = self.frozen_ewald_bounds_Ainv_rad
            if np.iscomplexobj(supplied):
                raise ValueError("frozen Ewald envelope must be real")
            bounds = tuple(float(v) for v in supplied)
            if (
                len(bounds) != 4
                or not np.all(np.isfinite(bounds))
                or not 0 <= bounds[0] < bounds[1]
                or not 0 < bounds[3] <= 2 * np.pi
            ):
                raise ValueError("frozen Ewald envelope requires finite Q and angular support")
            object.__setattr__(self, "frozen_ewald_bounds_Ainv_rad", bounds)


@dataclass(frozen=True, slots=True)
class ConditionalFiberBatch:
    """One shared geometry batch before SF, mosaic and detector-region reduction.

    Rods remain individual physical identities. Both signed SF sheets use the
    same spatial kernels and quadrature masses; their cone angles are c and pi-c.
    ``integrated_coefficient`` contains source/optical coefficients and d(azimuth).
    Regular weighted batches also contain physical W(u)du, including canonical
    strengths and rod populations; only S+/W and S-/W remain to contract. Local-m0
    unweighted batches contain du with unit geometric rod population and require
    the canonical strength table. Neither needs a second detector Jacobian.
    """

    source_state_index: int
    rods: tuple[Rod, ...]
    radial_Ainv: float
    positive_axial_Ainv: NDArray[np.float64]
    axial_index: NDArray[np.int64]
    transfer: FiberDetectorTransfer
    integrated_coefficient: NDArray[np.float64]
    local_m0: LocalM0DetectorTransfer | None = None
    signed_strength_fractions: NDArray[np.float64] | None = None
    angular_error_indicator: float = 0.0
    native_pixel_patch: tuple[tuple[int, int, int, int], NDArray[np.float64]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rods", tuple(self.rods))
        for name, dtype in (
            ("positive_axial_Ainv", np.float64),
            ("axial_index", np.int64),
            ("integrated_coefficient", np.float64),
        ):
            original = np.asarray(getattr(self, name))
            if name == "axial_index" and original.dtype.kind not in "iu":
                raise TypeError("axial indices must be integers")
            value = np.array(original, dtype=dtype, copy=True)
            if value.ndim != 1 or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError("fiber batch arrays must be finite nonnegative vectors")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        count = len(self.transfer.quadrature_index)
        if (
            self.axial_index.shape != (count,)
            or self.integrated_coefficient.shape != (count,)
            or np.any(self.axial_index >= len(self.positive_axial_Ainv))
            or type(self.source_state_index) is not int
            or self.source_state_index < 0
            or not np.isfinite(self.radial_Ainv)
            or self.radial_Ainv < 0
            or not self.rods
            or any(not isinstance(rod, Rod) for rod in self.rods)
        ):
            raise ValueError("fiber batch identities, nodes and transfer must align")
        if not np.isfinite(self.angular_error_indicator) or self.angular_error_indicator < 0:
            raise ValueError("angular error indicator must be finite and nonnegative")
        if self.signed_strength_fractions is not None:
            fractions = np.array(self.signed_strength_fractions, dtype=float, copy=True)
            if (
                fractions.shape != (2, len(self.positive_axial_Ainv))
                or np.any(~np.isfinite(fractions))
                or np.any(fractions < 0)
                or not np.allclose(fractions.sum(axis=0), 1, rtol=0, atol=5e-15)
            ):
                raise ValueError("weighted batch signed fractions must align")
            fractions.setflags(write=False)
            object.__setattr__(self, "signed_strength_fractions", fractions)
        if self.local_m0 is not None and self.local_m0.transfer is not self.transfer:
            raise ValueError("local metadata must describe this same transfer")


def iter_conditional_fiber_transfers(
    *,
    rods: tuple[Rod, ...],
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    native_bounds_px: ArrayLike,
    reference_mosaic: MosaicParameters,
    rule: FiberIntegrationRule,
    local_stitched_m0: bool,
    cancel_requested: Callable[[], bool] | None = None,
    source_state_indices: tuple[int, ...] | None = None,
    scattering_cache: FiberScatteringCache | None = None,
    include_source_mass: bool = True,
    regular_integrator: Callable[..., Iterator[ConditionalFiberBatch]] | None = None,
    include_local_m0: bool = True,
) -> Iterator[ConditionalFiberBatch]:
    """Stream the same continuous transfers for native fits and full-panel images.

    Bounds describe the requested observable. A renderer supplies its panel
    domain; it cannot reuse an ROI-pruned fit response. Unstitched m0 uses
    the same positive interior-node regular engine and planar kinematic optics. The local air-Ewald
    chart is reserved for the explicitly requested local-lamella composite.
    Source masses are kept intact when rows or directions have no valid support.
    Excluding the local composite retains original rod groups, group indices and
    the combined channel count used for the regular absolute error allowance.
    """
    if regular_integrator is None:
        raise ValueError("the strength-weighted regular integrator must be supplied explicitly")
    if not isinstance(rule, FiberIntegrationRule) or type(local_stitched_m0) is not bool:
        raise TypeError("an explicit integration rule and local-composite selection are required")
    if type(include_local_m0) is not bool:
        raise TypeError("include_local_m0 must be bool")
    validate_conditional_spatial_support(source, instrument)
    if (
        incident.states.source_revision != source.mean_rays.source_revision
        or incident.states.material_revision != material.material_revision
        or incident.states.sample_geometry_revision != instrument.sample_geometry_revision
    ):
        raise ValueError("source, material and geometry must match the incident states")
    physical_rods = tuple(rods)
    if not physical_rods or any(not isinstance(rod, Rod) for rod in physical_rods):
        raise ValueError("at least one physical rod is required")
    if len({(rod.h, rod.k) for rod in physical_rods}) != len(physical_rods):
        raise ValueError("physical rods must not be duplicated")
    basis, rotation = reciprocal_basis(reciprocal_basis_Ainv), proper_rotation(crystal_to_sample)
    if not np.allclose(rotation, instrument.sample_from_crystal.rotation, rtol=0, atol=1e-12):
        raise ValueError("crystal_to_sample must match the instrument mounting")
    b3 = np.linalg.norm(basis[:, 2])
    crystal_normal = basis[:, 2] / b3
    normal = rotation @ crystal_normal
    groups: list[tuple[float, list[Rod]]] = []
    for rod in physical_rods:
        anchor = rod.h * basis[:, 0] + rod.k * basis[:, 1]
        radius = float(np.linalg.norm(anchor - (anchor @ crystal_normal) * crystal_normal))
        for group_radius, group in groups:
            if abs(radius - group_radius) <= 64 * np.finfo(float).eps * max(1.0, radius):
                group.append(rod)
                break
        else:
            groups.append((radius, [rod]))
    states = incident.states
    valid_sources = [int(index) for index in np.flatnonzero(states.valid)]
    requested = valid_sources if source_state_indices is None else list(source_state_indices)
    if len(set(requested)) != len(requested) or any(
        type(index) is not int or not 0 <= index < len(states.valid) for index in requested
    ):
        raise ValueError("source state indices must be unique original source rows")
    active_sources = [index for index in requested if states.valid[index]]
    if not valid_sources:
        return
    context = dict(source=source, incident=incident, material=material, instrument=instrument)
    channel_bounds = {}
    angular_regions = {}
    for local in {radius == 0 and local_stitched_m0 for radius, _ in groups}:
        if local and not include_local_m0:
            continue
        channel_bounds[local] = {}
        for si in valid_sources:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            source_bounds = conditional_ewald_region_bounds(
                native_bounds_px=native_bounds_px,
                source_state_index=si,
                source_latent_radius=rule.source_latent_radius,
                local_m0=local,
                **context,
            )
            # Intersected Cartesian and elastic-shell intervals may be empty.
            # Such rectangles have no event in this channel; never reverse bounds.
            source_bounds = source_bounds[source_bounds[:, 1] >= source_bounds[:, 0]]
            frozen = rule.frozen_ewald_bounds_Ainv_rad
            if frozen is not None:
                low_q, high_q, start_angle, arc_width = frozen
                relative_start = np.mod(source_bounds[:, 2] - start_angle, 2 * np.pi)
                if (
                    np.any(source_bounds[:, 0] < low_q)
                    or np.any(source_bounds[:, 1] > high_q)
                    or (
                        arc_width < 2 * np.pi
                        and np.any(relative_start + source_bounds[:, 3] > arc_width)
                    )
                ):
                    raise ValueError(
                        "frozen Ewald envelope does not enclose current source-region bounds"
                    )
                source_bounds = np.asarray([frozen])
            channel_bounds[local][si] = source_bounds
            if (
                len(source_bounds)
                and si in active_sources
                and local
                and rule.local_m0_angular_rule == "resolved_cdf_gl8.v1"
            ):
                angular_regions[local, si] = native_angular_resolution_regions(
                    native_bounds_px=native_bounds_px,
                    source_state_index=si,
                    local_m0=local,
                    resolution_fraction=rule.local_m0_angular_resolution_fraction,
                    source_latent_radius=rule.source_latent_radius,
                    maximum_backward_probability=rule.maximum_backward_probability,
                    **context,
                )
    for gi, (radius, group) in enumerate(groups):
        if all(rod.population == 0 for rod in group):
            continue
        if cancel_requested is not None and cancel_requested():
            raise CancelledError
        local = radius == 0 and local_stitched_m0
        if local and not include_local_m0:
            continue
        bounds = channel_bounds[local]
        all_bounds = np.concatenate(tuple(bounds.values()))
        if not len(all_bounds) or all_bounds[:, 1].max() <= radius:
            continue
        q_lower, q_upper = float(all_bounds[:, 0].min()), float(all_bounds[:, 1].max())
        frozen = rule.local_m0_q_bounds_Ainv if local else rule.regular_q_bounds_Ainv
        if frozen is not None:
            if frozen[0] > q_lower or frozen[1] < q_upper:
                raise ValueError(
                    "frozen Q support does not enclose the actual source-region bounds"
                )
            q_lower, q_upper = frozen
        lower = np.sqrt(max(0.0, q_lower**2 - radius**2))
        upper = np.sqrt(max(0.0, q_upper**2 - radius**2))
        spacing = b3 * (rule.local_m0_peak_spacing_L if local else 1.0)
        centers = np.arange(np.floor(lower / spacing), np.ceil(upper / spacing) + 1) * spacing
        if local and rule.local_m0_axial_peak_coordinate == "film_phase_q_first_source":
            # Historical numerical proposal: first source wavelength, unchanged full support.
            wavelength = states.wavelength_A[0]
            index = np.flatnonzero(material.wavelength_A == wavelength)
            if len(index) != 1:
                raise ValueError("proposal reference wavelength must have unique material optics")
            k0, refractive_index = 2 * np.pi / wavelength, material.n_complex[index[0]]
            if upper > 2 * k0:
                raise ValueError("historical phase-Q proposal exceeds its reference elastic sphere")

            phase_limits = [local_m0_phase_q_Ainv(q, k0, refractive_index) for q in (lower, upper)]
            targets = centers[(centers >= phase_limits[0]) & (centers <= phase_limits[1])]
            centers = np.array(
                [
                    brentq(
                        lambda q, target, k0, n: local_m0_phase_q_Ainv(q, k0, n) - target,
                        lower,
                        upper,
                        args=(target, k0, refractive_index),
                    )
                    for target in targets
                ]
            )
        else:
            centers = centers[(centers >= lower) & (centers <= upper)]
        if not len(centers):
            centers = np.array([(lower + upper) / 2])
        for si in active_sources:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            if not len(bounds[si]):
                continue
            ki = (
                (2 * np.pi / states.wavelength_A[si]) * states.direction_sample[si]
                if local
                else states.k_film_phase_sample_Ainv[si]
            )
            coordinate_parameters = dict(
                axial_bounds_Ainv=(lower, upper),
                axial_peak_centers_Ainv=centers,
                axial_peak_half_width_Ainv=b3
                * (rule.local_m0_peak_half_width_L if local else 0.02),
                radial_Ainv=radius,
                ki_sample_Ainv=ki,
                normal_sample=normal,
                source_region_bounds=bounds[si],
                reference_mosaic=reference_mosaic,
                axial_power=rule.local_m0_axial_power,
                angular_power=(
                    rule.local_m0_angular_power if local else rule.angular_initial_power
                ),
                axial_seed=(
                    rule.local_m0_seed
                    if local and rule.local_m0_angular_rule == "cdf_stratified_importance.v1"
                    else 7919 * rule.local_m0_seed + 65537 * gi + 1009
                ),
                angular_resolution_regions=angular_regions.get((local, si)),
                maximum_angular_panel_nodes=rule.maximum_angular_panel_nodes,
                batch_size=rule.batch_size,
            )
            batch_parameters = dict(
                rods=tuple(group),
                radial_Ainv=radius,
                reciprocal_basis_Ainv=basis,
                crystal_to_sample=rotation,
                source_state_index=si,
                local=local,
                rule=rule,
                scattering_cache=scattering_cache,
                include_source_mass=include_source_mass,
                **context,
            )
            if not local and rule.regular_rule == "fixed_importance.v1":
                for nodes in iter_fixed_fiber_coordinates(
                    coordinate_parameters=coordinate_parameters,
                    rule=rule,
                    source_index=si,
                    group_index=gi,
                ):
                    if cancel_requested is not None and cancel_requested():
                        raise CancelledError
                    batch = compile_conditional_fiber_batch(nodes, **batch_parameters)
                    if batch is not None:
                        yield batch
                continue
            if not local:
                yield from regular_integrator(
                    coordinate_parameters=coordinate_parameters,
                    batch_parameters=batch_parameters,
                    cancel_requested=cancel_requested,
                    channel_count=len(groups) * len(valid_sources),
                )
                continue
            for nodes in iter_local_m0_coordinates(
                **coordinate_parameters, angular_rule=rule.local_m0_angular_rule
            ):
                if cancel_requested is not None and cancel_requested():
                    raise CancelledError
                batch = compile_conditional_fiber_batch(nodes, **batch_parameters)
                if batch is not None:
                    yield batch


def compile_conditional_fiber_batch(
    nodes: FiberQuadratureNodes,
    *,
    rods,
    radial_Ainv,
    reciprocal_basis_Ainv,
    crystal_to_sample,
    source_state_index,
    local,
    rule,
    scattering_cache,
    include_source_mass,
    source,
    incident,
    material,
    instrument,
) -> ConditionalFiberBatch | None:
    """One geometry/optics/projector for local-m0 du and regular W du rules.

    None means no retained evaluated events; it does not prove an empty interval.
    """
    if not len(nodes.axial_index):
        return None
    axial = nodes.positive_axial_Ainv[nodes.axial_index]
    azimuth = nodes.ewald_azimuth_rad
    args = dict(
        source_state_index=source_state_index,
        source=source,
        incident=incident,
        material=material,
        instrument=instrument,
    )
    if local:
        coordinates = dict(external_q_Ainv=axial, ewald_azimuth_rad=azimuth)
    else:
        basis = reciprocal_basis_Ainv
        b3 = np.linalg.norm(basis[:, 2])
        normal = basis[:, 2] / b3
        rod = replace(rods[0], population=1.0)
        offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ normal
        coordinates = dict(
            rod=rod,
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=crystal_to_sample,
            L=(axial - offset) / b3,
            ewald_azimuth_rad=azimuth,
        )
    if scattering_cache is None:
        scattering = (_compile_local_scattering if local else _compile_fiber_scattering)(
            **coordinates, **args
        )
    else:
        scattering = scattering_cache.compile(local=local, coordinates=coordinates, context=args)
    transfer, local_transfer = project_conditional_fiber_transfer(
        scattering,
        source=source,
        incident=incident,
        source_state_index=source_state_index,
        instrument=instrument,
        maximum_backward_probability=rule.maximum_backward_probability,
        include_source_mass=include_source_mass,
    )
    if transfer is None:
        return None
    selected = transfer.quadrature_index
    coefficient = transfer.coefficient_per_L_rad / (1 if local else b3)
    mass = (
        nodes.weight_Ainv_rad
        if nodes.strength_weighted_mass is None
        else nodes.strength_weighted_mass
    )
    return ConditionalFiberBatch(
        source_state_index,
        rods,
        radial_Ainv,
        nodes.positive_axial_Ainv,
        nodes.axial_index[selected],
        transfer,
        coefficient * mass[selected],
        local_transfer,
        nodes.signed_strength_fractions,
    )
