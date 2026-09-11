"""Independent crystal-azimuth integration on the physical Ewald sphere.

Signed rod coordinates resolve the structure factor and remove the radial fold.
The resulting conditional position kernels are shared by fitting and rendering.
All arrays returned here are one quadrature batch, not a retained event raster.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from dataclasses import dataclass, replace

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc

from painted_ewald import MosaicParameters, Rod
from painted_ewald.validation import proper_rotation, reciprocal_basis
from rasim_next.core.contracts import MaterialOptics
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
from rasim_next.pipeline._continuous_detector_kernel import local_m0_geometry
from rasim_next.pipeline.continuous_detector import _incident_phase_shell_offset_Ainv2
from rasim_next.pipeline.source_spatial import (
    DetectorSpatialKernels,
    compile_conditional_spatial_kernels,
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
    axis, first, second = _ewald_frame(ki)
    w = offset + b3 * ell
    q = np.hypot(r, w)
    supported = (q > 0) & (q <= 2 * k)
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


def compile_conditional_fiber_transfer(
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
    maximum_backward_probability: float,
) -> FiberDetectorTransfer | None:
    """Transport one rod batch through shared optics and conditional positions.

    None means no supported forward detector-plane ray. This is the macroscopic
    planar-interface channel; the local-lamella m0 stitch keeps its own chart.
    Neither detector-panel cropping nor structure-strength pruning occurs here.
    Rod population, source mass, illuminated path and optical factors enter once.
    """
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
    plane = _intersect_detector_plane(
        np.broadcast_to(states.sample_intersection_lab_m[si], outgoing_lab.shape),
        outgoing_lab,
        instrument,
    )
    forward = plane.status == ValidityCode.VALID
    indices, selected = indices[forward], selected[forward]
    if not len(selected):
        return None
    outgoing_sample, outgoing_lab = outgoing_sample[forward], outgoing_lab[forward]
    incident_sign = -1 if states.direction_sample[si, 2] < 0 else 1
    decay = mode_decay_constant(states.kz_film_Ainv[si], incident_sign) + mode_decay_constant(
        modes.kz_film_Ainv[indices], modes.propagation_direction[indices]
    )
    attenuation = uniform_depth_attenuation(decay, 0.0, instrument.film_thickness_A)
    coefficient = measure[selected] * scalar_optical_weight(
        states.entrance_amplitude[si], modes.exit_amplitude[indices], attenuation
    )
    coefficient *= (
        rod.population
        * states.source_weight[si]
        * states.footprint_acceptance[si]
        * incident_illuminated_path_weight(states.direction_sample[si])
    )
    coefficient *= scattering_polarization_weight(
        states.direction_sample[si], outgoing_sample, model_id=states.polarization_state_id[si]
    )
    q_norm = np.linalg.norm(q[selected], axis=1)
    polar = np.arccos(np.clip(q[selected] @ (rotation @ normal) / q_norm, -1.0, 1.0))
    spatial = compile_conditional_spatial_kernels(
        instrument=instrument,
        source=source,
        source_state_index=si,
        outgoing_direction_lab=outgoing_lab,
        maximum_backward_probability=maximum_backward_probability,
    )
    return FiberDetectorTransfer(
        selected,
        polar,
        cone[selected],
        coefficient,
        spatial,
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


def compile_conditional_local_m0_transfer(
    *,
    external_q_Ainv: ArrayLike,
    ewald_azimuth_rad: ArrayLike,
    source: ConditionalSourceSamples,
    incident: IncidentTransportResult,
    source_state_index: int,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    maximum_backward_probability: float,
) -> LocalM0DetectorTransfer | None:
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
    outgoing_lab = instrument.lab_from_sample.apply_vector(outgoing[indices])
    plane = _intersect_detector_plane(
        np.broadcast_to(states.sample_intersection_lab_m[si], outgoing_lab.shape),
        outgoing_lab,
        instrument,
    )
    indices = indices[plane.status == ValidityCode.VALID]
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
    spatial = compile_conditional_spatial_kernels(
        instrument=instrument,
        source=source,
        source_state_index=si,
        outgoing_direction_lab=instrument.lab_from_sample.apply_vector(outgoing[indices]),
        maximum_backward_probability=maximum_backward_probability,
    )
    coefficient = (
        g[:, 4]
        * states.source_weight[si]
        * states.footprint_acceptance[si]
        * incident_illuminated_path_weight(states.direction_sample[si])
    )
    transfer = FiberDetectorTransfer(
        indices, g[:, 1], np.zeros(len(indices)), coefficient, spatial, g[:, 5], g[:, 6]
    )
    return LocalM0DetectorTransfer(transfer, g[:, 2], g[:, 3])


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
        for _iteration in range(70):
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
            if abs(error) < 2e-15 or hi - lo < 2e-14 * max(1.0, upper - lower):
                break
            if error > 0:
                hi = v
            else:
                lo = v
            proposed = v - error / density
            v = proposed if lo < proposed < hi else 0.5 * (lo + hi)
        x[i] = v
        pdf[i] = density
    return x, pdf


@numba.njit(nogil=True)
def _wrapped_cauchy_cdf(value, width):
    turns = np.floor((value + np.pi) / (2 * np.pi))
    reduced = value - 2 * np.pi * turns
    return (
        turns
        + 0.5
        + np.arctan2(np.sin(reduced / 2), np.tanh(width / 2) * np.cos(reduced / 2)) / np.pi
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
def _bounded_angular_quantiles(quantiles, q, bounds, centers, widths, uniform_mass=0.2):
    """Condition the complete proposal on the conservative reachable arc union.

    Empty-support axial rows retain their attempted quadrature count with zero
    contribution. The returned PDF includes the union probability normalization.
    """
    x = np.zeros(quantiles.shape)
    pdf = np.zeros(quantiles.shape)
    lower = np.empty(2 * len(bounds))
    upper = np.empty(2 * len(bounds))
    for i in range(len(q)):
        count = 0
        for row in bounds:
            if q[i] < row[0] or q[i] > row[1] or row[3] == 0:
                continue
            start = row[2] % (2 * np.pi)
            width = row[3]
            if width >= 2 * np.pi:
                lower[0] = 0.0
                upper[0] = 2 * np.pi
                count = 1
                break
            stop = start + width
            lower[count] = start
            upper[count] = min(stop, 2 * np.pi)
            count += 1
            if stop > 2 * np.pi:
                lower[count] = 0.0
                upper[count] = stop - 2 * np.pi
                count += 1
        if count == 0:
            continue
        order = np.argsort(lower[:count])
        lo = lower[:count][order]
        hi = upper[:count][order]
        merged = 1
        for j in range(1, count):
            if lo[j] <= hi[merged - 1]:
                hi[merged - 1] = max(hi[merged - 1], hi[j])
            else:
                lo[merged] = lo[j]
                hi[merged] = hi[j]
                merged += 1
        offsets = np.empty(centers.shape[1])
        for j in range(len(offsets)):
            offsets[j] = _wrapped_cauchy_cdf(-centers[i, j], widths[i, j])
        base = np.empty(merged)
        mass = np.empty(merged)
        total = 0.0
        for j in range(merged):
            base[j] = _angular_cdf_density(lo[j], centers[i], widths[i], offsets, uniform_mass)[0]
            mass[j] = (
                _angular_cdf_density(hi[j], centers[i], widths[i], offsets, uniform_mass)[0]
                - base[j]
            )
            total += mass[j]
        if total <= 0:
            raise ArithmeticError("Nonpositive reachable angular proposal mass")
        for j in range(quantiles.shape[1]):
            target = quantiles[i, j] * total
            interval = 0
            while interval < merged - 1 and target > mass[interval]:
                target -= mass[interval]
                interval += 1
            target += base[interval]
            left, right = lo[interval], hi[interval]
            v = left + (right - left) * quantiles[i, j]
            for _iteration in range(70):
                cdf, density = _angular_cdf_density(v, centers[i], widths[i], offsets, uniform_mass)
                error = cdf - target
                if abs(error) < 2e-15 or right - left < 2e-14:
                    break
                if error > 0:
                    right = v
                else:
                    left = v
                proposed = v - error / density
                v = proposed if left < proposed < right else 0.5 * (left + right)
            x[i, j] = v
            pdf[i, j] = density / total
    return x, pdf


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
    Empty geometric support contributes zero while retaining the original number
    of attempted nodes. No SF, mosaic density, optical or source mass is included.
    """

    positive_axial_Ainv: NDArray[np.float64]
    axial_index: NDArray[np.int64]
    ewald_azimuth_rad: NDArray[np.float64]
    weight_Ainv_rad: NDArray[np.float64]

    def __post_init__(self) -> None:
        index = np.asarray(self.axial_index)
        if index.dtype.kind not in "iu":
            raise TypeError("axial_index must contain integers")
        for name in ("positive_axial_Ainv", "axial_index", "ewald_azimuth_rad", "weight_Ainv_rad"):
            dtype = np.int64 if name == "axial_index" else np.float64
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.ndim != 1 or np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError("quadrature arrays must be finite nonnegative vectors")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            self.ewald_azimuth_rad.shape != self.axial_index.shape
            or self.weight_Ainv_rad.shape != self.axial_index.shape
            or np.any(self.axial_index >= len(self.positive_axial_Ainv))
            or np.any(self.ewald_azimuth_rad > 2 * np.pi)
            or np.any(self.weight_Ainv_rad <= 0)
        ):
            raise ValueError(
                "retained quadrature indices, azimuths and positive weights must align"
            )


def sample_conditional_fiber_coordinates(
    *,
    axial_bounds_Ainv: tuple[float, float],
    axial_peak_centers_Ainv: ArrayLike,
    axial_peak_half_width_Ainv: float,
    radial_Ainv: float,
    ki_sample_Ainv: ArrayLike,
    normal_sample: ArrayLike,
    source_region_bounds: ArrayLike,
    reference_mosaic: MosaicParameters,
    axial_power: int,
    angular_power: int,
    axial_seed: int,
    angular_shift_seed: int,
) -> FiberQuadratureNodes:
    """Integrate a joint continuous rod/Ewald function with a proper 2-D net.

    Proposal centers and widths improve efficiency only; their complete normalized
    mixture probabilities are removed by the returned weights. Bounds come from
    ``conditional_ewald_region_bounds`` with its explicit source-tail budget.
    Reuse axial bounds and seed across source rows to share exact SF evaluations.
    The two Sobol dimensions must remain paired; separate 1-D sequences can retain
    correlated empty cells indefinitely despite apparent sample-count convergence.
    """
    if not isinstance(reference_mosaic, MosaicParameters):
        raise TypeError("reference_mosaic must be MosaicParameters")
    lower, upper = map(float, axial_bounds_Ainv)
    centers = np.asarray(axial_peak_centers_Ainv, dtype=np.float64)
    width, radius = float(axial_peak_half_width_Ainv), float(radial_Ainv)
    ki, normal = (
        np.asarray(ki_sample_Ainv, dtype=np.float64),
        np.asarray(normal_sample, dtype=np.float64),
    )
    bounds = np.asarray(source_region_bounds, dtype=np.float64)
    if (
        not np.all(np.isfinite([lower, upper, width, radius]))
        or lower < 0
        or upper <= lower
        or width <= 0
        or radius < 0
        or centers.ndim != 1
        or not len(centers)
        or np.any(~np.isfinite(centers))
        or ki.shape != (3,)
        or np.any(~np.isfinite(ki))
        or np.linalg.norm(ki) == 0
        or normal.shape != (3,)
        or np.any(~np.isfinite(normal))
        or not np.isclose(np.linalg.norm(normal), 1.0, rtol=0, atol=1e-12)
        or bounds.ndim != 2
        or bounds.shape[1] != 4
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[:, 0] < 0)
        or np.any(bounds[:, 1] < bounds[:, 0])
        or np.any((bounds[:, 3] < 0) | (bounds[:, 3] > 2 * np.pi))
    ):
        raise ValueError(
            "finite ordered axial bounds, positive proposal width and valid Ewald bounds required"
        )
    for value in (axial_power, angular_power, axial_seed, angular_shift_seed):
        if type(value) is not int or value < 0:
            raise ValueError("quadrature powers and seeds must be nonnegative integers")
    unit = qmc.Sobol(2, scramble=True, seed=axial_seed).random_base2(axial_power)
    axial, axial_pdf = _axial_mixture_quantiles(
        unit[:, 0], lower, upper, centers, width, np.ones(len(centers))
    )
    angular_centers, angular_widths = _angular_proposal_parameters(
        axial,
        radius,
        ki,
        normal,
        reference_mosaic.gaussian_sigma_rad,
        reference_mosaic.lorentzian_half_width_rad,
    )
    angular_count = 2**angular_power
    shift = unit[:, 1] + np.random.default_rng(angular_shift_seed).random()
    angular_quantiles = (shift[:, None] + (np.arange(angular_count) + 0.5) / angular_count) % 1
    azimuth, angular_pdf = _bounded_angular_quantiles(
        angular_quantiles, np.hypot(radius, axial), bounds, angular_centers, angular_widths
    )
    axial_index = np.repeat(np.arange(len(axial)), angular_count)
    supported = angular_pdf.ravel() > 0
    axial_index = axial_index[supported]
    weights = 1 / (
        len(axial) * angular_count * axial_pdf[axial_index] * angular_pdf.ravel()[supported]
    )
    return FiberQuadratureNodes(axial, axial_index, azimuth.ravel()[supported], weights)


@dataclass(frozen=True, slots=True)
class FiberIntegrationRule:
    """Numerical proposal and integration controls, never physical peak cutoffs.

    Peak spacing and width only concentrate the normalized importance proposal.
    The uniform component retains the complete geometrically bounded domain.
    A local m0 domain reaching Q=0 requires an integrable structure/optical model,
    such as the named Parratt composite; the coordinate rule cannot supply one.
    """

    axial_power: int = 12
    angular_power: int = 5
    seed: int = 0
    axial_peak_spacing_L: float = 1.0
    axial_peak_half_width_L: float = 0.02
    source_latent_radius: float = 8.0
    maximum_backward_probability: float = 1e-12
    batch_size: int = 16384

    def __post_init__(self) -> None:
        for name in ("axial_power", "angular_power", "seed", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "batch_size" else 0):
                raise ValueError(f"{name} must be a valid nonnegative integer")
        for name in ("axial_peak_spacing_L", "axial_peak_half_width_L", "source_latent_radius"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not 0 <= self.maximum_backward_probability < 1:
            raise ValueError("maximum_backward_probability must lie in [0,1)")


@dataclass(frozen=True, slots=True)
class ConditionalFiberBatch:
    """One shared geometry batch before SF, mosaic and detector-region reduction.

    Rods remain individual physical identities. Both signed SF sheets use the
    same spatial kernels and quadrature masses; their cone angles are c and pi-c.
    ``integrated_coefficient`` contains the source and optical coefficients times
    d(axial) d(azimuth), with unit rod population. It is an integrated event
    coefficient, not a density to multiply by a second detector Jacobian.
    """

    source_state_index: int
    rods: tuple[Rod, ...]
    radial_Ainv: float
    positive_axial_Ainv: NDArray[np.float64]
    axial_index: NDArray[np.int64]
    transfer: FiberDetectorTransfer
    integrated_coefficient: NDArray[np.float64]
    local_m0: LocalM0DetectorTransfer | None = None

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
) -> Iterator[ConditionalFiberBatch]:
    """Stream the same continuous transfers for native fits and full-panel images.

    Bounds describe the requested observable. A renderer supplies its panel
    domain; it cannot reuse an ROI-pruned fit response. Ordinary unstitched m0
    uses the same planar kinematic optics as every other rod. The local air-Ewald
    chart is reserved for the explicitly requested local-lamella composite.
    Source masses are kept intact when rows or directions have no valid support.
    """
    if not isinstance(rule, FiberIntegrationRule) or type(local_stitched_m0) is not bool:
        raise TypeError("an explicit integration rule and local-composite selection are required")
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
    for local in {radius == 0 and local_stitched_m0 for radius, _ in groups}:
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
            channel_bounds[local][si] = source_bounds[source_bounds[:, 1] >= source_bounds[:, 0]]
    for gi, (radius, group) in enumerate(groups):
        if cancel_requested is not None and cancel_requested():
            raise CancelledError
        local = radius == 0 and local_stitched_m0
        bounds = channel_bounds[local]
        all_bounds = np.concatenate(tuple(bounds.values()))
        if not len(all_bounds) or all_bounds[:, 1].max() <= radius:
            continue
        lower = np.sqrt(max(0.0, all_bounds[:, 0].min() ** 2 - radius**2))
        upper = np.sqrt(max(0.0, all_bounds[:, 1].max() ** 2 - radius**2))
        spacing = b3 * rule.axial_peak_spacing_L
        centers = np.arange(np.floor(lower / spacing), np.ceil(upper / spacing) + 1) * spacing
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
            nodes = sample_conditional_fiber_coordinates(
                axial_bounds_Ainv=(lower, upper),
                axial_peak_centers_Ainv=centers,
                axial_peak_half_width_Ainv=b3 * rule.axial_peak_half_width_L,
                radial_Ainv=radius,
                ki_sample_Ainv=ki,
                normal_sample=normal,
                source_region_bounds=bounds[si],
                reference_mosaic=reference_mosaic,
                axial_power=rule.axial_power,
                angular_power=rule.angular_power,
                axial_seed=7919 * rule.seed + 65537 * gi + 1009,
                angular_shift_seed=8191 * si + 7919 * rule.seed + 131 * gi + 973,
            )
            for first in range(0, len(nodes.axial_index), rule.batch_size):
                if cancel_requested is not None and cancel_requested():
                    raise CancelledError
                stop = first + rule.batch_size
                axial_index = nodes.axial_index[first:stop]
                axial = nodes.positive_axial_Ainv[axial_index]
                azimuth = nodes.ewald_azimuth_rad[first:stop]
                args = dict(
                    source_state_index=si,
                    maximum_backward_probability=rule.maximum_backward_probability,
                    **context,
                )
                local_transfer = None
                if local:
                    local_transfer = compile_conditional_local_m0_transfer(
                        external_q_Ainv=axial,
                        ewald_azimuth_rad=azimuth,
                        **args,
                    )
                    transfer = None if local_transfer is None else local_transfer.transfer
                else:
                    rod = replace(group[0], population=1.0)
                    offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ crystal_normal
                    transfer = compile_conditional_fiber_transfer(
                        rod=rod,
                        reciprocal_basis_Ainv=basis,
                        crystal_to_sample=rotation,
                        L=(axial - offset) / b3,
                        ewald_azimuth_rad=azimuth,
                        **args,
                    )
                if transfer is None:
                    continue
                selected = transfer.quadrature_index
                coefficient = transfer.coefficient_per_L_rad / (1 if local else b3)
                yield ConditionalFiberBatch(
                    si,
                    tuple(group),
                    radius,
                    nodes.positive_axial_Ainv,
                    axial_index[selected],
                    transfer,
                    coefficient * nodes.weight_Ainv_rad[first:stop][selected],
                    local_transfer,
                )
