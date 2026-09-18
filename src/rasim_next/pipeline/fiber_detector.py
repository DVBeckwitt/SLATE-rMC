"""Independent crystal-azimuth integration on the physical Ewald sphere.

Signed rod coordinates resolve the structure factor and remove the radial fold.
The resulting conditional position kernels are shared by fitting and rendering.
All arrays returned here are one quadrature batch, not a retained event raster.
"""

from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from time import perf_counter

import numba
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_legendre
from scipy.stats import qmc

from painted_ewald import MosaicParameters, Rod
from painted_ewald.validation import proper_rotation, reciprocal_basis
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


def seed_angular_panel_edges(
    source_region_bounds: ArrayLike, *, maximum_panel_width_rad: float, maximum_panels: int
) -> tuple[float, ...]:
    """Seed a fixed physical-angle partition from complete native geometry bounds.

    Supply the union of bounds over every required observation/source/probe.
    All geometry endpoints remain knots, including boundaries inside overlapping
    support arcs. Gaps outside those arcs remain panels, not omitted support.
    The width cap is an explicit accuracy control, not a convergence certificate.
    """
    bounds = np.asarray(source_region_bounds)
    width = maximum_panel_width_rad
    if (
        bounds.ndim != 2
        or bounds.shape[1] != 4
        or np.iscomplexobj(bounds)
        or np.any(~np.isfinite(bounds))
        or np.any(bounds[:, 0] < 0)
        or np.any(bounds[:, 3] < 0)
        or np.any(bounds[:, 3] > 2 * np.pi)
        or np.ndim(width) != 0
        or np.iscomplexobj(width)
        or not np.isfinite(width)
        or width <= 0
        or type(maximum_panels) is not int
        or maximum_panels < 1
    ):
        raise ValueError(
            "angular seeding requires finite geometry, positive width and panel budget"
        )
    bounds = bounds[(bounds[:, 1] >= bounds[:, 0]) & (bounds[:, 3] > 0)].astype(float)
    bounds[:, :2] = [0.0, 1.0]
    start = bounds[:, 2] % (2 * np.pi)
    stop = start + bounds[:, 3]
    cuts = np.unique(
        np.r_[
            0.0, 2 * np.pi, start, np.minimum(stop, 2 * np.pi), stop[stop > 2 * np.pi] - 2 * np.pi
        ]
    )
    lo, hi = _angular_intervals(0.0, bounds)
    lengths = np.diff(cuts)
    midpoint = cuts[:-1] + lengths / 2
    inside = np.zeros(len(midpoint), dtype=bool)
    for left, right in zip(lo, hi, strict=True):
        inside |= (midpoint >= left) & (midpoint <= right)
    counts = np.maximum(1.0, np.where(inside, np.ceil(lengths / width), 1.0))
    if np.sum(counts) > maximum_panels:
        raise ValueError("angular panel budget exceeded; no mesh or accuracy fallback returned")
    edges = np.r_[
        np.concatenate(
            [
                np.linspace(left, right, int(count) + 1)[:-1]
                for left, right, count in zip(cuts[:-1], cuts[1:], counts, strict=True)
            ]
        ),
        2 * np.pi,
    ]
    if np.any(np.diff(edges) <= 0):
        raise ValueError("requested angular panel width is below floating-point resolution")
    return tuple(edges)


def _validated_angular_edges(edges, quadrature_kind):
    if edges is None:
        return None
    value = np.asarray(edges)
    if (
        quadrature_kind != "composite_gauss"
        or np.iscomplexobj(value)
        or value.ndim != 1
        or len(value) < 2
        or np.any(~np.isfinite(value))
        or value[0] != 0
        or value[-1] != 2 * np.pi
        or np.any(np.diff(value) <= 0)
    ):
        raise ValueError(
            "angular panels require increasing finite [0,2*pi] edges and composite_gauss"
        )
    return np.asarray(value, dtype=float)


@numba.njit(nogil=True)
def _angular_panel_node_count(q, bounds, edges, order, maximum_nodes):
    count = 0
    for coordinate in q:
        lo, hi = _angular_intervals(coordinate, bounds)
        for j in range(len(lo)):
            left, right = lo[j], hi[j]
            count += (
                np.searchsorted(edges, right, side="left")
                - np.searchsorted(edges, left, side="right")
                + 1
            ) * order
            if count > maximum_nodes:
                raise ValueError("angular panel node budget exceeded before allocation")
    return count


@numba.njit(nogil=True)
def _physical_angular_panels(q, bounds, edges, nodes, weights, count):
    axial_index = np.empty(count, dtype=np.int64)
    phi, mass = np.empty(count), np.empty(count)
    count = 0
    for i, coordinate in enumerate(q):
        lo, hi = _angular_intervals(coordinate, bounds)
        for arc in range(len(lo)):
            left, right = lo[arc], hi[arc]
            first = np.searchsorted(edges, left, side="right")
            stop = np.searchsorted(edges, right, side="left")
            for j in range(first, stop + 1):
                end = right if j == stop else edges[j]
                half = (end - left) / 2
                for k in range(len(nodes)):
                    axial_index[count] = i
                    phi[count] = left + half * (nodes[k] + 1)
                    mass[count] = half * weights[k]
                    count += 1
                left = end
    return axial_index, phi, mass


@numba.njit(nogil=True)
def _bounded_angular_quantiles(
    quantiles, q, bounds, centers, widths, split_arcs=False, uniform_mass=0.2
):
    """Condition the complete proposal on the conservative reachable arc union.

    Empty-support axial rows retain their attempted quadrature count with zero
    contribution. The returned PDF includes the union probability normalization.
    """
    axial_indices = []
    angular_indices = []
    angles = []
    densities = []
    fractions = []
    for i in range(len(q)):
        lo, hi = _angular_intervals(q[i], bounds)
        merged = len(lo)
        if merged == 0:
            continue
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
            for arc in range(merged if split_arcs else 1):
                if split_arcs:
                    interval = arc
                    target = quantiles[i, j] * mass[interval]
                    fraction = mass[interval] / total
                else:
                    target = quantiles[i, j] * total
                    interval = 0
                    while interval < merged - 1 and target > mass[interval]:
                        target -= mass[interval]
                        interval += 1
                    fraction = 1.0
                if mass[interval] <= 0:
                    continue
                target += base[interval]
                left, right = lo[interval], hi[interval]
                v = left + (right - left) * quantiles[i, j]
                # Keep the Newton fast path, then guarantee bracket contraction.
                # Return only a tested coordinate and the density evaluated there.
                for _iteration in range(150):
                    cdf, density = _angular_cdf_density(
                        v, centers[i], widths[i], offsets, uniform_mass
                    )
                    error = cdf - target
                    if not np.isfinite(error) or not np.isfinite(density) or density <= 0:
                        raise ValueError("angular inverse CDF requires a finite positive density")
                    if abs(error) < 2e-15 or right - left < 2e-14:
                        break
                    if error > 0:
                        right = v
                    else:
                        left = v
                    proposed = v - error / density if _iteration < 70 else 0.5 * (left + right)
                    v = proposed if left < proposed < right else 0.5 * (left + right)
                else:
                    raise ValueError(
                        "angular inverse CDF did not converge within its iteration budget"
                    )
                axial_indices.append(i)
                angular_indices.append(j)
                angles.append(v)
                densities.append(density / total)
                fractions.append(fraction)
    return (
        np.array(axial_indices),
        np.array(angular_indices),
        np.array(angles),
        np.array(densities),
        np.array(fractions),
    )


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
    quadrature_kind: str = "sobol",
    maximum_axial_panel_width_Ainv: float | None = None,
    angular_support: str = "q_conditioned_union",
    axial_panel_edges_Ainv: ArrayLike | None = None,
    angular_panel_edges_rad: ArrayLike | None = None,
    maximum_angular_panel_nodes: int = 4194304,
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
    if angular_support not in {"q_conditioned_union", "fixed_union"}:
        raise ValueError("angular support must be q_conditioned_union or fixed_union")
    angular_edges = _validated_angular_edges(angular_panel_edges_rad, quadrature_kind)
    if type(maximum_angular_panel_nodes) is not int or maximum_angular_panel_nodes < 1:
        raise ValueError("angular panel node budget must be a positive integer")
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
    if angular_edges is not None and 2**angular_power > maximum_angular_panel_nodes:
        raise ValueError("angular panel node budget exceeded by per-panel order")
    if maximum_axial_panel_width_Ainv is not None and (
        quadrature_kind != "composite_gauss"
        or np.ndim(maximum_axial_panel_width_Ainv) != 0
        or np.iscomplexobj(maximum_axial_panel_width_Ainv)
        or not np.isfinite(maximum_axial_panel_width_Ainv)
        or maximum_axial_panel_width_Ainv <= 0
    ):
        raise ValueError("a finite positive axial panel width requires composite_gauss")
    if axial_panel_edges_Ainv is not None:
        supplied = np.asarray(axial_panel_edges_Ainv)
        if (
            quadrature_kind != "composite_gauss"
            or np.iscomplexobj(supplied)
            or supplied.ndim != 1
            or len(supplied) < 2
            or np.any(~np.isfinite(supplied))
            or np.any(np.diff(supplied) <= 0)
            or supplied[0] < 0
            or supplied[0] != lower
            or supplied[-1] != upper
        ):
            raise ValueError(
                "physical axial panels must enclose exactly the domain and require composite_gauss"
            )
        if maximum_axial_panel_width_Ainv is not None:
            raise ValueError("explicit axial panels cannot also request a panel-width cap")
        edges = np.asarray(supplied, dtype=float)
        axial_nodes, axial_weights = roots_legendre(2 ** min(axial_power, 3))
        half = np.diff(edges) / 2
        axial = (edges[:-1, None] + half[:, None] * (axial_nodes + 1)).ravel()
        axial_pdf = np.ones(len(axial))
        if angular_edges is not None:
            axial_mass = (half[:, None] * axial_weights).ravel()
        else:
            angular_nodes, angular_weights = roots_legendre(2**angular_power)
            angular_quantiles = np.broadcast_to(
                (angular_nodes + 1) / 2, (len(axial), len(angular_nodes))
            ).copy()
            probability_weights = (
                (half[:, None] * axial_weights).ravel()[:, None] * angular_weights / 2
            )
    elif quadrature_kind == "sobol":
        unit = qmc.Sobol(2, scramble=True, seed=axial_seed).random_base2(axial_power)
        axial_quantiles = unit[:, 0]
        shift = unit[:, 1] + np.random.default_rng(angular_shift_seed).random()
        angular_count = 2**angular_power
        angular_quantiles = (shift[:, None] + (np.arange(angular_count) + 0.5) / angular_count) % 1
        probability_weights = np.full(angular_quantiles.shape, 1 / (len(unit) * angular_count))
    elif quadrature_kind == "composite_gauss":
        order = 2 ** min(axial_power, 3)
        panels = 2**axial_power // order
        axial_nodes, axial_weights = roots_legendre(order)
        edges = np.linspace(0.0, 1.0, panels + 1)
        if maximum_axial_panel_width_Ainv is not None:
            while True:
                physical_edges, _ = _axial_mixture_quantiles(
                    edges, lower, upper, centers, width, np.ones(len(centers))
                )
                split = np.diff(physical_edges) > maximum_axial_panel_width_Ainv
                if not np.any(split):
                    break
                middle = (edges[:-1][split] + edges[1:][split]) / 2
                if np.any((middle <= edges[:-1][split]) | (middle >= edges[1:][split])):
                    raise ValueError(
                        "requested axial panel width is below floating-point resolution"
                    )
                edges = np.sort(np.concatenate((edges, middle)))
        panel_mass = np.diff(edges)
        axial_quantiles = (edges[:-1, None] + panel_mass[:, None] * (axial_nodes + 1) / 2).ravel()
        axial_weights = (panel_mass[:, None] * axial_weights).ravel()
        if angular_edges is not None:
            axial_mass = axial_weights / 2
        else:
            angular_nodes, angular_weights = roots_legendre(2**angular_power)
            angular_count = len(angular_nodes)
            angular_quantiles = np.broadcast_to(
                (angular_nodes + 1) / 2, (len(axial_quantiles), angular_count)
            ).copy()
            probability_weights = axial_weights[:, None] * angular_weights[None, :] / 4
    else:
        raise ValueError("quadrature kind must be sobol or composite_gauss")
    if axial_panel_edges_Ainv is None:
        axial, axial_pdf = _axial_mixture_quantiles(
            axial_quantiles, lower, upper, centers, width, np.ones(len(centers))
        )
    if angular_support == "fixed_union" and len(bounds):
        # Preserve one conservative angular union throughout the source's Q range.
        # Activating an observation must not remesh another observation's integral.
        bounds = bounds.copy()
        bounds[:, 0] = np.min(bounds[:, 0])
        bounds[:, 1] = np.max(bounds[:, 1])
    if angular_edges is not None:
        q = np.hypot(radius, axial)
        count = _angular_panel_node_count(
            q, bounds, angular_edges, 2**angular_power, maximum_angular_panel_nodes
        )
        if count == 0:
            return FiberQuadratureNodes(
                axial, np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
            )
        angular_nodes, angular_weights = roots_legendre(2**angular_power)
        index, phi, angular_mass = _physical_angular_panels(
            q,
            bounds,
            angular_edges,
            angular_nodes,
            angular_weights,
            count,
        )
        axial_mass = axial_mass / axial_pdf
        return FiberQuadratureNodes(axial, index, phi, axial_mass[index] * angular_mass)
    angular_centers, angular_widths = _angular_proposal_parameters(
        axial,
        radius,
        ki,
        normal,
        reference_mosaic.gaussian_sigma_rad,
        reference_mosaic.lorentzian_half_width_rad,
    )
    axial_index, angular_index, azimuth, angular_pdf, fractions = _bounded_angular_quantiles(
        angular_quantiles,
        np.hypot(radius, axial),
        bounds,
        angular_centers,
        angular_widths,
        quadrature_kind == "composite_gauss",
    )
    weights = (
        probability_weights[axial_index, angular_index]
        * fractions
        / (axial_pdf[axial_index] * angular_pdf)
    )
    return FiberQuadratureNodes(axial, axial_index, azimuth, weights)


@dataclass(frozen=True, slots=True)
class AxialPanelMesh:
    """Fixed physical panels for an explicit complete radial rod group.

    Local-lamella m0 uses external |Q|; regular rods use positive internal
    axial momentum. The mesh never declares intensity or detector symmetry.
    """

    rods_hk: tuple[tuple[int, int], ...]
    coordinate: str
    edges_Ainv: tuple[float, ...]

    def __post_init__(self):
        rods = tuple(sorted(tuple(row) for row in self.rods_hk))
        if (
            not rods
            or len(set(rods)) != len(rods)
            or any(len(row) != 2 or any(type(v) is not int for v in row) for row in rods)
        ):
            raise ValueError("axial mesh requires unique integer (h,k) rods")
        if self.coordinate not in {"positive_phase_axial", "external_local_m0_q"}:
            raise ValueError("axial mesh coordinate must name its physical measure")
        if self.coordinate == "external_local_m0_q" and rods != ((0, 0),):
            raise ValueError("external local-m0 mesh requires only the zero rod")
        supplied = np.asarray(self.edges_Ainv)
        if (
            np.iscomplexobj(supplied)
            or supplied.ndim != 1
            or len(supplied) < 2
            or np.any(~np.isfinite(supplied))
            or supplied[0] < 0
            or np.any(np.diff(supplied) <= 0)
        ):
            raise ValueError("axial mesh edges must be finite, nonnegative and strictly increasing")
        object.__setattr__(self, "rods_hk", rods)
        object.__setattr__(self, "edges_Ainv", tuple(float(v) for v in supplied))


@dataclass(frozen=True, slots=True)
class FiberIntegrationRule:
    """Numerical proposal and integration controls, never physical peak cutoffs.

    Peak spacing and width only concentrate the normalized importance proposal.
    The uniform component retains the complete geometrically bounded domain.
    A local m0 domain reaching Q=0 requires an integrable structure/optical model,
    such as the named Parratt composite; the coordinate rule cannot supply one.
    Local-m0 angular order and panel cap override their global values only in
    that channel; None inherits them. The physical integration measure is retained.
    """

    axial_power: int = 12
    angular_power: int = 5
    seed: int = 0
    axial_peak_spacing_L: float = 1.0
    axial_peak_half_width_L: float = 0.02
    source_latent_radius: float = 8.0
    maximum_backward_probability: float = 1e-12
    batch_size: int = 16384
    cone_quadrature_order: int = 16
    stitch_grid_size: int = 513
    regular_q_bounds_Ainv: tuple[float, float] | None = None
    local_m0_q_bounds_Ainv: tuple[float, float] | None = None
    quadrature_kind: str = "sobol"
    maximum_axial_panel_width_Ainv: float | None = None
    local_m0_maximum_axial_panel_width_Ainv: float | None = None
    local_m0_angular_power: int | None = None
    angular_support: str = "q_conditioned_union"
    frozen_ewald_bounds_Ainv_rad: tuple[float, float, float, float] | None = None
    axial_meshes: tuple[AxialPanelMesh, ...] = ()
    angular_panel_edges_rad: tuple[float, ...] | None = None
    maximum_angular_panel_nodes: int = 4194304

    def __post_init__(self) -> None:
        angular_edges = _validated_angular_edges(self.angular_panel_edges_rad, self.quadrature_kind)
        if angular_edges is not None:
            object.__setattr__(self, "angular_panel_edges_rad", tuple(angular_edges))
        if (
            type(self.maximum_angular_panel_nodes) is not int
            or self.maximum_angular_panel_nodes < 1
        ):
            raise ValueError("angular panel node budget must be a positive integer")
        meshes = tuple(AxialPanelMesh(**m) if isinstance(m, dict) else m for m in self.axial_meshes)
        if any(not isinstance(m, AxialPanelMesh) for m in meshes):
            raise TypeError("axial_meshes must contain declared AxialPanelMesh values")
        if meshes and (self.quadrature_kind != "composite_gauss" or self.axial_power != 3):
            raise ValueError(
                "frozen axial meshes require composite_gauss and axial_power=3; refine mesh edges instead"
            )
        if meshes and (
            self.maximum_axial_panel_width_Ainv is not None
            or self.local_m0_maximum_axial_panel_width_Ainv is not None
        ):
            raise ValueError("explicit axial meshes cannot also request panel-width caps")
        rods = [hk for mesh in meshes for hk in mesh.rods_hk]
        if len(set(rods)) != len(rods):
            raise ValueError("axial meshes must have disjoint rod groups")
        object.__setattr__(self, "axial_meshes", meshes)
        if self.local_m0_angular_power is not None and (
            type(self.local_m0_angular_power) is not int or self.local_m0_angular_power < 0
        ):
            raise ValueError("local_m0_angular_power must be a nonnegative integer or None")
        if self.frozen_ewald_bounds_Ainv_rad is not None:
            bounds = tuple(float(v) for v in self.frozen_ewald_bounds_Ainv_rad)
            if (
                len(bounds) != 4
                or not np.all(np.isfinite(bounds))
                or not 0 <= bounds[0] < bounds[1]
                or not 0 < bounds[3] <= 2 * np.pi
            ):
                raise ValueError(
                    "frozen Ewald envelope requires ordered Q and a positive angular arc"
                )
            object.__setattr__(self, "frozen_ewald_bounds_Ainv_rad", bounds)
        if self.angular_support not in {"q_conditioned_union", "fixed_union"}:
            raise ValueError("angular support must be q_conditioned_union or fixed_union")
        if self.quadrature_kind not in {"sobol", "composite_gauss"}:
            raise ValueError("quadrature kind must be sobol or composite_gauss")
        if self.quadrature_kind == "composite_gauss" and self.seed != 0:
            raise ValueError("Gauss-Legendre has no random seed; use order refinement")
        for name in (
            "maximum_axial_panel_width_Ainv",
            "local_m0_maximum_axial_panel_width_Ainv",
        ):
            value = getattr(self, name)
            if value is not None:
                if (
                    self.quadrature_kind != "composite_gauss"
                    or np.ndim(value) != 0
                    or np.iscomplexobj(value)
                    or not np.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("a finite positive axial panel width requires composite_gauss")
                object.__setattr__(self, name, float(value))
        for name in ("regular_q_bounds_Ainv", "local_m0_q_bounds_Ainv"):
            value = getattr(self, name)
            if value is not None:
                bounds = tuple(float(v) for v in value)
                if (
                    len(bounds) != 2
                    or not np.all(np.isfinite(bounds))
                    or not 0 <= bounds[0] < bounds[1]
                ):
                    raise ValueError("frozen Q bounds must be a finite increasing nonnegative pair")
                object.__setattr__(self, name, bounds)
        if type(self.stitch_grid_size) is not int or self.stitch_grid_size < 257:
            raise ValueError("stitch grid size must be an integer of at least 257")
        if type(self.cone_quadrature_order) is not int or self.cone_quadrature_order < 4:
            raise ValueError("cone quadrature order must be an integer of at least four")
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
    scattering_cache: FiberScatteringCache | None = None,
    include_source_mass: bool = True,
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
    if rule.axial_meshes:
        for radius, group in groups:
            group_key = tuple(sorted((r.h, r.k) for r in group))
            coordinate = (
                "external_local_m0_q"
                if radius == 0 and local_stitched_m0
                else "positive_phase_axial"
            )
            matches = [
                m
                for m in rule.axial_meshes
                if m.rods_hk == group_key and m.coordinate == coordinate
            ]
            if len(matches) != 1:
                raise ValueError(
                    "axial meshes must cover every complete current rod group and coordinate"
                )
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
    for gi, (radius, group) in enumerate(groups):
        if cancel_requested is not None and cancel_requested():
            raise CancelledError
        local = radius == 0 and local_stitched_m0
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
        spacing = b3 * rule.axial_peak_spacing_L
        centers = np.arange(np.floor(lower / spacing), np.ceil(upper / spacing) + 1) * spacing
        centers = centers[(centers >= lower) & (centers <= upper)]
        if not len(centers):
            centers = np.array([(lower + upper) / 2])
        group_key = tuple(sorted((r.h, r.k) for r in group))
        meshes = [m for m in rule.axial_meshes if any(hk in group_key for hk in m.rods_hk)]
        mesh = meshes[0] if meshes else None
        if mesh is not None and (
            len(meshes) != 1
            or mesh.rods_hk != group_key
            or mesh.coordinate != ("external_local_m0_q" if local else "positive_phase_axial")
        ):
            raise ValueError("axial mesh must match the complete rod group and coordinate")
        if mesh is not None:
            if mesh.edges_Ainv[0] > lower or mesh.edges_Ainv[-1] < upper:
                raise ValueError("frozen axial mesh does not enclose current source-region support")
            lower, upper = mesh.edges_Ainv[0], mesh.edges_Ainv[-1]
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
                angular_power=(
                    rule.local_m0_angular_power
                    if local and rule.local_m0_angular_power is not None
                    else rule.angular_power
                ),
                axial_seed=7919 * rule.seed + 65537 * gi + 1009,
                angular_shift_seed=8191 * si + 7919 * rule.seed + 131 * gi + 973,
                quadrature_kind=rule.quadrature_kind,
                maximum_axial_panel_width_Ainv=(
                    rule.local_m0_maximum_axial_panel_width_Ainv
                    if local and rule.local_m0_maximum_axial_panel_width_Ainv is not None
                    else rule.maximum_axial_panel_width_Ainv
                ),
                angular_support=rule.angular_support,
                axial_panel_edges_Ainv=None if mesh is None else mesh.edges_Ainv,
                angular_panel_edges_rad=rule.angular_panel_edges_rad,
                maximum_angular_panel_nodes=rule.maximum_angular_panel_nodes,
            )
            for first in range(0, len(nodes.axial_index), rule.batch_size):
                if cancel_requested is not None and cancel_requested():
                    raise CancelledError
                stop = first + rule.batch_size
                axial_index = nodes.axial_index[first:stop]
                axial = nodes.positive_axial_Ainv[axial_index]
                azimuth = nodes.ewald_azimuth_rad[first:stop]
                args = dict(source_state_index=si, **context)
                if local:
                    coordinates = dict(external_q_Ainv=axial, ewald_azimuth_rad=azimuth)
                else:
                    rod = replace(group[0], population=1.0)
                    offset = (rod.h * basis[:, 0] + rod.k * basis[:, 1]) @ crystal_normal
                    coordinates = dict(
                        rod=rod,
                        reciprocal_basis_Ainv=basis,
                        crystal_to_sample=rotation,
                        L=(axial - offset) / b3,
                        ewald_azimuth_rad=azimuth,
                    )
                if scattering_cache is None:
                    scattering = (
                        _compile_local_scattering if local else _compile_fiber_scattering
                    )(**coordinates, **args)
                else:
                    scattering = scattering_cache.compile(
                        local=local, coordinates=coordinates, context=args
                    )
                transfer, local_transfer = project_conditional_fiber_transfer(
                    scattering,
                    source=source,
                    incident=incident,
                    source_state_index=si,
                    instrument=instrument,
                    maximum_backward_probability=rule.maximum_backward_probability,
                    include_source_mass=include_source_mass,
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
