"""Position-prior-free detector peak discovery and reciprocal-space indexing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter, minimum_filter

from painted_ewald import Rod
from rasim_next.fitting import IntegerLMarkerKey, IntegerLMarkerObservations
from rasim_next.geometry import (
    AngleFrame,
    CompiledInstrument,
    angles_to_detector_coordinates,
    detector_coordinates_to_angles,
)
from rasim_next.pipeline.configured_simulation import (
    GeometryOnlyEwaldContext,
    NominalEwaldContext,
    solve_integer_l_ewald_roots,
)
from rasim_next.selection.indexing import (
    MarkerIndexingDecision,
    MarkerIndexingStatus,
    MeasuredImageIndexingResult,
    PeakIndexingPolicy,
    _array_hash,
    _Candidate,
    _finite,
    _float_token,
    _is_sha256_identifier,
    _merge_candidates,
    _native_refinement,
    _positive,
    _positive_integer,
    _robust_sigma,
    _sha256_payload,
)

FloatArray = NDArray[np.float64]
type _GeometryIndexingContext = NominalEwaldContext | GeometryOnlyEwaldContext


def _circular_distance(first: float, second: float) -> float:
    return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def _array_payload(value: ArrayLike) -> tuple[tuple[int, ...], tuple[str, ...]]:
    array = np.asarray(value, dtype=np.float64)
    return array.shape, tuple(float(item).hex() for item in array.flat)


@dataclass(frozen=True, slots=True)
class BlindIndexingPolicy:
    """Frozen gates for discovery that receives no marker positions.

    ``uncertainty_sigma`` is an operational multiplier on the localization
    weight, not a claim of calibrated probabilistic coverage.
    """

    track_policy: PeakIndexingPolicy = field(default_factory=PeakIndexingPolicy)
    maximum_candidate_count: int = 512
    sampling_tile_phi_count: int = 256
    minimum_scattering_radius_px: float = 12.0
    maximum_family_residual_Ainv: float = 0.15
    separation_fraction: float = 0.25
    uncertainty_sigma: float = 3.0
    maximum_anchor_distance_px: float = 32.0
    revision: str = "global-search-cake.q-space-integer-l.v2"

    def __post_init__(self) -> None:
        if not isinstance(self.track_policy, PeakIndexingPolicy):
            raise TypeError("track_policy must be a PeakIndexingPolicy")
        for name in ("maximum_candidate_count", "sampling_tile_phi_count"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        radius = _finite(self.minimum_scattering_radius_px, "minimum_scattering_radius_px")
        if radius < 0.0:
            raise ValueError("minimum_scattering_radius_px must be nonnegative")
        object.__setattr__(self, "minimum_scattering_radius_px", radius)
        for name in (
            "maximum_family_residual_Ainv",
            "uncertainty_sigma",
            "maximum_anchor_distance_px",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.maximum_anchor_distance_px > 32.0:
            raise ValueError("maximum_anchor_distance_px must not exceed 32")
        fraction = _finite(self.separation_fraction, "separation_fraction")
        if not 0.0 < fraction <= 0.25:
            raise ValueError("separation_fraction must lie in (0, 0.25]")
        object.__setattr__(self, "separation_fraction", fraction)
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a nonempty string")


@dataclass(frozen=True, slots=True)
class DiscoveredCakePeak:
    """One globally discovered peak, refined in detector-native coordinates.

    The two positive-definite matrices are conservative ridge-support and
    localization weights; neither is a calibrated statistical covariance.
    """

    column_px: float
    row_px: float
    two_theta_rad: float
    phi_rad: float
    covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    localization_covariance_px2: tuple[tuple[float, float], tuple[float, float]]
    z_score: float

    def __post_init__(self) -> None:
        for name in ("column_px", "row_px", "two_theta_rad", "phi_rad", "z_score"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.z_score < 0.0:
            raise ValueError("z_score must be nonnegative")
        for name in ("covariance_px2", "localization_covariance_px2"):
            covariance = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                covariance.shape != (2, 2)
                or not np.all(np.isfinite(covariance))
                or not np.array_equal(covariance, covariance.T)
                or np.any(np.linalg.eigvalsh(covariance) <= 0.0)
            ):
                raise ValueError(f"{name} must be finite, symmetric, and positive definite")
            object.__setattr__(
                self,
                name,
                tuple(tuple(float(item) for item in row) for row in covariance),
            )


def _peak_payload(peak: DiscoveredCakePeak) -> tuple[object, ...]:
    return (
        _float_token(peak.column_px),
        _float_token(peak.row_px),
        _float_token(peak.two_theta_rad),
        _float_token(peak.phi_rad),
        tuple(tuple(_float_token(item) for item in row) for row in peak.covariance_px2),
        tuple(
            tuple(_float_token(item) for item in row) for row in peak.localization_covariance_px2
        ),
        _float_token(peak.z_score),
    )


@dataclass(frozen=True, slots=True)
class MeasuredPeakDiscovery:
    """Hashed global cake-search output produced without a marker catalogue."""

    image_id: str
    detector_shape_rc: tuple[int, int]
    peaks: tuple[DiscoveredCakePeak, ...]
    detector_data_hash: str
    detector_mask_hash: str
    detector_mask_revision: str
    geometry_context_hash: str
    policy: BlindIndexingPolicy
    discovery_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError("image_id must be a nonempty string")
        detector_shape = tuple(self.detector_shape_rc)
        if len(detector_shape) != 2 or any(
            isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item < 1
            for item in detector_shape
        ):
            raise ValueError("detector_shape_rc must contain two positive integers")
        object.__setattr__(self, "detector_shape_rc", tuple(int(item) for item in detector_shape))
        peaks = tuple(sorted(self.peaks, key=lambda item: (item.row_px, item.column_px)))
        if any(not isinstance(item, DiscoveredCakePeak) for item in peaks):
            raise TypeError("peaks must contain DiscoveredCakePeak values")
        if len({(item.column_px, item.row_px) for item in peaks}) != len(peaks):
            raise ValueError("peaks must have unique detector-native coordinates")
        for name in ("detector_data_hash", "detector_mask_hash", "geometry_context_hash"):
            value = getattr(self, name)
            if not _is_sha256_identifier(value):
                raise ValueError(f"{name} must be a SHA-256 identifier")
        if (
            not isinstance(self.detector_mask_revision, str)
            or not self.detector_mask_revision.strip()
        ):
            raise ValueError("detector_mask_revision must be a nonempty string")
        if not isinstance(self.policy, BlindIndexingPolicy):
            raise TypeError("policy must be a BlindIndexingPolicy")
        object.__setattr__(self, "peaks", peaks)
        payload = {
            "schema": "measured-position-free-peak-discovery.v1",
            "image_id": self.image_id,
            "detector_shape_rc": self.detector_shape_rc,
            "detector_data_hash": self.detector_data_hash,
            "detector_mask_hash": self.detector_mask_hash,
            "detector_mask_revision": self.detector_mask_revision,
            "geometry_context_hash": self.geometry_context_hash,
            "policy": asdict(self.policy),
            "peaks": tuple(_peak_payload(item) for item in peaks),
        }
        object.__setattr__(self, "discovery_hash", _sha256_payload(payload))


def _discovery_geometry_hash(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
) -> str:
    payload = {
        "schema": "global-cake-discovery-geometry.v1",
        "detector_shape_rc": instrument.detector_shape_rc,
        "detector_row_pitch_m": _float_token(instrument.detector_row_pitch_m),
        "detector_column_pitch_m": _float_token(instrument.detector_column_pitch_m),
        "detector_reference_coordinate_px": tuple(
            _float_token(item) for item in instrument.detector_reference_coordinate_px
        ),
        "lab_from_detector_rotation": _array_payload(instrument.lab_from_detector.rotation),
        "lab_from_detector_translation_m": _array_payload(
            instrument.lab_from_detector.translation_m
        ),
        "angle_origin_lab_m": _array_payload(angle_frame.origin_lab_m),
        "angle_row_down_lab": _array_payload(angle_frame.row_down_lab),
        "angle_column_right_lab": _array_payload(angle_frame.column_right_lab),
        "angle_direct_beam_lab": _array_payload(angle_frame.direct_beam_lab),
        "angle_revision": angle_frame.revision,
    }
    return _sha256_payload(payload)


def _perimeter_coordinates(
    shape_rc: tuple[int, int], count: int = 256
) -> tuple[FloatArray, FloatArray]:
    rows, columns = shape_rc
    horizontal = np.linspace(-0.5, columns - 0.5, count, dtype=np.float64)
    vertical = np.linspace(-0.5, rows - 0.5, count, dtype=np.float64)
    column = np.concatenate(
        (horizontal, horizontal, np.full(count, -0.5), np.full(count, columns - 0.5))
    )
    row = np.concatenate((np.full(count, -0.5), np.full(count, rows - 0.5), vertical, vertical))
    return column, row


def _cake_axes(
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    cake_step_px: float,
) -> tuple[FloatArray, FloatArray]:
    perimeter_column, perimeter_row = _perimeter_coordinates(instrument.detector_shape_rc)
    perimeter_angles = detector_coordinates_to_angles(
        perimeter_column,
        perimeter_row,
        instrument=instrument,
        angle_frame=angle_frame,
    )
    valid = perimeter_angles.valid
    if not np.any(valid):
        raise ValueError("the detector perimeter has no valid scattering-angle support")
    maximum_two_theta = float(np.max(perimeter_angles.two_theta_rad[valid]))
    direct = angles_to_detector_coordinates(
        np.asarray((0.0,)),
        np.asarray((0.0,)),
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if bool(direct.valid):
        direct_column = float(direct.column_px[0])
        direct_row = float(direct.row_px[0])
    else:
        direct_column, direct_row = instrument.detector_reference_coordinate_px
    row_pitch = instrument.detector_row_pitch_m
    column_pitch = instrument.detector_column_pitch_m
    pitch = min(row_pitch, column_pitch)
    scaled_radius = np.hypot(
        (perimeter_column - direct_column) * column_pitch / pitch,
        (perimeter_row - direct_row) * row_pitch / pitch,
    )
    maximum_radius_px = max(float(np.max(scaled_radius)), 1.0)
    theta_count = max(2, math.ceil(maximum_radius_px / cake_step_px))
    phi_count = max(16, math.ceil(2.0 * np.pi * maximum_radius_px / cake_step_px))
    theta_step = maximum_two_theta / theta_count
    theta = (np.arange(theta_count, dtype=np.float64) + 0.5) * theta_step
    phi = -np.pi + (np.arange(phi_count, dtype=np.float64) + 0.5) * (2.0 * np.pi / phi_count)
    return theta, phi


def _scale_invariant_detector_counts(
    detector_counts: NDArray[np.generic],
    detector_valid_mask: NDArray[np.bool_],
) -> FloatArray:
    valid_values = np.asarray(detector_counts[detector_valid_mask], dtype=np.float64)
    if valid_values.size == 0:
        raise ValueError("detector_valid_mask must retain at least one pixel")
    maximum = float(np.max(valid_values, initial=0.0))
    if maximum == 0.0:
        return np.zeros(detector_counts.shape, dtype=np.float64)
    exponent = math.frexp(maximum)[1]
    scaled = np.ldexp(np.asarray(detector_counts, dtype=np.float64), -exponent)
    if not np.all(detector_valid_mask):
        scaled[~detector_valid_mask] = math.ldexp(float(np.median(valid_values)), -exponent)
    return scaled


def _sample_search_cake(
    detector_counts: NDArray[np.generic],
    detector_valid_mask: NDArray[np.bool_],
    detector_valid_mask_u8: NDArray[np.uint8],
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    theta: FloatArray,
    phi: FloatArray,
    tile_phi_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    valid_detector_values = detector_counts[detector_valid_mask]
    if valid_detector_values.size == 0:
        raise ValueError("detector_valid_mask must retain at least one pixel")
    fill = float(np.median(valid_detector_values))
    del valid_detector_values
    chart = np.empty((phi.size, theta.size), dtype=np.float32)
    support = np.zeros(chart.shape, dtype=np.bool_)
    for start in range(0, phi.size, tile_phi_count):
        stop = min(start + tile_phi_count, phi.size)
        phi_grid, theta_grid = np.meshgrid(phi[start:stop], theta, indexing="ij")
        coordinates = angles_to_detector_coordinates(
            theta_grid,
            phi_grid,
            instrument=instrument,
            angle_frame=angle_frame,
        )
        sampled_mask = map_coordinates(
            detector_valid_mask_u8,
            (coordinates.row_px, coordinates.column_px),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(np.bool_)
        tile_support = coordinates.valid & sampled_mask
        sampled = map_coordinates(
            detector_counts,
            (coordinates.row_px, coordinates.column_px),
            output=np.float64,
            order=1,
            mode="constant",
            cval=fill,
            prefilter=False,
        )
        chart[start:stop] = np.where(tile_support, sampled, fill).astype(np.float32)
        support[start:stop] = tile_support
    return chart, support


def _global_chart_candidates(
    detector_counts: NDArray[np.generic],
    detector_valid_mask: NDArray[np.bool_],
    detector_valid_mask_u8: NDArray[np.uint8],
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    policy: BlindIndexingPolicy,
) -> tuple[_Candidate, ...]:
    peak_policy = policy.track_policy
    theta, phi = _cake_axes(instrument, angle_frame, peak_policy.cake_step_px)
    chart, support = _sample_search_cake(
        detector_counts,
        detector_valid_mask,
        detector_valid_mask_u8,
        instrument=instrument,
        angle_frame=angle_frame,
        theta=theta,
        phi=phi,
        tile_phi_count=policy.sampling_tile_phi_count,
    )
    narrow_sigma = max(0.75, 1.3 / peak_policy.cake_step_px)
    broad_sigma = max(
        2.5 * narrow_sigma,
        0.30 * peak_policy.search_radius_px / peak_policy.cake_step_px,
    )
    narrow = gaussian_filter(chart, narrow_sigma, mode=("wrap", "nearest"))
    broad = gaussian_filter(chart, broad_sigma, mode=("wrap", "nearest"))
    np.subtract(narrow, broad, out=chart)
    response = chart
    del narrow
    core_support = minimum_filter(
        support,
        size=(3, 3),
        mode=("wrap", "constant"),
        cval=0,
    )
    del support
    center = np.zeros(theta.size, dtype=np.float64)
    scale = np.full(theta.size, np.inf, dtype=np.float64)
    response_scale = max(
        abs(float(np.min(response, initial=0.0))),
        abs(float(np.max(response, initial=0.0))),
    )
    numeric_floor = max(
        128.0 * np.finfo(np.float32).eps * response_scale,
        float(np.finfo(np.float32).tiny),
    )
    for theta_index in range(theta.size):
        values = response[core_support[:, theta_index], theta_index]
        if values.size < 20:
            continue
        center[theta_index] = float(np.median(values))
        scale[theta_index] = max(_robust_sigma(values), numeric_floor)
    maximum_filter(
        response,
        size=(3, 3),
        mode=("wrap", "nearest"),
        output=broad,
    )
    local_maximum = response == broad
    del broad
    local_maximum &= core_support
    del core_support
    minimum_theta_index = math.ceil(policy.minimum_scattering_radius_px / peak_policy.cake_step_px)
    theta_support = np.zeros(theta.size, dtype=np.bool_)
    theta_support[min(max(minimum_theta_index, 1), theta.size) : max(theta.size - 1, 1)] = True
    threshold = center + peak_policy.minimum_candidate_z * scale
    selected_phi = np.empty(0, dtype=np.int64)
    selected_theta = np.empty(0, dtype=np.int64)
    selected_z = np.empty(0, dtype=np.float64)
    candidate_theta_block = 32
    for start in range(0, theta.size, candidate_theta_block):
        stop = min(start + candidate_theta_block, theta.size)
        block_mask = local_maximum[:, start:stop]
        block_mask &= theta_support[None, start:stop]
        block_mask &= response[:, start:stop] >= threshold[None, start:stop]
        block_index = np.argwhere(block_mask)
        if block_index.size == 0:
            continue
        block_phi = block_index[:, 0]
        block_theta = block_index[:, 1] + start
        block_z = (response[block_phi, block_theta] - center[block_theta]) / scale[block_theta]
        block_order = np.lexsort((block_theta, block_phi, -block_z))[
            : policy.maximum_candidate_count
        ]
        combined_phi = np.concatenate((selected_phi, block_phi[block_order]))
        combined_theta = np.concatenate((selected_theta, block_theta[block_order]))
        combined_z = np.concatenate((selected_z, block_z[block_order]))
        combined_order = np.lexsort((combined_theta, combined_phi, -combined_z))[
            : policy.maximum_candidate_count
        ]
        selected_phi = combined_phi[combined_order]
        selected_theta = combined_theta[combined_order]
        selected_z = combined_z[combined_order]
    del local_maximum, response
    if selected_phi.size == 0:
        return ()
    coordinates = angles_to_detector_coordinates(
        theta[selected_theta],
        phi[selected_phi],
        instrument=instrument,
        angle_frame=angle_frame,
    )
    output: list[_Candidate] = []
    for index in range(selected_phi.size):
        if not bool(coordinates.valid[index]):
            continue
        refined = _native_refinement(
            detector_counts,
            detector_valid_mask,
            float(coordinates.column_px[index]),
            float(coordinates.row_px[index]),
            float(selected_z[index]),
            instrument=instrument,
            angle_frame=angle_frame,
            policy=peak_policy,
        )
        if refined is not None and refined.z_score >= peak_policy.minimum_candidate_z:
            output.append(refined)
    return _merge_candidates(tuple(output), peak_policy.candidate_merge_radius_px)


def discover_measured_cake_peaks(
    detector_native_counts: ArrayLike,
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    image_id: str,
    detector_valid_mask: ArrayLike | None = None,
    detector_mask_revision: str | None = None,
    policy: BlindIndexingPolicy | None = None,
) -> MeasuredPeakDiscovery:
    """Globally discover detector peaks without accepting marker or catalogue input."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be a CompiledInstrument")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    active_policy = BlindIndexingPolicy() if policy is None else policy
    if not isinstance(active_policy, BlindIndexingPolicy):
        raise TypeError("policy must be a BlindIndexingPolicy")
    counts = np.asarray(detector_native_counts)
    if (
        counts.shape != instrument.detector_shape_rc
        or np.iscomplexobj(counts)
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0.0)
    ):
        raise ValueError(
            "detector_native_counts must be finite, nonnegative, and match detector_shape_rc"
        )
    if detector_valid_mask is None:
        valid_mask = np.ones(counts.shape, dtype=np.bool_)
        mask_revision = "all-valid-detector-mask.v1"
    else:
        supplied_mask = np.asarray(detector_valid_mask)
        if supplied_mask.dtype.kind != "b" or supplied_mask.shape != counts.shape:
            raise ValueError("detector_valid_mask must be boolean with detector_shape_rc")
        valid_mask = np.asarray(supplied_mask, dtype=np.bool_)
        mask_revision = detector_mask_revision or "caller-supplied-bool-mask.v1"
    if detector_mask_revision is not None:
        if not isinstance(detector_mask_revision, str) or not detector_mask_revision.strip():
            raise ValueError("detector_mask_revision must be a nonempty string")
        mask_revision = detector_mask_revision
    search_counts = _scale_invariant_detector_counts(counts, valid_mask)
    candidates = _global_chart_candidates(
        search_counts,
        valid_mask,
        np.asarray(valid_mask, dtype=np.uint8),
        instrument=instrument,
        angle_frame=angle_frame,
        policy=active_policy,
    )
    peaks = tuple(
        DiscoveredCakePeak(
            column_px=item.column_px,
            row_px=item.row_px,
            two_theta_rad=item.two_theta_rad,
            phi_rad=item.phi_rad,
            covariance_px2=item.covariance_px2,
            localization_covariance_px2=item.localization_covariance_px2,
            z_score=item.z_score,
        )
        for item in candidates
    )
    return MeasuredPeakDiscovery(
        image_id=image_id,
        detector_shape_rc=instrument.detector_shape_rc,
        peaks=peaks,
        detector_data_hash=_array_hash(counts),
        detector_mask_hash=_array_hash(valid_mask),
        detector_mask_revision=mask_revision,
        geometry_context_hash=_discovery_geometry_hash(instrument, angle_frame),
        policy=active_policy,
    )


@dataclass(frozen=True, slots=True)
class _ReciprocalLabel:
    key: IntegerLMarkerKey
    representative_rod: Rod
    beta_rad: float
    score: float
    predicted_column_px: float
    predicted_row_px: float


def _transverse_radius_rod_shells(
    context: _GeometryIndexingContext,
) -> tuple[tuple[float, tuple[Rod, ...]], ...]:
    """Group physical rods only when the configured reciprocal metric says so."""

    basis = context.reciprocal_basis_Ainv
    b3 = basis[:, 2]
    mean_axis = b3 / np.linalg.norm(b3)
    entries: list[tuple[float, Rod]] = []
    for rod in context.rods:
        if rod.h == 0 and rod.k == 0:
            continue
        q_parallel = rod.h * basis[:, 0] + rod.k * basis[:, 1]
        transverse = q_parallel - float(q_parallel @ mean_axis) * mean_axis
        entries.append((float(np.linalg.norm(transverse)), rod))
    entries.sort(key=lambda item: (item[0], item[1].h, item[1].k))
    shell_radii: list[list[float]] = []
    shell_rods: list[list[Rod]] = []
    for radius, rod in entries:
        tolerance = 4096.0 * np.finfo(np.float64).eps * max(radius, 1.0)
        if shell_radii and abs(radius - shell_radii[-1][0]) <= tolerance:
            shell_radii[-1].append(radius)
            shell_rods[-1].append(rod)
        else:
            shell_radii.append([radius])
            shell_rods.append([rod])
    return tuple(
        (float(np.mean(radii)), tuple(rods))
        for radii, rods in zip(shell_radii, shell_rods, strict=True)
    )


def _label_q_point(
    q_sample_Ainv: FloatArray,
    covariance_q_sample_Ainv2: FloatArray,
    *,
    context: _GeometryIndexingContext,
    policy: BlindIndexingPolicy,
) -> _ReciprocalLabel | None:
    basis = context.reciprocal_basis_Ainv
    crystal_to_sample = context.crystal_to_sample
    ki_sample = context.ki_sample_Ainv
    q_crystal = crystal_to_sample.T @ q_sample_Ainv
    covariance_q_crystal = crystal_to_sample.T @ covariance_q_sample_Ainv2 @ crystal_to_sample
    b3 = basis[:, 2]
    b3_norm = float(np.linalg.norm(b3))
    mean_axis = b3 / b3_norm
    transverse_q = q_crystal - float(q_crystal @ mean_axis) * mean_axis
    transverse_norm = float(np.linalg.norm(transverse_q))
    if transverse_norm <= np.finfo(np.float64).eps:
        return None
    radial_shells = _transverse_radius_rod_shells(context)
    if not radial_shells:
        return None
    ordered_shell = sorted(
        enumerate(radial_shells),
        key=lambda item: (abs(transverse_norm - item[1][0]), item[0]),
    )
    _, (selected_radius, selected_rods) = ordered_shell[0]
    radial_residual = abs(transverse_norm - selected_radius)
    other_gap = min(
        (abs(selected_radius - shell[0]) for _, shell in ordered_shell[1:]),
        default=math.inf,
    )
    radial_gradient = transverse_q / transverse_norm
    radial_sigma = math.sqrt(
        max(float(radial_gradient @ covariance_q_crystal @ radial_gradient), 0.0)
    )
    radial_limit = policy.maximum_family_residual_Ainv
    if math.isfinite(other_gap):
        radial_limit = min(radial_limit, policy.separation_fraction * other_gap)
    if radial_residual + policy.uncertainty_sigma * radial_sigma > radial_limit:
        return None

    labels: list[tuple[IntegerLMarkerKey, Rod, float, float]] = []
    for rod in selected_rods:
        q_parallel = rod.h * basis[:, 0] + rod.k * basis[:, 1]
        transverse_parallel = q_parallel - float(q_parallel @ mean_axis) * mean_axis
        l_value = (float(q_crystal @ mean_axis) - float(q_parallel @ mean_axis)) / b3_norm
        integer_l = int(np.rint(l_value))
        l_sigma = math.sqrt(max(float(mean_axis @ covariance_q_crystal @ mean_axis), 0.0)) / b3_norm
        l_residual = abs(l_value - integer_l)
        if l_residual + policy.uncertainty_sigma * l_sigma > policy.separation_fraction:
            continue
        beta = math.atan2(
            float(mean_axis @ np.cross(transverse_parallel, transverse_q)),
            float(transverse_parallel @ transverse_q),
        ) % (2.0 * np.pi)
        roots = solve_integer_l_ewald_roots(
            rod=rod,
            integer_l=integer_l,
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=crystal_to_sample,
            ki_sample_Ainv=ki_sample,
        )
        if roots is None or len(roots.beta_rad) != 2 or roots.root_sign != (-1, 1):
            continue
        distances = tuple(_circular_distance(beta, item) for item in roots.beta_rad)
        root_index = int(np.argmin(distances))
        root_gap = _circular_distance(roots.beta_rad[0], roots.beta_rad[1])
        beta_gradient = np.cross(mean_axis, transverse_q) / (transverse_norm**2)
        beta_sigma = math.sqrt(
            max(float(beta_gradient @ covariance_q_crystal @ beta_gradient), 0.0)
        )
        beta_residual = distances[root_index]
        if (
            beta_residual + policy.uncertainty_sigma * beta_sigma
            > policy.separation_fraction * root_gap
        ):
            continue
        root_sign = roots.root_sign[root_index]
        key = IntegerLMarkerKey(
            family_m=rod.family_m,
            integer_L=integer_l,
            branch=roots.branch,
            root_sign=root_sign,
            representative_rod_hk=(rod.h, rod.k),
        )
        score = max(
            radial_residual / radial_limit,
            (l_residual + policy.uncertainty_sigma * l_sigma) / policy.separation_fraction,
            (beta_residual + policy.uncertainty_sigma * beta_sigma)
            / (policy.separation_fraction * root_gap),
        )
        labels.append((key, rod, roots.beta_rad[root_index], score))
    discrete = {(item[0].integer_L, item[0].branch, item[0].root_sign) for item in labels}
    if len(discrete) != 1:
        return None

    mapped_labels: list[tuple[IntegerLMarkerKey, Rod, float, float, float, float, FloatArray]] = []
    for key, rod, beta_rad, score in labels:
        mapped = context.map_latent_geometry(
            rod=rod,
            branch=key.branch,
            alpha_rad=0.0,
            beta_rad=beta_rad,
        )
        if not bool(mapped.valid):
            continue
        mapped_labels.append(
            (
                key,
                rod,
                beta_rad,
                score,
                float(mapped.column_px),
                float(mapped.row_px),
                np.asarray(mapped.ewald_geometry.q_sample_Ainv, dtype=np.float64),
            )
        )
    if not mapped_labels:
        return None
    coordinate_tolerance_px = (
        32768.0
        * np.finfo(np.float64).eps
        * max(float(max(context.instrument.detector_shape_rc)), 1.0)
    )
    q_tolerance_Ainv = (
        32768.0 * np.finfo(np.float64).eps * max(float(np.linalg.norm(ki_sample)), 1.0)
    )
    coincident_groups: list[
        list[tuple[IntegerLMarkerKey, Rod, float, float, float, float, FloatArray]]
    ] = []
    for candidate in sorted(mapped_labels, key=lambda item: (item[1].h, item[1].k)):
        matched_group = next(
            (
                group
                for group in coincident_groups
                if math.hypot(candidate[4] - group[0][4], candidate[5] - group[0][5])
                <= coordinate_tolerance_px
                and float(np.linalg.norm(candidate[6] - group[0][6])) <= q_tolerance_Ainv
            ),
            None,
        )
        if matched_group is None:
            coincident_groups.append([candidate])
        else:
            matched_group.append(candidate)
    if len(coincident_groups) != 1:
        return None
    coincident = coincident_groups[0]
    representative = min(coincident, key=lambda item: (item[1].h, item[1].k))
    representative_rod = representative[1]
    key = IntegerLMarkerKey(
        family_m=representative[0].family_m,
        integer_L=representative[0].integer_L,
        branch=representative[0].branch,
        root_sign=representative[0].root_sign,
        representative_rod_hk=(representative_rod.h, representative_rod.k),
    )
    return _ReciprocalLabel(
        key=key,
        representative_rod=representative_rod,
        beta_rad=representative[2],
        score=max(item[3] for item in coincident),
        predicted_column_px=representative[4],
        predicted_row_px=representative[5],
    )


def _indexing_context_hash(
    discovery: MeasuredPeakDiscovery,
    context: _GeometryIndexingContext,
    angle_frame: AngleFrame,
) -> str:
    states = context.incident.states
    rods = tuple((rod.h, rod.k, rod.family_m, _float_token(rod.population)) for rod in context.rods)
    air_k_Ainv = 2.0 * np.pi / float(states.wavelength_A[0])
    payload = {
        "schema": "position-free-reciprocal-indexing-context.v2",
        "discovery_hash": discovery.discovery_hash,
        "discovery_geometry_context_hash": discovery.geometry_context_hash,
        "sample_from_lab_rotation": _array_payload(context.instrument.sample_from_lab.rotation),
        "sample_intersection_lab_m": _array_payload(states.sample_intersection_lab_m),
        "incident_wavelength_A": _array_payload(states.wavelength_A),
        "incident_states": {
            "incident_state_id": _array_hash(states.incident_state_id),
            "incident_sample_id": _array_hash(states.incident_sample_id),
            "direction_sample": _array_hash(states.direction_sample),
            "k_air_sample_Ainv": _array_hash(states.k_air_sample_Ainv),
            "k_film_phase_sample_Ainv": _array_hash(states.k_film_phase_sample_Ainv),
            "kz_film_Ainv": _array_hash(states.kz_film_Ainv),
            "entrance_amplitude": _array_hash(states.entrance_amplitude),
            "footprint_acceptance": _array_hash(states.footprint_acceptance),
            "source_weight": _array_hash(states.source_weight),
            "valid": _array_hash(states.valid),
            "polarization_state_id": states.polarization_state_id,
            "status": tuple(item.value for item in states.status),
            "source_revision": states.source_revision,
            "sample_geometry_revision": states.sample_geometry_revision,
            "material_revision": states.material_revision,
            "incident_model_id": states.incident_model_id,
        },
        "instrument": {
            "lab_from_sample_rotation": _array_payload(context.instrument.lab_from_sample.rotation),
            "lab_from_sample_translation_m": _array_payload(
                context.instrument.lab_from_sample.translation_m
            ),
            "sample_geometry_revision": context.instrument.sample_geometry_revision,
            "film_thickness_A": _float_token(context.instrument.film_thickness_A),
        },
        "reciprocal_basis_Ainv": _array_payload(context.reciprocal_basis_Ainv),
        "crystal_to_sample": _array_payload(context.crystal_to_sample),
        "bragg_k_norm_Ainv": _float_token(air_k_Ainv),
        "ki_sample_Ainv": _array_payload(context.ki_sample_Ainv),
        "root_tolerance_rel": _float_token(0.0),
        "residual_tolerance_rel": _float_token(512.0 * np.finfo(np.float64).eps),
        "material_revision": states.material_revision,
        "rods": rods,
        "positive_b3_convention": True,
        "angle_frame_revision": angle_frame.revision,
    }
    return _sha256_payload(payload)


def _canonicalized_peak_angles(
    peaks: tuple[DiscoveredCakePeak, ...],
    *,
    instrument: CompiledInstrument,
    angle_frame: AngleFrame,
    validate_supplied: bool,
) -> tuple[DiscoveredCakePeak, ...]:
    if not peaks:
        return peaks
    canonical_angles = detector_coordinates_to_angles(
        np.asarray([peak.column_px for peak in peaks], dtype=np.float64),
        np.asarray([peak.row_px for peak in peaks], dtype=np.float64),
        instrument=instrument,
        angle_frame=angle_frame,
    )
    if not np.all(canonical_angles.valid & canonical_angles.azimuth_valid):
        raise ValueError("discovered peak detector coordinates have no valid canonical angles")
    canonical_two_theta = np.asarray(canonical_angles.two_theta_rad, dtype=np.float64)
    canonical_phi = np.asarray(canonical_angles.phi_rad, dtype=np.float64)
    if validate_supplied:
        supplied_two_theta = np.asarray([peak.two_theta_rad for peak in peaks], dtype=np.float64)
        supplied_phi = np.asarray([peak.phi_rad for peak in peaks], dtype=np.float64)
        tolerance = 32768.0 * np.finfo(np.float64).eps
        two_theta_error = np.abs(supplied_two_theta - canonical_two_theta)
        phi_error = np.abs((supplied_phi - canonical_phi + np.pi) % (2.0 * np.pi) - np.pi)
        if np.any(
            two_theta_error > tolerance * np.maximum(1.0, np.abs(canonical_two_theta))
        ) or np.any(phi_error > tolerance):
            raise ValueError("discovered peak detector and angle coordinates disagree")
    return tuple(
        replace(
            peak,
            two_theta_rad=float(two_theta),
            phi_rad=float(phi),
        )
        for peak, two_theta, phi in zip(
            peaks,
            canonical_two_theta,
            canonical_phi,
            strict=True,
        )
    )


def _index_peak_coordinates(
    peaks: tuple[DiscoveredCakePeak, ...],
    *,
    image_id: str,
    detector_data_hash: str,
    detector_mask_hash: str,
    detector_mask_revision: str,
    policy: BlindIndexingPolicy,
    context_hash: str,
    decision_reason: str,
    ewald_context: _GeometryIndexingContext,
    angle_frame: AngleFrame,
    incidence_angle_rad: float,
) -> MeasuredImageIndexingResult:
    """Label one immutable set of detector-native candidate coordinates."""

    states = ewald_context.incident.states
    if states.sample_geometry_revision != ewald_context.instrument.sample_geometry_revision:
        raise ValueError("incident and detector geometry sample revisions disagree")
    if peaks:
        column = np.asarray([item.column_px for item in peaks], dtype=np.float64)
        row = np.asarray([item.row_px for item in peaks], dtype=np.float64)
        sample_column = np.column_stack((column, column + 0.5, column - 0.5, column, column))
        sample_row = np.column_stack((row, row, row, row + 0.5, row - 0.5))
        geometry = ewald_context.evaluate_detector_geometry(
            sample_column,
            sample_row,
            include_surface_jacobian=False,
        )
    else:
        geometry = None
    indexed: list[tuple[DiscoveredCakePeak, _ReciprocalLabel, FloatArray]] = []
    for peak_index, peak in enumerate(peaks):
        assert geometry is not None
        if peak.z_score < policy.track_policy.minimum_site_z:
            continue
        if not np.all(geometry.valid[peak_index]):
            continue
        q = geometry.q_sample_Ainv[peak_index, 0]
        jacobian = np.column_stack(
            (
                geometry.q_sample_Ainv[peak_index, 1] - geometry.q_sample_Ainv[peak_index, 2],
                geometry.q_sample_Ainv[peak_index, 3] - geometry.q_sample_Ainv[peak_index, 4],
            )
        )
        covariance_px = np.asarray(peak.localization_covariance_px2, dtype=np.float64)
        covariance_q = jacobian @ covariance_px @ jacobian.T
        label = _label_q_point(
            q,
            covariance_q,
            context=ewald_context,
            policy=policy,
        )
        if label is not None:
            indexed.append((peak, label, covariance_q))
    ownership: dict[
        IntegerLMarkerKey,
        list[
            tuple[
                float,
                DiscoveredCakePeak,
                _ReciprocalLabel,
                float,
                float,
                float,
                float,
            ]
        ],
    ] = {}
    for peak, label, _ in indexed:
        key = label.key
        predicted_column = label.predicted_column_px
        predicted_row = label.predicted_row_px
        delta = np.asarray((peak.column_px - predicted_column, peak.row_px - predicted_row))
        anchor_distance = float(np.linalg.norm(delta))
        if anchor_distance > policy.maximum_anchor_distance_px:
            continue
        predicted_angles = detector_coordinates_to_angles(
            predicted_column,
            predicted_row,
            instrument=ewald_context.instrument,
            angle_frame=angle_frame,
        )
        if not bool(predicted_angles.valid) or not bool(predicted_angles.azimuth_valid):
            continue
        covariance = np.asarray(peak.covariance_px2) + (
            policy.track_policy.assignment_model_sigma_px**2 * np.eye(2)
        )
        cost = label.score * label.score + float(delta @ np.linalg.solve(covariance, delta))
        ownership.setdefault(key, []).append(
            (
                cost,
                peak,
                label,
                predicted_column,
                predicted_row,
                float(predicted_angles.two_theta_rad),
                float(predicted_angles.phi_rad),
            )
        )
    decisions: list[MarkerIndexingDecision] = []
    for key, owners in ownership.items():
        owners.sort(key=lambda item: (item[0], item[1].row_px, item[1].column_px))
        assignment_margin = None if len(owners) == 1 else owners[1][0] - owners[0][0]
        if (
            assignment_margin is not None
            and assignment_margin < policy.track_policy.minimum_assignment_margin
        ):
            continue
        (
            assignment_cost,
            peak,
            _,
            predicted_column,
            predicted_row,
            predicted_two_theta,
            predicted_phi,
        ) = owners[0]
        decisions.append(
            MarkerIndexingDecision(
                key=key,
                predicted_column_px=predicted_column,
                predicted_row_px=predicted_row,
                predicted_two_theta_rad=predicted_two_theta,
                predicted_phi_rad=predicted_phi,
                status=MarkerIndexingStatus.VISIBLE_CONFIDENT,
                reason=decision_reason,
                observed_column_px=peak.column_px,
                observed_row_px=peak.row_px,
                observed_two_theta_rad=peak.two_theta_rad,
                observed_phi_rad=peak.phi_rad,
                covariance_px2=peak.covariance_px2,
                z_score=peak.z_score,
                assignment_cost=assignment_cost,
                assignment_margin=assignment_margin,
            )
        )
    wavelength = float(ewald_context.incident.states.wavelength_A[0])
    return MeasuredImageIndexingResult(
        image_id=image_id,
        incidence_angle_rad=incidence_angle_rad,
        reference_wavelength_A=wavelength,
        marker_decisions=tuple(decisions),
        detector_data_hash=detector_data_hash,
        detector_mask_hash=detector_mask_hash,
        detector_mask_revision=detector_mask_revision,
        context_hash=context_hash,
        policy=policy.track_policy,
    )


def index_discovered_integer_l_peaks(
    discovery: MeasuredPeakDiscovery,
    *,
    ewald_context: _GeometryIndexingContext,
    angle_frame: AngleFrame,
    incidence_angle_rad: float,
) -> MeasuredImageIndexingResult:
    """Assign geometry-only labels after a position-free discovery is frozen."""

    if not isinstance(discovery, MeasuredPeakDiscovery):
        raise TypeError("discovery must be a MeasuredPeakDiscovery")
    if not isinstance(ewald_context, (NominalEwaldContext, GeometryOnlyEwaldContext)):
        raise TypeError("ewald_context must be a nominal geometry indexing context")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    if ewald_context.instrument.detector_shape_rc != discovery.detector_shape_rc:
        raise ValueError("discovery and Ewald context detector shapes disagree")
    expected_geometry_hash = _discovery_geometry_hash(
        ewald_context.instrument,
        angle_frame,
    )
    if discovery.geometry_context_hash != expected_geometry_hash:
        raise ValueError("discovery geometry does not match the Ewald instrument and angle frame")
    peaks = _canonicalized_peak_angles(
        discovery.peaks,
        instrument=ewald_context.instrument,
        angle_frame=angle_frame,
        validate_supplied=True,
    )
    return _index_peak_coordinates(
        peaks,
        image_id=discovery.image_id,
        detector_data_hash=discovery.detector_data_hash,
        detector_mask_hash=discovery.detector_mask_hash,
        detector_mask_revision=discovery.detector_mask_revision,
        policy=discovery.policy,
        context_hash=_indexing_context_hash(discovery, ewald_context, angle_frame),
        decision_reason=(
            "globally discovered before q-space label; exact alpha-zero anchor "
            "generated only after the label was frozen"
        ),
        ewald_context=ewald_context,
        angle_frame=angle_frame,
        incidence_angle_rad=incidence_angle_rad,
    )


def _frozen_reindexing_context_hash(
    discovery: MeasuredPeakDiscovery,
    frozen_observations: IntegerLMarkerObservations,
    ewald_context: _GeometryIndexingContext,
    angle_frame: AngleFrame,
) -> str:
    corrected_geometry_hash = _discovery_geometry_hash(
        ewald_context.instrument,
        angle_frame,
    )
    return _sha256_payload(
        {
            "schema": "frozen-position-free-coordinate-reindexing-context.v1",
            "source_discovery_hash": discovery.discovery_hash,
            "source_discovery_geometry_context_hash": discovery.geometry_context_hash,
            "corrected_discovery_geometry_context_hash": corrected_geometry_hash,
            "corrected_reciprocal_context_hash": _indexing_context_hash(
                discovery,
                ewald_context,
                angle_frame,
            ),
            "frozen_keys": tuple(
                (
                    key.family_m,
                    key.integer_L,
                    key.branch,
                    key.root_sign,
                    key.representative_rod_hk,
                )
                for key in frozen_observations.keys
            ),
            "frozen_coordinates_px": _array_hash(frozen_observations.coordinates_px),
            "frozen_covariance_px2": _array_hash(frozen_observations.covariance_px2),
        }
    )


def reindex_frozen_discovery_coordinates(
    discovery: MeasuredPeakDiscovery,
    *,
    frozen_observations: IntegerLMarkerObservations,
    ewald_context: _GeometryIndexingContext,
    angle_frame: AngleFrame,
    incidence_angle_rad: float,
) -> MeasuredImageIndexingResult:
    """Relabel immutable native candidates under corrected geometry.

    This is an identity audit of the selected position-free candidates. It
    performs no cake search and preserves every selected native coordinate,
    covariance, z-score, image hash, mask hash, and policy from ``discovery``.
    """

    if not isinstance(discovery, MeasuredPeakDiscovery):
        raise TypeError("discovery must be a MeasuredPeakDiscovery")
    if not isinstance(frozen_observations, IntegerLMarkerObservations):
        raise TypeError("frozen_observations must be IntegerLMarkerObservations")
    if not isinstance(ewald_context, (NominalEwaldContext, GeometryOnlyEwaldContext)):
        raise TypeError("ewald_context must be a nominal geometry indexing context")
    if not isinstance(angle_frame, AngleFrame):
        raise TypeError("angle_frame must be an AngleFrame")
    if ewald_context.instrument.detector_shape_rc != discovery.detector_shape_rc:
        raise ValueError("discovery and Ewald context detector shapes disagree")
    wavelength = float(ewald_context.incident.states.wavelength_A[0])
    if not math.isclose(
        frozen_observations.reference_wavelength_A,
        wavelength,
        rel_tol=0.0,
        abs_tol=256.0
        * np.finfo(np.float64).eps
        * max(frozen_observations.reference_wavelength_A, wavelength, 1.0),
    ):
        raise ValueError("frozen observations and corrected geometry wavelengths disagree")
    discovery_peaks = {(peak.column_px, peak.row_px): peak for peak in discovery.peaks}
    selected_peaks = []
    for index, coordinate in enumerate(frozen_observations.coordinates_px):
        peak = discovery_peaks.get((float(coordinate[0]), float(coordinate[1])))
        if peak is None:
            raise ValueError("a frozen observation is absent from its source discovery")
        if not np.array_equal(
            np.asarray(peak.covariance_px2),
            frozen_observations.covariance_px2[index],
        ):
            raise ValueError("a frozen observation covariance differs from its source discovery")
        selected_peaks.append(peak)
    peaks = _canonicalized_peak_angles(
        tuple(selected_peaks),
        instrument=ewald_context.instrument,
        angle_frame=angle_frame,
        validate_supplied=False,
    )
    context_hash = _frozen_reindexing_context_hash(
        discovery,
        frozen_observations,
        ewald_context,
        angle_frame,
    )
    return _index_peak_coordinates(
        peaks,
        image_id=discovery.image_id,
        detector_data_hash=discovery.detector_data_hash,
        detector_mask_hash=discovery.detector_mask_hash,
        detector_mask_revision=discovery.detector_mask_revision,
        policy=discovery.policy,
        context_hash=context_hash,
        decision_reason=(
            "original position-free native candidate relabeled under corrected geometry; "
            "no global discovery or coordinate refinement rerun"
        ),
        ewald_context=ewald_context,
        angle_frame=angle_frame,
        incidence_angle_rad=incidence_angle_rad,
    )
