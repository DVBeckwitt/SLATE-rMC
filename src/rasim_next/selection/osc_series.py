"""Strict OSC-series ingestion and measured integer-L indexing."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rasim_next.fitting import ExactTagGeometryModel, IndexedGeometryImage
from rasim_next.geometry import AngleFrame
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.io.osc import read_osc
from rasim_next.pipeline.configured_simulation import (
    ConfiguredGeometryInputs,
    GeometryOnlyEwaldContext,
    SimulationConfiguration,
    build_configured_geometry_inputs,
    build_geometry_only_ewald_context,
    load_simulation_config,
    load_strict_yaml_mapping,
    rebind_configured_geometry_instrument,
)
from rasim_next.selection.blind import (
    BlindIndexingPolicy,
    MeasuredPeakDiscovery,
    _discovery_geometry_hash,
    _indexing_context_hash,
    discover_measured_cake_peaks,
    index_discovered_integer_l_peaks,
    reindex_frozen_discovery_coordinates,
)
from rasim_next.selection.indexing import (
    FrozenMarkerVisibilityAudit,
    MeasuredIndexingResult,
    PeakIndexingPolicy,
    audit_frozen_marker_visibility,
    select_confident_branch_tracks,
)

_SCHEMA_VERSION = "rasim-osc-geometry-fit-v1"
DETECTOR_VALID_MASK_REVISION = "wholly-zero-rows-and-columns.v1"
M0_PEAK_EVIDENCE_REVISION = "expected-point.circular-core-annulus-median-mad.excess-pixel-gates.v1"


def _mapping(
    value: object,
    name: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ValueError(f"{name} keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _finite_angles(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} must contain only numeric scalars")
    angles = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in angles):
        raise ValueError(f"{name} must contain finite values")
    return angles


@dataclass(frozen=True, slots=True)
class OscGeometryImageConfiguration:
    image_id: str
    osc_path: Path
    axis_rotation_angles_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        image_id = str(self.image_id).strip()
        path = Path(self.osc_path).resolve()
        supplied_angles = tuple(self.axis_rotation_angles_deg)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in supplied_angles
        ):
            raise ValueError("axis_rotation_angles_deg must contain only numeric scalars")
        angles = tuple(float(value) for value in supplied_angles)
        if not image_id:
            raise ValueError("OSC geometry image_id must be nonempty")
        if not path.is_file():
            raise ValueError(f"OSC geometry image does not exist: {path}")
        if not angles or not all(math.isfinite(value) for value in angles):
            raise ValueError("axis_rotation_angles_deg must contain finite values")
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "osc_path", path)
        object.__setattr__(self, "axis_rotation_angles_deg", angles)


@dataclass(frozen=True, slots=True)
class OscGeometrySeriesConfiguration:
    config_path: Path
    incidence_axis_index: int
    images: tuple[OscGeometryImageConfiguration, ...]
    qualification_profile: str | None = None
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        config_path = Path(self.config_path).resolve()
        images = tuple(self.images)
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        if not config_path.is_file():
            raise ValueError(f"simulation config does not exist: {config_path}")
        if (
            isinstance(self.incidence_axis_index, bool)
            or not isinstance(self.incidence_axis_index, int)
            or self.incidence_axis_index < 0
        ):
            raise ValueError("incidence_axis_index must be a nonnegative integer")
        if not images or any(
            not isinstance(image, OscGeometryImageConfiguration) for image in images
        ):
            raise ValueError("images must contain at least one OSC geometry image")
        ids = tuple(image.image_id for image in images)
        paths = tuple(image.osc_path for image in images)
        if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
            raise ValueError("OSC geometry image IDs and paths must be unique")
        incidences = tuple(
            image.axis_rotation_angles_deg[self.incidence_axis_index]
            if self.incidence_axis_index < len(image.axis_rotation_angles_deg)
            else math.nan
            for image in images
        )
        if not all(math.isfinite(value) for value in incidences):
            raise ValueError("incidence_axis_index lies outside an image axis-angle tuple")
        qualification_profile = self.qualification_profile
        if qualification_profile is not None and (
            not isinstance(qualification_profile, str) or not qualification_profile.strip()
        ):
            raise ValueError("qualification_profile must be a nonempty string when supplied")
        object.__setattr__(self, "config_path", config_path)
        object.__setattr__(self, "images", images)
        object.__setattr__(
            self,
            "qualification_profile",
            None if qualification_profile is None else qualification_profile.strip(),
        )


def load_osc_geometry_series(
    path: str | Path,
) -> OscGeometrySeriesConfiguration:
    """Load one strict, manifest-relative OSC geometry series."""

    manifest_path = Path(path).resolve()
    document = _mapping(
        load_strict_yaml_mapping(manifest_path),
        "OSC geometry series",
        required={"schema_version", "simulation_config", "incidence_axis_index", "images"},
        optional={"qualification_profile"},
    )
    schema_version = document["schema_version"]
    simulation_config = document["simulation_config"]
    incidence_axis_index = document["incidence_axis_index"]
    raw_images = document["images"]
    if not isinstance(schema_version, str) or not isinstance(simulation_config, str):
        raise ValueError("schema_version and simulation_config must be strings")
    if (
        isinstance(incidence_axis_index, bool)
        or not isinstance(incidence_axis_index, int)
        or incidence_axis_index < 0
    ):
        raise ValueError("incidence_axis_index must be a nonnegative integer")
    if not isinstance(raw_images, list):
        raise ValueError("images must be a list")
    images = []
    for index, raw_image in enumerate(raw_images):
        data = _mapping(
            raw_image,
            f"images[{index}]",
            required={"image_id", "osc_path", "axis_rotation_angles_deg"},
        )
        if not isinstance(data["image_id"], str) or not isinstance(data["osc_path"], str):
            raise ValueError(f"images[{index}] image_id and osc_path must be strings")
        images.append(
            OscGeometryImageConfiguration(
                image_id=data["image_id"],
                osc_path=manifest_path.parent / data["osc_path"],
                axis_rotation_angles_deg=_finite_angles(
                    data["axis_rotation_angles_deg"],
                    f"images[{index}].axis_rotation_angles_deg",
                ),
            )
        )
    return OscGeometrySeriesConfiguration(
        config_path=manifest_path.parent / simulation_config,
        incidence_axis_index=incidence_axis_index,
        images=tuple(images),
        qualification_profile=document.get("qualification_profile"),
        schema_version=schema_version,
    )


def simulation_config_for_osc_image(
    base: SimulationConfiguration,
    image: OscGeometryImageConfiguration,
) -> SimulationConfiguration:
    """Apply one image's complete commanded-axis tuple to a shared simulation config."""

    if not isinstance(base, SimulationConfiguration):
        raise TypeError("base must be SimulationConfiguration")
    if not isinstance(image, OscGeometryImageConfiguration):
        raise TypeError("image must be OscGeometryImageConfiguration")
    if len(image.axis_rotation_angles_deg) != len(base.instrument.axis_rotations):
        raise ValueError("image axis-angle count must match the simulation configuration")
    rotations = tuple(
        replace(rotation, angle_deg=angle_deg)
        for rotation, angle_deg in zip(
            base.instrument.axis_rotations,
            image.axis_rotation_angles_deg,
            strict=True,
        )
    )
    return replace(
        base,
        instrument=replace(base.instrument, axis_rotations=rotations),
    )


def build_osc_angle_frame(
    *,
    mean_direction_lab: ArrayLike,
    instrument: CompiledInstrument,
    sample_intersection_lab_m: ArrayLike,
    revision: str,
) -> AngleFrame:
    """Build the detector-oriented chart frame at the beam/sample intersection."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    direct_beam = np.asarray(mean_direction_lab, dtype=np.float64)
    if direct_beam.shape != (3,) or not np.all(np.isfinite(direct_beam)):
        raise ValueError("mean_direction_lab must be a finite three-vector")
    norm = float(np.linalg.norm(direct_beam))
    if norm <= 0.0:
        raise ValueError("mean_direction_lab must be nonzero")
    direct_beam = direct_beam / norm
    detector_column_lab = instrument.lab_from_detector.apply_vector(np.asarray((1.0, 0.0, 0.0)))
    column_right = detector_column_lab - float(detector_column_lab @ direct_beam) * direct_beam
    if np.linalg.norm(column_right) <= 1.0e-12:
        raise ValueError("detector column axis is parallel to the direct beam")
    column_right /= np.linalg.norm(column_right)
    row_down = np.cross(direct_beam, column_right)
    return AngleFrame(
        origin_lab_m=sample_intersection_lab_m,
        row_down_lab=row_down,
        column_right_lab=column_right,
        direct_beam_lab=direct_beam,
        revision=revision,
    )


def detector_valid_mask_from_counts(counts: ArrayLike) -> NDArray[np.bool_]:
    """Mask wholly zero detector rows and columns in detector-native orientation."""

    array = np.asarray(counts)
    if array.ndim != 2:
        raise ValueError("detector counts must be two-dimensional")
    mask = np.ones(array.shape, dtype=np.bool_)
    mask[np.all(array == 0, axis=1), :] = False
    mask[:, np.all(array == 0, axis=0)] = False
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True, slots=True)
class ExpectedM0Peak:
    """One caller-supplied expected branchless 00L site under fixed detector geometry."""

    integer_L: int
    column_px: float
    row_px: float

    def __post_init__(self) -> None:
        integer_l = self.integer_L
        if (
            isinstance(integer_l, bool)
            or not isinstance(integer_l, (int, np.integer))
            or int(integer_l) <= 0
        ):
            raise ValueError("integer_L must be a positive crystallographic integer")
        column = float(self.column_px)
        row = float(self.row_px)
        if not math.isfinite(column) or not math.isfinite(row):
            raise ValueError("expected m=0 detector coordinates must be finite")
        object.__setattr__(self, "integer_L", int(integer_l))
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)


@dataclass(frozen=True, slots=True)
class M0PeakEvidencePolicy:
    """Frozen circular core/annulus gates for expected 00L signal in a native OSC image."""

    core_radius_px: float = 5.0
    background_inner_radius_px: float = 10.0
    background_outer_radius_px: float = 25.0
    minimum_peak_z: float = 5.0
    minimum_integrated_z: float = 5.0
    minimum_valid_fraction: float = 0.8
    mad_scale: float = 1.4826
    sigma_floor_counts: float = 1.0
    excess_pixel_z: float = 3.0
    minimum_excess_pixel_count: int = 3
    revision: str = M0_PEAK_EVIDENCE_REVISION

    def __post_init__(self) -> None:
        values = {
            name: float(getattr(self, name))
            for name in (
                "core_radius_px",
                "background_inner_radius_px",
                "background_outer_radius_px",
                "minimum_peak_z",
                "minimum_integrated_z",
                "minimum_valid_fraction",
                "mad_scale",
                "sigma_floor_counts",
                "excess_pixel_z",
            )
        }
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("m=0 evidence policy values must be finite and positive")
        if not (
            values["core_radius_px"]
            < values["background_inner_radius_px"]
            < values["background_outer_radius_px"]
        ):
            raise ValueError("m=0 evidence radii must be strictly increasing")
        if values["minimum_valid_fraction"] > 1.0:
            raise ValueError("minimum_valid_fraction must not exceed one")
        if self.revision != M0_PEAK_EVIDENCE_REVISION:
            raise ValueError("m=0 evidence policy revision does not match the implementation")
        if (
            isinstance(self.minimum_excess_pixel_count, bool)
            or not isinstance(self.minimum_excess_pixel_count, (int, np.integer))
            or int(self.minimum_excess_pixel_count) <= 0
        ):
            raise ValueError("minimum_excess_pixel_count must be a positive integer")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "minimum_excess_pixel_count",
            int(self.minimum_excess_pixel_count),
        )


@dataclass(frozen=True, slots=True)
class M0PeakEvidence:
    """Raw-count evidence for one expected branchless 00L profile."""

    candidate: ExpectedM0Peak
    classification: str
    core_pixel_count: int
    background_pixel_count: int
    excess_pixel_count: int
    background_counts: float | None
    robust_sigma_counts: float | None
    maximum_core_counts: float | None
    peak_z: float | None
    integrated_z: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ExpectedM0Peak):
            raise TypeError("candidate must be ExpectedM0Peak")
        allowed = {
            "LOCAL_SIGNAL_SUPPORTED",
            "LOCAL_SIGNAL_BELOW_GATE",
            "OUTSIDE_PANEL",
            "INSUFFICIENT_VALID_SUPPORT",
        }
        if self.classification not in allowed:
            raise ValueError("unsupported m=0 peak-evidence classification")
        for name in ("core_pixel_count", "background_pixel_count", "excess_pixel_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        numeric = (
            self.background_counts,
            self.robust_sigma_counts,
            self.maximum_core_counts,
            self.peak_z,
            self.integrated_z,
        )
        if self.classification in {"OUTSIDE_PANEL", "INSUFFICIENT_VALID_SUPPORT"}:
            if any(value is not None for value in numeric):
                raise ValueError("unmeasured m=0 evidence fields must be None")
        elif any(value is None or not math.isfinite(float(value)) for value in numeric):
            raise ValueError("measured m=0 evidence fields must be finite")

    @property
    def integer_L(self) -> int:
        return self.candidate.integer_L

    @property
    def accepted(self) -> bool:
        return self.classification == "LOCAL_SIGNAL_SUPPORTED"


def evaluate_expected_m0_peak_evidence(
    detector_native_counts: ArrayLike,
    *,
    candidates: tuple[ExpectedM0Peak, ...],
    detector_valid_mask: ArrayLike | None = None,
    policy: M0PeakEvidencePolicy | None = None,
) -> tuple[M0PeakEvidence, ...]:
    """Test structure-selected fixed-geometry 00L sites against raw native counts.

    This is an eligibility gate, not an intensity residual. Its z statistics and candidate
    structure-factor magnitude are not residual weights; a configured response may still contain
    within-profile structure-factor variation.
    """

    frozen = tuple(candidates)
    if any(not isinstance(item, ExpectedM0Peak) for item in frozen):
        raise ValueError("candidates must contain ExpectedM0Peak values")
    if not frozen:
        return ()
    frozen = tuple(sorted(frozen, key=lambda item: item.integer_L))
    if len({item.integer_L for item in frozen}) != len(frozen):
        raise ValueError("expected m=0 integer L identities must be unique")

    supplied_counts = np.asarray(detector_native_counts)
    if (
        supplied_counts.ndim != 2
        or np.iscomplexobj(supplied_counts)
        or not np.all(np.isfinite(supplied_counts))
        or np.any(supplied_counts < 0.0)
    ):
        raise ValueError("detector_native_counts must be a finite nonnegative real 2-D array")
    counts = supplied_counts
    if detector_valid_mask is None:
        valid_mask = np.ones(counts.shape, dtype=np.bool_)
    else:
        supplied_mask = np.asarray(detector_valid_mask)
        if supplied_mask.dtype.kind != "b" or supplied_mask.shape != counts.shape:
            raise ValueError("detector_valid_mask must be boolean with the detector shape")
        valid_mask = np.asarray(supplied_mask, dtype=np.bool_)
    active_policy = M0PeakEvidencePolicy() if policy is None else policy
    if not isinstance(active_policy, M0PeakEvidencePolicy):
        raise TypeError("policy must be M0PeakEvidencePolicy")

    rows, columns = counts.shape
    outer = active_policy.background_outer_radius_px
    evidence: list[M0PeakEvidence] = []
    for candidate in frozen:
        if not (-0.5 <= candidate.column_px <= columns - 0.5) or not (
            -0.5 <= candidate.row_px <= rows - 0.5
        ):
            evidence.append(
                M0PeakEvidence(candidate, "OUTSIDE_PANEL", 0, 0, 0, None, None, None, None, None)
            )
            continue
        full_column_start = math.floor(candidate.column_px - outer)
        full_column_stop = math.ceil(candidate.column_px + outer) + 1
        full_row_start = math.floor(candidate.row_px - outer)
        full_row_stop = math.ceil(candidate.row_px + outer) + 1
        full_row, full_column = np.ogrid[
            full_row_start:full_row_stop,
            full_column_start:full_column_stop,
        ]
        full_radius_squared = (full_column - candidate.column_px) ** 2 + (
            full_row - candidate.row_px
        ) ** 2
        expected_core_count = int(
            np.count_nonzero(full_radius_squared <= active_policy.core_radius_px**2)
        )
        expected_background_count = int(
            np.count_nonzero(
                (full_radius_squared >= active_policy.background_inner_radius_px**2)
                & (full_radius_squared <= outer**2)
            )
        )
        column_start = max(0, full_column_start)
        column_stop = min(columns, full_column_stop)
        row_start = max(0, full_row_start)
        row_stop = min(rows, full_row_stop)
        local_row, local_column = np.ogrid[row_start:row_stop, column_start:column_stop]
        radius_squared = (local_column - candidate.column_px) ** 2 + (
            local_row - candidate.row_px
        ) ** 2
        core_support = radius_squared <= active_policy.core_radius_px**2
        background_support = (radius_squared >= active_policy.background_inner_radius_px**2) & (
            radius_squared <= outer**2
        )
        local_valid = valid_mask[row_start:row_stop, column_start:column_stop]
        core_valid = core_support & local_valid
        background_valid = background_support & local_valid
        core_count = int(np.count_nonzero(core_valid))
        background_count = int(np.count_nonzero(background_valid))
        if (
            expected_core_count == 0
            or expected_background_count == 0
            or core_count < active_policy.minimum_valid_fraction * expected_core_count
            or background_count < active_policy.minimum_valid_fraction * expected_background_count
        ):
            evidence.append(
                M0PeakEvidence(
                    candidate,
                    "INSUFFICIENT_VALID_SUPPORT",
                    core_count,
                    background_count,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
            continue
        crop = np.asarray(
            counts[row_start:row_stop, column_start:column_stop],
            dtype=np.float64,
        )
        core = crop[core_valid]
        background = crop[background_valid]
        background_counts = float(np.median(background))
        mad = float(np.median(np.abs(background - background_counts)))
        robust_sigma = max(
            active_policy.mad_scale * mad,
            active_policy.sigma_floor_counts,
        )
        maximum_core = float(np.max(core))
        peak_z = (maximum_core - background_counts) / robust_sigma
        integrated_z = float(np.sum(core - background_counts)) / (
            robust_sigma * math.sqrt(core_count)
        )
        excess_pixel_count = int(
            np.count_nonzero(core > background_counts + active_policy.excess_pixel_z * robust_sigma)
        )
        classification = (
            "LOCAL_SIGNAL_SUPPORTED"
            if peak_z >= active_policy.minimum_peak_z
            and integrated_z >= active_policy.minimum_integrated_z
            and excess_pixel_count >= active_policy.minimum_excess_pixel_count
            else "LOCAL_SIGNAL_BELOW_GATE"
        )
        evidence.append(
            M0PeakEvidence(
                candidate,
                classification,
                core_count,
                background_count,
                excess_pixel_count,
                background_counts,
                robust_sigma,
                maximum_core,
                peak_z,
                integrated_z,
            )
        )
    return tuple(evidence)


@dataclass(frozen=True, slots=True)
class OscGeometryIndexingRun:
    series: OscGeometrySeriesConfiguration
    selection: MeasuredIndexingResult
    discoveries: tuple[MeasuredPeakDiscovery, ...]
    geometry_inputs: tuple[ConfiguredGeometryInputs, ...]
    geometry_contexts: tuple[GeometryOnlyEwaldContext, ...]
    indexed_images: tuple[IndexedGeometryImage, ...] | None
    geometry_setup_seconds: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.series, OscGeometrySeriesConfiguration):
            raise TypeError("series must be OscGeometrySeriesConfiguration")
        if not isinstance(self.selection, MeasuredIndexingResult):
            raise TypeError("selection must be MeasuredIndexingResult")
        discoveries = tuple(self.discoveries)
        if any(not isinstance(item, MeasuredPeakDiscovery) for item in discoveries):
            raise TypeError("discoveries must contain MeasuredPeakDiscovery values")
        declared_ids = tuple(item.image_id for item in self.series.images)
        if tuple(item.image_id for item in discoveries) != declared_ids:
            raise ValueError("discovery image IDs must match the declared OSC series order")
        geometry_inputs = tuple(self.geometry_inputs)
        geometry_contexts = tuple(self.geometry_contexts)
        if len(geometry_inputs) != len(declared_ids) or any(
            not isinstance(item, ConfiguredGeometryInputs) for item in geometry_inputs
        ):
            raise ValueError("geometry_inputs must match the declared OSC series order")
        if len(geometry_contexts) != len(declared_ids) or any(
            not isinstance(item, GeometryOnlyEwaldContext) for item in geometry_contexts
        ):
            raise ValueError("geometry_contexts must match the declared OSC series order")
        selected_by_id = {item.image_id: item for item in self.selection.image_results}
        if set(selected_by_id) != set(declared_ids):
            raise ValueError("selection image IDs must match the declared OSC series")
        declared_by_id = {item.image_id: item for item in self.series.images}
        for discovery, inputs, context in zip(
            discoveries,
            geometry_inputs,
            geometry_contexts,
            strict=True,
        ):
            selected = selected_by_id[discovery.image_id]
            declared = declared_by_id[discovery.image_id]
            declared_incidence = declared.axis_rotation_angles_deg[self.series.incidence_axis_index]
            if (
                selected.detector_data_hash != discovery.detector_data_hash
                or selected.detector_mask_hash != discovery.detector_mask_hash
                or selected.detector_mask_revision != discovery.detector_mask_revision
                or selected.policy != discovery.policy.track_policy
            ):
                raise ValueError("selection and discovery image provenance disagree")
            if not math.isclose(
                selected.incidence_angle_rad,
                math.radians(declared_incidence),
                rel_tol=0.0,
                abs_tol=1.0e-14,
            ):
                raise ValueError("selection incidence disagrees with the declared OSC series")
            configured_angle = inputs.config.instrument.axis_rotations[
                self.series.incidence_axis_index
            ].angle_deg
            if not math.isclose(
                configured_angle,
                declared_incidence,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            ):
                raise ValueError("geometry inputs disagree with the declared OSC series")
            frame = build_osc_angle_frame(
                mean_direction_lab=inputs.config.source.mean_direction_lab,
                instrument=context.instrument,
                sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
                revision=f"osc-geometry-angle-frame.{discovery.image_id}.v1",
            )
            if discovery.geometry_context_hash != _discovery_geometry_hash(
                context.instrument,
                frame,
            ) or selected.context_hash != _indexing_context_hash(
                discovery,
                context,
                frame,
            ):
                raise ValueError("selection geometry context disagrees with its source discovery")
        images = None if self.indexed_images is None else tuple(self.indexed_images)
        if images is not None:
            if tuple(item.image_id for item in images) != declared_ids:
                raise ValueError("indexed image IDs must match the declared OSC series order")
            for declared, image in zip(self.series.images, images, strict=True):
                if not isinstance(image, IndexedGeometryImage):
                    raise TypeError("indexed_images must contain IndexedGeometryImage values")
                incidence = declared.axis_rotation_angles_deg[self.series.incidence_axis_index]
                if not math.isclose(
                    image.commanded_angle_rad,
                    math.radians(incidence),
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                ):
                    raise ValueError("indexed image commanded angle disagrees with the series")
                discovery = next(item for item in discoveries if item.image_id == image.image_id)
                selected = selected_by_id[image.image_id]
                model_context = build_geometry_only_ewald_context(image.model.inputs)
                model_frame = build_osc_angle_frame(
                    mean_direction_lab=image.model.inputs.config.source.mean_direction_lab,
                    instrument=model_context.instrument,
                    sample_intersection_lab_m=(
                        model_context.incident.states.sample_intersection_lab_m[0]
                    ),
                    revision=f"osc-geometry-angle-frame.{image.image_id}.v1",
                )
                if discovery.geometry_context_hash != _discovery_geometry_hash(
                    model_context.instrument,
                    model_frame,
                ) or selected.context_hash != _indexing_context_hash(
                    discovery,
                    model_context,
                    model_frame,
                ):
                    raise ValueError(
                        "indexed image model geometry disagrees with its indexing context"
                    )
                observations = self.selection.observations_for(image.image_id)
                if (
                    image.observations.keys != observations.keys
                    or not np.array_equal(
                        image.observations.coordinates_px, observations.coordinates_px
                    )
                    or not np.array_equal(
                        image.observations.covariance_px2, observations.covariance_px2
                    )
                ):
                    raise ValueError("indexed images must retain the run's frozen observations")
        setup = float(self.geometry_setup_seconds)
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(setup) or setup < 0.0:
            raise ValueError("geometry_setup_seconds must be finite and nonnegative")
        if not math.isfinite(elapsed) or elapsed < setup:
            raise ValueError("elapsed_seconds must be finite and at least geometry setup time")
        object.__setattr__(self, "discoveries", discoveries)
        object.__setattr__(self, "geometry_inputs", geometry_inputs)
        object.__setattr__(self, "geometry_contexts", geometry_contexts)
        object.__setattr__(self, "indexed_images", images)
        object.__setattr__(self, "geometry_setup_seconds", setup)
        object.__setattr__(self, "elapsed_seconds", elapsed)


@dataclass(frozen=True, slots=True)
class FrozenOscGeometryReindexing:
    """A corrected-geometry relabeling bound to its frozen OSC discovery run."""

    selection: MeasuredIndexingResult
    source_manifest_hash: str
    source_discovery_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selection, MeasuredIndexingResult):
            raise TypeError("selection must be a MeasuredIndexingResult")
        if (
            not isinstance(self.source_manifest_hash, str)
            or len(self.source_manifest_hash) != 71
            or not self.source_manifest_hash.startswith("sha256-")
        ):
            raise ValueError("source_manifest_hash must be a SHA-256 identifier")
        hashes = tuple(sorted(self.source_discovery_hashes))
        if not hashes or len({image_id for image_id, _ in hashes}) != len(hashes):
            raise ValueError("source_discovery_hashes must have unique image IDs")
        if any(
            not image_id
            or not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256-")
            for image_id, digest in hashes
        ):
            raise ValueError("source_discovery_hashes must contain SHA-256 identifiers")
        object.__setattr__(self, "source_discovery_hashes", hashes)


def index_osc_geometry_series(
    series: OscGeometrySeriesConfiguration,
    *,
    blind_policy: BlindIndexingPolicy | None = None,
    track_policy: PeakIndexingPolicy | None = None,
    instrument_by_image_id: dict[str, CompiledInstrument] | None = None,
) -> OscGeometryIndexingRun:
    """Index one declared OSC series, retaining a preflight run when admission is incomplete."""

    if not isinstance(series, OscGeometrySeriesConfiguration):
        raise TypeError("series must be OscGeometrySeriesConfiguration")
    blind = (
        BlindIndexingPolicy(track_policy=PeakIndexingPolicy(cake_step_px=1.0))
        if blind_policy is None
        else blind_policy
    )
    if not isinstance(blind, BlindIndexingPolicy):
        raise TypeError("blind_policy must be BlindIndexingPolicy")
    if track_policy is not None and not isinstance(track_policy, PeakIndexingPolicy):
        raise TypeError("track_policy must be PeakIndexingPolicy")
    if track_policy is not None:
        blind = replace(blind, track_policy=track_policy)
    track = blind.track_policy
    overrides = {} if instrument_by_image_id is None else dict(instrument_by_image_id)
    expected_ids = {image.image_id for image in series.images}
    if set(overrides) - expected_ids:
        raise ValueError("instrument overrides contain unknown image IDs")

    base = load_simulation_config(series.config_path)
    if len(base.instrument.axis_rotations) != 1 or series.incidence_axis_index != 0:
        raise ValueError(
            "OSC geometry fitting v1 requires one configured goniometer axis at index 0"
        )
    started = perf_counter()
    geometry_started = perf_counter()
    first_config = simulation_config_for_osc_image(base, series.images[0])
    shared_inputs = build_configured_geometry_inputs(first_config)
    geometry_setup_seconds = perf_counter() - geometry_started
    image_results = []
    discoveries = []
    models = []
    geometry_inputs = []
    geometry_contexts = []
    for index, image in enumerate(series.images):
        config = simulation_config_for_osc_image(base, image)
        inputs = (
            shared_inputs
            if index == 0
            else rebind_configured_geometry_instrument(shared_inputs, config)
        )
        override = overrides.get(image.image_id)
        if override is not None and not isinstance(override, CompiledInstrument):
            raise TypeError("instrument overrides must be CompiledInstrument values")
        context = build_geometry_only_ewald_context(inputs, instrument=override)
        geometry_inputs.append(inputs)
        geometry_contexts.append(context)
        osc = read_osc(image.osc_path)
        frame = build_osc_angle_frame(
            mean_direction_lab=inputs.config.source.mean_direction_lab,
            instrument=context.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"osc-geometry-angle-frame.{image.image_id}.v1",
        )
        discovery = discover_measured_cake_peaks(
            osc.detector_native_counts,
            instrument=context.instrument,
            angle_frame=frame,
            image_id=image.image_id,
            detector_valid_mask=detector_valid_mask_from_counts(osc.detector_native_counts),
            detector_mask_revision="osc-all-zero-edge-mask.v1",
            policy=blind,
        )
        discoveries.append(discovery)
        incidence_deg = image.axis_rotation_angles_deg[series.incidence_axis_index]
        image_results.append(
            index_discovered_integer_l_peaks(
                discovery,
                ewald_context=context,
                angle_frame=frame,
                incidence_angle_rad=math.radians(incidence_deg),
            )
        )
        models.append(ExactTagGeometryModel(inputs))
    selection = select_confident_branch_tracks(tuple(image_results), policy=track)
    observation_packs = []
    if not overrides:
        for image in series.images:
            try:
                observation_packs.append(selection.observations_for(image.image_id))
            except ValueError as error:
                expected = f"image {image.image_id!r} has no accepted visible marker observations"
                if str(error) != expected:
                    raise
                observation_packs = []
                break
    indexed_images = (
        tuple(
            IndexedGeometryImage(
                image_id=image.image_id,
                commanded_angle_rad=math.radians(
                    image.axis_rotation_angles_deg[series.incidence_axis_index]
                ),
                model=model,
                observations=observations,
            )
            for image, model, observations in zip(
                series.images,
                models,
                observation_packs,
                strict=True,
            )
        )
        if observation_packs
        else None
    )
    return OscGeometryIndexingRun(
        series=series,
        selection=selection,
        discoveries=tuple(discoveries),
        geometry_inputs=tuple(geometry_inputs),
        geometry_contexts=tuple(geometry_contexts),
        indexed_images=indexed_images,
        geometry_setup_seconds=geometry_setup_seconds,
        elapsed_seconds=perf_counter() - started,
    )


def reindex_frozen_osc_geometry_series(
    run: OscGeometryIndexingRun,
    *,
    instrument_by_image_id: dict[str, CompiledInstrument],
) -> FrozenOscGeometryReindexing:
    """Relabel one run's immutable native candidates without OSC I/O or discovery."""

    if not isinstance(run, OscGeometryIndexingRun):
        raise TypeError("run must be an OscGeometryIndexingRun")
    if run.indexed_images is None:
        raise ValueError("run must retain its provenance-bound geometry models")
    instruments = dict(instrument_by_image_id)
    expected_ids = {image.image_id for image in run.series.images}
    if set(instruments) != expected_ids:
        raise ValueError("corrected instruments must match the complete OSC image-ID set")
    indexed_by_id = {image.image_id: image for image in run.indexed_images}
    results = []
    for declared, discovery in zip(run.series.images, run.discoveries, strict=True):
        instrument = instruments[declared.image_id]
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("corrected instruments must be CompiledInstrument values")
        indexed = indexed_by_id[declared.image_id]
        context = build_geometry_only_ewald_context(
            indexed.model.inputs,
            instrument=instrument,
        )
        frame = build_osc_angle_frame(
            mean_direction_lab=indexed.model.inputs.config.source.mean_direction_lab,
            instrument=context.instrument,
            sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
            revision=f"osc-geometry-angle-frame.{declared.image_id}.v1",
        )
        results.append(
            reindex_frozen_discovery_coordinates(
                discovery,
                frozen_observations=run.selection.observations_for(declared.image_id),
                ewald_context=context,
                angle_frame=frame,
                incidence_angle_rad=indexed.commanded_angle_rad,
            )
        )
    return FrozenOscGeometryReindexing(
        selection=select_confident_branch_tracks(tuple(results), policy=run.selection.policy),
        source_manifest_hash=run.selection.manifest_hash,
        source_discovery_hashes=tuple(
            (discovery.image_id, discovery.discovery_hash) for discovery in run.discoveries
        ),
    )


def audit_frozen_osc_geometry_reindexing(
    run: OscGeometryIndexingRun,
    reindexed: FrozenOscGeometryReindexing,
    *,
    instrument_by_image_id: dict[str, CompiledInstrument],
) -> FrozenMarkerVisibilityAudit:
    """Recompute and verify exact frozen-reindex lineage before visibility comparison."""

    if not isinstance(run, OscGeometryIndexingRun):
        raise TypeError("run must be an OscGeometryIndexingRun")
    if not isinstance(reindexed, FrozenOscGeometryReindexing):
        raise TypeError("reindexed must be a FrozenOscGeometryReindexing")
    if run.indexed_images is None:
        raise ValueError("run must retain its provenance-bound geometry models")
    instruments = dict(instrument_by_image_id)
    expected = reindex_frozen_osc_geometry_series(
        run,
        instrument_by_image_id=instruments,
    )
    if (
        reindexed.source_manifest_hash != expected.source_manifest_hash
        or reindexed.source_discovery_hashes != expected.source_discovery_hashes
        or reindexed.selection.manifest_hash != expected.selection.manifest_hash
    ):
        raise ValueError(
            "frozen reindexing result is not the exact source-coordinate relabeling for the "
            "corrected geometry"
        )
    return audit_frozen_marker_visibility(run.selection, reindexed.selection)


__all__ = [
    "DETECTOR_VALID_MASK_REVISION",
    "M0_PEAK_EVIDENCE_REVISION",
    "ExpectedM0Peak",
    "FrozenOscGeometryReindexing",
    "M0PeakEvidence",
    "M0PeakEvidencePolicy",
    "OscGeometryImageConfiguration",
    "OscGeometryIndexingRun",
    "OscGeometrySeriesConfiguration",
    "audit_frozen_osc_geometry_reindexing",
    "build_osc_angle_frame",
    "detector_valid_mask_from_counts",
    "evaluate_expected_m0_peak_evidence",
    "index_osc_geometry_series",
    "load_osc_geometry_series",
    "reindex_frozen_osc_geometry_series",
    "simulation_config_for_osc_image",
]
