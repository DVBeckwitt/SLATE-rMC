"""Recreate the Bi2Se3 Figure-7 layout from a verified staged-fit checkpoint.

Measured profiles are finite-bin reductions of the native OSC pixels.  Fitted profiles are
independent continuous detector-coordinate cubatures over the same reciprocal-space regions;
the model is never rasterized or reconstructed from the measured image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
SCRIPTS_ROOT = ROOT / "scripts"
for import_root in (SOURCE_ROOT, SCRIPTS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_REPLAY_CASE = ROOT / "examples" / "bi2se3" / "experiment" / "staged_fit_replay.toml"
DEFAULT_FIGURE_CONFIG = ROOT / "examples" / "bi2se3" / "experiment" / "figure7_recreation.toml"
DIAGNOSTIC_SCHEMA = "rasim-fitted-figure7-recreation-diagnostic-v2"
OUTPUT_SCHEMA = "rasim-fitted-figure7-recreation-output-v2"


@dataclass(frozen=True, slots=True)
class RegionSettings:
    """One explicitly bounded reciprocal-space region drawn on the detector."""

    family_m: int
    h: int
    k: int
    qr_half_width_Ainv: float
    minimum_L: float
    maximum_L: float
    color: str


@dataclass(frozen=True, slots=True)
class LabelSettings:
    """One detector annotation anchored in raw top-origin native coordinates."""

    text: str
    column_px: float
    row_px: float


@dataclass(frozen=True, slots=True)
class ProfileSettings:
    """One side-specific measured/model profile derived from a displayed region."""

    identity: str
    label: str
    family_m: int
    minimum_column_px: float
    maximum_column_px: float
    minimum_L: float
    maximum_L: float
    bin_width_L: float
    sideband_gap_Ainv: float
    sideband_width_Ainv: float


@dataclass(frozen=True, slots=True)
class CubatureSettings:
    """Deterministic continuous detector-coordinate integration controls."""

    coarse_tile_size_px: int
    fine_tile_size_px: int
    coarse_gauss_order: int
    fine_gauss_order: int
    refinement_gauss_order: int
    minimum_refinement_tile_size_px: int
    maximum_state_block_count: int
    cuda_coordinate_chunk_size: int
    maximum_profile_relative_l2: float
    maximum_integrated_mass_relative_error: float


@dataclass(frozen=True, slots=True)
class FigureSettings:
    """Immutable numerical and presentation recipe for one Figure-7 recreation."""

    material_id: str
    display_dataset_id: str
    display_incidence_deg: float
    crop_shape_rc: tuple[int, int]
    profile_shape_rc: tuple[int, int]
    geometry_row_chunk_size: int
    detector_low_percentile: float
    detector_high_percentile: float
    regions: tuple[RegionSettings, ...]
    profiles: tuple[ProfileSettings, ...]
    labels: tuple[LabelSettings, ...]
    cubature: CubatureSettings


def _strict_keys(mapping: dict[str, Any], expected: set[str], name: str) -> None:
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(f"{name} keys changed; missing={missing}, extra={extra}")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def load_figure_settings(path: str | Path) -> FigureSettings:
    """Load a strict recipe that resolves the ambiguous legacy region widths."""

    document = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _strict_keys(
        document,
        {
            "schema_version",
            "material_id",
            "display_dataset_id",
            "display_incidence_deg",
            "detector",
            "model_cubature",
            "region",
            "profile",
            "label",
        },
        "figure recipe",
    )
    if document["schema_version"] != "rasim-fitted-figure7-recreation-v2":
        raise ValueError("unsupported fitted Figure-7 recipe schema")
    detector = document["detector"]
    if not isinstance(detector, dict):
        raise ValueError("detector recipe must be a table")
    _strict_keys(
        detector,
        {
            "crop_rows",
            "crop_columns",
            "profile_rows",
            "profile_columns",
            "geometry_row_chunk_size",
            "low_percentile",
            "high_percentile",
        },
        "detector recipe",
    )
    rows = _integer(detector["crop_rows"], "detector.crop_rows")
    columns = _integer(detector["crop_columns"], "detector.crop_columns")
    profile_rows = _integer(detector["profile_rows"], "detector.profile_rows")
    profile_columns = _integer(detector["profile_columns"], "detector.profile_columns")
    chunk = _integer(detector["geometry_row_chunk_size"], "detector.geometry_row_chunk_size")
    low = _number(detector["low_percentile"], "detector.low_percentile")
    high = _number(detector["high_percentile"], "detector.high_percentile")
    incidence = _number(document["display_incidence_deg"], "display_incidence_deg")
    if (
        rows <= 0
        or columns <= 0
        or profile_rows < rows
        or profile_columns < columns
        or chunk <= 0
        or not 0.0 <= low < high <= 100.0
    ):
        raise ValueError("detector display settings contain invalid values")

    cubature_data = document["model_cubature"]
    if not isinstance(cubature_data, dict):
        raise ValueError("model_cubature must be a table")
    _strict_keys(
        cubature_data,
        {
            "coarse_tile_size_px",
            "fine_tile_size_px",
            "coarse_gauss_order",
            "fine_gauss_order",
            "refinement_gauss_order",
            "minimum_refinement_tile_size_px",
            "maximum_state_block_count",
            "cuda_coordinate_chunk_size",
            "maximum_profile_relative_l2",
            "maximum_integrated_mass_relative_error",
        },
        "model cubature",
    )
    cubature = CubatureSettings(
        coarse_tile_size_px=_integer(
            cubature_data["coarse_tile_size_px"], "model_cubature.coarse_tile_size_px"
        ),
        fine_tile_size_px=_integer(
            cubature_data["fine_tile_size_px"], "model_cubature.fine_tile_size_px"
        ),
        coarse_gauss_order=_integer(
            cubature_data["coarse_gauss_order"], "model_cubature.coarse_gauss_order"
        ),
        fine_gauss_order=_integer(
            cubature_data["fine_gauss_order"], "model_cubature.fine_gauss_order"
        ),
        refinement_gauss_order=_integer(
            cubature_data["refinement_gauss_order"],
            "model_cubature.refinement_gauss_order",
        ),
        minimum_refinement_tile_size_px=_integer(
            cubature_data["minimum_refinement_tile_size_px"],
            "model_cubature.minimum_refinement_tile_size_px",
        ),
        maximum_state_block_count=_integer(
            cubature_data["maximum_state_block_count"],
            "model_cubature.maximum_state_block_count",
        ),
        cuda_coordinate_chunk_size=_integer(
            cubature_data["cuda_coordinate_chunk_size"],
            "model_cubature.cuda_coordinate_chunk_size",
        ),
        maximum_profile_relative_l2=_number(
            cubature_data["maximum_profile_relative_l2"],
            "model_cubature.maximum_profile_relative_l2",
        ),
        maximum_integrated_mass_relative_error=_number(
            cubature_data["maximum_integrated_mass_relative_error"],
            "model_cubature.maximum_integrated_mass_relative_error",
        ),
    )
    if (
        cubature.coarse_tile_size_px < cubature.fine_tile_size_px
        or cubature.fine_tile_size_px <= 0
        or cubature.coarse_gauss_order < 2
        or cubature.coarse_gauss_order % 2
        or cubature.fine_gauss_order <= cubature.coarse_gauss_order
        or cubature.fine_gauss_order % 2
        or cubature.refinement_gauss_order < cubature.fine_gauss_order
        or cubature.refinement_gauss_order % 2
        or cubature.minimum_refinement_tile_size_px <= 0
        or cubature.minimum_refinement_tile_size_px >= cubature.fine_tile_size_px
        or cubature.fine_tile_size_px % cubature.minimum_refinement_tile_size_px
        or cubature.maximum_state_block_count <= 0
        or cubature.cuda_coordinate_chunk_size <= 0
        or cubature.maximum_profile_relative_l2 <= 0.0
        or cubature.maximum_integrated_mass_relative_error <= 0.0
    ):
        raise ValueError("model cubature settings are invalid")

    regions_data = document["region"]
    profiles_data = document["profile"]
    labels_data = document["label"]
    if not all(isinstance(value, list) for value in (regions_data, profiles_data, labels_data)):
        raise ValueError("region, profile, and label recipes must be arrays of tables")
    regions: list[RegionSettings] = []
    for record in regions_data:
        if not isinstance(record, dict):
            raise ValueError("each reciprocal region must be a table")
        _strict_keys(
            record,
            {
                "family_m",
                "h",
                "k",
                "qr_half_width_Ainv",
                "minimum_L",
                "maximum_L",
                "color",
            },
            "reciprocal region",
        )
        region = RegionSettings(
            family_m=_integer(record["family_m"], "region.family_m"),
            h=_integer(record["h"], "region.h"),
            k=_integer(record["k"], "region.k"),
            qr_half_width_Ainv=_number(record["qr_half_width_Ainv"], "region.qr_half_width_Ainv"),
            minimum_L=_number(record["minimum_L"], "region.minimum_L"),
            maximum_L=_number(record["maximum_L"], "region.maximum_L"),
            color=_text(record["color"], "region.color"),
        )
        if (
            region.family_m < 0
            or region.qr_half_width_Ainv <= 0.0
            or region.minimum_L >= region.maximum_L
            or len(region.color) != 7
            or region.color[0] != "#"
            or any(character not in "0123456789abcdefABCDEF" for character in region.color[1:])
        ):
            raise ValueError("reciprocal region contains invalid values")
        regions.append(region)
    if not regions or len({region.family_m for region in regions}) != len(regions):
        raise ValueError("reciprocal regions must contain unique families")
    if tuple(region.family_m for region in regions) != tuple(
        sorted(region.family_m for region in regions)
    ):
        raise ValueError("reciprocal regions must be sorted by family_m")

    profiles: list[ProfileSettings] = []
    region_by_family = {region.family_m: region for region in regions}
    for record in profiles_data:
        if not isinstance(record, dict):
            raise ValueError("each reciprocal profile must be a table")
        _strict_keys(
            record,
            {
                "identity",
                "label",
                "family_m",
                "minimum_column_px",
                "maximum_column_px",
                "minimum_L",
                "maximum_L",
                "bin_width_L",
                "sideband_gap_Ainv",
                "sideband_width_Ainv",
            },
            "reciprocal profile",
        )
        profile = ProfileSettings(
            identity=_text(record["identity"], "profile.identity"),
            label=_text(record["label"], "profile.label"),
            family_m=_integer(record["family_m"], "profile.family_m"),
            minimum_column_px=_number(record["minimum_column_px"], "profile.minimum_column_px"),
            maximum_column_px=_number(record["maximum_column_px"], "profile.maximum_column_px"),
            minimum_L=_number(record["minimum_L"], "profile.minimum_L"),
            maximum_L=_number(record["maximum_L"], "profile.maximum_L"),
            bin_width_L=_number(record["bin_width_L"], "profile.bin_width_L"),
            sideband_gap_Ainv=_number(record["sideband_gap_Ainv"], "profile.sideband_gap_Ainv"),
            sideband_width_Ainv=_number(
                record["sideband_width_Ainv"], "profile.sideband_width_Ainv"
            ),
        )
        span_in_bins = (profile.maximum_L - profile.minimum_L) / profile.bin_width_L
        if (
            profile.family_m not in region_by_family
            or profile.minimum_column_px < -0.5
            or profile.maximum_column_px > profile_columns - 0.5
            or profile.minimum_column_px >= profile.maximum_column_px
            or profile.minimum_L < region_by_family[profile.family_m].minimum_L
            or profile.maximum_L > region_by_family[profile.family_m].maximum_L
            or profile.minimum_L >= profile.maximum_L
            or profile.bin_width_L <= 0.0
            or not math.isclose(span_in_bins, round(span_in_bins), rel_tol=0.0, abs_tol=1.0e-10)
            or profile.sideband_gap_Ainv < 0.0
            or profile.sideband_width_Ainv <= 0.0
        ):
            raise ValueError("reciprocal profile contains invalid values")
        profiles.append(profile)
    if len(profiles) != 7 or len({profile.identity for profile in profiles}) != len(profiles):
        raise ValueError("Figure 7 requires seven uniquely identified profiles")

    labels: list[LabelSettings] = []
    for record in labels_data:
        if not isinstance(record, dict):
            raise ValueError("each detector label must be a table")
        _strict_keys(record, {"text", "column_px", "row_px"}, "detector label")
        label = LabelSettings(
            text=_text(record["text"], "label.text"),
            column_px=_number(record["column_px"], "label.column_px"),
            row_px=_number(record["row_px"], "label.row_px"),
        )
        if not 0.0 <= label.column_px <= columns or not 0.0 <= label.row_px <= rows:
            raise ValueError("detector label contains invalid values")
        labels.append(label)
    return FigureSettings(
        material_id=_text(document["material_id"], "material_id"),
        display_dataset_id=_text(document["display_dataset_id"], "display_dataset_id"),
        display_incidence_deg=incidence,
        crop_shape_rc=(rows, columns),
        profile_shape_rc=(profile_rows, profile_columns),
        geometry_row_chunk_size=chunk,
        detector_low_percentile=low,
        detector_high_percentile=high,
        regions=tuple(regions),
        profiles=tuple(profiles),
        labels=tuple(labels),
        cubature=cubature,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return document


def _external_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError("figure artifacts must resolve outside the repository")
    return resolved


def _verified_checkpoint_chain(
    checkpoint_directory: Path,
    *,
    case: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from rasim_next.core.staged_fit import staged_fit_scientific_revision

    certificate_path = checkpoint_directory / "replay_certificate.json"
    certificate = _json_document(certificate_path)
    if (
        certificate.get("schema_version") != "rasim-staged-fit-replay-certificate-v2"
        or certificate.get("verified") is not True
        or certificate.get("through") != "ordered_intensity"
        or certificate.get("case_id") != case.case_id
        or certificate.get("material_id") != case.material_id
        or certificate.get("case_sha256") != _sha256(case.path)
    ):
        raise ValueError("checkpoint lacks a matching verified ordered-intensity certificate")
    stages = certificate.get("stages")
    revisions = certificate.get("stage_scientific_revisions")
    expected_names = ("geometry", "mosaic", "ordered_intensity")
    if (
        not isinstance(stages, dict)
        or tuple(stages) != expected_names
        or not isinstance(revisions, dict)
        or set(revisions) != set(expected_names)
    ):
        raise ValueError("verified checkpoint certificate has an invalid stage chain")
    envelopes: dict[str, dict[str, Any]] = {}
    upstream_revision: str | None = None
    for stage in expected_names:
        envelope = _json_document(checkpoint_directory / f"{stage}.json")
        certificate_envelope = stages.get(stage)
        if envelope != certificate_envelope:
            raise ValueError(f"checkpoint {stage} envelope differs from its certificate")
        revision = envelope.get("scientific_revision")
        payload = {name: value for name, value in envelope.items() if name != "scientific_revision"}
        if (
            not isinstance(revision, str)
            or revision != staged_fit_scientific_revision(stage, payload)
            or revision != revisions[stage]
            or envelope.get("upstream_scientific_revision") != upstream_revision
        ):
            raise ValueError(f"checkpoint {stage} scientific revision or lineage changed")
        upstream_revision = revision
        envelopes[stage] = envelope
    return certificate, envelopes


def _checkpoint_artifact(
    checkpoint_directory: Path,
    *,
    stage: str,
    relative_directory: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    stage_path = checkpoint_directory / f"{stage}.json"
    envelope = _json_document(stage_path)
    if (
        envelope.get("schema_version") != "rasim-staged-fit-replay-stage-v2"
        or envelope.get("stage") != stage
    ):
        raise ValueError(f"checkpoint {stage} envelope has an unsupported contract")
    state = envelope.get("state")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint {stage} envelope lacks state")
    declared = Path(str(state.get("artifact", "")))
    artifact = (
        declared
        if declared.is_file()
        else checkpoint_directory / relative_directory / declared.name
    )
    expected_hash = state.get("artifact_sha256")
    if (
        not artifact.is_file()
        or not isinstance(expected_hash, str)
        or _sha256(artifact) != expected_hash
    ):
        raise ValueError(f"checkpoint {stage} artifact is missing or changed")
    return envelope, artifact.resolve(), _json_document(artifact)


def _detector_qr_l_maps(
    inputs: Any,
    *,
    shape_rc: tuple[int, int],
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map final native detector pixels to crystal-frame Qr and L without intensity work."""

    from rasim_next.measurement.reciprocal_profiles import LayeredReciprocalFrame
    from rasim_next.pipeline.configured_simulation import build_nominal_ewald_context

    rows, columns = shape_rc
    detector_rows, detector_columns = inputs.instrument.detector_shape_rc
    if rows > detector_rows or columns > detector_columns:
        raise ValueError("figure crop exceeds the native detector")
    context = build_nominal_ewald_context(inputs)
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    qr_Ainv = np.full(shape_rc, np.nan, dtype=np.float32)
    L = np.full(shape_rc, np.nan, dtype=np.float32)
    valid = np.zeros(shape_rc, dtype=np.bool_)
    column = np.arange(columns, dtype=np.float64)
    for start in range(0, rows, row_chunk_size):
        stop = min(rows, start + row_chunk_size)
        row = np.arange(start, stop, dtype=np.float64)
        column_grid, row_grid = np.broadcast_arrays(column[None, :], row[:, None])
        geometry = context.evaluate_detector_geometry(
            column_grid,
            row_grid,
            include_surface_jacobian=False,
        )
        accepted = np.asarray(geometry.valid, dtype=np.bool_)
        chunk_qr, chunk_L = frame.coordinates(geometry.q_sample_Ainv)
        finite = accepted & np.isfinite(chunk_qr) & np.isfinite(chunk_L)
        qr_Ainv[start:stop] = np.where(finite, chunk_qr, np.nan).astype(np.float32)
        L[start:stop] = np.where(finite, chunk_L, np.nan).astype(np.float32)
        valid[start:stop] = finite
    return qr_Ainv, L, valid


def _profile_regions(settings: FigureSettings, inputs: Any) -> tuple[Any, ...]:
    from rasim_next.measurement.reciprocal_profiles import (
        LayeredReciprocalFrame,
        ReciprocalProfileRegion,
    )

    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    display_region = {region.family_m: region for region in settings.regions}
    profiles: list[ReciprocalProfileRegion] = []
    for profile in settings.profiles:
        family = display_region[profile.family_m]
        bin_count = round((profile.maximum_L - profile.minimum_L) / profile.bin_width_L)
        edges = np.linspace(profile.minimum_L, profile.maximum_L, bin_count + 1)
        profiles.append(
            ReciprocalProfileRegion(
                identity=profile.identity,
                qr_center_Ainv=frame.rod_radial_coordinate_Ainv((family.h, family.k, 0)),
                qr_half_width_Ainv=family.qr_half_width_Ainv,
                axial_bin_edges=edges,
                detector_column_interval_px=(
                    profile.minimum_column_px,
                    profile.maximum_column_px,
                ),
                sideband_gap_Ainv=profile.sideband_gap_Ainv,
                sideband_width_Ainv=profile.sideband_width_Ainv,
            )
        )
    return tuple(profiles)


def _measured_profile_reduction(
    *,
    detector_counts: np.ndarray,
    qr_Ainv: np.ndarray,
    axial_L: np.ndarray,
    detector_valid: np.ndarray,
    profiles: Sequence[Any],
) -> dict[str, dict[str, np.ndarray]]:
    from rasim_next.measurement.reciprocal_profiles import (
        accumulate_binned_samples,
        reciprocal_profile_membership,
    )

    rows, columns = detector_counts.shape
    column_px = np.broadcast_to(np.arange(columns, dtype=np.float64)[None, :], (rows, columns))
    unit_measure = np.ones((rows, columns), dtype=np.float64)
    reduced: dict[str, dict[str, np.ndarray]] = {}
    for profile in profiles:
        membership = reciprocal_profile_membership(
            qr_Ainv,
            axial_L,
            column_px,
            detector_valid,
            region=profile,
        )
        signal = accumulate_binned_samples(
            values=detector_counts,
            measure_weights=unit_measure,
            bin_index=membership.signal_bin_index,
            bin_count=profile.bin_count,
        )
        sideband = accumulate_binned_samples(
            values=detector_counts,
            measure_weights=unit_measure,
            bin_index=membership.sideband_bin_index,
            bin_count=profile.bin_count,
        )
        signal_mean = signal.mean
        background_mean = sideband.mean
        valid = (
            np.isfinite(signal_mean)
            & np.isfinite(background_mean)
            & (signal.measure_sum > 0.0)
            & (sideband.measure_sum > 0.0)
        )
        reduced[profile.identity] = {
            "L": profile.axial_bin_centers,
            "raw_mean": signal_mean,
            "background_mean": background_mean,
            "background_subtracted_mean": signal_mean - background_mean,
            "signal_pixel_count": signal.measure_sum,
            "sideband_pixel_count": sideband.measure_sum,
            "valid": valid,
        }
    return reduced


def _detector_cubature_nodes(
    shape_rc: tuple[int, int],
    *,
    tile_size_px: int,
    gauss_order: int,
    candidate_pixel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Gauss nodes on candidate tiles plus a one-tile boundary halo."""

    rows, columns = shape_rc
    if (
        isinstance(rows, bool)
        or not isinstance(rows, (int, np.integer))
        or isinstance(columns, bool)
        or not isinstance(columns, (int, np.integer))
        or rows <= 0
        or columns <= 0
    ):
        raise ValueError("shape_rc must contain two positive integers")
    if (
        isinstance(tile_size_px, bool)
        or not isinstance(tile_size_px, (int, np.integer))
        or tile_size_px <= 0
        or isinstance(gauss_order, bool)
        or not isinstance(gauss_order, (int, np.integer))
        or gauss_order < 2
    ):
        raise ValueError("tile_size_px and gauss_order must be valid positive integers")
    candidate = np.asarray(candidate_pixel_mask, dtype=np.bool_)
    if candidate.shape != shape_rc or not np.any(candidate):
        raise ValueError("candidate_pixel_mask must select native detector pixels")
    node, weight = np.polynomial.legendre.leggauss(gauss_order)
    tile_row_count = (rows + tile_size_px - 1) // tile_size_px
    tile_column_count = (columns + tile_size_px - 1) // tile_size_px
    selected_row, selected_column = np.nonzero(candidate)
    selected_tile = np.unique(
        (selected_row // tile_size_px) * tile_column_count + selected_column // tile_size_px
    )
    tile_row = selected_tile // tile_column_count
    tile_column = selected_tile % tile_column_count
    halo_tiles: list[np.ndarray] = []
    for row_offset in (-1, 0, 1):
        for column_offset in (-1, 0, 1):
            halo_row = tile_row + row_offset
            halo_column = tile_column + column_offset
            accepted = (
                (halo_row >= 0)
                & (halo_row < tile_row_count)
                & (halo_column >= 0)
                & (halo_column < tile_column_count)
            )
            halo_tiles.append(halo_row[accepted] * tile_column_count + halo_column[accepted])
    selected_tile = np.unique(np.concatenate(halo_tiles))
    tile_row = selected_tile // tile_column_count
    tile_column = selected_tile % tile_column_count
    lower_row = tile_row.astype(np.float64) * tile_size_px - 0.5
    lower_column = tile_column.astype(np.float64) * tile_size_px - 0.5
    upper_row = np.minimum(lower_row + tile_size_px, rows - 0.5)
    upper_column = np.minimum(lower_column + tile_size_px, columns - 0.5)
    center_row = 0.5 * (lower_row + upper_row)
    center_column = 0.5 * (lower_column + upper_column)
    half_row = 0.5 * (upper_row - lower_row)
    half_column = 0.5 * (upper_column - lower_column)
    column = center_column[:, None, None] + half_column[:, None, None] * node[None, :, None]
    row = center_row[:, None, None] + half_row[:, None, None] * node[None, None, :]
    area = (
        half_column[:, None, None]
        * half_row[:, None, None]
        * weight[None, :, None]
        * weight[None, None, :]
    )
    shape = (selected_tile.size, gauss_order, gauss_order)
    return (
        np.broadcast_to(column, shape).reshape(-1),
        np.broadcast_to(row, shape).reshape(-1),
        np.broadcast_to(area, shape).reshape(-1),
    )


def _cubature_profile_pass(
    *,
    inputs: Any,
    detector_by_family: dict[int, Any],
    family_by_identity: dict[str, int],
    profiles: Sequence[Any],
    shape_rc: tuple[int, int],
    tile_size_px: int,
    gauss_order: int,
    execution_backend: str,
    cuda_coordinate_chunk_size: int,
    candidate_pixel_mask_by_family: dict[int, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    from rasim_next.measurement.reciprocal_profiles import (
        LayeredReciprocalFrame,
        accumulate_binned_samples,
        reciprocal_profile_membership,
    )
    from rasim_next.pipeline.configured_simulation import build_nominal_ewald_context

    start = perf_counter()
    context = build_nominal_ewald_context(inputs)
    frame = LayeredReciprocalFrame(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        sample_from_crystal_rotation=inputs.instrument.sample_from_crystal.rotation,
        axial_basis_index=2,
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    execution_records: list[dict[str, Any]] = []
    candidate_coordinate_count = 0
    for family_m, detector in detector_by_family.items():
        family_profiles = tuple(
            profile for profile in profiles if family_by_identity[profile.identity] == family_m
        )
        column_px, row_px, measure_weight = _detector_cubature_nodes(
            shape_rc,
            tile_size_px=tile_size_px,
            gauss_order=gauss_order,
            candidate_pixel_mask=candidate_pixel_mask_by_family[family_m],
        )
        candidate_coordinate_count += int(column_px.size)
        geometry = context.evaluate_detector_geometry(
            column_px,
            row_px,
            include_surface_jacobian=False,
        )
        qr_Ainv, axial_L = frame.coordinates(geometry.q_sample_Ainv)
        membership_by_identity = {
            profile.identity: reciprocal_profile_membership(
                qr_Ainv,
                axial_L,
                column_px,
                geometry.valid,
                region=profile,
            )
            for profile in family_profiles
        }
        selected = np.zeros(column_px.shape, dtype=np.bool_)
        for profile in family_profiles:
            selected |= membership_by_identity[profile.identity].signal_bin_index >= 0
        selected_count = int(np.count_nonzero(selected))
        if selected_count == 0:
            raise RuntimeError(f"continuous cubature found no nodes for family m={family_m}")
        evaluated = detector.evaluate_detector_density_all_roots(
            column_px[selected],
            row_px[selected],
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=(
                cuda_coordinate_chunk_size if execution_backend == "cuda" else None
            ),
        )
        density = np.asarray(evaluated.density_A2_per_px2, dtype=np.float64)
        if np.any(~np.isfinite(density)) or np.any(evaluated.caustic):
            raise RuntimeError(
                "continuous detector cubature encountered a caustic or nonfinite value"
            )
        execution_records.append(
            {
                "family_m": family_m,
                "candidate_coordinate_count": int(column_px.size),
                "selected_coordinate_count": selected_count,
                "backend": evaluated.execution_backend,
                "device": evaluated.execution_device,
                "minimum_valid_source_count": int(np.min(evaluated.valid_source_count)),
                "maximum_valid_source_count": int(np.max(evaluated.valid_source_count)),
            }
        )
        for profile in family_profiles:
            indices = membership_by_identity[profile.identity].signal_bin_index[selected]
            integral = accumulate_binned_samples(
                values=density,
                measure_weights=measure_weight[selected],
                bin_index=indices,
                bin_count=profile.bin_count,
            )
            result[profile.identity] = {
                "signal_sum_A2": integral.signal_sum,
                "measure_sum_px2": integral.measure_sum,
                "mean_A2_per_px2": integral.mean,
            }
    return result, {
        "tile_size_px": tile_size_px,
        "gauss_order": gauss_order,
        "candidate_coordinate_count": candidate_coordinate_count,
        "elapsed_seconds": perf_counter() - start,
        "family_execution": execution_records,
    }


def _continuous_fitted_profiles(
    *,
    inputs: Any,
    structure_representative: dict[str, Any],
    profiles: Sequence[Any],
    family_by_identity: dict[str, int],
    detector_qr_Ainv: np.ndarray,
    detector_L: np.ndarray,
    detector_valid: np.ndarray,
    settings: FigureSettings,
    execution_backend: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    from rasim_next.measurement.reciprocal_profiles import reciprocal_profile_membership
    from rasim_next.ordered.motifs import Bi2Se3QuintupleLayerParameters
    from rasim_next.pipeline.configured_simulation import build_source_averaged_detector

    baseline = Bi2Se3QuintupleLayerParameters.from_crystal(inputs.crystal)
    parameters = replace(
        baseline,
        bi_fractional_z=float(structure_representative["bi_fractional_z"]),
        se2_fractional_z=float(structure_representative["se2_fractional_z"]),
        bi_occupancy=float(structure_representative["bi_occupancy"]),
        se1_occupancy=float(structure_representative["se1_occupancy"]),
        se2_occupancy=float(structure_representative["se2_occupancy"]),
        u_radial_A2=float(structure_representative["u_radial_A2"]),
        u_normal_A2=float(structure_representative["u_normal_A2"]),
    )
    strength = replace(inputs.strength, structure_parameters=parameters)
    full_detector = build_source_averaged_detector(replace(inputs, strength=strength))
    families = sorted(set(family_by_identity.values()))
    detector_by_family = {
        family_m: full_detector.restrict_rods(
            tuple(rod for rod in full_detector.rods if rod.family_m == family_m)
        ).with_maximum_state_block_count(settings.cubature.maximum_state_block_count)
        for family_m in families
    }
    rows, columns = settings.profile_shape_rc
    column_px = np.broadcast_to(np.arange(columns, dtype=np.float64)[None, :], (rows, columns))
    candidate_pixel_mask_by_family = {
        family_m: np.zeros(settings.profile_shape_rc, dtype=np.bool_) for family_m in families
    }
    for profile in profiles:
        membership = reciprocal_profile_membership(
            detector_qr_Ainv,
            detector_L,
            column_px,
            detector_valid,
            region=profile,
        )
        candidate_pixel_mask_by_family[family_by_identity[profile.identity]] |= (
            membership.signal_bin_index >= 0
        )
    passes: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    execution: dict[str, Any] = {}
    for name, tile_size, gauss_order in (
        (
            "coarse",
            settings.cubature.coarse_tile_size_px,
            settings.cubature.coarse_gauss_order,
        ),
        (
            "fine",
            settings.cubature.fine_tile_size_px,
            settings.cubature.fine_gauss_order,
        ),
    ):
        passes[name], execution[name] = _cubature_profile_pass(
            inputs=inputs,
            detector_by_family=detector_by_family,
            family_by_identity=family_by_identity,
            profiles=profiles,
            shape_rc=settings.profile_shape_rc,
            tile_size_px=tile_size,
            gauss_order=gauss_order,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=settings.cubature.cuda_coordinate_chunk_size,
            candidate_pixel_mask_by_family=candidate_pixel_mask_by_family,
        )

    def convergence_record(
        profile: Any,
        lower: dict[str, np.ndarray],
        higher: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        lower_mean = lower["mean_A2_per_px2"]
        higher_mean = higher["mean_A2_per_px2"]
        valid = np.isfinite(lower_mean) & np.isfinite(higher_mean)
        if np.count_nonzero(valid) < 3 or not np.any(higher_mean[valid] > 0.0):
            raise RuntimeError(
                f"continuous cubature has insufficient support for {profile.identity}"
            )
        denominator = float(np.linalg.norm(higher_mean[valid]))
        relative_l2 = float(np.linalg.norm(higher_mean[valid] - lower_mean[valid]) / denominator)
        lower_mass = float(np.sum(lower["signal_sum_A2"], dtype=np.float64))
        higher_mass = float(np.sum(higher["signal_sum_A2"], dtype=np.float64))
        mass_relative = abs(lower_mass - higher_mass) / max(abs(higher_mass), np.finfo(float).tiny)
        return {
            "identity": profile.identity,
            "profile_relative_l2": relative_l2,
            "integrated_mass_relative_error": mass_relative,
            "passed": (
                relative_l2 <= settings.cubature.maximum_profile_relative_l2
                and mass_relative <= settings.cubature.maximum_integrated_mass_relative_error
            ),
        }

    initial_convergence = [
        convergence_record(
            profile,
            passes["coarse"][profile.identity],
            passes["fine"][profile.identity],
        )
        for profile in profiles
    ]
    failed_identities = {
        record["identity"] for record in initial_convergence if not record["passed"]
    }
    accepted = dict(passes["fine"])
    refinement_convergence: list[dict[str, Any]] = []
    refinement_tile_size = settings.cubature.fine_tile_size_px // 2
    while failed_identities and (
        refinement_tile_size >= settings.cubature.minimum_refinement_tile_size_px
    ):
        refinement_profiles = tuple(
            profile for profile in profiles if profile.identity in failed_identities
        )
        refinement_families = {
            family_by_identity[profile.identity] for profile in refinement_profiles
        }
        pass_name = f"refinement_tile_{refinement_tile_size}px"
        refinement, execution[pass_name] = _cubature_profile_pass(
            inputs=inputs,
            detector_by_family={
                family_m: detector_by_family[family_m] for family_m in sorted(refinement_families)
            },
            family_by_identity=family_by_identity,
            profiles=refinement_profiles,
            shape_rc=settings.profile_shape_rc,
            tile_size_px=refinement_tile_size,
            gauss_order=settings.cubature.refinement_gauss_order,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=settings.cubature.cuda_coordinate_chunk_size,
            candidate_pixel_mask_by_family={
                family_m: candidate_pixel_mask_by_family[family_m]
                for family_m in sorted(refinement_families)
            },
        )
        level_records = [
            {
                **convergence_record(
                    profile,
                    accepted[profile.identity],
                    refinement[profile.identity],
                ),
                "higher_order_tile_size_px": refinement_tile_size,
                "gauss_order": settings.cubature.refinement_gauss_order,
            }
            for profile in refinement_profiles
        ]
        refinement_convergence.extend(level_records)
        accepted.update(refinement)
        failed_identities = {record["identity"] for record in level_records if not record["passed"]}
        refinement_tile_size //= 2
    if failed_identities:
        failures = [
            record for record in refinement_convergence if record["identity"] in failed_identities
        ]
        raise RuntimeError(
            "continuous profile refinement did not converge: "
            + json.dumps(failures, sort_keys=True)
        )
    return accepted, {
        "measure": "continuous_detector_coordinate_density_integrated_before_area_normalization.v1",
        "model_raster_created": False,
        "passes": execution,
        "initial_convergence": initial_convergence,
        "refinement_convergence": refinement_convergence,
        "qualified": True,
    }


def prepare_figure_diagnostic(
    *,
    replay_case_path: Path,
    checkpoint_directory: Path,
    figure_config_path: Path,
    destination: Path,
    execution_backend: str,
) -> Path:
    """Prepare one resumable detector/fit diagnostic from verified stage artifacts."""

    import recover_bi2se3_ordered_intensity as ordered_runner
    from replay_staged_fit import load_replay_case
    from staged_fit_ordered_intensity import project_bi2se3_ordered_artifact

    from rasim_next.io.osc import read_osc
    from rasim_next.proof.diagnostics import write_diagnostic

    output = _external_path(destination)
    if not output.name.endswith(".ra_diag.npz"):
        raise ValueError("figure diagnostic must end in .ra_diag.npz")
    if output.exists():
        raise FileExistsError(f"diagnostic already exists: {output}")
    if execution_backend not in {"cpu", "cuda"}:
        raise ValueError("execution_backend must be 'cpu' or 'cuda'")
    start = perf_counter()
    case = load_replay_case(replay_case_path, repository_root=ROOT)
    settings = load_figure_settings(figure_config_path)
    if case.material_id != settings.material_id or case.material_id != "Bi2Se3":
        raise ValueError("the Bi2Se3 preparer cannot consume another material")
    checkpoint = checkpoint_directory.resolve()
    if not checkpoint.is_dir():
        raise ValueError("checkpoint directory does not exist")
    certificate, verified_envelopes = _verified_checkpoint_chain(checkpoint, case=case)
    mosaic_envelope, mosaic_path, mosaic_document = _checkpoint_artifact(
        checkpoint,
        stage="mosaic",
        relative_directory="mosaic_artifacts",
    )
    ordered_envelope, ordered_path, ordered_document = _checkpoint_artifact(
        checkpoint,
        stage="ordered_intensity",
        relative_directory="ordered_intensity_artifacts",
    )
    if (
        mosaic_envelope != verified_envelopes["mosaic"]
        or ordered_envelope != verified_envelopes["ordered_intensity"]
    ):
        raise ValueError("checkpoint artifacts are not the certified stage envelopes")
    for envelope in (mosaic_envelope, ordered_envelope):
        if (
            envelope.get("case_id") != case.case_id
            or envelope.get("material_id") != case.material_id
            or envelope.get("source_state_count") != case.source_state_count
        ):
            raise ValueError("checkpoint envelope does not match the replay case")
    if ordered_envelope.get("upstream_scientific_revision") != mosaic_envelope.get(
        "scientific_revision"
    ):
        raise ValueError("ordered checkpoint is not descended from the supplied mosaic stage")

    projection = project_bi2se3_ordered_artifact(ordered_document, case=case)
    projected_state = projection["state"]
    ordered_state = ordered_envelope["state"]
    for name in ("fixed_position", "parameters", "profile_scales", "structure_representative"):
        if ordered_state.get(name) != projected_state[name]:
            raise ValueError(f"ordered checkpoint changed projected field {name}")
    mosaic_state = mosaic_envelope["state"]
    if projected_state["fixed_position"] != mosaic_state.get("fixed_position") or projection[
        "fixed_mosaic"
    ] != dict(
        zip(
            ("gaussian_sigma_deg", "lorentzian_hwhm_deg", "lorentzian_probability"),
            mosaic_state.get("parameters", ()),
            strict=True,
        )
    ):
        raise ValueError("ordered checkpoint changed its position or mosaic predecessor")

    ordered_case_path = case.input_paths["ordered_intensity_case"]
    ordered_case = tomllib.loads(ordered_case_path.read_text(encoding="utf-8"))
    mosaic_case_path = case.input_paths["mosaic_case"]
    mosaic_case = tomllib.loads(mosaic_case_path.read_text(encoding="utf-8"))
    profile_config = ordered_case.get("profiles")
    if not isinstance(profile_config, dict):
        raise ValueError("ordered case lacks its profile contract")
    prepared = ordered_runner.prepare_measured_ordered_inputs(
        mosaic_document,
        mosaic_case_path=mosaic_case_path,
        mosaic_case=mosaic_case,
        profile_config=profile_config,
        source_sample_count=case.source_state_count,
    )
    if (
        prepared.source_revision != ordered_envelope.get("source_revision")
        or prepared.fixed_position_record != projected_state["fixed_position"]
        or prepared.mosaic_parameters != projection["fixed_mosaic"]
    ):
        raise ValueError("rebuilt figure inputs do not match the fitted checkpoint")

    observations = mosaic_case.get("m0_observations")
    if not isinstance(observations, list) or len(observations) != len(prepared.series):
        raise ValueError("mosaic case lacks the incidence observation catalog")
    dataset_order = tuple(str(record["dataset_id"]) for record in observations)
    fit_arrays = fit_observable_arrays(ordered_document, dataset_order=dataset_order)
    if settings.display_dataset_id not in dataset_order:
        raise ValueError("display dataset is not present in the fitted incidence series")
    display_index = dataset_order.index(settings.display_dataset_id)
    if case.incidence_angles_deg[display_index] != settings.display_incidence_deg:
        raise ValueError("display dataset and incidence angle do not align")
    inputs = prepared.series[display_index]

    fixed_position = projected_state["fixed_position"]
    commanded_incidence_angles_deg = tuple(
        float(value) for value in fixed_position["commanded_incidence_angles_deg"]
    )
    effective_incidence_angles_deg = tuple(
        float(value) for value in fixed_position["effective_incidence_angles_deg"]
    )
    incidence_angle_delta_rad = float(fixed_position["incidence_angle_delta_rad"])
    if (
        len(commanded_incidence_angles_deg) != len(dataset_order)
        or len(effective_incidence_angles_deg) != len(dataset_order)
        or not np.allclose(
            np.asarray(effective_incidence_angles_deg) - np.asarray(commanded_incidence_angles_deg),
            incidence_angle_delta_rad * 180.0 / math.pi,
            rtol=0.0,
            atol=2.0e-14,
        )
    ):
        raise ValueError("fixed position does not contain one common incidence-angle delta")
    observation_fields = (
        "observation_model",
        "measured_detector_observation_source",
        "background_inheritance",
        "claim_boundary",
    )
    if any(not isinstance(ordered_document.get(name), str) for name in observation_fields):
        raise ValueError("ordered artifact lacks its observation claim boundary")
    fit_document = ordered_document.get("fit")
    if not isinstance(fit_document, dict):
        raise ValueError("ordered artifact lacks its fit record")
    parameters = fit_document.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {
        "se1_over_bi",
        "se2_over_bi",
        "u_radial_A2",
        "u_normal_A2",
    }:
        raise ValueError("ordered artifact fitted-parameter contract changed")
    occupancy_ratio_reference = fit_document.get("occupancy_ratio_reference")
    if occupancy_ratio_reference != "bi_occupancy":
        raise ValueError("ordered artifact occupancy-ratio gauge changed")
    structure_factor = inputs.config.structure_factor
    if structure_factor.model_id != "bi2se3_finite_2h.v1":
        raise ValueError("Figure-7 recreation requires the finite Bi2Se3 2H model")

    role = f"osc_{settings.display_incidence_deg:g}deg"
    osc_path = case.input_paths.get(role)
    if osc_path is None:
        raise ValueError("replay case lacks the selected OSC role")
    native_counts = read_osc(osc_path).detector_native_counts
    profile_rows, profile_columns = settings.profile_shape_rc
    if profile_rows > native_counts.shape[0] or profile_columns > native_counts.shape[1]:
        raise ValueError("configured profile detector extent exceeds the measured OSC")
    profile_counts = np.asarray(
        native_counts[:profile_rows, :profile_columns],
        dtype=np.int32,
    )
    profile_qr_Ainv, profile_axial_L, profile_detector_valid = _detector_qr_l_maps(
        inputs,
        shape_rc=settings.profile_shape_rc,
        row_chunk_size=settings.geometry_row_chunk_size,
    )
    rows, columns = settings.crop_shape_rc
    detector_counts = profile_counts[:rows, :columns]
    qr_Ainv = profile_qr_Ainv[:rows, :columns]
    axial_L = profile_axial_L[:rows, :columns]
    detector_valid = profile_detector_valid[:rows, :columns]
    profiles = _profile_regions(settings, inputs)
    family_by_identity = {
        profile.identity: configured.family_m
        for profile, configured in zip(profiles, settings.profiles, strict=True)
    }
    measured_profiles = _measured_profile_reduction(
        detector_counts=profile_counts,
        qr_Ainv=profile_qr_Ainv,
        axial_L=profile_axial_L,
        detector_valid=profile_detector_valid,
        profiles=profiles,
    )
    fitted_profiles, profile_execution = _continuous_fitted_profiles(
        inputs=inputs,
        structure_representative=projected_state["structure_representative"],
        profiles=profiles,
        family_by_identity=family_by_identity,
        detector_qr_Ainv=profile_qr_Ainv,
        detector_L=profile_axial_L,
        detector_valid=profile_detector_valid,
        settings=settings,
        execution_backend=execution_backend,
    )
    dataset_scales = fit_document.get("dataset_scales")
    if not isinstance(dataset_scales, dict) or settings.display_dataset_id not in dataset_scales:
        raise ValueError("ordered artifact lacks the selected incidence scale")
    dataset_scale = float(dataset_scales[settings.display_dataset_id])
    if not math.isfinite(dataset_scale) or dataset_scale <= 0.0:
        raise ValueError("ordered artifact contains an invalid selected incidence scale")

    configured_hk = {(rod.h, rod.k): rod.family_m for rod in inputs.rods}
    region_records: list[dict[str, Any]] = []
    for region in settings.regions:
        if configured_hk.get((region.h, region.k)) != region.family_m:
            raise ValueError(
                f"configured rod ({region.h}, {region.k}) does not represent m={region.family_m}"
            )
        representative_profile = next(
            profile
            for profile, configured in zip(profiles, settings.profiles, strict=True)
            if configured.family_m == region.family_m
        )
        target_qr_Ainv = representative_profile.qr_center_Ainv
        mask = detector_region_mask(
            qr_Ainv,
            axial_L,
            detector_valid,
            target_qr_Ainv=target_qr_Ainv,
            qr_half_width_Ainv=region.qr_half_width_Ainv,
            minimum_L=region.minimum_L,
            maximum_L=region.maximum_L,
        )
        region_records.append(
            {
                **asdict(region),
                "target_qr_Ainv": target_qr_Ainv,
                "selected_detector_pixel_count": int(np.count_nonzero(mask)),
            }
        )

    profile_records: list[dict[str, Any]] = []
    flattened: dict[str, list[np.ndarray]] = {
        "profile_index": [],
        "profile_L": [],
        "profile_measured_raw_mean": [],
        "profile_measured_background_mean": [],
        "profile_measured_signal_mean": [],
        "profile_measured_signal_pixel_count": [],
        "profile_measured_sideband_pixel_count": [],
        "profile_measured_valid": [],
        "profile_model_mean_A2_per_px2": [],
        "profile_model_scaled_mean": [],
        "profile_model_measure_px2": [],
    }
    for profile_index, (profile, configured) in enumerate(
        zip(profiles, settings.profiles, strict=True)
    ):
        measured = measured_profiles[profile.identity]
        model = fitted_profiles[profile.identity]
        profile_records.append(
            {
                **asdict(configured),
                "qr_center_Ainv": profile.qr_center_Ainv,
                "qr_half_width_Ainv": profile.qr_half_width_Ainv,
                "axial_bin_edges": profile.axial_bin_edges.tolist(),
                "measured_reduction": (
                    "native_pixel_center_signal_mean_minus_declared_radial_sideband_mean.v1"
                ),
                "model_reduction": (
                    "continuous_detector_coordinate_gauss_cubature_signal_over_area.v1"
                ),
            }
        )
        bin_count = profile.bin_count
        flattened["profile_index"].append(np.full(bin_count, profile_index, dtype=np.int64))
        flattened["profile_L"].append(measured["L"])
        flattened["profile_measured_raw_mean"].append(measured["raw_mean"])
        flattened["profile_measured_background_mean"].append(measured["background_mean"])
        flattened["profile_measured_signal_mean"].append(measured["background_subtracted_mean"])
        flattened["profile_measured_signal_pixel_count"].append(measured["signal_pixel_count"])
        flattened["profile_measured_sideband_pixel_count"].append(measured["sideband_pixel_count"])
        flattened["profile_measured_valid"].append(measured["valid"])
        flattened["profile_model_mean_A2_per_px2"].append(model["mean_A2_per_px2"])
        flattened["profile_model_scaled_mean"].append(model["mean_A2_per_px2"] * dataset_scale)
        flattened["profile_model_measure_px2"].append(model["measure_sum_px2"])

    manifest = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "material_id": case.material_id,
        "case_id": case.case_id,
        "display_dataset_id": settings.display_dataset_id,
        "display_incidence_commanded_deg": settings.display_incidence_deg,
        "display_incidence_effective_deg": effective_incidence_angles_deg[display_index],
        "incidence_angle_model": {
            "model_id": fixed_position["incidence_angle_model_id"],
            "common_delta_rad": incidence_angle_delta_rad,
            "common_delta_deg": incidence_angle_delta_rad * 180.0 / math.pi,
            "commanded_angles_deg": commanded_incidence_angles_deg,
            "effective_angles_deg": effective_incidence_angles_deg,
        },
        "coordinate_convention": "detector-native arrays [row,column]; saved rows are top-origin",
        "measured_profile_detector_shape_rc": list(settings.profile_shape_rc),
        "figure_settings": {
            **asdict(settings),
            "regions": region_records,
            "profiles": profile_records,
        },
        "fit_observable_contract": ("saved_continuous_source_averaged_peak_center_predictions.v1"),
        "fit_observable_count": int(fit_arrays["observed_signal"].size),
        "fitted_families_m": sorted({int(value) for value in fit_arrays["family_m"]}),
        "profile_families_not_used_in_fit_m": sorted(
            {region.family_m for region in settings.regions}
            - {int(value) for value in fit_arrays["family_m"]}
        ),
        "observation_model": ordered_document["observation_model"],
        "measured_detector_observation_source": ordered_document[
            "measured_detector_observation_source"
        ],
        "background_inheritance": ordered_document["background_inheritance"],
        "claim_boundary": ordered_document["claim_boundary"],
        "detector_overlay_contract": (
            "nominal_source_crystal_frame_qr_l_shared_profile_regions.v2"
        ),
        "detector_overlay_is_profile_projector": True,
        "measured_profile_pixel_membership": "native_detector_pixel_centers.v1",
        "continuous_profile_execution": profile_execution,
        "display_dataset_scale": dataset_scale,
        "fitted_distribution_pixelized": False,
        "simulated_detector_rasterization_used": False,
        "measured_detector_pixels_used": True,
        "smoothing_applied_to_data_or_model": False,
        "legacy_classification": "CORRECTED",
        "legacy_layout_relation": "publication panel geometry and labels retained",
        "legacy_first_divergent_stage": (
            "legacy smoothed measured-data trace replaced by continuous physical-model cubature"
        ),
        "legacy_numerical_oracle": "none; surviving legacy curves are not physical simulations",
        "fit_classification": ordered_document["status"],
        "fit_adequacy": ordered_document["fit_adequacy"],
        "fitted_structure_parameters": {
            **{name: float(value) for name, value in parameters.items()},
            "occupancy_ratio_reference": occupancy_ratio_reference,
            "parameters_on_bounds": ordered_document["fit_adequacy"]["parameters_on_bounds"],
        },
        "fitted_structure_representative": projected_state["structure_representative"],
        "structure_representative_claim_boundary": (
            "normalized occupancy gauge only; not an absolute occupancy determination"
        ),
        "fixed_structure_assumption": {
            "model_id": structure_factor.model_id,
            "layers": structure_factor.layers,
            "normalization": structure_factor.normalization,
            "shared_disorder_epsilon": structure_factor.shared_disorder_epsilon,
            "polytype": "2H",
            "includes_4H_or_6H_components": False,
            "stacking_parameters_fitted": False,
        },
        "fitted_mosaic": projection["fixed_mosaic"],
        "provenance": {
            "replay_case": str(case.path),
            "replay_case_sha256": _sha256(case.path),
            "replay_certificate": str(checkpoint / "replay_certificate.json"),
            "replay_certificate_sha256": _sha256(checkpoint / "replay_certificate.json"),
            "replay_certificate_backend": certificate["backend"],
            "figure_config": str(figure_config_path.resolve()),
            "figure_config_sha256": _sha256(figure_config_path.resolve()),
            "preparation_renderer": str(Path(__file__).resolve()),
            "preparation_renderer_sha256": _sha256(Path(__file__).resolve()),
            "mosaic_envelope_sha256": _sha256(checkpoint / "mosaic.json"),
            "mosaic_artifact": str(mosaic_path),
            "mosaic_artifact_sha256": _sha256(mosaic_path),
            "ordered_envelope_sha256": _sha256(checkpoint / "ordered_intensity.json"),
            "ordered_artifact": str(ordered_path),
            "ordered_artifact_sha256": _sha256(ordered_path),
            "osc": str(osc_path),
            "osc_sha256": _sha256(osc_path),
        },
        "preparation_seconds": perf_counter() - start,
    }
    arrays = {
        "detector_counts": detector_counts,
        "detector_qr_Ainv": qr_Ainv,
        "detector_L": axial_L,
        "detector_valid": detector_valid,
        **{f"fit_{name}": value for name, value in fit_arrays.items()},
        **{name: np.concatenate(parts) for name, parts in flattened.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_diagnostic(output, arrays=arrays, manifest=manifest, repository_root=ROOT)


def _validate_figure_diagnostic(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    """Fail closed on the complete detector, fitted-state, and seven-profile contract."""

    detector_names = {
        "detector_counts",
        "detector_qr_Ainv",
        "detector_L",
        "detector_valid",
    }
    fit_names = {
        "fit_dataset_id",
        "fit_family_m",
        "fit_integer_L",
        "fit_branch_id",
        "fit_observed_signal",
        "fit_fitted_signal",
        "fit_relative_residual",
    }
    profile_names = {
        "profile_index",
        "profile_L",
        "profile_measured_raw_mean",
        "profile_measured_background_mean",
        "profile_measured_signal_mean",
        "profile_measured_signal_pixel_count",
        "profile_measured_sideband_pixel_count",
        "profile_measured_valid",
        "profile_model_mean_A2_per_px2",
        "profile_model_scaled_mean",
        "profile_model_measure_px2",
    }
    if set(arrays) != detector_names | fit_names | profile_names:
        raise ValueError("fitted Figure-7 diagnostic array contract changed")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != DIAGNOSTIC_SCHEMA:
        raise ValueError("unsupported fitted Figure-7 diagnostic")

    settings = manifest.get("figure_settings")
    if not isinstance(settings, dict):
        raise ValueError("fitted Figure-7 diagnostic lacks figure settings")
    crop_shape = settings.get("crop_shape_rc")
    if (
        not isinstance(crop_shape, list)
        or len(crop_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in crop_shape)
    ):
        raise ValueError("fitted Figure-7 detector crop contract changed")
    detector_shape = tuple(crop_shape)
    profile_shape = settings.get("profile_shape_rc")
    if (
        not isinstance(profile_shape, list)
        or len(profile_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in profile_shape)
        or profile_shape[0] < crop_shape[0]
        or profile_shape[1] < crop_shape[1]
        or manifest.get("measured_profile_detector_shape_rc") != profile_shape
    ):
        raise ValueError("fitted Figure-7 profile detector extent changed")
    counts = arrays["detector_counts"]
    qr_Ainv = arrays["detector_qr_Ainv"]
    axial_L = arrays["detector_L"]
    detector_valid = arrays["detector_valid"]
    if (
        counts.dtype != np.int32
        or counts.shape != detector_shape
        or np.any(counts < 0)
        or qr_Ainv.dtype != np.float32
        or qr_Ainv.shape != detector_shape
        or axial_L.dtype != np.float32
        or axial_L.shape != detector_shape
        or detector_valid.dtype != np.bool_
        or detector_valid.shape != detector_shape
    ):
        raise ValueError("fitted Figure-7 detector arrays are inconsistent")
    expected_detector_valid = np.isfinite(qr_Ainv) & np.isfinite(axial_L) & (qr_Ainv >= 0.0)
    if not np.array_equal(detector_valid, expected_detector_valid):
        raise ValueError("fitted Figure-7 detector validity contract changed")
    regions = settings.get("regions")
    if not isinstance(regions, list) or len(regions) != 4:
        raise ValueError("fitted Figure-7 detector regions changed")
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("fitted Figure-7 detector region is invalid")
        selected = detector_region_mask(
            qr_Ainv,
            axial_L,
            detector_valid,
            target_qr_Ainv=float(region["target_qr_Ainv"]),
            qr_half_width_Ainv=float(region["qr_half_width_Ainv"]),
            minimum_L=float(region["minimum_L"]),
            maximum_L=float(region["maximum_L"]),
        )
        if int(region.get("selected_detector_pixel_count", -1)) != int(np.count_nonzero(selected)):
            raise ValueError("fitted Figure-7 detector region count changed")

    fit_count = manifest.get("fit_observable_count")
    if isinstance(fit_count, bool) or not isinstance(fit_count, int) or fit_count <= 0:
        raise ValueError("fitted Figure-7 observable count is invalid")
    if any(arrays[name].shape != (fit_count,) for name in fit_names):
        raise ValueError("fitted Figure-7 observable arrays are inconsistent")
    if (
        arrays["fit_dataset_id"].dtype.kind != "U"
        or arrays["fit_family_m"].dtype != np.int64
        or arrays["fit_integer_L"].dtype != np.int64
        or arrays["fit_branch_id"].dtype != np.int64
        or arrays["fit_observed_signal"].dtype != np.float64
        or arrays["fit_fitted_signal"].dtype != np.float64
        or arrays["fit_relative_residual"].dtype != np.float64
    ):
        raise ValueError("fitted Figure-7 observable dtypes changed")
    fit_family = arrays["fit_family_m"]
    fit_branch = arrays["fit_branch_id"]
    observed = arrays["fit_observed_signal"]
    fitted = arrays["fit_fitted_signal"]
    residual = arrays["fit_relative_residual"]
    if (
        np.any(arrays["fit_dataset_id"] == "")
        or np.any(fit_family < 0)
        or np.any(arrays["fit_integer_L"] <= 0)
        or np.any(~np.isin(fit_branch, (0, 1, 2)))
        or np.any((fit_family == 0) != (fit_branch == 0))
        or np.any(~np.isfinite(observed))
        or np.any(observed <= 0.0)
        or np.any(~np.isfinite(fitted))
        or np.any(fitted < 0.0)
        or np.any(~np.isfinite(residual))
        or not np.allclose(residual, (fitted - observed) / observed, rtol=0.0, atol=2.0e-15)
    ):
        raise ValueError("fitted Figure-7 observables are semantically inconsistent")
    fitted_families = sorted({int(value) for value in fit_family})
    if manifest.get("fitted_families_m") != fitted_families:
        raise ValueError("fitted Figure-7 family metadata changed")

    profiles = settings.get("profiles")
    expected_identities = (
        "m1_minus",
        "m1_plus",
        "m3_minus",
        "m3_plus",
        "m4_minus",
        "m4_plus",
        "m0",
    )
    if (
        not isinstance(profiles, list)
        or tuple(
            record.get("identity") if isinstance(record, dict) else None for record in profiles
        )
        != expected_identities
    ):
        raise ValueError("fitted Figure-7 profile identity order changed")
    expected_profile_keys = {
        "identity",
        "label",
        "family_m",
        "minimum_column_px",
        "maximum_column_px",
        "minimum_L",
        "maximum_L",
        "bin_width_L",
        "sideband_gap_Ainv",
        "sideband_width_Ainv",
        "qr_center_Ainv",
        "qr_half_width_Ainv",
        "axial_bin_edges",
        "measured_reduction",
        "model_reduction",
    }
    expected_indices: list[np.ndarray] = []
    expected_centers: list[np.ndarray] = []
    profile_families: set[int] = set()
    for index, record in enumerate(profiles):
        if set(record) != expected_profile_keys:
            raise ValueError("fitted Figure-7 profile manifest contract changed")
        edges = np.asarray(record["axial_bin_edges"], dtype=np.float64)
        width = float(record["bin_width_L"])
        if (
            edges.ndim != 1
            or edges.size < 2
            or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0.0)
            or not np.allclose(np.diff(edges), width, rtol=0.0, atol=2.0e-14)
            or not math.isclose(edges[0], float(record["minimum_L"]), abs_tol=2.0e-14)
            or not math.isclose(edges[-1], float(record["maximum_L"]), abs_tol=2.0e-14)
            or record["measured_reduction"]
            != "native_pixel_center_signal_mean_minus_declared_radial_sideband_mean.v1"
            or record["model_reduction"]
            != "continuous_detector_coordinate_gauss_cubature_signal_over_area.v1"
        ):
            raise ValueError("fitted Figure-7 profile bin contract changed")
        profile_families.add(int(record["family_m"]))
        expected_indices.append(np.full(edges.size - 1, index, dtype=np.int64))
        expected_centers.append(0.5 * (edges[:-1] + edges[1:]))
    detector_column_lower = -0.5
    detector_column_upper = profile_shape[1] - 0.5
    for left, right in ((0, 1), (2, 3), (4, 5)):
        intervals = sorted(
            (
                float(profiles[index]["minimum_column_px"]),
                float(profiles[index]["maximum_column_px"]),
            )
            for index in (left, right)
        )
        if (
            intervals[0][0] != detector_column_lower
            or intervals[1][1] != detector_column_upper
            or not math.isclose(intervals[0][1], intervals[1][0], abs_tol=1.0e-12)
        ):
            raise ValueError("fitted Figure-7 branch intervals no longer partition the detector")
    if (
        float(profiles[6]["minimum_column_px"]) != detector_column_lower
        or float(profiles[6]["maximum_column_px"]) != detector_column_upper
    ):
        raise ValueError("fitted Figure-7 m=0 profile no longer spans the detector")
    concatenated_indices = np.concatenate(expected_indices)
    concatenated_centers = np.concatenate(expected_centers)
    profile_count = concatenated_indices.size
    if any(arrays[name].shape != (profile_count,) for name in profile_names):
        raise ValueError("fitted Figure-7 profile arrays are inconsistent")
    float_profile_names = profile_names - {"profile_index", "profile_measured_valid"}
    if (
        arrays["profile_index"].dtype != np.int64
        or arrays["profile_measured_valid"].dtype != np.bool_
        or any(arrays[name].dtype != np.float64 for name in float_profile_names)
        or not np.array_equal(arrays["profile_index"], concatenated_indices)
        or not np.allclose(arrays["profile_L"], concatenated_centers, rtol=0.0, atol=2.0e-14)
    ):
        raise ValueError("fitted Figure-7 flattened profile layout changed")

    signal_count = arrays["profile_measured_signal_pixel_count"]
    sideband_count = arrays["profile_measured_sideband_pixel_count"]
    raw_mean = arrays["profile_measured_raw_mean"]
    background_mean = arrays["profile_measured_background_mean"]
    measured_signal = arrays["profile_measured_signal_mean"]
    measured_valid = arrays["profile_measured_valid"]
    if (
        np.any(~np.isfinite(signal_count))
        or np.any(~np.isfinite(sideband_count))
        or np.any(signal_count < 0.0)
        or np.any(sideband_count < 0.0)
        or not np.array_equal(signal_count, np.rint(signal_count))
        or not np.array_equal(sideband_count, np.rint(sideband_count))
        or not np.array_equal(np.isfinite(raw_mean), signal_count > 0.0)
        or not np.array_equal(np.isfinite(background_mean), sideband_count > 0.0)
        or not np.array_equal(measured_valid, (signal_count > 0.0) & (sideband_count > 0.0))
        or not np.array_equal(np.isfinite(measured_signal), measured_valid)
        or not np.allclose(
            measured_signal[measured_valid],
            raw_mean[measured_valid] - background_mean[measured_valid],
            rtol=0.0,
            atol=2.0e-12,
        )
    ):
        raise ValueError("fitted Figure-7 measured profiles are semantically inconsistent")

    model_mean = arrays["profile_model_mean_A2_per_px2"]
    model_scaled = arrays["profile_model_scaled_mean"]
    model_measure = arrays["profile_model_measure_px2"]
    model_support = model_measure > 0.0
    dataset_scale = float(manifest.get("display_dataset_scale", math.nan))
    if (
        not math.isfinite(dataset_scale)
        or dataset_scale <= 0.0
        or np.any(~np.isfinite(model_measure))
        or np.any(model_measure < 0.0)
        or not np.array_equal(np.isfinite(model_mean), model_support)
        or not np.array_equal(np.isfinite(model_scaled), model_support)
        or np.any(model_mean[model_support] < 0.0)
        or np.any(model_scaled[model_support] < 0.0)
        or np.any(measured_valid & ~model_support)
        or not np.allclose(
            model_scaled[model_support],
            model_mean[model_support] * dataset_scale,
            rtol=2.0e-15,
            atol=2.0e-12,
        )
    ):
        raise ValueError("fitted Figure-7 model profiles are semantically inconsistent")
    extrapolated_families = sorted(profile_families - set(fitted_families))
    if manifest.get("profile_families_not_used_in_fit_m") != extrapolated_families:
        raise ValueError("fitted Figure-7 extrapolated-family metadata changed")

    incidence = manifest.get("incidence_angle_model")
    if not isinstance(incidence, dict):
        raise ValueError("fitted Figure-7 incidence model is missing")
    commanded = np.asarray(incidence.get("commanded_angles_deg"), dtype=np.float64)
    effective = np.asarray(incidence.get("effective_angles_deg"), dtype=np.float64)
    delta_rad = float(incidence.get("common_delta_rad", math.nan))
    delta_deg = float(incidence.get("common_delta_deg", math.nan))
    if (
        commanded.ndim != 1
        or commanded.size == 0
        or effective.shape != commanded.shape
        or not np.all(np.isfinite(commanded))
        or not np.all(np.isfinite(effective))
        or not math.isclose(delta_deg, math.degrees(delta_rad), abs_tol=2.0e-14)
        or not np.allclose(effective - commanded, delta_deg, rtol=0.0, atol=2.0e-14)
        or not math.isclose(
            float(manifest.get("display_incidence_effective_deg", math.nan)),
            float(manifest.get("display_incidence_commanded_deg", math.nan)) + delta_deg,
            abs_tol=2.0e-14,
        )
    ):
        raise ValueError("fitted Figure-7 common incidence delta changed")
    structure = manifest.get("fixed_structure_assumption")
    if (
        not isinstance(structure, dict)
        or structure.get("layers") != 52
        or structure.get("polytype") != "2H"
        or structure.get("shared_disorder_epsilon") != 0.001
        or structure.get("includes_4H_or_6H_components") is not False
        or structure.get("stacking_parameters_fitted") is not False
    ):
        raise ValueError("fitted Figure-7 fixed 2H structure assumption changed")
    if (
        manifest.get("fitted_distribution_pixelized") is not False
        or manifest.get("simulated_detector_rasterization_used") is not False
        or manifest.get("smoothing_applied_to_data_or_model") is not False
        or manifest.get("detector_overlay_is_profile_projector") is not True
    ):
        raise ValueError("fitted Figure-7 sampling or smoothing contract changed")

    execution = manifest.get("continuous_profile_execution")
    cubature = settings.get("cubature")
    if (
        not isinstance(execution, dict)
        or execution.get("qualified") is not True
        or execution.get("model_raster_created") is not False
        or not isinstance(cubature, dict)
    ):
        raise ValueError("fitted Figure-7 continuous execution is unqualified")
    maximum_l2 = float(cubature["maximum_profile_relative_l2"])
    maximum_mass = float(cubature["maximum_integrated_mass_relative_error"])
    initial = execution.get("initial_convergence")
    refinements = execution.get("refinement_convergence")
    if (
        not isinstance(initial, list)
        or not isinstance(refinements, list)
        or tuple(record.get("identity") for record in initial) != expected_identities
    ):
        raise ValueError("fitted Figure-7 convergence record changed")
    latest: dict[str, bool] = {}
    failed: set[str] = set()
    for record in initial:
        relative_l2 = float(record.get("profile_relative_l2", math.nan))
        relative_mass = float(record.get("integrated_mass_relative_error", math.nan))
        passed = relative_l2 <= maximum_l2 and relative_mass <= maximum_mass
        if (
            not math.isfinite(relative_l2)
            or not math.isfinite(relative_mass)
            or record.get("passed") is not passed
        ):
            raise ValueError("fitted Figure-7 convergence decision changed")
        identity = str(record["identity"])
        latest[identity] = passed
        if not passed:
            failed.add(identity)
    for record in refinements:
        identity = str(record.get("identity"))
        if identity not in failed:
            raise ValueError("fitted Figure-7 refinement does not follow a failed profile")
        relative_l2 = float(record.get("profile_relative_l2", math.nan))
        relative_mass = float(record.get("integrated_mass_relative_error", math.nan))
        passed = relative_l2 <= maximum_l2 and relative_mass <= maximum_mass
        if (
            not math.isfinite(relative_l2)
            or not math.isfinite(relative_mass)
            or record.get("passed") is not passed
        ):
            raise ValueError("fitted Figure-7 refinement decision changed")
        latest[identity] = passed
        if passed:
            failed.remove(identity)
    if failed or not all(latest.values()):
        raise ValueError("fitted Figure-7 continuous profiles did not converge")


def _load_figure_diagnostic(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    diagnostic = _external_path(path)
    with np.load(diagnostic, allow_pickle=False) as archive:
        if "manifest_json" not in archive.files:
            raise ValueError("figure diagnostic lacks its embedded manifest")
        if any(archive[name].dtype.hasobject for name in archive.files):
            raise ValueError("figure diagnostic may not contain object arrays")
        manifest_array = np.asarray(archive["manifest_json"])
        if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
            raise ValueError("figure diagnostic has an invalid embedded manifest")
        manifest = json.loads(manifest_array.tobytes().decode("utf-8"))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "manifest_json"
        }
    _validate_figure_diagnostic(arrays, manifest)
    return arrays, manifest


def _draw_detector_panel(
    axis: Any, arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> tuple[float, float]:
    import matplotlib

    counts = arrays["detector_counts"]
    qr_Ainv = arrays["detector_qr_Ainv"]
    axial_L = arrays["detector_L"]
    valid = arrays["detector_valid"]
    rows, columns = counts.shape
    settings = manifest["figure_settings"]
    display_valid = valid & np.isfinite(axial_L) & (axial_L >= 0.0)
    if not np.any(display_valid):
        raise RuntimeError("detector crop has no valid positive-L pixels")
    low = float(np.percentile(counts[display_valid], settings["detector_low_percentile"]))
    high = float(np.percentile(counts[display_valid], settings["detector_high_percentile"]))
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise RuntimeError("detector display percentiles are invalid")
    color_map = matplotlib.colormaps["magma"].copy()
    color_map.set_bad("#050505")
    displayed = np.ma.array(counts, mask=~display_valid, copy=False)
    axis.imshow(
        displayed,
        origin="upper",
        extent=(-0.5, columns - 0.5, rows - 0.5, -0.5),
        cmap=color_map,
        vmin=low,
        vmax=high,
        interpolation="nearest",
        rasterized=True,
    )
    column_coordinate = np.arange(columns, dtype=np.float64)
    row_coordinate = np.arange(rows, dtype=np.float64)
    for region in settings["regions"]:
        target = float(region["target_qr_Ainv"])
        half_width = float(region["qr_half_width_Ainv"])
        minimum_L = float(region["minimum_L"])
        maximum_L = float(region["maximum_L"])
        color = str(region["color"])
        family_m = int(region["family_m"])
        mask = detector_region_mask(
            qr_Ainv,
            axial_L,
            valid,
            target_qr_Ainv=target,
            qr_half_width_Ainv=half_width,
            minimum_L=minimum_L,
            maximum_L=maximum_L,
        )
        alpha = 0.32 if family_m == 0 else 0.11
        axis.contourf(
            column_coordinate,
            row_coordinate,
            mask.astype(np.uint8),
            levels=(0.5, 1.5),
            colors=(color,),
            alpha=alpha,
            antialiased=False,
        )
        axis.contour(
            column_coordinate,
            row_coordinate,
            mask.astype(np.uint8),
            levels=(0.5,),
            colors=(color,),
            linewidths=(1.45,),
            linestyles=("--",),
            alpha=1.0 if family_m == 0 else 0.72,
        )
        center_support = (
            valid
            & np.isfinite(qr_Ainv)
            & np.isfinite(axial_L)
            & (axial_L >= minimum_L)
            & (axial_L <= maximum_L)
        )
        center_field = np.where(center_support, qr_Ainv - target, np.nan)
        finite_center = center_field[np.isfinite(center_field)]
        if finite_center.size and np.min(finite_center) <= 0.0 <= np.max(finite_center):
            axis.contour(
                column_coordinate,
                row_coordinate,
                center_field,
                levels=(0.0,),
                colors=(color,),
                linewidths=(1.1,),
                linestyles=("-",),
            )
    for label in settings["labels"]:
        axis.text(
            float(label["column_px"]),
            float(label["row_px"]),
            str(label["text"]),
            color="white",
            fontsize=8.6,
            fontweight="bold",
            ha="center",
            va="center",
        )
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_xticks(np.linspace(0.0, columns - 1.0, 5))
    axis.set_yticks(np.linspace(0.0, rows - 1.0, 5))
    axis.set_xlabel("Detector-native column (px)")
    axis.set_ylabel("Detector-native row (px; top-origin)")
    axis.set_title(
        "Measured 5° OSC with the integrated reciprocal-space regions\n"
        f"effective $\\theta_i$ = {manifest['display_incidence_effective_deg']:.3f}° "
        f"(shared $\\Delta\\theta_i$ = {manifest['incidence_angle_model']['common_delta_deg']:+.3f}°)",
        fontsize=11,
    )
    axis.text(-0.055, 1.03, "(a)", transform=axis.transAxes, fontsize=14, fontweight="bold")
    return low, high


def _draw_profile_panels(
    axes: Sequence[Any], arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> None:
    from matplotlib.lines import Line2D

    if len(axes) != 7:
        raise ValueError("Figure 7 requires six off-specular axes and one m=0 axis")
    profile_index = arrays["profile_index"]
    L = arrays["profile_L"]
    measured = arrays["profile_measured_signal_mean"]
    measured_valid = arrays["profile_measured_valid"]
    model = arrays["profile_model_scaled_mean"]
    records = manifest["figure_settings"]["profiles"]
    plotted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, (axis, record) in enumerate(zip(axes, records, strict=True)):
        selected = profile_index == index
        x = L[selected]
        data_y = np.where(measured_valid[selected], measured[selected], np.nan)
        model_y = np.where(np.isfinite(model[selected]), model[selected], np.nan)
        if index == 6:
            data_y = np.where(data_y > 0.0, data_y, np.nan)
            model_y = np.where(model_y > 0.0, model_y, np.nan)
        plotted.append((x, data_y, model_y))
        axis.plot(
            x,
            data_y,
            color="0.16",
            linewidth=1.1,
            linestyle="-",
            marker=".",
            markersize=2.8,
            label="Measured OSC profile",
        )
        axis.plot(
            x,
            model_y,
            color="#D55E00",
            linewidth=1.45,
            linestyle="--",
            label="Fitted physical model",
        )
        axis.set_xlim(float(record["minimum_L"]), float(record["maximum_L"]))
        axis.grid(True, color="0.87", linewidth=0.5)
        axis.tick_params(direction="in", top=True, right=True)
        axis.text(
            0.54 if index < 6 else 0.5,
            0.94,
            str(record["label"]),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
    for left, right in ((0, 1), (2, 3), (4, 5)):
        values = np.concatenate(
            tuple(
                value[np.isfinite(value)] for index in (left, right) for value in plotted[index][1:]
            )
        )
        lower = float(np.min(values))
        upper = float(np.max(values))
        span = max(upper - lower, np.finfo(float).eps)
        lower -= 0.06 * span
        upper += 0.06 * span
        axes[left].set_ylim(lower, upper)
        axes[right].set_ylim(lower, upper)
        axes[left].axhline(0.0, color="0.72", linewidth=0.55, zorder=0)
        axes[right].axhline(0.0, color="0.72", linewidth=0.55, zorder=0)
        axes[right].tick_params(labelleft=False)
    for index, axis in enumerate(axes[:6]):
        if index % 2 == 0:
            axis.set_ylabel("Signal (a.u.)")
        if index < 4:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("$L$")
    m0_axis = axes[6]
    m0_values = np.concatenate(tuple(value[np.isfinite(value)] for value in plotted[6][1:]))
    m0_axis.set_yscale("log")
    m0_axis.set_ylim(float(np.min(m0_values)) * 0.7, float(np.max(m0_values)) * 1.35)
    m0_axis.set_xlabel("$L$")
    m0_axis.set_ylabel("Signal (a.u.)")
    axes[1].legend(
        handles=(
            Line2D(
                (0,),
                (0,),
                color="0.16",
                linestyle="-",
                marker=".",
                label="Measured OSC profile",
            ),
            Line2D(
                (0,),
                (0,),
                color="#D55E00",
                linestyle="--",
                label="Fitted physical model",
            ),
        ),
        loc="upper right",
        fontsize=8.5,
        frameon=False,
    )
    axes[0].text(-0.12, 1.04, "(c)", transform=axes[0].transAxes, fontsize=14, fontweight="bold")


def _render_output_paths(output_directory: Path) -> tuple[Path, Path, Path, Path]:
    output = _external_path(output_directory)
    return (
        output,
        output / "bi2se3_figure7_recreated.png",
        output / "bi2se3_figure7_recreated.pdf",
        output / "bi2se3_figure7_recreated.json",
    )


def _preflight_render_outputs(output_directory: Path) -> tuple[Path, Path, Path, Path]:
    paths = _render_output_paths(output_directory)
    for path in paths[1:]:
        if path.exists():
            raise FileExistsError(f"figure output already exists: {path}")
    return paths


def render_figure_diagnostic(*, diagnostic_path: Path, output_directory: Path) -> dict[str, Path]:
    """Render publication PNG/PDF and one provenance manifest from a prepared diagnostic."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output, png_path, pdf_path, manifest_path = _preflight_render_outputs(output_directory)
    diagnostic = _external_path(diagnostic_path)
    if not diagnostic.is_file():
        raise FileNotFoundError(f"figure diagnostic does not exist: {diagnostic}")
    diagnostic_hash = _sha256(diagnostic)
    arrays, embedded_manifest = _load_figure_diagnostic(diagnostic)
    output.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(13.2, 16.8), layout="constrained")
    grid = figure.add_gridspec(3, 1, height_ratios=(0.84, 1.16, 0.11))
    detector_axis = figure.add_subplot(grid[0])
    profile_grid = grid[1].subgridspec(4, 2, height_ratios=(1.0, 1.0, 1.0, 1.12))
    profile_axes = (
        *(figure.add_subplot(profile_grid[row, column]) for row in range(3) for column in range(2)),
        figure.add_subplot(profile_grid[3, :]),
    )
    count_limits = _draw_detector_panel(detector_axis, arrays, embedded_manifest)
    _draw_profile_panels(profile_axes, arrays, embedded_manifest)
    caption_axis = figure.add_subplot(grid[2])
    caption_axis.axis("off")
    caption_axis.text(
        0.5,
        0.9,
        "Measured curves are background-subtracted native detector-pixel-bin means.  Orange "
        "curves are continuous detector-coordinate cubatures of the fitted 52-layer nearly-perfect "
        "2H model over the same Qr/L regions, multiplied only by the saved 5° fit scale.  Connecting "
        "segments are visual guides between finite-bin means; no smoothing or model raster was "
        "used.  m=3 and m=4 are extrapolations (only m=0 and m=1 entered the structure fit).  "
        "No 4H/6H component or stacking refinement is included; the fit remains unqualified.",
        ha="center",
        va="top",
        fontsize=8.8,
        wrap=True,
    )
    figure.suptitle(
        "Bi$_2$Se$_3$: Figure 7 recreated with the staged physical-model fit",
        fontsize=15,
        fontweight="bold",
    )
    metadata = {"Creator": "SLATE-rMC fitted Figure-7 renderer"}
    process_id = os.getpid()
    temporary_png = output / f"bi2se3_figure7_recreated.{process_id}.tmp.png"
    temporary_pdf = output / f"bi2se3_figure7_recreated.{process_id}.tmp.pdf"
    temporary_manifest = output / f"bi2se3_figure7_recreated.{process_id}.tmp.json"
    temporary_paths = (temporary_png, temporary_pdf, temporary_manifest)
    if any(path.exists() for path in temporary_paths):
        raise FileExistsError("figure rendering temporary path already exists")
    try:
        try:
            figure.savefig(temporary_png, dpi=220, bbox_inches="tight", metadata=metadata)
            figure.savefig(temporary_pdf, bbox_inches="tight", metadata=metadata)
        finally:
            plt.close(figure)
        if _sha256(diagnostic) != diagnostic_hash:
            raise RuntimeError("figure diagnostic changed while it was being rendered")
        output_manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "diagnostic": str(diagnostic),
            "diagnostic_sha256": diagnostic_hash,
            "renderer": str(Path(__file__).resolve()),
            "renderer_sha256": _sha256(Path(__file__).resolve()),
            "scientific_manifest": embedded_manifest,
            "outputs": {
                "png": {"path": str(png_path), "sha256": _sha256(temporary_png)},
                "pdf": {"path": str(pdf_path), "sha256": _sha256(temporary_pdf)},
            },
            "rendering": {
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "data_or_model_smoothing": "none",
                "detector_interpolation": "nearest",
                "detector_array_orientation": "unchanged detector-native top-origin rows",
                "detector_count_limits": list(count_limits),
                "profile_y_scale": (
                    "saved 5-degree fit scale; no panel normalization; linear off-specular and "
                    "logarithmic m=0"
                ),
                "profile_line_segments": "visual guides between finite-bin means",
            },
        }
        temporary_manifest.write_text(
            json.dumps(output_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        promoted: list[Path] = []
        try:
            for source, destination in (
                (temporary_png, png_path),
                (temporary_pdf, pdf_path),
                (temporary_manifest, manifest_path),
            ):
                os.replace(source, destination)
                promoted.append(destination)
        except Exception:
            for destination in promoted:
                destination.unlink(missing_ok=True)
            raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    return {"png": png_path, "pdf": pdf_path, "manifest": manifest_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and render the artifact-backed Bi2Se3 Figure-7 recreation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="build the resumable numeric diagnostic")
    prepare.add_argument("--case", type=Path, default=DEFAULT_REPLAY_CASE)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=DEFAULT_FIGURE_CONFIG)
    prepare.add_argument("--diagnostic", type=Path, required=True)
    prepare.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")

    render = subparsers.add_parser("render", help="render PNG/PDF from a prepared diagnostic")
    render.add_argument("--diagnostic", type=Path, required=True)
    render.add_argument("--output-directory", type=Path, required=True)

    all_steps = subparsers.add_parser("all", help="prepare the diagnostic, then render it")
    all_steps.add_argument("--case", type=Path, default=DEFAULT_REPLAY_CASE)
    all_steps.add_argument("--checkpoint", type=Path, required=True)
    all_steps.add_argument("--config", type=Path, default=DEFAULT_FIGURE_CONFIG)
    all_steps.add_argument("--output-directory", type=Path, required=True)
    all_steps.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        diagnostic = prepare_figure_diagnostic(
            replay_case_path=arguments.case,
            checkpoint_directory=arguments.checkpoint,
            figure_config_path=arguments.config,
            destination=arguments.diagnostic,
            execution_backend=arguments.backend,
        )
        print(json.dumps({"diagnostic": str(diagnostic)}, sort_keys=True))
        return 0
    if arguments.command == "render":
        outputs = render_figure_diagnostic(
            diagnostic_path=arguments.diagnostic,
            output_directory=arguments.output_directory,
        )
        print(json.dumps({name: str(path) for name, path in outputs.items()}, sort_keys=True))
        return 0

    output_directory = _external_path(arguments.output_directory)
    diagnostic = output_directory / "bi2se3_figure7_inputs.ra_diag.npz"
    _preflight_render_outputs(output_directory)
    if diagnostic.exists():
        raise FileExistsError(f"diagnostic already exists: {diagnostic}")
    prepared = prepare_figure_diagnostic(
        replay_case_path=arguments.case,
        checkpoint_directory=arguments.checkpoint,
        figure_config_path=arguments.config,
        destination=diagnostic,
        execution_backend=arguments.backend,
    )
    outputs = render_figure_diagnostic(
        diagnostic_path=prepared,
        output_directory=output_directory,
    )
    print(
        json.dumps(
            {
                "diagnostic": str(prepared),
                **{name: str(path) for name, path in outputs.items()},
            },
            sort_keys=True,
        )
    )
    return 0


def fit_observable_arrays(
    document: dict[str, Any],
    *,
    dataset_order: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return identity-aligned measured and forward-model fitted peak-center observables."""

    fit = document.get("fit")
    profiles = fit.get("profiles") if isinstance(fit, dict) else None
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(isinstance(record, dict) for record in profiles)
    ):
        raise ValueError("ordered artifact lacks fitted profile records")
    order = tuple(str(value) for value in dataset_order)
    if not order or len(set(order)) != len(order):
        raise ValueError("dataset_order must contain unique dataset identities")
    order_index = {dataset_id: index for index, dataset_id in enumerate(order)}
    if {str(record.get("dataset_id")) for record in profiles} != set(order):
        raise ValueError("fitted profile datasets do not match dataset_order")

    def record_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
        branch = record.get("root_side_branch_id")
        return (
            order_index[str(record["dataset_id"])],
            int(record["family_m"]),
            0 if branch is None else int(branch),
            int(record["integer_L"]),
        )

    records = sorted(profiles, key=record_key)
    dataset_id = np.asarray([str(record["dataset_id"]) for record in records], dtype="U64")
    family_m = np.asarray([int(record["family_m"]) for record in records], dtype=np.int64)
    integer_L = np.asarray([int(record["integer_L"]) for record in records], dtype=np.int64)
    branch_id = np.asarray(
        [
            0 if record.get("root_side_branch_id") is None else int(record["root_side_branch_id"])
            for record in records
        ],
        dtype=np.int64,
    )
    observed = np.asarray(
        [float(record["transferred_observed_signal_density_A2_per_rad2"]) for record in records],
        dtype=np.float64,
    )
    fitted = np.asarray(
        [float(record["fitted_signal_density_A2_per_rad2"]) for record in records],
        dtype=np.float64,
    )
    residual = np.asarray(
        [float(record["relative_residual"]) for record in records],
        dtype=np.float64,
    )
    if (
        np.any(family_m < 0)
        or np.any(integer_L <= 0)
        or np.any(~np.isin(branch_id, (0, 1, 2)))
        or np.any((family_m == 0) != (branch_id == 0))
        or np.any(~np.isfinite(observed))
        or np.any(~np.isfinite(fitted))
        or np.any(observed <= 0.0)
        or np.any(fitted <= 0.0)
        or np.any(~np.isfinite(residual))
    ):
        raise ValueError("ordered artifact contains invalid fitted profile values")
    expected_residual = (fitted - observed) / observed
    if not np.allclose(residual, expected_residual, rtol=0.0, atol=2.0e-15):
        raise ValueError("ordered artifact profile residuals are inconsistent")
    return {
        "dataset_id": dataset_id,
        "family_m": family_m,
        "integer_L": integer_L,
        "branch_id": branch_id,
        "observed_signal": observed,
        "fitted_signal": fitted,
        "relative_residual": residual,
    }


def detector_region_mask(
    qr_Ainv: np.ndarray,
    L: np.ndarray,
    valid: np.ndarray,
    *,
    target_qr_Ainv: float,
    qr_half_width_Ainv: float,
    minimum_L: float,
    maximum_L: float,
) -> np.ndarray:
    """Select native detector pixels inside one explicit radial reciprocal-space band."""

    qr = np.asarray(qr_Ainv, dtype=np.float64)
    axial = np.asarray(L, dtype=np.float64)
    accepted = np.asarray(valid, dtype=np.bool_)
    if qr.shape != axial.shape or qr.shape != accepted.shape:
        raise ValueError("Qr, L, and valid arrays must have one common detector shape")
    scalars = np.asarray(
        (target_qr_Ainv, qr_half_width_Ainv, minimum_L, maximum_L), dtype=np.float64
    )
    if (
        not np.all(np.isfinite(scalars))
        or target_qr_Ainv < 0.0
        or qr_half_width_Ainv <= 0.0
        or minimum_L >= maximum_L
    ):
        raise ValueError("reciprocal-region bounds must be finite and ordered")
    return (
        accepted
        & np.isfinite(qr)
        & np.isfinite(axial)
        & (np.abs(qr - target_qr_Ainv) <= qr_half_width_Ainv)
        & (axial >= minimum_L)
        & (axial <= maximum_L)
    )


if __name__ == "__main__":
    raise SystemExit(main())
