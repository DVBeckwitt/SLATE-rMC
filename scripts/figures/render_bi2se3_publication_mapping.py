"""Render a publication-ready Bi2Se3 reciprocal-to-detector mapping sequence.

The reciprocal panel samples the continuous configured field. The Ewald panel point-samples the
exact almost-everywhere intrinsic density only on the internal-film sphere patch seen by the
configured active detector, including regular nonzero m=0 support. Spawned CPU processes evaluate
independent row batches. The two detector panels independently evaluate the weighted configured
spectral lines at one mean ray geometry per line; this minimum display quadrature does not sample
the configured spatial/divergence widths. The configured and incrementally tilted images are
evaluated separately rather than warped from one another. Generated artifacts must remain outside
the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "src"
for import_root in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from rasim_next.geometry import (  # noqa: E402
    CompiledInstrument,
    compose_intrinsic_xy_rotation,
    project_detector_rays,
)
from rasim_next.pipeline.bragg_space import finite_stack_integer_l_display_nodes  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    ConfiguredSimulationInputs,
    NominalEwaldContext,
    ReciprocalSpaceDisplay,
    SimulationConfiguration,
)

DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"
DEFAULT_OUTPUT_DIRECTORY = Path.home() / ".rasim-next" / "bi2se3-publication-mapping"
RECIPROCAL_POINT_BUDGET = 420_000

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class DetectorDensityModel(Protocol):
    def evaluate_detector_density_all_roots(
        self,
        column_px: object,
        row_px: object,
        *,
        execution_backend: str = "cpu",
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class FigureSettings:
    """Numerical and display settings for the reusable publication figure."""

    incidence_deg: float = 10.0
    gaussian_sigma_deg: float = 2.0
    lorentzian_hwhm_deg: float = 0.2
    lorentzian_probability: float = 0.1
    tilt_column_deg: float = 0.0
    tilt_row_deg: float = 20.0
    mosaic_alpha_count: int = 32
    reciprocal_beta_count: int = 48
    reciprocal_u_background_count: int = 48
    ewald_image_size: int = 1440
    ewald_tile_row_count: int = 32
    ewald_worker_count: int = min(4, os.cpu_count() or 1)
    detector_image_size: int = 1440
    detector_tile_row_count: int = 32
    schematic_detector_cell_count: int = 480
    dpi: int = 320

    def __post_init__(self) -> None:
        finite_names = (
            "incidence_deg",
            "gaussian_sigma_deg",
            "lorentzian_hwhm_deg",
            "lorentzian_probability",
            "tilt_column_deg",
            "tilt_row_deg",
        )
        for name in finite_names:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(float(value))
            ):
                raise ValueError("all angles, widths, and probabilities must be finite numbers")
            object.__setattr__(self, name, float(value))
        if self.gaussian_sigma_deg < 0.0 or self.lorentzian_hwhm_deg < 0.0:
            raise ValueError("mosaic component widths must be nonnegative")
        if not 0.0 <= self.lorentzian_probability <= 1.0:
            raise ValueError("lorentzian_probability must lie in [0, 1]")
        if self.lorentzian_probability < 1.0 and self.gaussian_sigma_deg == 0.0:
            raise ValueError("an active Gaussian component requires positive sigma")
        if self.lorentzian_probability > 0.0 and self.lorentzian_hwhm_deg == 0.0:
            raise ValueError("an active Lorentzian component requires positive HWHM")
        count_names = (
            "mosaic_alpha_count",
            "reciprocal_beta_count",
            "reciprocal_u_background_count",
            "ewald_image_size",
            "ewald_tile_row_count",
            "ewald_worker_count",
            "detector_image_size",
            "detector_tile_row_count",
            "schematic_detector_cell_count",
            "dpi",
        )
        for name in count_names:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError("sampling counts and tile size must be positive integers")
            object.__setattr__(self, name, int(value))
        if self.ewald_image_size < 4:
            raise ValueError("ewald_image_size must be at least four")
        if self.mosaic_alpha_count < 16:
            raise ValueError("mosaic_alpha_count must be at least 16")
        if self.reciprocal_u_background_count < 8:
            raise ValueError("reciprocal_u_background_count must be at least 8")
        if self.detector_image_size < 64:
            raise ValueError("detector_image_size must be at least 64")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Source identities captured before any expensive publication calculation."""

    config_path: Path
    config_identity: str
    config_sha256: str
    renderer_path: Path
    renderer_identity: str
    renderer_sha256: str


@dataclass(frozen=True, slots=True)
class DetectorDisplay:
    density_A2_per_px2: FloatArray
    valid: BoolArray
    caustic: BoolArray
    column_edges_px: FloatArray
    row_edges_px: FloatArray
    measure_id: str
    execution_backend: str

    def __post_init__(self) -> None:
        density = np.array(self.density_A2_per_px2, dtype=np.float64, copy=True, order="C")
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        column_edges = np.array(self.column_edges_px, dtype=np.float64, copy=True, order="C")
        row_edges = np.array(self.row_edges_px, dtype=np.float64, copy=True, order="C")
        if (
            density.ndim != 2
            or min(density.shape, default=0) < 1
            or np.any(np.isnan(density))
            or np.any(density < 0.0)
        ):
            raise ValueError("detector density must be a nonempty, nonnegative, non-NaN 2D array")
        if valid.shape != density.shape or caustic.shape != density.shape:
            raise ValueError("detector masks must match the density shape")
        if np.any(np.isposinf(density) & ~caustic):
            raise ValueError("only declared caustic cells may carry infinite density")
        if (
            column_edges.shape != (density.shape[1] + 1,)
            or row_edges.shape != (density.shape[0] + 1,)
            or not np.all(np.isfinite(column_edges))
            or not np.all(np.isfinite(row_edges))
            or np.any(np.diff(column_edges) <= 0.0)
            or np.any(np.diff(row_edges) <= 0.0)
        ):
            raise ValueError("detector edges must be finite, increasing, and bound every cell")
        if not isinstance(self.measure_id, str) or not self.measure_id:
            raise ValueError("detector measure_id must be nonempty")
        if not isinstance(self.execution_backend, str) or not self.execution_backend:
            raise ValueError("detector execution_backend must be nonempty")
        for array in (density, valid, caustic, column_edges, row_edges):
            array.setflags(write=False)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "column_edges_px", column_edges)
        object.__setattr__(self, "row_edges_px", row_edges)


@dataclass(frozen=True, slots=True)
class DetectorSurfaceRaster:
    """One detector-native density field carried by its physical LAB pose."""

    vertices_lab_m: FloatArray
    density_A2_per_px2: FloatArray
    valid: BoolArray
    caustic: BoolArray
    measure_id: str

    def __post_init__(self) -> None:
        density = np.array(self.density_A2_per_px2, dtype=np.float64, copy=True, order="C")
        vertices = np.array(self.vertices_lab_m, dtype=np.float64, copy=True, order="C")
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if (
            density.ndim != 2
            or min(density.shape, default=0) < 1
            or np.any(np.isnan(density))
            or np.any(density < 0.0)
        ):
            raise ValueError(
                "detector surface density must be nonempty, nonnegative, non-NaN, and 2D"
            )
        if vertices.shape != (density.shape[0] + 1, density.shape[1] + 1, 3):
            raise ValueError("detector surface vertices must bound every native density cell")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("detector surface vertices must be finite")
        if valid.shape != density.shape or caustic.shape != density.shape:
            raise ValueError("detector validity masks must match the native density shape")
        if np.any(np.isposinf(density) & ~caustic):
            raise ValueError("only declared caustic cells may carry infinite density")
        if not isinstance(self.measure_id, str) or not self.measure_id:
            raise ValueError("detector surface measure_id must be nonempty")
        for array in (vertices, density, valid, caustic):
            array.setflags(write=False)
        object.__setattr__(self, "vertices_lab_m", vertices)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "caustic", caustic)


@dataclass(frozen=True, slots=True)
class DisplaySurfaceRaster:
    """Detector texture after diagrammatic translation and scaling for the schematic."""

    vertices_display: FloatArray
    density_A2_per_px2: FloatArray
    valid: BoolArray
    caustic: BoolArray
    measure_id: str

    def __post_init__(self) -> None:
        density = np.array(self.density_A2_per_px2, dtype=np.float64, copy=True, order="C")
        vertices = np.array(self.vertices_display, dtype=np.float64, copy=True, order="C")
        valid = np.array(self.valid, dtype=np.bool_, copy=True, order="C")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if density.ndim != 2 or min(density.shape) < 1 or np.any(np.isnan(density)):
            raise ValueError("display surface density must be a nonempty non-NaN 2D array")
        if vertices.shape != (density.shape[0] + 1, density.shape[1] + 1, 3):
            raise ValueError("display vertices must bound every texture cell")
        if (
            not np.all(np.isfinite(vertices))
            or np.any(density < 0.0)
            or valid.shape != density.shape
            or caustic.shape != density.shape
            or np.any(np.isinf(density) & ~caustic)
        ):
            raise ValueError("display surface arrays have invalid values")
        if not isinstance(self.measure_id, str) or not self.measure_id:
            raise ValueError("display surface measure_id must be nonempty")
        for array in (vertices, density, valid, caustic):
            array.setflags(write=False)
        object.__setattr__(self, "vertices_display", vertices)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "caustic", caustic)


@dataclass(frozen=True, slots=True)
class DetectorVisibleEwaldPatchDisplay:
    """Continuous intrinsic density on the configured detector-visible sphere patch."""

    ki_sample_Ainv: FloatArray
    q_vertices_sample_Ainv: FloatArray
    outgoing_direction_centers_sample: FloatArray
    intensity_density_A2_per_sr: FloatArray
    m0_intensity_density_A2_per_sr: FloatArray
    inverse_branch_count: NDArray[np.int64]
    caustic: BoolArray
    detector_visible: BoolArray
    column_edges_px: FloatArray
    row_edges_px: FloatArray
    detector_visible_m0_q_gap_Ainv: float
    maximum_ewald_residual_Ainv: float
    requested_worker_count: int
    worker_count: int
    execution_backend: str
    multiprocessing_start_method: str | None
    measure_id: str = "detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"
    selection_id: str = "configured_active_panel_regular_all_inverse_preimages_including_m0.v1"

    def __post_init__(self) -> None:
        density = np.array(
            self.intensity_density_A2_per_sr,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        m0_density = np.array(
            self.m0_intensity_density_A2_per_sr,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        ki_sample = np.array(self.ki_sample_Ainv, dtype=np.float64, copy=True, order="C")
        vertices = np.array(self.q_vertices_sample_Ainv, dtype=np.float64, copy=True, order="C")
        directions = np.array(
            self.outgoing_direction_centers_sample,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        inverse_count = np.array(self.inverse_branch_count, dtype=np.int64, copy=True, order="C")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        visible = np.array(self.detector_visible, dtype=np.bool_, copy=True, order="C")
        column_edges = np.array(self.column_edges_px, dtype=np.float64, copy=True, order="C")
        row_edges = np.array(self.row_edges_px, dtype=np.float64, copy=True, order="C")
        if density.ndim != 2 or min(density.shape) < 2 or not np.any(visible):
            raise ValueError("Ewald patch must contain a nonempty two-dimensional visible raster")
        if ki_sample.shape != (3,) or not np.all(np.isfinite(ki_sample)):
            raise ValueError("Ewald incident wavevector must be a finite three-vector")
        if vertices.shape != (density.shape[0] + 1, density.shape[1] + 1, 3):
            raise ValueError("Ewald vertices must bound every detector-native cell")
        if directions.shape != (*density.shape, 3):
            raise ValueError("Ewald directions must provide one center for every cell")
        if any(
            array.shape != density.shape for array in (m0_density, inverse_count, caustic, visible)
        ):
            raise ValueError("Ewald density, branch, caustic, and visibility rasters must align")
        if (
            column_edges.shape != (density.shape[1] + 1,)
            or row_edges.shape != (density.shape[0] + 1,)
            or np.any(np.diff(column_edges) <= 0.0)
            or np.any(np.diff(row_edges) <= 0.0)
        ):
            raise ValueError("detector-native edges must be finite, increasing cell boundaries")
        if (
            not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(directions))
            or not np.all(np.isfinite(column_edges))
            or not np.all(np.isfinite(row_edges))
            or np.any(np.isnan(density))
            or np.any(np.isnan(m0_density))
            or np.any(density < 0.0)
            or np.any(m0_density < 0.0)
            or np.any(m0_density > density)
            or np.any(inverse_count < 0)
        ):
            raise ValueError("Ewald patch arrays have invalid values")
        if np.any((~visible) & ((density != 0.0) | (m0_density != 0.0))):
            raise ValueError("only detector-visible cells may carry Ewald intensity")
        if np.any(np.isinf(density) & ~caustic):
            raise ValueError("only declared Ewald caustics may carry infinite density")
        if np.any(directions[~visible] != 0.0) or not np.allclose(
            np.linalg.norm(directions[visible], axis=-1),
            1.0,
            rtol=0.0,
            atol=4096.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("only visible cells may carry unit outgoing directions")

        used_vertices = np.zeros(vertices.shape[:2], dtype=np.bool_)
        visible_row, visible_column = np.nonzero(visible)
        for row_offset, column_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
            used_vertices[visible_row + row_offset, visible_column + column_offset] = True
        k_magnitude_Ainv = float(np.linalg.norm(ki_sample))
        if k_magnitude_Ainv == 0.0 or not np.allclose(
            np.linalg.norm(vertices[used_vertices] + ki_sample, axis=-1),
            k_magnitude_Ainv,
            rtol=0.0,
            atol=4096.0 * np.finfo(np.float64).eps * max(k_magnitude_Ainv, 1.0),
        ):
            raise ValueError("visible Ewald vertices must lie on the incident sphere")
        if not math.isfinite(float(self.detector_visible_m0_q_gap_Ainv)) or (
            self.detector_visible_m0_q_gap_Ainv <= 0.0
        ):
            raise ValueError("the visible m=0 patch requires a positive reciprocal Q gap")
        if not math.isfinite(float(self.maximum_ewald_residual_Ainv)) or (
            self.maximum_ewald_residual_Ainv < 0.0
        ):
            raise ValueError("maximum Ewald residual must be finite and nonnegative")
        for value, name in (
            (self.requested_worker_count, "requested_worker_count"),
            (self.worker_count, "worker_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.worker_count > self.requested_worker_count:
            raise ValueError("used worker count cannot exceed the requested worker count")
        if self.execution_backend == "serial_cpu.v1":
            if self.worker_count != 1 or self.multiprocessing_start_method is not None:
                raise ValueError("serial Ewald execution must use one worker and no start method")
        elif self.execution_backend == "process_cpu_spawn.v1":
            if self.multiprocessing_start_method != "spawn":
                raise ValueError("parallel Ewald execution requires the spawn start method")
        else:
            raise ValueError("unsupported Ewald execution backend")
        if self.measure_id != ("detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"):
            raise ValueError("unsupported Ewald patch measure")
        if self.selection_id != (
            "configured_active_panel_regular_all_inverse_preimages_including_m0.v1"
        ):
            raise ValueError("unsupported Ewald patch selection")
        for array in (
            ki_sample,
            vertices,
            directions,
            density,
            m0_density,
            inverse_count,
            caustic,
            visible,
            column_edges,
            row_edges,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "ki_sample_Ainv", ki_sample)
        object.__setattr__(self, "q_vertices_sample_Ainv", vertices)
        object.__setattr__(self, "outgoing_direction_centers_sample", directions)
        object.__setattr__(self, "intensity_density_A2_per_sr", density)
        object.__setattr__(self, "m0_intensity_density_A2_per_sr", m0_density)
        object.__setattr__(self, "inverse_branch_count", inverse_count)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "detector_visible", visible)
        object.__setattr__(self, "column_edges_px", column_edges)
        object.__setattr__(self, "row_edges_px", row_edges)


@dataclass(frozen=True, slots=True)
class SchematicRays:
    intrinsic_direction_lab: FloatArray
    direction_lab: FloatArray
    intrinsic_density_A2_rad2_inv: FloatArray
    ideal_point_lab_m: FloatArray
    tilted_point_lab_m: FloatArray
    origin_lab_m: FloatArray
    ideal_instrument: CompiledInstrument
    tilted_instrument: CompiledInstrument

    def __post_init__(self) -> None:
        intrinsic_direction = np.array(
            self.intrinsic_direction_lab,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        direction = np.array(self.direction_lab, dtype=np.float64, copy=True, order="C")
        density = np.array(
            self.intrinsic_density_A2_rad2_inv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        ideal_point = np.array(self.ideal_point_lab_m, dtype=np.float64, copy=True, order="C")
        tilted_point = np.array(self.tilted_point_lab_m, dtype=np.float64, copy=True, order="C")
        origin = np.array(self.origin_lab_m, dtype=np.float64, copy=True, order="C")
        if direction.ndim != 2 or direction.shape[1:] != (3,) or not direction.size:
            raise ValueError("schematic directions must have shape (ray, 3)")
        shape = (direction.shape[0],)
        if (
            intrinsic_direction.shape != (*shape, 3)
            or density.shape != shape
            or ideal_point.shape != (*shape, 3)
            or tilted_point.shape != (*shape, 3)
            or origin.shape != (3,)
            or not np.all(np.isfinite(intrinsic_direction))
            or not np.all(np.isfinite(direction))
            or not np.all(np.isfinite(density))
            or not np.all(np.isfinite(ideal_point))
            or not np.all(np.isfinite(tilted_point))
            or not np.all(np.isfinite(origin))
            or np.any(density < 0.0)
        ):
            raise ValueError("schematic ray arrays must be finite and shape-compatible")
        if not np.allclose(
            np.stack(
                (
                    np.linalg.norm(intrinsic_direction, axis=-1),
                    np.linalg.norm(direction, axis=-1),
                )
            ),
            1.0,
            rtol=0.0,
            atol=2.0e-14,
        ):
            raise ValueError(
                "schematic intrinsic and refracted LAB directions must be unit vectors"
            )
        if not isinstance(self.ideal_instrument, CompiledInstrument) or not isinstance(
            self.tilted_instrument,
            CompiledInstrument,
        ):
            raise TypeError("schematic instruments must be compiled instruments")
        for array in (intrinsic_direction, direction, density, ideal_point, tilted_point, origin):
            array.setflags(write=False)
        object.__setattr__(self, "intrinsic_direction_lab", intrinsic_direction)
        object.__setattr__(self, "direction_lab", direction)
        object.__setattr__(self, "intrinsic_density_A2_rad2_inv", density)
        object.__setattr__(self, "ideal_point_lab_m", ideal_point)
        object.__setattr__(self, "tilted_point_lab_m", tilted_point)
        object.__setattr__(self, "origin_lab_m", origin)


@dataclass(frozen=True, slots=True)
class PublicationData:
    settings: FigureSettings
    inputs: ConfiguredSimulationInputs
    nominal: NominalEwaldContext
    reciprocal: ReciprocalSpaceDisplay
    ewald: DetectorVisibleEwaldPatchDisplay
    mosaic_alpha_nodes_rad: FloatArray
    reciprocal_logical_candidate_count: int
    reciprocal_candidate_evaluation_count: int
    ideal_detector: DetectorDisplay
    tilted_detector: DetectorDisplay
    tilted_instrument: CompiledInstrument
    schematic_rays: SchematicRays
    timings_s: dict[str, float]


@dataclass(frozen=True, slots=True)
class EwaldOnlyData:
    settings: FigureSettings
    inputs: ConfiguredSimulationInputs
    nominal: NominalEwaldContext
    ewald: DetectorVisibleEwaldPatchDisplay
    timings_s: dict[str, float]


@dataclass(frozen=True, slots=True)
class SchematicOnlyData:
    settings: FigureSettings
    inputs: ConfiguredSimulationInputs
    nominal: NominalEwaldContext
    ewald: DetectorVisibleEwaldPatchDisplay
    ideal_detector: DetectorDisplay
    tilted_detector: DetectorDisplay
    tilted_instrument: CompiledInstrument
    schematic_rays: SchematicRays
    timings_s: dict[str, float]


def apply_figure_settings(
    base: SimulationConfiguration,
    settings: FigureSettings,
) -> SimulationConfiguration:
    """Apply explicit figure overrides without changing the Bi2Se3 model authority."""

    if base.material.phase_id != "bi2se3":
        raise ValueError("the publication renderer accepts only the Bi2Se3 phase configuration")
    if not base.instrument.axis_rotations:
        raise ValueError("the publication configuration requires a declared incidence axis")
    first_axis = replace(
        base.instrument.axis_rotations[0],
        angle_deg=settings.incidence_deg,
    )
    instrument = replace(
        base.instrument,
        axis_rotations=(first_axis, *base.instrument.axis_rotations[1:]),
    )
    source = replace(
        base.source,
        sample_count=base.source.minimum_physical_sample_count,
    )
    mosaic = replace(
        base.mosaic,
        gaussian_sigma_deg=settings.gaussian_sigma_deg,
        lorentzian_hwhm_deg=settings.lorentzian_hwhm_deg,
        lorentzian_probability=settings.lorentzian_probability,
    )
    numerics = replace(
        base.numerics,
        reciprocal_alpha_count=settings.mosaic_alpha_count,
        reciprocal_beta_count=settings.reciprocal_beta_count,
        reciprocal_u_count=settings.reciprocal_u_background_count,
        reciprocal_alpha_max_deg=180.0,
        ewald_alpha_count=settings.mosaic_alpha_count,
        ewald_alpha_max_deg=180.0,
        detector_execution_backend="cpu",
    )
    return replace(
        base,
        source=source,
        instrument=instrument,
        mosaic=mosaic,
        numerics=numerics,
    )


def select_scale_resolved_mosaic_alpha_nodes(
    inputs: ConfiguredSimulationInputs,
    *,
    count: int,
) -> FloatArray:
    """Thin the compiled full-support mosaic quadrature for deterministic display."""

    candidates = np.unique(np.asarray(inputs.bragg_space.mosaic_space.alpha_rad))
    if count < 2 or count > candidates.size:
        raise ValueError(f"mosaic alpha count must lie in [2, {candidates.size}]")
    indices = np.rint(np.linspace(0, candidates.size - 1, count)).astype(np.int64)
    if np.unique(indices).size != count:
        raise RuntimeError("scale-resolved mosaic thinning produced duplicate indices")
    selected = np.array(candidates[indices], dtype=np.float64, copy=True, order="C")
    selected.setflags(write=False)
    return selected


def structure_resolved_axial_nodes_Ainv(
    inputs: ConfiguredSimulationInputs,
    rod: object,
    *,
    background_count: int,
) -> FloatArray:
    """Combine coarse rod coverage with integer-L peaks and +/-0.5/N shoulders."""

    lower, upper = inputs.bragg_space.rod_u_bounds_Ainv(rod)
    b3_norm = float(np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2]))
    lower_l = lower / b3_norm
    upper_l = upper / b3_norm
    axial = (
        finite_stack_integer_l_display_nodes(
            lower_l,
            upper_l,
            layer_count=inputs.config.structure_factor.layers,
            background_count=background_count,
        )
        * b3_norm
    )
    axial = np.array(axial, dtype=np.float64, copy=True, order="C")
    axial.setflags(write=False)
    return axial


def sample_scale_resolved_reciprocal_space(
    inputs: ConfiguredSimulationInputs,
    *,
    alpha_nodes_rad: FloatArray,
    beta_count: int,
    u_background_count: int,
    maximum_points: int = RECIPROCAL_POINT_BUDGET,
) -> tuple[ReciprocalSpaceDisplay, int, int]:
    """Sample SF peaks and full-support mosaic scales through the canonical field."""

    beta = (np.arange(beta_count, dtype=np.float64) + 0.5) * (2.0 * np.pi / beta_count)
    rods = inputs.bragg_space.config.rods
    per_rod_budget = max(1, maximum_points // len(rods))
    points: list[FloatArray] = []
    intensities: list[FloatArray] = []
    families: list[NDArray[np.int64]] = []
    logical_candidate_count = 0
    evaluated_candidate_count = 0
    for rod in rods:
        axial = structure_resolved_axial_nodes_Ainv(
            inputs,
            rod,
            background_count=u_background_count,
        )
        logical_candidate_count += alpha_nodes_rad.size * beta.size * axial.size
        alpha_probe, axial_probe = np.meshgrid(
            alpha_nodes_rad,
            axial,
            indexing="ij",
        )
        probe = inputs.bragg_space.evaluate_latent(
            rod=rod,
            alpha_rad=alpha_probe,
            beta_rad=np.zeros_like(alpha_probe),
            u_Ainv=axial_probe,
        )
        evaluated_candidate_count += alpha_probe.size
        probe_density = probe.intensity_density_A2_rad2_inv
        positive_probe = np.isfinite(probe_density) & (probe_density > 0.0)
        if not np.any(positive_probe):
            continue
        local_high = float(np.max(probe_density[positive_probe]))
        resolved = positive_probe & (probe_density >= local_high * 1.0e-8)
        pair_alpha_index, pair_axial_index = np.nonzero(resolved)
        if not pair_alpha_index.size:
            continue

        represented_axial = np.any(resolved, axis=0)
        axial_index = np.flatnonzero(represented_axial)
        best_alpha = np.argmax(
            np.where(
                resolved[:, represented_axial],
                probe_density[:, represented_axial],
                -np.inf,
            ),
            axis=0,
        )
        reserve_alpha = np.repeat(alpha_nodes_rad[best_alpha], beta.size)
        reserve_beta = np.tile(beta, axial_index.size)
        reserve_axial = np.repeat(axial[axial_index], beta.size)
        reserve_evaluated = inputs.bragg_space.evaluate_latent(
            rod=rod,
            alpha_rad=reserve_alpha,
            beta_rad=reserve_beta,
            u_Ainv=reserve_axial,
        )
        evaluated_candidate_count += reserve_alpha.size
        reserve_q = reserve_evaluated.q_sample_Ainv.reshape(axial_index.size, beta.size, 3)
        reserve_density = reserve_evaluated.intensity_density_A2_rad2_inv.reshape(
            axial_index.size,
            beta.size,
        )
        reserve_valid = (
            np.all(np.isfinite(reserve_q), axis=-1)
            & np.isfinite(reserve_density)
            & (reserve_density >= local_high * 1.0e-8)
            & (reserve_q[..., 2] > 0.0)
        )
        has_reserve = np.any(reserve_valid, axis=1)
        reserve_rows = np.flatnonzero(has_reserve)
        reserve_beta_index = np.argmax(
            np.where(reserve_valid[has_reserve], reserve_q[has_reserve, :, 2], -np.inf),
            axis=1,
        )
        reserved_points = reserve_q[reserve_rows, reserve_beta_index]
        reserved_density = reserve_density[reserve_rows, reserve_beta_index]
        reserved_keys = (
            best_alpha[reserve_rows] * beta.size + reserve_beta_index
        ) * axial.size + axial_index[reserve_rows]
        if reserved_points.shape[0] > per_rod_budget:
            raise RuntimeError("reciprocal point budget cannot retain every resolved axial node")

        fill_budget = per_rod_budget - reserved_points.shape[0]
        total_fill_candidates = pair_alpha_index.size * beta.size
        fill_evaluation_count = min(total_fill_candidates, 3 * fill_budget)
        if fill_evaluation_count:
            logical_position = np.floor(
                (np.arange(fill_evaluation_count, dtype=np.float64) + 0.5)
                * total_fill_candidates
                / fill_evaluation_count
            ).astype(np.int64)
            pair_position, beta_index = np.divmod(logical_position, beta.size)
            fill_alpha_index = pair_alpha_index[pair_position]
            fill_axial_index = pair_axial_index[pair_position]
            fill_evaluated = inputs.bragg_space.evaluate_latent(
                rod=rod,
                alpha_rad=alpha_nodes_rad[fill_alpha_index],
                beta_rad=beta[beta_index],
                u_Ainv=axial[fill_axial_index],
            )
            evaluated_candidate_count += fill_evaluation_count
            fill_q = fill_evaluated.q_sample_Ainv
            fill_density = fill_evaluated.intensity_density_A2_rad2_inv
            fill_keys = (fill_alpha_index * beta.size + beta_index) * axial.size + fill_axial_index
            fill_valid = (
                np.all(np.isfinite(fill_q), axis=-1)
                & np.isfinite(fill_density)
                & (fill_density >= local_high * 1.0e-8)
                & (fill_q[:, 2] > 0.0)
                & ~np.isin(fill_keys, reserved_keys)
            )
            fill_selected = np.flatnonzero(fill_valid)
            if fill_selected.size > fill_budget:
                positions = np.rint(np.linspace(0, fill_selected.size - 1, fill_budget)).astype(
                    np.int64
                )
                fill_selected = fill_selected[positions]
            rod_points = np.concatenate((reserved_points, fill_q[fill_selected]))
            rod_density = np.concatenate((reserved_density, fill_density[fill_selected]))
        else:
            rod_points = reserved_points
            rod_density = reserved_density
        if not rod_points.size:
            continue
        points.append(rod_points)
        intensities.append(rod_density)
        families.append(np.full(rod_density.size, rod.family_m, dtype=np.int64))
    if not points:
        raise RuntimeError("scale-resolved reciprocal display has no positive-Qz points")
    return (
        ReciprocalSpaceDisplay(
            q_sample_Ainv=np.concatenate(points),
            intensity_density_A2_rad2_inv=np.concatenate(intensities),
            family_m=np.concatenate(families),
            reference_wavelength_A=2.0 * np.pi / inputs.bragg_space.config.k_norm_Ainv,
        ),
        logical_candidate_count,
        evaluated_candidate_count,
    )


def _equal_solid_angle_q_vertices(
    ki_sample_Ainv: FloatArray,
    *,
    mu_count: int,
    phi_count: int,
) -> FloatArray:
    """Build a closed, ki-aligned Q-sphere vertex grid with equal-area cells."""

    basis = _incident_aligned_basis(ki_sample_Ainv)
    mu_edges = np.linspace(-1.0, 1.0, mu_count + 1)
    phi_edges = np.linspace(0.0, 2.0 * np.pi, phi_count + 1)
    mu, phi = np.meshgrid(mu_edges, phi_edges, indexing="ij")
    transverse = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    kf_hat_components = np.stack(
        (transverse * np.cos(phi), transverse * np.sin(phi), mu),
        axis=-1,
    )
    kf_hat_sample = kf_hat_components @ basis.T
    k_magnitude_Ainv = float(np.linalg.norm(ki_sample_Ainv))
    q_vertices = k_magnitude_Ainv * kf_hat_sample - ki_sample_Ainv
    q_vertices[:, -1] = q_vertices[:, 0]
    return q_vertices


@dataclass(frozen=True, slots=True)
class _EwaldPatchTile:
    start_row: int
    stop_row: int
    density_A2_per_sr: FloatArray
    m0_density_A2_per_sr: FloatArray
    outgoing_direction_sample: FloatArray
    inverse_branch_count: NDArray[np.int64]
    caustic: BoolArray
    detector_visible: BoolArray
    maximum_ewald_residual_Ainv: float
    detector_visible_m0_q_gap_Ainv: float
    measure_id: str


def _evaluate_ewald_patch_tile(
    context: NominalEwaldContext,
    column_centers_px: FloatArray,
    row_centers_px: FloatArray,
    start_row: int,
    stop_row: int,
) -> _EwaldPatchTile:
    column_grid, row_grid = np.broadcast_arrays(
        column_centers_px[None, :],
        row_centers_px[start_row:stop_row, None],
    )
    evaluated = context.geometry.evaluate_detector_visible_ewald_directions(
        column_grid,
        row_grid,
        rods=context.rods,
    )
    m0_indices = tuple(
        rod_index for rod_index, rod in enumerate(evaluated.rods) if rod.family_m == 0
    )
    if len(m0_indices) != 1 or evaluated.detector_visible_m0_q_gap_Ainv is None:
        raise RuntimeError(
            "the configured publication rod catalog must contain exactly one m=0 rod"
        )
    m0_density = evaluated.per_rod_density_A2_per_sr[..., m0_indices[0]]
    return _EwaldPatchTile(
        start_row=start_row,
        stop_row=stop_row,
        density_A2_per_sr=evaluated.density_A2_per_sr,
        m0_density_A2_per_sr=m0_density,
        outgoing_direction_sample=evaluated.outgoing_direction_sample,
        inverse_branch_count=np.sum(
            evaluated.per_rod_inverse_branch_count,
            axis=-1,
            dtype=np.int64,
        ),
        caustic=np.any(evaluated.caustic, axis=-1),
        detector_visible=evaluated.detector_visible,
        maximum_ewald_residual_Ainv=float(
            np.max(evaluated.geometry.ewald_residual_Ainv, initial=0.0)
        ),
        detector_visible_m0_q_gap_Ainv=evaluated.detector_visible_m0_q_gap_Ainv,
        measure_id=evaluated.measure_id,
    )


def _evaluate_ewald_patch_batch(
    config: SimulationConfiguration,
    column_centers_px: FloatArray,
    row_centers_px: FloatArray,
    row_ranges: tuple[tuple[int, int], ...],
) -> tuple[_EwaldPatchTile, ...]:
    """Build immutable worker-local physics once, then evaluate assigned row tiles."""

    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
    )

    context = build_nominal_ewald_context(build_configured_simulation_inputs(config))
    return tuple(
        _evaluate_ewald_patch_tile(
            context,
            column_centers_px,
            row_centers_px,
            start_row,
            stop_row,
        )
        for start_row, stop_row in row_ranges
    )


@contextmanager
def _single_threaded_child_blas() -> Iterator[None]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _evaluate_ewald_patch_tiles(
    config: SimulationConfiguration,
    context: NominalEwaldContext,
    column_centers_px: FloatArray,
    row_centers_px: FloatArray,
    *,
    tile_row_count: int,
    worker_count: int,
) -> tuple[tuple[_EwaldPatchTile, ...], str, str | None, int]:
    row_count = row_centers_px.size
    row_ranges = tuple(
        (start_row, min(start_row + tile_row_count, row_count))
        for start_row in range(0, row_count, tile_row_count)
    )
    used_worker_count = min(worker_count, len(row_ranges))
    if used_worker_count == 1:
        tiles = tuple(
            _evaluate_ewald_patch_tile(
                context,
                column_centers_px,
                row_centers_px,
                start_row,
                stop_row,
            )
            for start_row, stop_row in row_ranges
        )
        return tiles, "serial_cpu.v1", None, 1

    batches = tuple(
        tuple(row_ranges[worker_index::used_worker_count])
        for worker_index in range(used_worker_count)
    )
    completed: list[_EwaldPatchTile] = []
    spawn_context = get_context("spawn")
    with (
        _single_threaded_child_blas(),
        ProcessPoolExecutor(
            max_workers=used_worker_count,
            mp_context=spawn_context,
        ) as executor,
    ):
        futures = tuple(
            executor.submit(
                _evaluate_ewald_patch_batch,
                config,
                column_centers_px,
                row_centers_px,
                batch,
            )
            for batch in batches
        )
        try:
            for future in as_completed(futures):
                completed.extend(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return (
        tuple(sorted(completed, key=lambda tile: tile.start_row)),
        "process_cpu_spawn.v1",
        "spawn",
        used_worker_count,
    )


def evaluate_detector_visible_ewald_patch(
    config: SimulationConfiguration,
    context: NominalEwaldContext,
    *,
    image_size: int,
    tile_row_count: int,
    worker_count: int,
) -> DetectorVisibleEwaldPatchDisplay:
    """Sample only the configured active-panel patch of the internal Ewald sphere."""

    for value, name, minimum in (
        (image_size, "image_size", 4),
        (tile_row_count, "tile_row_count", 1),
        (worker_count, "worker_count", 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    image_size = int(image_size)
    tile_row_count = int(tile_row_count)
    worker_count = int(worker_count)
    detector_rows, detector_columns = context.instrument.detector_shape_rc
    full_column_edges = np.linspace(-0.5, detector_columns - 0.5, image_size + 1)
    full_row_edges = np.linspace(-0.5, detector_rows - 0.5, image_size + 1)
    column_centers = 0.5 * (full_column_edges[:-1] + full_column_edges[1:])
    row_centers = 0.5 * (full_row_edges[:-1] + full_row_edges[1:])
    tiles, backend, start_method, used_worker_count = _evaluate_ewald_patch_tiles(
        config,
        context,
        column_centers,
        row_centers,
        tile_row_count=tile_row_count,
        worker_count=worker_count,
    )

    full_shape = (image_size, image_size)
    density = np.zeros(full_shape, dtype=np.float64)
    m0_density = np.zeros(full_shape, dtype=np.float64)
    directions = np.zeros((*full_shape, 3), dtype=np.float64)
    inverse_count = np.zeros(full_shape, dtype=np.int64)
    caustic = np.zeros(full_shape, dtype=np.bool_)
    center_visible = np.zeros(full_shape, dtype=np.bool_)
    coverage = np.zeros(image_size, dtype=np.int64)
    measure_id: str | None = None
    m0_gap_Ainv: float | None = None
    maximum_residual_Ainv = 0.0
    for tile in tiles:
        start_row, stop_row = tile.start_row, tile.stop_row
        tile_shape = (stop_row - start_row, image_size)
        if (
            not (0 <= start_row < stop_row <= image_size)
            or any(
                array.shape != tile_shape
                for array in (
                    tile.density_A2_per_sr,
                    tile.m0_density_A2_per_sr,
                    tile.inverse_branch_count,
                    tile.caustic,
                    tile.detector_visible,
                )
            )
            or tile.outgoing_direction_sample.shape != (*tile_shape, 3)
        ):
            raise ValueError("Ewald worker returned a malformed row tile")
        coverage[start_row:stop_row] += 1
        density[start_row:stop_row] = tile.density_A2_per_sr
        m0_density[start_row:stop_row] = tile.m0_density_A2_per_sr
        directions[start_row:stop_row] = tile.outgoing_direction_sample
        inverse_count[start_row:stop_row] = tile.inverse_branch_count
        caustic[start_row:stop_row] = tile.caustic
        center_visible[start_row:stop_row] = tile.detector_visible
        if measure_id is None:
            measure_id = tile.measure_id
            m0_gap_Ainv = tile.detector_visible_m0_q_gap_Ainv
        elif tile.measure_id != measure_id or tile.detector_visible_m0_q_gap_Ainv != m0_gap_Ainv:
            raise ValueError("Ewald worker metadata changed between row tiles")
        maximum_residual_Ainv = max(
            maximum_residual_Ainv,
            tile.maximum_ewald_residual_Ainv,
        )
    if not np.all(coverage == 1):
        raise ValueError("Ewald worker rows must cover the raster exactly once")
    if measure_id is None or m0_gap_Ainv is None or not np.any(center_visible):
        raise RuntimeError("the configured active detector has no internal Ewald sphere patch")
    if maximum_residual_Ainv > 2.0e-13:
        raise FloatingPointError("detector-visible intrinsic coating violates the Ewald identity")

    visible_row, visible_column = np.nonzero(center_visible)
    row_start, row_stop = int(np.min(visible_row)), int(np.max(visible_row)) + 1
    column_start, column_stop = int(np.min(visible_column)), int(np.max(visible_column)) + 1
    column_edges = full_column_edges[column_start : column_stop + 1]
    row_edges = full_row_edges[row_start : row_stop + 1]
    column_vertex_grid, row_vertex_grid = np.meshgrid(column_edges, row_edges)
    vertex_geometry = context.geometry.evaluate_detector_visible_geometry(
        column_vertex_grid,
        row_vertex_grid,
    )
    vertex_valid = vertex_geometry.valid
    visible = center_visible[row_start:row_stop, column_start:column_stop].copy()
    visible &= (
        vertex_valid[:-1, :-1]
        & vertex_valid[:-1, 1:]
        & vertex_valid[1:, :-1]
        & vertex_valid[1:, 1:]
    )
    if not np.any(visible):
        raise RuntimeError("no complete detector-native cell lies on the visible Ewald patch")

    painted_row, painted_column = np.nonzero(visible)
    local_row_start, local_row_stop = int(np.min(painted_row)), int(np.max(painted_row)) + 1
    local_column_start = int(np.min(painted_column))
    local_column_stop = int(np.max(painted_column)) + 1
    row_slice = slice(row_start + local_row_start, row_start + local_row_stop)
    column_slice = slice(column_start + local_column_start, column_start + local_column_stop)
    vertex_row_slice = slice(local_row_start, local_row_stop + 1)
    vertex_column_slice = slice(local_column_start, local_column_stop + 1)
    visible = visible[
        local_row_start:local_row_stop,
        local_column_start:local_column_stop,
    ]
    density = density[row_slice, column_slice].copy()
    m0_density = m0_density[row_slice, column_slice].copy()
    directions = directions[row_slice, column_slice].copy()
    inverse_count = inverse_count[row_slice, column_slice].copy()
    caustic = caustic[row_slice, column_slice].copy()
    density[~visible] = 0.0
    m0_density[~visible] = 0.0
    directions[~visible] = 0.0
    inverse_count[~visible] = 0
    caustic[~visible] = False
    if not np.any(density > 0.0) or not np.any(m0_density > 0.0):
        raise RuntimeError("the visible Ewald patch lacks positive all-rod or m=0 support")

    return DetectorVisibleEwaldPatchDisplay(
        ki_sample_Ainv=context.ki_sample_Ainv,
        q_vertices_sample_Ainv=vertex_geometry.q_sample_Ainv[
            vertex_row_slice,
            vertex_column_slice,
        ],
        outgoing_direction_centers_sample=directions,
        intensity_density_A2_per_sr=density,
        m0_intensity_density_A2_per_sr=m0_density,
        inverse_branch_count=inverse_count,
        caustic=caustic,
        detector_visible=visible,
        column_edges_px=column_edges[local_column_start : local_column_stop + 1],
        row_edges_px=row_edges[local_row_start : local_row_stop + 1],
        detector_visible_m0_q_gap_Ainv=m0_gap_Ainv,
        maximum_ewald_residual_Ainv=maximum_residual_Ainv,
        requested_worker_count=worker_count,
        worker_count=used_worker_count,
        execution_backend=backend,
        multiprocessing_start_method=start_method,
        measure_id=measure_id,
    )


def _coarsened_cell_count(count: int, maximum: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2
        for value in (count, maximum)
    ):
        raise ValueError("surface cell counts and limits must be integers of at least two")
    return min(int(count), int(maximum))


def _ewald_render_texture(
    display: DetectorVisibleEwaldPatchDisplay,
    *,
    maximum_row_count: int,
    maximum_column_count: int,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Block-average the detector-native patch without inventing sphere vertices."""

    source_rows, source_columns = display.intensity_density_A2_per_sr.shape
    target_rows = _coarsened_cell_count(source_rows, maximum_row_count)
    target_columns = _coarsened_cell_count(source_columns, maximum_column_count)
    row_bounds = np.rint(np.linspace(0, source_rows, target_rows + 1)).astype(np.int64)
    column_bounds = np.rint(np.linspace(0, source_columns, target_columns + 1)).astype(np.int64)
    if np.any(np.diff(row_bounds) <= 0) or np.any(np.diff(column_bounds) <= 0):
        raise ValueError("Ewald render boundaries must partition every source cell")

    def block_sum(values: FloatArray) -> FloatArray:
        integral = np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        return (
            integral[row_bounds[1:, None], column_bounds[None, 1:]]
            - integral[row_bounds[:-1, None], column_bounds[None, 1:]]
            - integral[row_bounds[1:, None], column_bounds[None, :-1]]
            + integral[row_bounds[:-1, None], column_bounds[None, :-1]]
        )

    finite_density = np.where(
        np.isfinite(display.intensity_density_A2_per_sr),
        display.intensity_density_A2_per_sr,
        0.0,
    )
    cell_count = np.diff(row_bounds)[:, None] * np.diff(column_bounds)[None, :]
    density = block_sum(finite_density) / cell_count
    visible = block_sum(display.detector_visible.astype(np.float64)) == cell_count
    caustic = block_sum(display.caustic.astype(np.float64)) > 0.0
    density[caustic] = np.inf
    density[~visible] = 0.0
    vertices = display.q_vertices_sample_Ainv[
        row_bounds[:, None],
        column_bounds[None, :],
    ]
    return vertices, density, visible


def tilt_detector_about_reference(
    instrument: CompiledInstrument,
    *,
    column_tilt_deg: float,
    row_tilt_deg: float,
) -> CompiledInstrument:
    """Tilt a compiled detector while keeping its reference-coordinate point fixed."""

    rotation = compose_intrinsic_xy_rotation(
        instrument.lab_from_detector.rotation,
        math.radians(float(column_tilt_deg)),
        math.radians(float(row_tilt_deg)),
    )
    pose = replace(instrument.lab_from_detector, rotation=rotation)
    return replace(instrument, lab_from_detector=pose)


def evaluate_detector_display(
    detector: DetectorDensityModel,
    *,
    instrument: CompiledInstrument,
    image_size: int,
    tile_row_count: int,
) -> DetectorDisplay:
    """Center-sample one continuous detector density on a bounded display grid."""

    for value, name in ((image_size, "image_size"), (tile_row_count, "tile_row_count")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    image_size = int(image_size)
    tile_row_count = int(tile_row_count)
    rows, columns = instrument.detector_shape_rc
    column_edges = np.linspace(-0.5, columns - 0.5, image_size + 1)
    row_edges = np.linspace(-0.5, rows - 0.5, image_size + 1)
    column = 0.5 * (column_edges[:-1] + column_edges[1:])
    row = 0.5 * (row_edges[:-1] + row_edges[1:])
    density = np.zeros((image_size, image_size), dtype=np.float64)
    valid = np.zeros_like(density, dtype=np.bool_)
    caustic = np.zeros_like(density, dtype=np.bool_)
    measure_id: str | None = None
    execution_backend: str | None = None
    for start in range(0, image_size, tile_row_count):
        stop = min(start + tile_row_count, image_size)
        column_grid, row_grid = np.broadcast_arrays(
            column[None, :],
            row[start:stop, None],
        )
        evaluated = detector.evaluate_detector_density_all_roots(
            column_grid,
            row_grid,
            execution_backend="cpu",
        )
        tile_shape = column_grid.shape
        tile_density = np.asarray(evaluated.density_A2_per_px2)
        tile_valid_count = np.asarray(evaluated.valid_source_count)
        tile_caustic = np.asarray(evaluated.caustic)
        if any(
            array.shape != tile_shape for array in (tile_density, tile_valid_count, tile_caustic)
        ):
            raise ValueError(
                "detector tile arrays must exactly match the requested coordinate grid"
            )
        tile_measure_id = str(evaluated.measure_id)
        tile_backend = str(evaluated.execution_backend)
        if measure_id is not None and (
            tile_measure_id != measure_id or tile_backend != execution_backend
        ):
            raise ValueError("detector evaluator metadata changed between display tiles")
        density[start:stop] = tile_density
        valid[start:stop] = tile_valid_count > 0
        caustic[start:stop] = tile_caustic
        measure_id = tile_measure_id
        execution_backend = tile_backend
    if measure_id is None or execution_backend is None:
        raise RuntimeError("detector display grid was not evaluated")
    for array in (density, valid, caustic, column_edges, row_edges):
        array.setflags(write=False)
    return DetectorDisplay(
        density_A2_per_px2=density,
        valid=valid,
        caustic=caustic,
        column_edges_px=column_edges,
        row_edges_px=row_edges,
        measure_id=measure_id,
        execution_backend=execution_backend,
    )


def detector_surface_raster(
    display: DetectorDisplay,
    instrument: CompiledInstrument,
) -> DetectorSurfaceRaster:
    """Carry a detector-native cell density onto the detector plane in LAB coordinates."""

    column_grid, row_grid = np.meshgrid(
        display.column_edges_px,
        display.row_edges_px,
        indexing="xy",
    )
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    detector_local_m = np.stack(
        (
            (column_grid - reference_column) * instrument.detector_column_pitch_m,
            (row_grid - reference_row) * instrument.detector_row_pitch_m,
            np.zeros_like(column_grid),
        ),
        axis=-1,
    )
    return DetectorSurfaceRaster(
        vertices_lab_m=instrument.lab_from_detector.apply_point(detector_local_m),
        density_A2_per_px2=display.density_A2_per_px2,
        valid=display.valid,
        caustic=display.caustic,
        measure_id=display.measure_id,
    )


def _coarsen_detector_display(
    display: DetectorDisplay,
    *,
    maximum_cell_count: int,
) -> DetectorDisplay:
    """Block-average a detector density in linear space for a tractable 3D texture."""

    if maximum_cell_count <= 0:
        raise ValueError("maximum_cell_count must be positive")
    row_count, column_count = display.density_A2_per_px2.shape
    row_factor = math.ceil(row_count / maximum_cell_count)
    column_factor = math.ceil(column_count / maximum_cell_count)
    row_starts = np.arange(0, row_count, row_factor)
    column_starts = np.arange(0, column_count, column_factor)
    row_sizes = np.diff(np.append(row_starts, row_count))
    column_sizes = np.diff(np.append(column_starts, column_count))

    def block_sum(values: NDArray[np.generic]) -> NDArray[np.generic]:
        return np.add.reduceat(
            np.add.reduceat(values, row_starts, axis=0),
            column_starts,
            axis=1,
        )

    density = block_sum(display.density_A2_per_px2) / (row_sizes[:, None] * column_sizes[None, :])
    valid = block_sum(display.valid.astype(np.int64)) > 0
    caustic = block_sum(display.caustic.astype(np.int64)) > 0
    column_edges = display.column_edges_px[np.append(column_starts, column_count)]
    row_edges = display.row_edges_px[np.append(row_starts, row_count)]
    for array in (density, valid, caustic, column_edges, row_edges):
        array.setflags(write=False)
    return DetectorDisplay(
        density_A2_per_px2=density,
        valid=valid,
        caustic=caustic,
        column_edges_px=column_edges,
        row_edges_px=row_edges,
        measure_id=display.measure_id,
        execution_backend=display.execution_backend,
    )


def _sample_schematic_rays(
    nominal: NominalEwaldContext,
    *,
    tilted_instrument: CompiledInstrument,
    alpha_nodes_rad: FloatArray,
) -> SchematicRays:
    from rasim_next.core.validity import ValidityCode

    beta_count = 48
    beta = (np.arange(beta_count) + 0.5) * 2.0 * np.pi / beta_count
    alpha_grid, beta_grid = np.meshgrid(alpha_nodes_rad, beta, indexing="ij")
    intrinsic_directions: list[FloatArray] = []
    directions: list[FloatArray] = []
    densities: list[FloatArray] = []
    for rod in nominal.rods:
        if rod.family_m == 0:
            continue
        for branch in (1, 2):
            evaluated = nominal.geometry.map_detector_visible_coating(
                rod=rod,
                branch=branch,
                alpha_rad=alpha_grid,
                beta_rad=beta_grid,
            )
            density = evaluated.coating_intensity_density_A2_rad2_inv
            ewald = evaluated.geometry.ewald_geometry
            valid = (
                ewald.valid
                & (ewald.q_sample_Ainv[..., 2] > 0.0)
                & (evaluated.geometry.exit_status == ValidityCode.VALID.value)
                & np.isfinite(density)
                & (density > 0.0)
            )
            if not np.any(valid):
                continue
            kf_film_lab = nominal.instrument.lab_from_sample.apply_vector(
                ewald.kf_sample_Ainv[valid]
            )
            intrinsic_directions.append(
                kf_film_lab / np.linalg.norm(kf_film_lab, axis=-1, keepdims=True)
            )
            kf_air_lab = evaluated.geometry.kf_air_lab_Ainv[valid]
            directions.append(kf_air_lab / np.linalg.norm(kf_air_lab, axis=-1, keepdims=True))
            densities.append(density[valid])
    if not directions:
        raise RuntimeError("no exit-valid positive-Qz intrinsic Ewald rays are available")
    intrinsic_direction = np.concatenate(intrinsic_directions)
    direction = np.concatenate(directions)
    density = np.concatenate(densities)

    origin = nominal.incident.states.sample_intersection_lab_m[0]
    schematic_ideal = nominal.instrument
    schematic_tilted = tilted_instrument
    origins = np.broadcast_to(origin, direction.shape)
    ideal_projection = project_detector_rays(origins, direction, schematic_ideal)
    tilted_projection = project_detector_rays(origins, direction, schematic_tilted)
    valid = ideal_projection.valid & tilted_projection.valid
    if not np.any(valid):
        raise RuntimeError("no common rays reach both physical detector poses")
    intrinsic_direction = intrinsic_direction[valid]
    direction = direction[valid]
    density = density[valid]
    return SchematicRays(
        intrinsic_direction_lab=intrinsic_direction,
        direction_lab=direction,
        intrinsic_density_A2_rad2_inv=density,
        ideal_point_lab_m=ideal_projection.point_lab_m[valid],
        tilted_point_lab_m=tilted_projection.point_lab_m[valid],
        origin_lab_m=origin,
        ideal_instrument=schematic_ideal,
        tilted_instrument=schematic_tilted,
    )


def _configure_matplotlib() -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise SystemExit(
            "This renderer requires Matplotlib; install the visualization extra first."
        ) from error
    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "figure.titlesize": 15.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _positive_log_bounds(
    values: Sequence[FloatArray],
    *,
    decades: float,
    upper_quantile: float = 1.0,
) -> tuple[float, float]:
    positive_parts = [
        np.asarray(value)[np.isfinite(value) & (np.asarray(value) > 0.0)] for value in values
    ]
    positive_parts = [value for value in positive_parts if value.size]
    if not positive_parts:
        raise RuntimeError("the requested figure has no positive finite intensity")
    positive = np.concatenate(positive_parts)
    high = float(np.quantile(positive, upper_quantile))
    low = max(float(np.min(positive)), high * 10.0 ** (-decades))
    return low, high


def _set_equal_3d_limits(axis: object, points: FloatArray, *, padding: float = 0.04) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    half_span = 0.5 * float(np.max(maximum - minimum)) * (1.0 + padding)
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _incident_aligned_basis(ki_sample_Ainv: FloatArray) -> FloatArray:
    beam_axis = np.asarray(ki_sample_Ainv, dtype=np.float64)
    beam_axis = beam_axis / np.linalg.norm(beam_axis)
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(reference @ beam_axis)) > 0.9:
        reference = np.asarray((1.0, 0.0, 0.0))
    transverse_1 = np.cross(reference, beam_axis)
    transverse_1 /= np.linalg.norm(transverse_1)
    transverse_2 = np.cross(beam_axis, transverse_1)
    return np.column_stack((transverse_1, transverse_2, beam_axis))


def _draw_reciprocal_panel(axis: object, data: ReciprocalSpaceDisplay) -> object:
    from matplotlib.colors import LogNorm

    positive_qz = data.q_sample_Ainv[:, 2] > 0.0
    points = data.q_sample_Ainv[positive_qz]
    intensity = data.intensity_density_A2_rad2_inv[positive_qz]
    low, high = _positive_log_bounds((intensity,), decades=8.0)
    shown = axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=intensity,
        s=0.65,
        alpha=0.34,
        linewidths=0.0,
        cmap="magma",
        norm=LogNorm(vmin=low, vmax=high),
        rasterized=True,
    )
    axis.set_xlabel(r"$Q_x$ ($\AA^{-1}$)")
    axis.set_ylabel(r"$Q_y$ ($\AA^{-1}$)")
    axis.set_zlabel(r"$Q_z$ ($\AA^{-1}$)")
    axis.set_title("a  Reciprocal-space intensity", loc="left", fontweight="bold")
    axis.view_init(elev=22.0, azim=-52.0)
    axis.set_proj_type("ortho")
    _set_equal_3d_limits(axis, points)
    return shown


def _ewald_norm(data: DetectorVisibleEwaldPatchDisplay) -> object:
    from matplotlib.colors import LogNorm

    low, high = _positive_log_bounds(
        (data.intensity_density_A2_per_sr,),
        decades=8.0,
        upper_quantile=0.9995,
    )
    return LogNorm(vmin=low, vmax=high, clip=False)


def _surface_facecolors(
    density: FloatArray,
    *,
    norm: object,
    valid: BoolArray | None = None,
    invalid_color: object = "#e8e8e3",
) -> tuple[object, FloatArray]:
    import matplotlib
    from matplotlib.colors import to_rgba

    color_map = matplotlib.colormaps["magma"].copy()
    color_map.set_under("#07040c")
    color_map.set_over("#fff4b5")
    color_map.set_bad("#e8e8e3")
    values = np.asarray(density, dtype=np.float64)
    safe_values = np.where(values > 0.0, values, float(norm.vmin))
    colors = np.asarray(color_map(norm(safe_values)), dtype=np.float64)
    colors[values <= 0.0] = to_rgba("#777a7d")
    if valid is not None:
        colors[~np.asarray(valid, dtype=np.bool_)] = to_rgba(invalid_color)
    return color_map, colors


def _surface_scalar_mappable(color_map: object, norm: object, values: FloatArray) -> object:
    from matplotlib.cm import ScalarMappable

    shown = ScalarMappable(norm=norm, cmap=color_map)
    shown.set_array(values)
    return shown


def _draw_ewald_panel(
    axis: object,
    data: DetectorVisibleEwaldPatchDisplay,
    *,
    ki_sample_Ainv: FloatArray,
    maximum_row_count: int = 720,
    maximum_column_count: int = 1440,
) -> object:
    basis = _incident_aligned_basis(ki_sample_Ainv)
    q_vertices, display_density, display_visible = _ewald_render_texture(
        data,
        maximum_row_count=maximum_row_count,
        maximum_column_count=maximum_column_count,
    )
    points_ki = q_vertices @ basis
    norm = _ewald_norm(data)
    color_map, facecolors = _surface_facecolors(
        display_density,
        norm=norm,
        valid=display_visible,
        invalid_color=(0.0, 0.0, 0.0, 0.0),
    )
    context_sphere = (
        _equal_solid_angle_q_vertices(
            ki_sample_Ainv,
            mu_count=24,
            phi_count=48,
        )
        @ basis
    )
    axis.plot_wireframe(
        context_sphere[..., 0],
        context_sphere[..., 1],
        context_sphere[..., 2],
        rstride=2,
        cstride=4,
        color="#8c969f",
        alpha=0.18,
        linewidth=0.35,
    )
    surface = axis.plot_surface(
        points_ki[..., 0],
        points_ki[..., 1],
        points_ki[..., 2],
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )
    surface.set_rasterized(True)
    for edge in (
        points_ki[0],
        points_ki[-1],
        points_ki[:, 0],
        points_ki[:, -1],
    ):
        axis.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#315f70", linewidth=0.8)
    shown = _surface_scalar_mappable(
        color_map,
        norm,
        display_density,
    )
    radius = float(np.linalg.norm(ki_sample_Ainv))
    axis.scatter((0.0,), (0.0,), (0.0,), color="#1d2329", s=14, depthshade=False)
    axis.quiver(
        0.0,
        0.0,
        -radius,
        0.0,
        0.0,
        radius,
        color="#2a9d76",
        linewidth=1.8,
        arrow_length_ratio=0.12,
    )
    axis.text(0.0, 0.0, -0.46 * radius, r"$\mathbf{k}_i$", color="#167457")
    axis.set_xlabel(r"$Q_{\perp 1}$ ($\AA^{-1}$)")
    axis.set_ylabel(r"$Q_{\perp 2}$ ($\AA^{-1}$)")
    axis.set_zlabel(r"$Q_{\parallel k_i}$ ($\AA^{-1}$)")
    axis.set_title("b  Detector-visible Ewald intensity", loc="left", fontweight="bold")
    axis.text2D(
        0.02,
        0.025,
        r"painted patch: configured active panel; regular nonzero $m=0$ included",
        transform=axis.transAxes,
        fontsize=7.8,
        color="#315f70",
    )
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.set_zlim(-2.0 * radius, 0.05 * radius)
    axis.set_box_aspect((1.0, 1.0, 1.08))
    axis.view_init(elev=18.0, azim=-52.0)
    axis.set_proj_type("ortho")
    return shown


def _detector_norm(ideal: DetectorDisplay, tilted: DetectorDisplay) -> object:
    from matplotlib.colors import LogNorm

    low, high = _positive_log_bounds(
        (ideal.density_A2_per_px2, tilted.density_A2_per_px2),
        decades=8.0,
        upper_quantile=0.9995,
    )
    return LogNorm(vmin=low, vmax=high, clip=False)


def _log_scale_metadata(norm: object, arrays: Sequence[FloatArray]) -> dict[str, object]:
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)
    has_under = any(
        bool(np.any(np.isfinite(values) & (values > 0.0) & (values < vmin))) for values in arrays
    )
    has_over = any(
        bool(np.any(np.isposinf(values) | (np.isfinite(values) & (values > vmax))))
        for values in arrays
    )
    extension = (
        "both"
        if has_under and has_over
        else "min"
        if has_under
        else "max"
        if has_over
        else "neither"
    )
    return {
        "vmin": vmin,
        "vmax": vmax,
        "upper_quantile": 0.9995,
        "maximum_decades": 8.0,
        "has_under_range_positive_values": has_under,
        "has_over_range_or_positive_infinite_values": has_over,
        "colorbar_extend": extension,
    }


def _draw_detector_panel(
    axis: object,
    data: DetectorDisplay,
    *,
    instrument: CompiledInstrument,
    norm: object,
    title: str,
    subtitle: str,
) -> object:
    import matplotlib

    color_map = matplotlib.colormaps["magma"].copy()
    color_map.set_bad("#f1f1ee")
    color_map.set_under("#050109")
    color_map.set_over("#fff4b5")
    display_values = np.where(
        np.isposinf(data.density_A2_per_px2),
        10.0 * float(norm.vmax),
        data.density_A2_per_px2,
    )
    displayed = np.ma.masked_where(~data.valid | (display_values <= 0.0), display_values)
    extent = (
        float(data.column_edges_px[0]),
        float(data.column_edges_px[-1]),
        float(data.row_edges_px[-1]),
        float(data.row_edges_px[0]),
    )
    shown = axis.imshow(
        displayed,
        origin="upper",
        extent=extent,
        cmap=color_map,
        norm=norm,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    axis.scatter(
        (reference_column,),
        (reference_row,),
        marker="+",
        s=58,
        linewidth=1.3,
        color="white",
        zorder=4,
    )
    axis.set_xlabel("detector column (px)")
    axis.set_ylabel("detector row (px)")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.text(
        0.02,
        0.02,
        subtitle,
        transform=axis.transAxes,
        fontsize=8.5,
        color="#f8f8f6",
        bbox={"facecolor": "#141414", "alpha": 0.68, "edgecolor": "none", "pad": 3.0},
    )
    return shown


def _draw_detector_surface(
    axis: object,
    surface_data: DisplaySurfaceRaster,
    *,
    norm: object,
    outline_color: str,
) -> object:
    color_map, facecolors = _surface_facecolors(
        surface_data.density_A2_per_px2,
        norm=norm,
        valid=surface_data.valid,
    )
    vertices = surface_data.vertices_display
    surface = axis.plot_surface(
        vertices[..., 0],
        vertices[..., 1],
        vertices[..., 2],
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )
    surface.set_rasterized(True)
    corners = np.asarray(
        (vertices[0, 0], vertices[0, -1], vertices[-1, -1], vertices[-1, 0], vertices[0, 0])
    )
    axis.plot(
        corners[:, 0],
        corners[:, 1],
        corners[:, 2],
        color=outline_color,
        linewidth=1.35,
    )
    return _surface_scalar_mappable(color_map, norm, surface_data.density_A2_per_px2)


def _reposition_detector_surface(
    surface_data: DetectorSurfaceRaster,
    *,
    source_center_lab_m: FloatArray,
    display_center: FloatArray,
    scale: float,
) -> DisplaySurfaceRaster:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("schematic detector scale must be finite and positive")
    vertices = display_center + scale * (surface_data.vertices_lab_m - source_center_lab_m)
    return DisplaySurfaceRaster(
        vertices_display=vertices,
        density_A2_per_px2=surface_data.density_A2_per_px2,
        valid=surface_data.valid,
        caustic=surface_data.caustic,
        measure_id=surface_data.measure_id,
    )


def _reposition_schematic_points(
    points_lab_m: FloatArray,
    *,
    source_center_lab_m: FloatArray,
    display_center: FloatArray,
    scale: float,
) -> FloatArray:
    return display_center + scale * (points_lab_m - source_center_lab_m)


def _ewald_direction_sphere_vertices_lab_m(
    q_vertices_sample_Ainv: FloatArray,
    *,
    nominal: NominalEwaldContext,
    origin_lab_m: FloatArray,
    radius_m: float,
) -> FloatArray:
    kf_sample_Ainv = q_vertices_sample_Ainv + nominal.ki_sample_Ainv
    direction_sample = kf_sample_Ainv / np.linalg.norm(kf_sample_Ainv, axis=-1, keepdims=True)
    direction_lab = nominal.instrument.lab_from_sample.apply_vector(direction_sample)
    direction_lab /= np.linalg.norm(direction_lab, axis=-1, keepdims=True)
    return origin_lab_m + radius_m * direction_lab


def _representative_ray_indices(rays: SchematicRays, *, count: int) -> NDArray[np.int64]:
    rows, columns = rays.ideal_instrument.detector_shape_rc
    projection = project_detector_rays(
        np.broadcast_to(rays.origin_lab_m, rays.direction_lab.shape),
        rays.direction_lab,
        rays.ideal_instrument,
    )
    column_bin = np.clip((projection.column_px / columns * 5).astype(np.int64), 0, 4)
    row_bin = np.clip((projection.row_px / rows * 4).astype(np.int64), 0, 3)
    selected: list[int] = []
    for bin_id in range(20):
        members = np.flatnonzero(row_bin * 5 + column_bin == bin_id)
        if members.size:
            selected.append(int(members[np.argmax(rays.intrinsic_density_A2_rad2_inv[members])]))
    selected.sort(key=lambda index: rays.intrinsic_density_A2_rad2_inv[index], reverse=True)
    return np.asarray(selected[:count], dtype=np.int64)


def _render_projection_schematic(data: PublicationData | SchematicOnlyData) -> object:
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(17.2, 8.8))
    axis = figure.add_axes((0.015, 0.045, 0.77, 0.91), projection="3d")
    rays = data.schematic_rays
    sphere_q_vertices, sphere_density, sphere_visible = _ewald_render_texture(
        data.ewald,
        maximum_row_count=360,
        maximum_column_count=720,
    )
    sphere_center = np.asarray((-0.235, -0.015, -0.035))
    sphere_radius_m = 0.052
    sphere_vertices = _ewald_direction_sphere_vertices_lab_m(
        sphere_q_vertices,
        nominal=data.nominal,
        origin_lab_m=sphere_center,
        radius_m=sphere_radius_m,
    )
    context_sphere_vertices = _ewald_direction_sphere_vertices_lab_m(
        _equal_solid_angle_q_vertices(
            data.nominal.ki_sample_Ainv,
            mu_count=18,
            phi_count=36,
        ),
        nominal=data.nominal,
        origin_lab_m=sphere_center,
        radius_m=sphere_radius_m,
    )
    axis.plot_wireframe(
        context_sphere_vertices[..., 0],
        context_sphere_vertices[..., 1],
        context_sphere_vertices[..., 2],
        rstride=2,
        cstride=3,
        color="#7b858e",
        alpha=0.16,
        linewidth=0.3,
    )
    sphere_norm = _ewald_norm(data.ewald)
    sphere_color_map, sphere_facecolors = _surface_facecolors(
        sphere_density,
        norm=sphere_norm,
        valid=sphere_visible,
        invalid_color=(0.0, 0.0, 0.0, 0.0),
    )
    sphere_surface = axis.plot_surface(
        sphere_vertices[..., 0],
        sphere_vertices[..., 1],
        sphere_vertices[..., 2],
        facecolors=sphere_facecolors,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0.0,
    )
    sphere_surface.set_rasterized(True)
    sphere_shown = _surface_scalar_mappable(
        sphere_color_map,
        sphere_norm,
        sphere_density,
    )

    ideal_display = _coarsen_detector_display(
        data.ideal_detector,
        maximum_cell_count=data.settings.schematic_detector_cell_count,
    )
    tilted_display = _coarsen_detector_display(
        data.tilted_detector,
        maximum_cell_count=data.settings.schematic_detector_cell_count,
    )
    detector_norm = _detector_norm(data.ideal_detector, data.tilted_detector)
    detector_scale = 0.66
    ideal_display_center = np.asarray((-0.025, 0.060, 0.0))
    tilted_display_center = np.asarray((0.205, 0.080, 0.0))
    ideal_source_center = rays.ideal_instrument.lab_from_detector.translation_m
    tilted_source_center = rays.tilted_instrument.lab_from_detector.translation_m
    ideal_surface = _reposition_detector_surface(
        detector_surface_raster(ideal_display, rays.ideal_instrument),
        source_center_lab_m=ideal_source_center,
        display_center=ideal_display_center,
        scale=detector_scale,
    )
    tilted_surface = _reposition_detector_surface(
        detector_surface_raster(tilted_display, rays.tilted_instrument),
        source_center_lab_m=tilted_source_center,
        display_center=tilted_display_center,
        scale=detector_scale,
    )
    _draw_detector_surface(
        axis,
        ideal_surface,
        norm=detector_norm,
        outline_color="#28a4c8",
    )
    detector_shown = _draw_detector_surface(
        axis,
        tilted_surface,
        norm=detector_norm,
        outline_color="#e68427",
    )

    ideal_display_points = _reposition_schematic_points(
        rays.ideal_point_lab_m,
        source_center_lab_m=ideal_source_center,
        display_center=ideal_display_center,
        scale=detector_scale,
    )
    tilted_display_points = _reposition_schematic_points(
        rays.tilted_point_lab_m,
        source_center_lab_m=tilted_source_center,
        display_center=tilted_display_center,
        scale=detector_scale,
    )
    sphere_anchor_points = sphere_center + sphere_radius_m * rays.intrinsic_direction_lab
    selected = _representative_ray_indices(rays, count=3)
    for index in selected:
        for endpoint in (ideal_display_points[index], tilted_display_points[index]):
            line = np.vstack((sphere_anchor_points[index], endpoint))
            axis.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color="#3d434a",
                alpha=0.34,
                linewidth=0.62,
            )
    incident_film_direction_lab = data.nominal.instrument.lab_from_sample.apply_vector(
        data.nominal.ki_sample_Ainv / np.linalg.norm(data.nominal.ki_sample_Ainv)
    )
    incident_film_direction_lab /= np.linalg.norm(incident_film_direction_lab)
    beam_start = sphere_center - 0.062 * incident_film_direction_lab
    axis.quiver(
        beam_start[0],
        beam_start[1],
        beam_start[2],
        *(0.057 * incident_film_direction_lab),
        color="#218c69",
        linewidth=2.7,
        arrow_length_ratio=0.14,
    )
    axis.text(
        *(beam_start + np.asarray((-0.020, 0.0, 0.025))),
        r"internal $\mathbf{k}_{i,\mathrm{film}}$",
        color="#167457",
    )

    axis.text(
        *(sphere_center + np.asarray((-0.050, -0.002, 0.056))),
        "intrinsic film\nEwald sphere",
        color="#4b1b52",
    )
    axis.text(
        *(ideal_display_center + np.asarray((-0.092, 0.0, 0.115))),
        "configured planar map",
        color="#147c9b",
    )
    axis.text(
        *(tilted_display_center + np.asarray((-0.040, 0.0, 0.115))),
        "tilted-detector map",
        color="#ad5c14",
    )
    axis.text2D(
        0.018,
        0.955,
        "Continuous intrinsic Ewald intensity → exit refraction → two detector-native densities",
        transform=axis.transAxes,
        fontsize=12.2,
        fontweight="bold",
    )
    axis.text2D(
        0.018,
        0.905,
        "The colored sphere patch contains only rays reaching the configured active panel; both detector fields are independently evaluated continuous textures. Positions and scales are diagrammatic; physical orientations are retained.",
        transform=axis.transAxes,
        fontsize=9.3,
        color="#41474d",
    )
    axis.legend(
        handles=(
            Line2D(
                (0,),
                (0,),
                color="#218c69",
                linewidth=2.4,
                label=r"internal $\mathbf{k}_{i,\mathrm{film}}$",
            ),
            Line2D(
                (0,),
                (0,),
                color="#3d434a",
                linewidth=0.9,
                label="diagrammatic exit-refraction guides",
            ),
        ),
        loc="lower left",
        frameon=False,
        fontsize=8.8,
    )
    sphere_colorbar_axis = figure.add_axes((0.805, 0.205, 0.016, 0.59))
    sphere_colorbar = figure.colorbar(
        sphere_shown,
        cax=sphere_colorbar_axis,
        extend=_log_scale_metadata(
            sphere_norm,
            (data.ewald.intensity_density_A2_per_sr,),
        )["colorbar_extend"],
    )
    sphere_colorbar.set_label(
        r"intrinsic Ewald density ($\AA^2\,sr^{-1}$; log, 8 decades)",
        labelpad=7,
    )
    detector_colorbar_axis = figure.add_axes((0.902, 0.205, 0.016, 0.59))
    detector_colorbar = figure.colorbar(
        detector_shown,
        cax=detector_colorbar_axis,
        extend=_log_scale_metadata(
            detector_norm,
            (
                data.ideal_detector.density_A2_per_px2,
                data.tilted_detector.density_A2_per_px2,
            ),
        )["colorbar_extend"],
    )
    detector_colorbar.set_label(
        r"detector density ($\AA^2\,px^{-2}$; log, 8 decades)",
        labelpad=7,
    )
    axis.set_xlim(-0.305, 0.315)
    axis.set_ylim(-0.085, 0.155)
    axis.set_zlim(-0.135, 0.135)
    axis.set_box_aspect((2.1, 1.0, 1.15))
    axis.view_init(elev=17.0, azim=-63.0)
    axis.set_proj_type("persp", focal_length=0.9)
    axis.set_axis_off()
    return figure


def _render_four_panel(data: PublicationData) -> object:
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(16.8, 13.0), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.12))
    reciprocal_axis = figure.add_subplot(grid[0, 0], projection="3d")
    ewald_axis = figure.add_subplot(grid[0, 1], projection="3d")
    ideal_axis = figure.add_subplot(grid[1, 0])
    tilted_axis = figure.add_subplot(grid[1, 1])
    caption_axis = figure.add_subplot(grid[2, :])
    caption_axis.set_axis_off()

    reciprocal_shown = _draw_reciprocal_panel(reciprocal_axis, data.reciprocal)
    ewald_shown = _draw_ewald_panel(
        ewald_axis,
        data.ewald,
        ki_sample_Ainv=data.nominal.ki_sample_Ainv,
    )
    detector_norm = _detector_norm(data.ideal_detector, data.tilted_detector)
    ideal_shown = _draw_detector_panel(
        ideal_axis,
        data.ideal_detector,
        instrument=data.inputs.instrument,
        norm=detector_norm,
        title="c  Intermediate planar mapping",
        subtitle="configured reference pose (default YAML tilt 0°); independently evaluated density",
    )
    _draw_detector_panel(
        tilted_axis,
        data.tilted_detector,
        instrument=data.tilted_instrument,
        norm=detector_norm,
        title="d  Mapping on a tilted detector",
        subtitle=(
            f"configured pose + intrinsic increment: column {data.settings.tilt_column_deg:+g}°, "
            f"row {data.settings.tilt_row_deg:+g}°"
        ),
    )
    figure.colorbar(
        reciprocal_shown,
        ax=reciprocal_axis,
        shrink=0.66,
        pad=0.055,
        label=r"latent density ($\AA^2\,rad^{-2}$; log)",
    )
    figure.colorbar(
        ewald_shown,
        ax=ewald_axis,
        shrink=0.66,
        pad=0.055,
        extend=_log_scale_metadata(
            _ewald_norm(data.ewald),
            (data.ewald.intensity_density_A2_per_sr,),
        )["colorbar_extend"],
        label=r"intrinsic Ewald density ($\AA^2\,sr^{-1}$; log, 8 decades)",
    )
    detector_colorbar = figure.colorbar(
        ideal_shown,
        ax=(ideal_axis, tilted_axis),
        shrink=0.82,
        pad=0.025,
        extend=_log_scale_metadata(
            detector_norm,
            (
                data.ideal_detector.density_A2_per_px2,
                data.tilted_detector.density_A2_per_px2,
            ),
        )["colorbar_extend"],
    )
    detector_colorbar.set_label(
        r"raw detector-coordinate density ($\AA^2\,px^{-2}$; log; 8-decade window)"
    )
    figure.suptitle(
        "Bi$_2$Se$_3$ reciprocal-to-detector intensity mapping at "
        rf"$\alpha_i={data.settings.incidence_deg:g}^\circ$ "
        r"(a--b: mean Cu K$\alpha$ reference; c--d: weighted K$\alpha$ doublet)"
    )
    caption_axis.text(
        0.5,
        0.62,
        (
            rf"Mosaic: Gaussian $\sigma={data.settings.gaussian_sigma_deg:g}^\circ$, "
            rf"Lorentzian HWHM $={data.settings.lorentzian_hwhm_deg:g}^\circ$, "
            rf"$\eta_L={data.settings.lorentzian_probability:g}$.  "
            "Panels c-d include canonical exit refraction, optical/attenuation factors, and the "
            "detector-coordinate Jacobian once; detector solid angle is not reapplied. The shared "
            "detector scale shows eight decades: its lower tail uses the under-range color, while "
            "the finite top 0.05% and any exact caustics use the over-range color."
        ),
        ha="center",
        va="center",
        wrap=True,
        fontsize=9.2,
    )
    caption_axis.text(
        0.5,
        0.12,
        (
            f"Panel a uses {data.mosaic_alpha_nodes_rad.size} scale-resolved display nodes. Panel b "
            f"directly samples the exact a.e. density at {data.ewald.intensity_density_A2_per_sr.size:,} "
            "detector-native centers on the configured active-panel patch of the internal-film "
            "Ewald sphere; the faint wireframe supplies whole-sphere context only. Regular nonzero "
            "m=0 support is included, while the collapsed direct Q=0 root is excluded by the strict "
            f"visible-patch gap |Q|>{data.ewald.detector_visible_m0_q_gap_Ainv:.3g} A^-1. Positive "
            "values below the eight-decade window use the under-range color, and exact positive "
            "caustics use the over-range color. Each detector map point-samples its continuous "
            "detector-coordinate "
            "density at its stated physical pose; the tilted map is not warped from panel c."
        ),
        ha="center",
        va="center",
        wrap=True,
        fontsize=8.5,
        color="#454b50",
    )
    return figure


def _render_ewald_standalone(data: PublicationData | EwaldOnlyData) -> object:
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(9.2, 8.0), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    shown = _draw_ewald_panel(
        axis,
        data.ewald,
        ki_sample_Ainv=data.nominal.ki_sample_Ainv,
        maximum_row_count=720,
        maximum_column_count=1440,
    )
    figure.colorbar(
        shown,
        ax=axis,
        shrink=0.70,
        pad=0.06,
        extend=_log_scale_metadata(
            _ewald_norm(data.ewald),
            (data.ewald.intensity_density_A2_per_sr,),
        )["colorbar_extend"],
        label=r"intrinsic Ewald density ($\AA^2\,sr^{-1}$; log, 8 decades)",
    )
    return figure


def _render_standalone_figures(data: PublicationData) -> Iterator[tuple[str, object]]:
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(9.2, 8.0), layout="constrained")
    axis = figure.add_subplot(111, projection="3d")
    shown = _draw_reciprocal_panel(axis, data.reciprocal)
    figure.colorbar(
        shown,
        ax=axis,
        shrink=0.70,
        pad=0.06,
        label=r"latent density ($\AA^2\,rad^{-2}$; log)",
    )
    yield "01-reciprocal-space", figure

    yield "02-painted-ewald-surface", _render_ewald_standalone(data)

    norm = _detector_norm(data.ideal_detector, data.tilted_detector)
    for stem, detector, instrument, title, subtitle in (
        (
            "03-untilted-planar-mapping",
            data.ideal_detector,
            data.inputs.instrument,
            "Intermediate planar mapping",
            "configured reference pose (default YAML tilt 0°); weighted K-alpha lines at mean ray geometry",
        ),
        (
            "04-tilted-detector-mapping",
            data.tilted_detector,
            data.tilted_instrument,
            "Tilted-detector mapping",
            f"configured pose + column {data.settings.tilt_column_deg:+g}°, row {data.settings.tilt_row_deg:+g}° increment",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8.6, 7.5), layout="constrained")
        shown = _draw_detector_panel(
            axis,
            detector,
            instrument=instrument,
            norm=norm,
            title=title,
            subtitle=subtitle,
        )
        figure.colorbar(
            shown,
            ax=axis,
            extend=_log_scale_metadata(
                norm,
                (
                    data.ideal_detector.density_A2_per_px2,
                    data.tilted_detector.density_A2_per_px2,
                ),
            )["colorbar_extend"],
            label=(
                r"raw detector-coordinate density "
                r"($\AA^2\,px^{-2}$; log; 8-decade display window)"
            ),
        )
        yield stem, figure


def build_publication_data(
    config_path: Path,
    settings: FigureSettings,
) -> PublicationData:
    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        build_source_averaged_detector,
        load_simulation_config,
    )

    timings: dict[str, float] = {}
    started = perf_counter()
    base = load_simulation_config(config_path, repository_root=ROOT)
    config = apply_figure_settings(base, settings)
    inputs = build_configured_simulation_inputs(config)
    alpha_nodes = select_scale_resolved_mosaic_alpha_nodes(
        inputs,
        count=settings.mosaic_alpha_count,
    )
    timings["configured_build"] = perf_counter() - started

    started = perf_counter()
    reciprocal, reciprocal_logical_count, reciprocal_evaluated_count = (
        sample_scale_resolved_reciprocal_space(
            inputs,
            alpha_nodes_rad=alpha_nodes,
            beta_count=settings.reciprocal_beta_count,
            u_background_count=settings.reciprocal_u_background_count,
        )
    )
    timings["reciprocal_sample"] = perf_counter() - started

    started = perf_counter()
    nominal = build_nominal_ewald_context(inputs)
    ewald = evaluate_detector_visible_ewald_patch(
        config,
        nominal,
        image_size=settings.ewald_image_size,
        tile_row_count=settings.ewald_tile_row_count,
        worker_count=settings.ewald_worker_count,
    )
    timings["ewald_pointwise_density"] = perf_counter() - started

    started = perf_counter()
    detector = build_source_averaged_detector(inputs)
    ideal_display = evaluate_detector_display(
        detector,
        instrument=inputs.instrument,
        image_size=settings.detector_image_size,
        tile_row_count=settings.detector_tile_row_count,
    )
    timings["untilted_detector_density"] = perf_counter() - started

    started = perf_counter()
    tilted_instrument = tilt_detector_about_reference(
        inputs.instrument,
        column_tilt_deg=settings.tilt_column_deg,
        row_tilt_deg=settings.tilt_row_deg,
    )
    tilted_detector = detector.rebind_geometry(
        incident=inputs.incident,
        instrument=tilted_instrument,
    )
    tilted_display = evaluate_detector_display(
        tilted_detector,
        instrument=tilted_instrument,
        image_size=settings.detector_image_size,
        tile_row_count=settings.detector_tile_row_count,
    )
    timings["tilted_detector_density"] = perf_counter() - started

    started = perf_counter()
    schematic_rays = _sample_schematic_rays(
        nominal,
        tilted_instrument=tilted_instrument,
        alpha_nodes_rad=alpha_nodes,
    )
    timings["schematic_rays"] = perf_counter() - started
    return PublicationData(
        settings=settings,
        inputs=inputs,
        nominal=nominal,
        reciprocal=reciprocal,
        ewald=ewald,
        mosaic_alpha_nodes_rad=alpha_nodes,
        reciprocal_logical_candidate_count=reciprocal_logical_count,
        reciprocal_candidate_evaluation_count=reciprocal_evaluated_count,
        ideal_detector=ideal_display,
        tilted_detector=tilted_display,
        tilted_instrument=tilted_instrument,
        schematic_rays=schematic_rays,
        timings_s=timings,
    )


def build_ewald_only_data(
    config_path: Path,
    settings: FigureSettings,
) -> EwaldOnlyData:
    """Build only the detector-visible Ewald field, without detector-map recomputation."""

    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        load_simulation_config,
    )

    started = perf_counter()
    base = load_simulation_config(config_path, repository_root=ROOT)
    config = apply_figure_settings(base, settings)
    inputs = build_configured_simulation_inputs(config)
    nominal = build_nominal_ewald_context(inputs)
    configured_build_s = perf_counter() - started
    started = perf_counter()
    ewald = evaluate_detector_visible_ewald_patch(
        config,
        nominal,
        image_size=settings.ewald_image_size,
        tile_row_count=settings.ewald_tile_row_count,
        worker_count=settings.ewald_worker_count,
    )
    return EwaldOnlyData(
        settings=settings,
        inputs=inputs,
        nominal=nominal,
        ewald=ewald,
        timings_s={
            "configured_build": configured_build_s,
            "ewald_pointwise_density": perf_counter() - started,
        },
    )


def build_schematic_only_data(
    config_path: Path,
    settings: FigureSettings,
) -> SchematicOnlyData:
    """Build the prior projection schematic without reciprocal or full detector maps."""

    from rasim_next.pipeline.configured_simulation import (
        build_configured_simulation_inputs,
        build_nominal_ewald_context,
        build_source_averaged_detector,
        load_simulation_config,
    )

    timings: dict[str, float] = {}
    started = perf_counter()
    base = load_simulation_config(config_path, repository_root=ROOT)
    config = apply_figure_settings(base, settings)
    inputs = build_configured_simulation_inputs(config)
    alpha_nodes = select_scale_resolved_mosaic_alpha_nodes(
        inputs,
        count=settings.mosaic_alpha_count,
    )
    nominal = build_nominal_ewald_context(inputs)
    timings["configured_build"] = perf_counter() - started

    started = perf_counter()
    ewald = evaluate_detector_visible_ewald_patch(
        config,
        nominal,
        image_size=settings.ewald_image_size,
        tile_row_count=settings.ewald_tile_row_count,
        worker_count=settings.ewald_worker_count,
    )
    timings["ewald_pointwise_density"] = perf_counter() - started

    detector_grid_size = settings.schematic_detector_cell_count
    detector_tile_rows = min(settings.detector_tile_row_count, detector_grid_size)
    started = perf_counter()
    detector = build_source_averaged_detector(inputs)
    ideal_display = evaluate_detector_display(
        detector,
        instrument=inputs.instrument,
        image_size=detector_grid_size,
        tile_row_count=detector_tile_rows,
    )
    timings["untilted_schematic_detector_density"] = perf_counter() - started

    started = perf_counter()
    tilted_instrument = tilt_detector_about_reference(
        inputs.instrument,
        column_tilt_deg=settings.tilt_column_deg,
        row_tilt_deg=settings.tilt_row_deg,
    )
    tilted_detector = detector.rebind_geometry(
        incident=inputs.incident,
        instrument=tilted_instrument,
    )
    tilted_display = evaluate_detector_display(
        tilted_detector,
        instrument=tilted_instrument,
        image_size=detector_grid_size,
        tile_row_count=detector_tile_rows,
    )
    timings["tilted_schematic_detector_density"] = perf_counter() - started

    started = perf_counter()
    schematic_rays = _sample_schematic_rays(
        nominal,
        tilted_instrument=tilted_instrument,
        alpha_nodes_rad=alpha_nodes,
    )
    timings["schematic_rays"] = perf_counter() - started
    return SchematicOnlyData(
        settings=settings,
        inputs=inputs,
        nominal=nominal,
        ewald=ewald,
        ideal_detector=ideal_display,
        tilted_detector=tilted_display,
        tilted_instrument=tilted_instrument,
        schematic_rays=schematic_rays,
        timings_s=timings,
    )


def _external_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or resolved.is_relative_to(ROOT):
        raise ValueError("generated publication artifacts must remain outside the repository")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def _staging_directory(output_directory: Path, case_stem: str) -> Iterator[Path]:
    case_id = hashlib.sha256(case_stem.encode("utf-8")).hexdigest()[:10]
    path = output_directory / f".rmp-{case_id}-{uuid4().hex[:10]}"
    path.mkdir()
    try:
        yield path
    except BaseException:
        backup = path / "prior-case-inventory"
        preserve_recovery = backup.is_dir() and any(backup.iterdir())
        if not preserve_recovery:
            shutil.rmtree(path)
        raise
    else:
        shutil.rmtree(path)


@contextmanager
def _case_output_lock(output_directory: Path, case_stem: str) -> Iterator[None]:
    case_id = hashlib.sha256(case_stem.encode("utf-8")).hexdigest()[:16]
    path = output_directory / f".rmp-{case_id}.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"publication case is already being installed: {case_stem}; if no renderer is active, "
            f"remove the stale lock {path}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()} case={case_stem}\n".encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _save_figure(
    figure: object,
    *,
    output_directory: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    from matplotlib import pyplot as plt

    outputs: list[Path] = []
    try:
        for suffix in formats:
            path = output_directory / f"{stem}.{suffix}"
            options: dict[str, object] = {"bbox_inches": "tight", "dpi": dpi}
            figure.savefig(path, **options)
            outputs.append(path)
    finally:
        plt.close(figure)
    return outputs


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _capture_source_provenance(config_path: Path) -> SourceProvenance:
    resolved_config = config_path.expanduser().resolve(strict=True)
    renderer_path = Path(__file__).resolve(strict=True)
    config_identity = (
        resolved_config.relative_to(ROOT).as_posix()
        if resolved_config.is_relative_to(ROOT)
        else str(resolved_config)
    )
    return SourceProvenance(
        config_path=resolved_config,
        config_identity=config_identity,
        config_sha256=_sha256(resolved_config),
        renderer_path=renderer_path,
        renderer_identity=renderer_path.relative_to(ROOT).as_posix(),
        renderer_sha256=_sha256(renderer_path),
    )


def _require_source_provenance_unchanged(provenance: SourceProvenance) -> None:
    for label, path, expected_hash in (
        ("configuration", provenance.config_path, provenance.config_sha256),
        ("renderer", provenance.renderer_path, provenance.renderer_sha256),
    ):
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"{label} changed during publication rendering")


def _case_stem(settings: FigureSettings, case_name: str | None) -> str:
    if case_name is not None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", case_name) is None:
            raise ValueError(
                "case_name must contain only letters, digits, '.', '_', or '-' and start alphanumeric"
            )
        return case_name
    angle = f"{settings.incidence_deg:g}".replace("-", "m").replace(".", "p")
    return f"bi2se3-{angle}deg"


def _validated_formats(formats: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(formats)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= {"png", "pdf"}:
        raise ValueError(
            "formats must be a nonempty unique sequence containing only 'png' or 'pdf'"
        )
    return selected


def _figure_stems(case_stem: str, *, include_standalone: bool) -> tuple[str, ...]:
    stems = [f"{case_stem}-publication-panels", f"{case_stem}-projection-schematic"]
    if include_standalone:
        stems.extend(
            f"{case_stem}-{suffix}"
            for suffix in (
                "01-reciprocal-space",
                "02-painted-ewald-surface",
                "03-untilted-planar-mapping",
                "04-tilted-detector-mapping",
            )
        )
    return tuple(stems)


def _figure_paths(
    output_directory: Path,
    *,
    case_stem: str,
    formats: Sequence[str],
    include_standalone: bool,
) -> tuple[Path, ...]:
    return tuple(
        output_directory / f"{stem}.{suffix}"
        for stem in _figure_stems(case_stem, include_standalone=include_standalone)
        for suffix in formats
    )


def _known_case_artifact_paths(output_directory: Path, case_stem: str) -> tuple[Path, ...]:
    figure_paths = _figure_paths(
        output_directory,
        case_stem=case_stem,
        formats=("png", "pdf"),
        include_standalone=True,
    )
    manifest = output_directory / f"{case_stem}-publication-manifest.json"
    return (manifest, *figure_paths)


def _ewald_only_artifact_paths(
    output_directory: Path,
    *,
    case_stem: str,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    figures = tuple(
        output_directory / f"{case_stem}-visible-m0-ewald-surface.{suffix}" for suffix in formats
    )
    return (*figures, output_directory / f"{case_stem}-ewald-only-manifest.json")


def _schematic_only_artifact_paths(
    output_directory: Path,
    *,
    case_stem: str,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    figures = tuple(
        output_directory / f"{case_stem}-projection-schematic-only.{suffix}" for suffix in formats
    )
    return (*figures, output_directory / f"{case_stem}-schematic-only-manifest.json")


def _require_available_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    unsafe = [path for path in existing if not path.is_file() and not path.is_symlink()]
    if unsafe:
        names = ", ".join(path.name for path in unsafe)
        raise ValueError(f"artifact paths must not be directories or special files ({names})")
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"outputs already exist ({names}); pass --overwrite or --case-name")


def _install_staged_artifacts(
    staged_paths: Sequence[Path],
    *,
    desired_paths: Sequence[Path],
    known_case_paths: Sequence[Path],
    overwrite: bool,
    backup_directory: Path,
) -> tuple[Path, ...]:
    """Install a complete case inventory and restore the prior inventory on failure."""

    if not staged_paths or len({path.name for path in staged_paths}) != len(staged_paths):
        raise ValueError("staged artifact names must be nonempty and unique")
    if tuple(path.name for path in staged_paths) != tuple(path.name for path in desired_paths):
        raise ValueError("staged artifacts must exactly match the desired case inventory")
    if not staged_paths[-1].name.endswith("-manifest.json"):
        raise ValueError("the manifest must be installed last")
    if not known_case_paths[0].name.endswith("-manifest.json"):
        raise ValueError("the prior manifest must be backed up first")
    if any(not path.is_file() for path in staged_paths):
        raise FileNotFoundError("every staged publication artifact must exist before installation")
    _require_available_outputs(known_case_paths, overwrite=overwrite)
    backup_directory.mkdir(parents=False, exist_ok=False)
    final_paths = tuple(desired_paths)
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        if overwrite:
            for old_path in known_case_paths:
                if old_path.exists() or old_path.is_symlink():
                    backup_path = backup_directory / old_path.name
                    backups.append((backup_path, old_path))
                    old_path.replace(backup_path)
        for staged_path, final_path in zip(staged_paths, final_paths, strict=True):
            installed.append(final_path)
            staged_path.replace(final_path)
    except BaseException as install_error:
        rollback_errors: list[OSError] = []
        for final_path in reversed(installed):
            try:
                final_path.unlink(missing_ok=True)
            except OSError as error:
                rollback_errors.append(error)
        for backup_path, old_path in reversed(backups):
            if not (backup_path.exists() or backup_path.is_symlink()):
                continue
            try:
                backup_path.replace(old_path)
            except OSError as error:
                rollback_errors.append(error)
        if rollback_errors:
            details = "; ".join(str(error) for error in rollback_errors)
            raise RuntimeError(
                f"publication install failed and rollback was incomplete; "
                f"recovery files remain in {backup_directory}: {details}"
            ) from install_error
        raise
    return final_paths


def _render_staged_figures(
    data: PublicationData,
    *,
    output_directory: Path,
    formats: Sequence[str],
    include_standalone: bool,
    case_stem: str,
) -> tuple[Path, ...]:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    formats = _validated_formats(formats)
    expected = _figure_paths(
        output_directory,
        case_stem=case_stem,
        formats=formats,
        include_standalone=include_standalone,
    )
    _require_available_outputs(expected, overwrite=False)
    outputs: list[Path] = []
    outputs.extend(
        _save_figure(
            _render_four_panel(data),
            output_directory=output_directory,
            stem=f"{case_stem}-publication-panels",
            formats=formats,
            dpi=data.settings.dpi,
        )
    )
    outputs.extend(
        _save_figure(
            _render_projection_schematic(data),
            output_directory=output_directory,
            stem=f"{case_stem}-projection-schematic",
            formats=formats,
            dpi=data.settings.dpi,
        )
    )
    if include_standalone:
        for stem, figure in _render_standalone_figures(data):
            outputs.extend(
                _save_figure(
                    figure,
                    output_directory=output_directory,
                    stem=f"{case_stem}-{stem}",
                    formats=formats,
                    dpi=data.settings.dpi,
                )
            )
    return tuple(outputs)


def _render_staged_ewald_only(
    data: EwaldOnlyData,
    *,
    output_directory: Path,
    formats: Sequence[str],
    case_stem: str,
) -> tuple[Path, ...]:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    return tuple(
        _save_figure(
            _render_ewald_standalone(data),
            output_directory=output_directory,
            stem=f"{case_stem}-visible-m0-ewald-surface",
            formats=_validated_formats(formats),
            dpi=data.settings.dpi,
        )
    )


def _render_staged_schematic_only(
    data: SchematicOnlyData,
    *,
    output_directory: Path,
    formats: Sequence[str],
    case_stem: str,
) -> tuple[Path, ...]:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    return tuple(
        _save_figure(
            _render_projection_schematic(data),
            output_directory=output_directory,
            stem=f"{case_stem}-projection-schematic-only",
            formats=_validated_formats(formats),
            dpi=data.settings.dpi,
        )
    )


def _source_policy_records(
    inputs: ConfiguredSimulationInputs,
    nominal: NominalEwaldContext,
) -> dict[str, dict[str, object]]:
    reference = nominal.incident.states
    physical = inputs.samples
    return {
        "reciprocal_ewald_reference": {
            "policy": "nominal_mean_geometry_reference.v1",
            "source_sampling_model_id": reference.source_sampling_model_id,
            "source_revision": reference.source_revision,
            "state_count": int(reference.incident_sample_id.size),
            "wavelength_A": reference.wavelength_A.tolist(),
            "source_weight": reference.source_weight.tolist(),
        },
        "detector_intensity": {
            "policy": "weighted_discrete_lines_at_mean_ray_geometry.v1",
            "phase_space_sampling": "one_mean_geometry_row_per_line; configured spatial and divergence widths are not sampled",
            "source_sampling_model_id": physical.source_sampling_model_id,
            "source_revision": physical.source_revision,
            "state_count": int(physical.incident_sample_id.size),
            "wavelength_A": physical.wavelength_A.tolist(),
            "source_weight": physical.source_weight.tolist(),
        },
    }


def _publication_manifest_payload(
    data: PublicationData,
    *,
    provenance: SourceProvenance,
    outputs: Sequence[Path],
) -> dict[str, object]:
    return {
        "schema_version": "rasim-bi2se3-publication-mapping-v6",
        "config_path": provenance.config_identity,
        "config_sha256": provenance.config_sha256,
        "renderer_path": provenance.renderer_identity,
        "renderer_sha256": provenance.renderer_sha256,
        "physics_revision": data.inputs.config.physics_revision,
        "source_policies": _source_policy_records(data.inputs, data.nominal),
        "settings": asdict(data.settings),
        "tilt_policy": "configured_pose_plus_intrinsic_column_then_current_row_increment_about_reference.v1",
        "detector_poses": {
            "configured_rotation_lab_from_detector": data.inputs.instrument.lab_from_detector.rotation.tolist(),
            "tilted_rotation_lab_from_detector": data.tilted_instrument.lab_from_detector.rotation.tolist(),
            "shared_reference_translation_lab_m": data.inputs.instrument.lab_from_detector.translation_m.tolist(),
        },
        "measures": {
            "reciprocal": data.reciprocal.measure_id,
            "ewald": data.ewald.measure_id,
            "ewald_selection": data.ewald.selection_id,
            "untilted_detector": data.ideal_detector.measure_id,
            "tilted_detector": data.tilted_detector.measure_id,
            "schematic_sphere": data.ewald.measure_id,
            "schematic_planes": data.ideal_detector.measure_id,
        },
        "display_policies": {
            "mosaic_alpha": "compiled_full_support_quadrature_index_thinning.v1",
            "reciprocal": "positive_Qz_integer_L_plus_0p5_over_N_shoulders_top_8_decades_per_rod_budget.v1",
            "ewald": "configured_active_detector_native_centers_mapped_to_internal_film_kf_patch.v1",
            "schematic": "lab_oriented_internal_kf_direction_glyph_and_independent_detector_textures_with_diagrammatic_exit_refraction_guides.v1",
            "detector": "shared_log_8_decades_finite_q0.9995_under_tail_over_top_and_caustic.v1",
        },
        "mosaic_alpha_nodes_deg": np.rad2deg(data.mosaic_alpha_nodes_rad).tolist(),
        "counts": {
            "physical_rods": len(data.nominal.rods),
            "reciprocal_points": int(data.reciprocal.q_sample_Ainv.shape[0]),
            "reciprocal_logical_candidates": data.reciprocal_logical_candidate_count,
            "reciprocal_candidate_evaluations": data.reciprocal_candidate_evaluation_count,
            "ewald_detector_native_patch_grid_rc": list(
                data.ewald.intensity_density_A2_per_sr.shape
            ),
            "ewald_visible_samples": int(np.count_nonzero(data.ewald.detector_visible)),
            "ewald_unpainted_bbox_samples": int(np.count_nonzero(~data.ewald.detector_visible)),
            "ewald_positive_samples": int(
                np.count_nonzero(data.ewald.intensity_density_A2_per_sr > 0.0)
            ),
            "ewald_m0_positive_samples": int(
                np.count_nonzero(data.ewald.m0_intensity_density_A2_per_sr > 0.0)
            ),
            "ewald_zero_samples": int(
                np.count_nonzero(data.ewald.intensity_density_A2_per_sr == 0.0)
            ),
            "ewald_finite_samples": int(
                np.count_nonzero(np.isfinite(data.ewald.intensity_density_A2_per_sr))
            ),
            "ewald_caustic_samples": int(np.count_nonzero(data.ewald.caustic)),
            "ewald_regular_inverse_preimages": int(np.sum(data.ewald.inverse_branch_count)),
            "schematic_common_rays": int(data.schematic_rays.direction_lab.shape[0]),
            "detector_grid_shape_rc": list(data.ideal_detector.density_A2_per_px2.shape),
            "untilted_caustic_samples": int(np.count_nonzero(data.ideal_detector.caustic)),
            "tilted_caustic_samples": int(np.count_nonzero(data.tilted_detector.caustic)),
        },
        "ewald_point_sampling": {
            "q_coordinate_frame": "sample",
            "solid_angle": "internal_film_outgoing_kf_direction_sr",
            "center_parameterization": "uniform_configured_detector_native_column_row_centers",
            "column_edge_range_px": [
                float(data.ewald.column_edges_px[0]),
                float(data.ewald.column_edges_px[-1]),
            ],
            "row_edge_range_px": [
                float(data.ewald.row_edges_px[0]),
                float(data.ewald.row_edges_px[-1]),
            ],
            "sphere_policy": "canonical_top_exit_and_configured_active_panel_visibility.v1",
            "m0_policy": "regular_nonzero_support_with_positive_Q_gap;collapsed_direct_Q0_excluded.v1",
            "detector_visible_m0_q_gap_Ainv": (data.ewald.detector_visible_m0_q_gap_Ainv),
            "caustic_policy": "positive_numerator_is_infinite_and_marked;zero_numerator_uses_zero_a.e._representative",
            "maximum_ewald_residual_Ainv": data.ewald.maximum_ewald_residual_Ainv,
        },
        "execution": {
            "ewald": {
                "backend": data.ewald.execution_backend,
                "requested_worker_count": data.ewald.requested_worker_count,
                "used_worker_count": data.ewald.worker_count,
                "start_method": data.ewald.multiprocessing_start_method,
                "child_blas_thread_limit": 1,
            },
        },
        "frames": {
            "reciprocal_and_ewald_Q": "sample_frame_Ainv",
            "ewald_direction": "internal_film_outgoing_unit_direction_in_sample_frame",
            "detector_surfaces": "physical_lab_frame_m_at_each_declared_pose",
            "schematic": "exploded_diagram_frame_without_shared_length_scale",
            "schematic_connectors": "diagrammatic_correspondence_guides_after_exit_refraction",
        },
        "color_scales": {
            "ewald": _log_scale_metadata(
                _ewald_norm(data.ewald),
                (data.ewald.intensity_density_A2_per_sr,),
            ),
            "shared_detector": _log_scale_metadata(
                _detector_norm(data.ideal_detector, data.tilted_detector),
                (
                    data.ideal_detector.density_A2_per_px2,
                    data.tilted_detector.density_A2_per_px2,
                ),
            ),
        },
        "timings_s": data.timings_s,
        "outputs": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }


def _ewald_only_manifest_payload(
    data: EwaldOnlyData,
    *,
    provenance: SourceProvenance,
    outputs: Sequence[Path],
) -> dict[str, object]:
    return {
        "schema_version": "rasim-bi2se3-detector-visible-ewald-v2",
        "config_path": provenance.config_identity,
        "config_sha256": provenance.config_sha256,
        "renderer_path": provenance.renderer_identity,
        "renderer_sha256": provenance.renderer_sha256,
        "physics_revision": data.inputs.config.physics_revision,
        "source_policy": _source_policy_records(data.inputs, data.nominal)[
            "reciprocal_ewald_reference"
        ],
        "settings": asdict(data.settings),
        "measure": data.ewald.measure_id,
        "selection": data.ewald.selection_id,
        "sphere_policy": "canonical_top_exit_and_configured_active_panel_visibility.v1",
        "m0_policy": "regular_nonzero_support_with_positive_Q_gap;collapsed_direct_Q0_excluded.v1",
        "detector_visible_m0_q_gap_Ainv": data.ewald.detector_visible_m0_q_gap_Ainv,
        "grid": {
            "shape_rc": list(data.ewald.intensity_density_A2_per_sr.shape),
            "visible_samples": int(np.count_nonzero(data.ewald.detector_visible)),
            "positive_samples": int(np.count_nonzero(data.ewald.intensity_density_A2_per_sr > 0.0)),
            "m0_positive_samples": int(
                np.count_nonzero(data.ewald.m0_intensity_density_A2_per_sr > 0.0)
            ),
            "maximum_ewald_residual_Ainv": data.ewald.maximum_ewald_residual_Ainv,
        },
        "execution": {
            "backend": data.ewald.execution_backend,
            "requested_worker_count": data.ewald.requested_worker_count,
            "used_worker_count": data.ewald.worker_count,
            "start_method": data.ewald.multiprocessing_start_method,
            "child_blas_thread_limit": 1,
        },
        "color_scale": _log_scale_metadata(
            _ewald_norm(data.ewald),
            (data.ewald.intensity_density_A2_per_sr,),
        ),
        "timings_s": data.timings_s,
        "outputs": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }


def _schematic_only_manifest_payload(
    data: SchematicOnlyData,
    *,
    provenance: SourceProvenance,
    outputs: Sequence[Path],
) -> dict[str, object]:
    detector_norm = _detector_norm(data.ideal_detector, data.tilted_detector)
    return {
        "schema_version": "rasim-bi2se3-projection-schematic-v2",
        "config_path": provenance.config_identity,
        "config_sha256": provenance.config_sha256,
        "renderer_path": provenance.renderer_identity,
        "renderer_sha256": provenance.renderer_sha256,
        "physics_revision": data.inputs.config.physics_revision,
        "source_policies": _source_policy_records(data.inputs, data.nominal),
        "settings": asdict(data.settings),
        "build_scope": {
            "reciprocal_space": "not_evaluated",
            "ewald_patch": "evaluated_at_requested_schematic_resolution",
            "detector_maps": "evaluated_directly_at_schematic_detector_cell_count",
        },
        "tilt_policy": "configured_pose_plus_intrinsic_column_then_current_row_increment_about_reference.v1",
        "detector_poses": {
            "configured_rotation_lab_from_detector": data.inputs.instrument.lab_from_detector.rotation.tolist(),
            "tilted_rotation_lab_from_detector": data.tilted_instrument.lab_from_detector.rotation.tolist(),
            "shared_reference_translation_lab_m": data.inputs.instrument.lab_from_detector.translation_m.tolist(),
        },
        "measures": {
            "ewald": data.ewald.measure_id,
            "ewald_selection": data.ewald.selection_id,
            "configured_planar_detector": data.ideal_detector.measure_id,
            "tilted_detector": data.tilted_detector.measure_id,
        },
        "coordinates": {
            "ewald_frame": "internal_film_sample_frame_outgoing_kf_direction",
            "ewald_measure_denominator": "internal_film_outgoing_solid_angle_sr",
            "ewald_parameterization": "configured_active_detector_native_cell_centers",
            "detector_coordinate_order": "(column_px, row_px)",
            "array_index_order": "[row, column]",
            "ewald_column_edge_range_px": [
                float(data.ewald.column_edges_px[0]),
                float(data.ewald.column_edges_px[-1]),
            ],
            "ewald_row_edge_range_px": [
                float(data.ewald.row_edges_px[0]),
                float(data.ewald.row_edges_px[-1]),
            ],
        },
        "ewald_proof": {
            "detector_visible_m0_q_gap_Ainv": data.ewald.detector_visible_m0_q_gap_Ainv,
            "maximum_ewald_residual_Ainv": data.ewald.maximum_ewald_residual_Ainv,
        },
        "display_policies": {
            "sphere": "canonical_top_exit_and_configured_active_panel_visibility.v1",
            "m0": "regular_nonzero_support_with_positive_Q_gap;collapsed_direct_Q0_excluded.v1",
            "planes": "independently_evaluated_detector_native_continuous_textures.v1",
            "guide_rays": "representative_common_exit_valid_nonzero_m_correspondence_only.v1",
            "layout": "prior_exploded_projection_schematic_geometry_and_camera.v1",
        },
        "counts": {
            "ewald_patch_grid_shape_rc": list(data.ewald.intensity_density_A2_per_sr.shape),
            "ewald_visible_samples": int(np.count_nonzero(data.ewald.detector_visible)),
            "ewald_m0_positive_samples": int(
                np.count_nonzero(data.ewald.m0_intensity_density_A2_per_sr > 0.0)
            ),
            "ewald_positive_samples": int(
                np.count_nonzero(data.ewald.intensity_density_A2_per_sr > 0.0)
            ),
            "ewald_caustic_samples": int(np.count_nonzero(data.ewald.caustic)),
            "ewald_regular_inverse_preimages": int(np.sum(data.ewald.inverse_branch_count)),
            "configured_detector_grid_shape_rc": list(data.ideal_detector.density_A2_per_px2.shape),
            "tilted_detector_grid_shape_rc": list(data.tilted_detector.density_A2_per_px2.shape),
            "configured_detector_valid_samples": int(np.count_nonzero(data.ideal_detector.valid)),
            "tilted_detector_valid_samples": int(np.count_nonzero(data.tilted_detector.valid)),
            "configured_detector_caustic_samples": int(
                np.count_nonzero(data.ideal_detector.caustic)
            ),
            "tilted_detector_caustic_samples": int(np.count_nonzero(data.tilted_detector.caustic)),
            "schematic_common_guide_rays": int(data.schematic_rays.direction_lab.shape[0]),
        },
        "execution": {
            "ewald": {
                "backend": data.ewald.execution_backend,
                "requested_worker_count": data.ewald.requested_worker_count,
                "used_worker_count": data.ewald.worker_count,
                "start_method": data.ewald.multiprocessing_start_method,
                "child_blas_thread_limit": 1,
            },
            "detectors": {
                "configured_backend": data.ideal_detector.execution_backend,
                "tilted_backend": data.tilted_detector.execution_backend,
                "grid_policy": "direct_schematic_resolution_evaluation.v1",
            },
        },
        "color_scales": {
            "ewald": _log_scale_metadata(
                _ewald_norm(data.ewald),
                (data.ewald.intensity_density_A2_per_sr,),
            ),
            "detectors": _log_scale_metadata(
                detector_norm,
                (
                    data.ideal_detector.density_A2_per_px2,
                    data.tilted_detector.density_A2_per_px2,
                ),
            ),
        },
        "timings_s": data.timings_s,
        "outputs": [
            {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }


def _write_manifest(
    data: PublicationData,
    *,
    provenance: SourceProvenance,
    output_directory: Path,
    outputs: Sequence[Path],
    case_stem: str,
) -> Path:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    payload = _publication_manifest_payload(
        data,
        provenance=provenance,
        outputs=outputs,
    )
    path = output_directory / f"{case_stem}-publication-manifest.json"
    _require_available_outputs((path,), overwrite=False)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_ewald_only_manifest(
    data: EwaldOnlyData,
    *,
    provenance: SourceProvenance,
    output_directory: Path,
    outputs: Sequence[Path],
    case_stem: str,
) -> Path:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    payload = _ewald_only_manifest_payload(
        data,
        provenance=provenance,
        outputs=outputs,
    )
    path = output_directory / f"{case_stem}-ewald-only-manifest.json"
    _require_available_outputs((path,), overwrite=False)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_schematic_only_manifest(
    data: SchematicOnlyData,
    *,
    provenance: SourceProvenance,
    output_directory: Path,
    outputs: Sequence[Path],
    case_stem: str,
) -> Path:
    output_directory = _external_output_directory(output_directory)
    case_stem = _case_stem(data.settings, case_stem)
    payload = _schematic_only_manifest_payload(
        data,
        provenance=provenance,
        outputs=outputs,
    )
    path = output_directory / f"{case_stem}-schematic-only-manifest.json"
    _require_available_outputs((path,), overwrite=False)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--case-name", help="safe filename stem for a distinct parameter case")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing named case")
    parser.add_argument("--incidence-deg", type=float, default=10.0)
    parser.add_argument("--gaussian-sigma-deg", type=float, default=2.0)
    parser.add_argument("--lorentzian-hwhm-deg", type=float, default=0.2)
    parser.add_argument("--lorentzian-probability", type=float, default=0.1)
    parser.add_argument("--tilt-column-deg", type=float, default=0.0)
    parser.add_argument("--tilt-row-deg", type=float, default=20.0)
    parser.add_argument("--mosaic-alpha-count", type=int, default=32)
    parser.add_argument("--reciprocal-beta-count", type=int, default=48)
    parser.add_argument("--reciprocal-u-background-count", type=int, default=48)
    parser.add_argument("--ewald-image-size", type=int, default=1440)
    parser.add_argument("--ewald-tile-row-count", type=int, default=32)
    parser.add_argument(
        "--ewald-worker-count",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="spawned CPU processes used for the detector-visible Ewald patch",
    )
    parser.add_argument("--detector-image-size", type=int, default=1440)
    parser.add_argument("--detector-tile-row-count", type=int, default=32)
    parser.add_argument("--schematic-detector-cell-count", type=int, default=480)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--skip-standalone", action="store_true")
    partial_mode = parser.add_mutually_exclusive_group()
    partial_mode.add_argument(
        "--only-ewald",
        action="store_true",
        help="render only the visible m=0-inclusive Ewald surface; skip reciprocal and detector maps",
    )
    partial_mode.add_argument(
        "--only-schematic",
        action="store_true",
        help="render only the prior projection schematic at direct reduced detector resolution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = FigureSettings(
        incidence_deg=arguments.incidence_deg,
        gaussian_sigma_deg=arguments.gaussian_sigma_deg,
        lorentzian_hwhm_deg=arguments.lorentzian_hwhm_deg,
        lorentzian_probability=arguments.lorentzian_probability,
        tilt_column_deg=arguments.tilt_column_deg,
        tilt_row_deg=arguments.tilt_row_deg,
        mosaic_alpha_count=arguments.mosaic_alpha_count,
        reciprocal_beta_count=arguments.reciprocal_beta_count,
        reciprocal_u_background_count=arguments.reciprocal_u_background_count,
        ewald_image_size=arguments.ewald_image_size,
        ewald_tile_row_count=arguments.ewald_tile_row_count,
        ewald_worker_count=arguments.ewald_worker_count,
        detector_image_size=arguments.detector_image_size,
        detector_tile_row_count=arguments.detector_tile_row_count,
        schematic_detector_cell_count=arguments.schematic_detector_cell_count,
        dpi=arguments.dpi,
    )
    output_directory = _external_output_directory(arguments.output_directory)
    case_stem = _case_stem(settings, arguments.case_name)
    formats = _validated_formats(tuple(arguments.formats))
    if arguments.only_ewald:
        desired_paths = _ewald_only_artifact_paths(
            output_directory,
            case_stem=case_stem,
            formats=formats,
        )
        all_known = _ewald_only_artifact_paths(
            output_directory,
            case_stem=case_stem,
            formats=("png", "pdf"),
        )
        known_case_paths = (all_known[-1], *all_known[:-1])
        _require_available_outputs(known_case_paths, overwrite=arguments.overwrite)
        provenance = _capture_source_provenance(arguments.config)
        _configure_matplotlib()
        started = perf_counter()
        data = build_ewald_only_data(provenance.config_path, settings)
        with _staging_directory(output_directory, case_stem) as staging_directory:
            staged_outputs = _render_staged_ewald_only(
                data,
                output_directory=staging_directory,
                formats=formats,
                case_stem=case_stem,
            )
            _require_source_provenance_unchanged(provenance)
            staged_manifest = _write_ewald_only_manifest(
                data,
                provenance=provenance,
                output_directory=staging_directory,
                outputs=staged_outputs,
                case_stem=case_stem,
            )
            _require_source_provenance_unchanged(provenance)
            with _case_output_lock(output_directory, case_stem):
                installed = _install_staged_artifacts(
                    (*staged_outputs, staged_manifest),
                    desired_paths=desired_paths,
                    known_case_paths=known_case_paths,
                    overwrite=arguments.overwrite,
                    backup_directory=staging_directory / "prior-case-inventory",
                )
        print(
            json.dumps(
                {
                    "output_directory": str(output_directory),
                    "figures": [str(path) for path in installed[:-1]],
                    "manifest": str(installed[-1]),
                    "wall_time_s": perf_counter() - started,
                },
                indent=2,
            )
        )
        return 0
    if arguments.only_schematic:
        desired_paths = _schematic_only_artifact_paths(
            output_directory,
            case_stem=case_stem,
            formats=formats,
        )
        all_known = _schematic_only_artifact_paths(
            output_directory,
            case_stem=case_stem,
            formats=("png", "pdf"),
        )
        known_case_paths = (all_known[-1], *all_known[:-1])
        _require_available_outputs(known_case_paths, overwrite=arguments.overwrite)
        provenance = _capture_source_provenance(arguments.config)
        _configure_matplotlib()
        started = perf_counter()
        data = build_schematic_only_data(provenance.config_path, settings)
        with _staging_directory(output_directory, case_stem) as staging_directory:
            staged_outputs = _render_staged_schematic_only(
                data,
                output_directory=staging_directory,
                formats=formats,
                case_stem=case_stem,
            )
            _require_source_provenance_unchanged(provenance)
            staged_manifest = _write_schematic_only_manifest(
                data,
                provenance=provenance,
                output_directory=staging_directory,
                outputs=staged_outputs,
                case_stem=case_stem,
            )
            _require_source_provenance_unchanged(provenance)
            with _case_output_lock(output_directory, case_stem):
                installed = _install_staged_artifacts(
                    (*staged_outputs, staged_manifest),
                    desired_paths=desired_paths,
                    known_case_paths=known_case_paths,
                    overwrite=arguments.overwrite,
                    backup_directory=staging_directory / "prior-case-inventory",
                )
        print(
            json.dumps(
                {
                    "output_directory": str(output_directory),
                    "figures": [str(path) for path in installed[:-1]],
                    "manifest": str(installed[-1]),
                    "wall_time_s": perf_counter() - started,
                },
                indent=2,
            )
        )
        return 0
    include_standalone = not arguments.skip_standalone
    known_case_paths = _known_case_artifact_paths(output_directory, case_stem)
    desired_paths = (
        *_figure_paths(
            output_directory,
            case_stem=case_stem,
            formats=formats,
            include_standalone=include_standalone,
        ),
        output_directory / f"{case_stem}-publication-manifest.json",
    )
    _require_available_outputs(known_case_paths, overwrite=arguments.overwrite)
    provenance = _capture_source_provenance(arguments.config)
    _configure_matplotlib()
    started = perf_counter()
    data = build_publication_data(provenance.config_path, settings)
    with _staging_directory(output_directory, case_stem) as staging_directory:
        staged_outputs = _render_staged_figures(
            data,
            output_directory=staging_directory,
            formats=formats,
            include_standalone=include_standalone,
            case_stem=case_stem,
        )
        _require_source_provenance_unchanged(provenance)
        staged_manifest = _write_manifest(
            data,
            provenance=provenance,
            output_directory=staging_directory,
            outputs=staged_outputs,
            case_stem=case_stem,
        )
        _require_source_provenance_unchanged(provenance)
        with _case_output_lock(output_directory, case_stem):
            installed = _install_staged_artifacts(
                (*staged_outputs, staged_manifest),
                desired_paths=desired_paths,
                known_case_paths=known_case_paths,
                overwrite=arguments.overwrite,
                backup_directory=staging_directory / "prior-case-inventory",
            )
    outputs = installed[:-1]
    manifest = installed[-1]
    print(
        json.dumps(
            {
                "output_directory": str(output_directory),
                "figures": [str(path) for path in outputs],
                "manifest": str(manifest),
                "wall_time_s": perf_counter() - started,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
