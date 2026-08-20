"""Render the final Bi2Se3 reciprocal-cylinder and continuous-mosaic figures.

The zero-mosaic figure colors the first three physical reciprocal shells by the incoherent
family sum of the configured finite-stack structure strength. The continuous-mosaic figure
pushes the normalized folded-alpha/full-beta mosaic measure and the same per-rod strength into
axisymmetric reciprocal-space finite volumes, then applies the declared display smoothing.
Generated figures and optional caches must live outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from painted_ewald.mosaic import build_mosaic_space  # noqa: E402
from painted_ewald.rotations import axis_angle_rotation_batch, mosaic_axes  # noqa: E402
from painted_ewald.types import MosaicParameters, Rod  # noqa: E402
from rasim_next.pipeline.bragg_space import finite_stack_integer_l_display_nodes  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    ConfiguredSimulationInputs,
    build_configured_simulation_inputs,
    load_simulation_config,
)

FloatArray = NDArray[np.float64]
DEFAULT_CONFIG = ROOT / "configs" / "bi2se3_simulation.yaml"
DEFAULT_OUTPUT_DIRECTORY = Path.home() / ".rasim-next" / "bi2se3-reciprocal-intensity"
CACHE_SCHEMA = "rasim-bi2se3-axisymmetric-reciprocal-display-cache-v3"
VOLUME_ALGORITHM = "folded-alpha-full-beta-dense-axial-finite-volume-gaussian-display-v3"
PHYSICAL_FAMILIES = (0, 1, 3)
DISPLAY_SMOOTHING_SIGMA_BINS = (0.85, 0.75)
PROFILE_BACKGROUND_COUNT = 2000
AXIAL_TRANSFORM_CHUNK_SIZE = 256
AXIAL_QUADRATURE_POINT_COUNT = 32000
OUTPUT_STEMS = {
    "zero-mosaic": "bi2se3-first-three-sf-cylinders",
    "continuous-mosaic": "bi2se3-continuous-mosaic-volume",
}
CHILD_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class ReciprocalFigureSettings:
    """Numerical and display controls for the two final reciprocal-space figures."""

    incidence_deg: float = 10.0
    gaussian_sigma_deg: float = 2.0
    lorentzian_hwhm_deg: float = 0.2
    lorentzian_probability: float = 0.1
    radial_max_Ainv: float = 3.85
    axial_max_Ainv: float = 8.25
    radial_bin_count: int = 280
    axial_bin_count: int = 380
    alpha_panel_count: int = 20
    alpha_gauss_order: int = 12
    worker_count: int = min(8, os.cpu_count() or 1)
    surface_angle_count: int = 181
    dpi: int = 320

    def __post_init__(self) -> None:
        finite_values = {
            "incidence_deg": self.incidence_deg,
            "gaussian_sigma_deg": self.gaussian_sigma_deg,
            "lorentzian_hwhm_deg": self.lorentzian_hwhm_deg,
            "lorentzian_probability": self.lorentzian_probability,
            "radial_max_Ainv": self.radial_max_Ainv,
            "axial_max_Ainv": self.axial_max_Ainv,
        }
        for name, raw_value in finite_values.items():
            if isinstance(raw_value, (bool, np.bool_)) or not math.isfinite(float(raw_value)):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(raw_value))
        if self.gaussian_sigma_deg <= 0.0 or self.lorentzian_hwhm_deg <= 0.0:
            raise ValueError("both active mosaic widths must be positive")
        if not 0.0 <= self.lorentzian_probability <= 1.0:
            raise ValueError("lorentzian_probability must lie in [0, 1]")
        if self.radial_max_Ainv <= 0.0 or self.axial_max_Ainv <= 0.0:
            raise ValueError("reciprocal-space limits must be positive")
        integer_values = {
            "radial_bin_count": self.radial_bin_count,
            "axial_bin_count": self.axial_bin_count,
            "alpha_panel_count": self.alpha_panel_count,
            "alpha_gauss_order": self.alpha_gauss_order,
            "worker_count": self.worker_count,
            "surface_angle_count": self.surface_angle_count,
            "dpi": self.dpi,
        }
        for name, raw_value in integer_values.items():
            if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                raw_value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be a positive integer")
            value = int(raw_value)
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.radial_bin_count < 16 or self.axial_bin_count < 16:
            raise ValueError("reciprocal-volume grids require at least 16 bins per axis")
        if self.surface_angle_count < 16:
            raise ValueError("surface_angle_count must be at least 16")
        if self.dpi < 72:
            raise ValueError("dpi must be at least 72")


@dataclass(frozen=True, slots=True)
class FamilyStrengthProfile:
    family_m: int
    radial_Ainv: float
    axial_Ainv: FloatArray
    strength_A2: FloatArray
    physical_rod_count: int


@dataclass(frozen=True, slots=True)
class ReciprocalDisplayVolume:
    radial_centers_Ainv: FloatArray
    axial_centers_Ainv: FloatArray
    family_display_density_A4: FloatArray
    family_m: NDArray[np.int64]
    family_display_peaks_A4: FloatArray

    def __post_init__(self) -> None:
        radial = np.asarray(self.radial_centers_Ainv, dtype=np.float64)
        axial = np.asarray(self.axial_centers_Ainv, dtype=np.float64)
        density = np.asarray(self.family_display_density_A4, dtype=np.float64)
        families = np.asarray(self.family_m, dtype=np.int64)
        peaks = np.asarray(self.family_display_peaks_A4, dtype=np.float64)
        if radial.ndim != 1 or axial.ndim != 1 or radial.size < 2 or axial.size < 2:
            raise ValueError("reciprocal-volume centers must be one-dimensional grids")
        if not np.all(np.isfinite(radial)) or not np.all(np.diff(radial) > 0.0):
            raise ValueError("radial centers must be finite and increasing")
        if not np.all(np.isfinite(axial)) or not np.all(np.diff(axial) > 0.0):
            raise ValueError("axial centers must be finite and increasing")
        if density.shape != (families.size, radial.size, axial.size):
            raise ValueError("family display density shape does not match its identities and grids")
        if peaks.shape != families.shape:
            raise ValueError("family peaks must align with family identities")
        if (
            not np.all(np.isfinite(density))
            or np.any(density < 0.0)
            or not np.all(np.isfinite(peaks))
            or np.any(peaks < 0.0)
        ):
            raise ValueError(
                "reciprocal display densities and peaks must be finite and nonnegative"
            )
        observed_peaks = np.max(density, axis=(1, 2))
        if not np.allclose(peaks, observed_peaks, rtol=2.0e-7, atol=1.0e-12):
            raise ValueError("stored family peaks do not match the reciprocal display density")
        if tuple(int(value) for value in families) != PHYSICAL_FAMILIES:
            raise ValueError("the final reciprocal figure requires physical families m=0,1,3")


@dataclass(frozen=True, slots=True)
class _RodHistogramTask:
    family_m: int
    q_parallel_crystal_Ainv: FloatArray
    axial_Ainv: FloatArray
    alpha_probability_mass: FloatArray
    axial_structure_mass_A: FloatArray
    tilt_rotations_crystal: FloatArray
    mean_axis_crystal: FloatArray
    radial_edges_Ainv: FloatArray
    axial_edges_Ainv: FloatArray


@contextmanager
def _single_threaded_child_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in CHILD_THREAD_ENVIRONMENT_NAMES}
    try:
        for name in CHILD_THREAD_ENVIRONMENT_NAMES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _configured_inputs(
    config_path: Path,
    settings: ReciprocalFigureSettings,
) -> ConfiguredSimulationInputs:
    base = load_simulation_config(config_path, repository_root=ROOT)
    if base.material.phase_id != "bi2se3":
        raise ValueError("the reciprocal publication renderer requires the Bi2Se3 configuration")
    if not base.instrument.axis_rotations:
        raise ValueError("the Bi2Se3 configuration must declare its incidence rotation")
    incidence = replace(base.instrument.axis_rotations[0], angle_deg=settings.incidence_deg)
    instrument = replace(
        base.instrument,
        axis_rotations=(incidence, *base.instrument.axis_rotations[1:]),
    )
    mosaic = replace(
        base.mosaic,
        gaussian_sigma_deg=settings.gaussian_sigma_deg,
        lorentzian_hwhm_deg=settings.lorentzian_hwhm_deg,
        lorentzian_probability=settings.lorentzian_probability,
    )
    configured = replace(
        base,
        source=replace(
            base.source,
            sample_count=base.source.minimum_physical_sample_count,
        ),
        instrument=instrument,
        mosaic=mosaic,
    )
    return build_configured_simulation_inputs(configured)


def _physical_rods_by_family(
    inputs: ConfiguredSimulationInputs,
) -> dict[int, tuple[Rod, ...]]:
    result = {
        family_m: tuple(rod for rod in inputs.rods if rod.family_m == family_m)
        for family_m in PHYSICAL_FAMILIES
    }
    missing = [family_m for family_m, rods in result.items() if not rods]
    if missing:
        raise ValueError(f"configured physical rod catalog is missing families {missing}")
    if any(rod.family_m == 2 for rod in inputs.rods):
        raise RuntimeError("hexagonal indexing unexpectedly produced a physical m=2 family")
    return result


def _structure_resolved_axial_nodes_Ainv(
    inputs: ConfiguredSimulationInputs,
    rod: Rod,
    *,
    background_count: int,
) -> FloatArray:
    lower, upper = inputs.bragg_space.rod_u_bounds_Ainv(rod)
    b3_norm = float(np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2]))
    lower_l = lower / b3_norm
    upper_l = upper / b3_norm
    return (
        finite_stack_integer_l_display_nodes(
            lower_l,
            upper_l,
            layer_count=inputs.config.structure_factor.layers,
            background_count=background_count,
        )
        * b3_norm
    )


def _structure_quadrature_axial_Ainv(
    inputs: ConfiguredSimulationInputs,
    rod: Rod,
) -> tuple[FloatArray, FloatArray]:
    """Return the converged dense trapezoid rule over one complete elastic rod."""

    lower_Ainv, upper_Ainv = inputs.bragg_space.rod_u_bounds_Ainv(rod)
    axial_Ainv = np.linspace(
        lower_Ainv,
        upper_Ainv,
        AXIAL_QUADRATURE_POINT_COUNT,
        dtype=np.float64,
    )
    axial_weight_A = _trapezoid_weights(axial_Ainv)
    if (
        not np.all(np.diff(axial_Ainv) > 0.0)
        or np.any(axial_weight_A <= 0.0)
        or not math.isclose(
            float(np.sum(axial_weight_A)),
            upper_Ainv - lower_Ainv,
            rel_tol=2.0e-14,
            abs_tol=1.0e-14,
        )
    ):
        raise RuntimeError("axial structure-factor quadrature is not a valid positive rule")
    return axial_Ainv, axial_weight_A


def _trapezoid_weights(nodes: FloatArray) -> FloatArray:
    values = np.asarray(nodes, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise ValueError("trapezoid nodes must be a one-dimensional increasing grid")
    weights = np.empty_like(values)
    weights[0] = 0.5 * (values[1] - values[0])
    weights[-1] = 0.5 * (values[-1] - values[-2])
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    return weights


def _mosaic_structure_mass_A(
    alpha_probability_mass: FloatArray,
    axial_structure_mass_A: FloatArray,
) -> FloatArray:
    alpha_mass = np.asarray(alpha_probability_mass, dtype=np.float64)
    structure_mass = np.asarray(axial_structure_mass_A, dtype=np.float64)
    if (
        alpha_mass.ndim != 1
        or structure_mass.ndim != 1
        or not np.all(np.isfinite(alpha_mass))
        or not np.all(np.isfinite(structure_mass))
        or np.any(alpha_mass < 0.0)
        or np.any(structure_mass < 0.0)
    ):
        raise ValueError("mosaic and structure masses must be finite nonnegative vectors")
    if not math.isclose(float(np.sum(alpha_mass)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("mosaic probability mass must sum to one")
    return alpha_mass[:, None] * structure_mass[None, :]


def _family_strength_profiles(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
) -> tuple[FamilyStrengthProfile, ...]:
    rods_by_family = _physical_rods_by_family(inputs)
    k_Ainv = 2.0 * math.pi / inputs.config.source.mean_wavelength_A
    profiles: list[FamilyStrengthProfile] = []
    for family_m in PHYSICAL_FAMILIES:
        rods = rods_by_family[family_m]
        axial = _structure_resolved_axial_nodes_Ainv(
            inputs,
            rods[0],
            background_count=PROFILE_BACKGROUND_COUNT,
        )
        b3_norm = float(np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2]))
        ell = axial / b3_norm
        strength = np.zeros_like(axial)
        for rod in rods:
            strength += rod.population * inputs.strength.evaluate_profile(
                rod=rod,
                L=ell,
                k_norm_Ainv=k_Ainv,
            )
        radial = float(inputs.reciprocal.qr_Ainv(np.array((rods[0].h, rods[0].k))))
        positive = (axial >= 0.0) & (axial <= settings.axial_max_Ainv)
        profiles.append(
            FamilyStrengthProfile(
                family_m=family_m,
                radial_Ainv=radial,
                axial_Ainv=np.asarray(axial[positive], dtype=np.float64),
                strength_A2=np.asarray(strength[positive], dtype=np.float64),
                physical_rod_count=len(rods),
            )
        )
    return tuple(profiles)


def _deposit_rod_histogram(task: _RodHistogramTask) -> tuple[int, FloatArray]:
    q_parallel_tilted = np.einsum(
        "aij,j->ai",
        task.tilt_rotations_crystal,
        task.q_parallel_crystal_Ainv,
        optimize=True,
    )
    axis_tilted = np.einsum(
        "aij,j->ai",
        task.tilt_rotations_crystal,
        task.mean_axis_crystal,
        optimize=True,
    )
    histogram = np.zeros(
        (task.radial_edges_Ainv.size - 1, task.axial_edges_Ainv.size - 1),
        dtype=np.float64,
    )
    for start in range(0, task.axial_Ainv.size, AXIAL_TRANSFORM_CHUNK_SIZE):
        stop = min(start + AXIAL_TRANSFORM_CHUNK_SIZE, task.axial_Ainv.size)
        q_crystal = (
            q_parallel_tilted[:, None, :]
            + task.axial_Ainv[None, start:stop, None] * axis_tilted[:, None, :]
        )
        qz = np.einsum("aui,i->au", q_crystal, task.mean_axis_crystal, optimize=True)
        radial_vector = q_crystal - qz[:, :, None] * task.mean_axis_crystal[None, None, :]
        qr = np.linalg.norm(radial_vector, axis=-1)
        weights = _mosaic_structure_mass_A(
            task.alpha_probability_mass,
            task.axial_structure_mass_A[start:stop],
        )
        valid = (
            np.isfinite(qr)
            & np.isfinite(qz)
            & np.isfinite(weights)
            & (weights > 0.0)
            & (qr >= task.radial_edges_Ainv[0])
            & (qr < task.radial_edges_Ainv[-1])
            & (qz >= task.axial_edges_Ainv[0])
            & (qz < task.axial_edges_Ainv[-1])
        )
        chunk_histogram, _, _ = np.histogram2d(
            qr[valid],
            qz[valid],
            bins=(task.radial_edges_Ainv, task.axial_edges_Ainv),
            weights=weights[valid],
        )
        histogram += chunk_histogram
    return task.family_m, histogram


def _annular_density_A4(
    family_mass_A: FloatArray,
    radial_edges_Ainv: FloatArray,
    axial_edges_Ainv: FloatArray,
) -> FloatArray:
    mass = np.asarray(family_mass_A, dtype=np.float64)
    radial_edges = np.asarray(radial_edges_Ainv, dtype=np.float64)
    axial_edges = np.asarray(axial_edges_Ainv, dtype=np.float64)
    expected_shape = (radial_edges.size - 1, axial_edges.size - 1)
    if mass.shape != expected_shape or np.any(mass < 0.0) or not np.all(np.isfinite(mass)):
        raise ValueError("family mass must be finite and match the annular grid")
    if not np.all(np.diff(radial_edges) > 0.0) or not np.all(np.diff(axial_edges) > 0.0):
        raise ValueError("annular grid edges must be increasing")
    annular_area_Ainv2 = math.pi * (radial_edges[1:] ** 2 - radial_edges[:-1] ** 2)
    reciprocal_volume_Ainv3 = annular_area_Ainv2[:, None] * np.diff(axial_edges)[None, :]
    return mass / reciprocal_volume_Ainv3


def _smoothed_annular_display_density_A4(
    family_mass_A: FloatArray,
    radial_edges_Ainv: FloatArray,
    axial_edges_Ainv: FloatArray,
) -> FloatArray:
    mass = np.asarray(family_mass_A, dtype=np.float64)
    smoothed_mass = gaussian_filter(
        mass,
        sigma=DISPLAY_SMOOTHING_SIGMA_BINS,
        mode="reflect",
    )
    if not math.isclose(
        float(np.sum(smoothed_mass)),
        float(np.sum(mass)),
        rel_tol=2.0e-14,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError("display smoothing failed to conserve reciprocal-space mass")
    return _annular_density_A4(smoothed_mass, radial_edges_Ainv, axial_edges_Ainv)


def compute_axisymmetric_display_volume(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
) -> ReciprocalDisplayVolume:
    """Push mosaic x SF into annular cells and form the declared smoothed display field."""

    mosaic_parameters = MosaicParameters(
        gaussian_sigma_rad=math.radians(settings.gaussian_sigma_deg),
        lorentzian_half_width_rad=math.radians(settings.lorentzian_hwhm_deg),
        lorentzian_probability=settings.lorentzian_probability,
        alpha_panel_count=settings.alpha_panel_count,
        alpha_gauss_order=settings.alpha_gauss_order,
        azimuth_count=1,
        azimuth_phase_rad=0.0,
    )
    mosaic = build_mosaic_space(
        reciprocal_basis_Ainv=inputs.reciprocal.basis_Ainv,
        crystal_to_sample=inputs.bragg_space.config.crystal_to_sample,
        parameters=mosaic_parameters,
    )
    alpha_rad = np.asarray(mosaic.alpha_rad, dtype=np.float64)
    alpha_mass = np.asarray(mosaic.probability_mass, dtype=np.float64)
    mean_axis, tilt_axis = mosaic_axes(inputs.reciprocal.basis_Ainv)
    tilt_rotations = axis_angle_rotation_batch(tilt_axis, alpha_rad)
    radial_edges = np.linspace(0.0, settings.radial_max_Ainv, settings.radial_bin_count + 1)
    axial_edges = np.linspace(0.0, settings.axial_max_Ainv, settings.axial_bin_count + 1)
    rods_by_family = _physical_rods_by_family(inputs)
    k_Ainv = 2.0 * math.pi / inputs.config.source.mean_wavelength_A
    b3_norm = float(np.linalg.norm(inputs.reciprocal.basis_Ainv[:, 2]))
    tasks: list[_RodHistogramTask] = []
    for family_m in PHYSICAL_FAMILIES:
        for rod in rods_by_family[family_m]:
            axial, du = _structure_quadrature_axial_Ainv(inputs, rod)
            strength_A2 = rod.population * inputs.strength.evaluate_profile(
                rod=rod,
                L=axial / b3_norm,
                k_norm_Ainv=k_Ainv,
            )
            q_parallel = (
                rod.h * inputs.reciprocal.basis_Ainv[:, 0]
                + rod.k * inputs.reciprocal.basis_Ainv[:, 1]
            )
            tasks.append(
                _RodHistogramTask(
                    family_m=family_m,
                    q_parallel_crystal_Ainv=np.asarray(q_parallel, dtype=np.float64),
                    axial_Ainv=np.asarray(axial, dtype=np.float64),
                    alpha_probability_mass=alpha_mass,
                    axial_structure_mass_A=np.asarray(strength_A2 * du, dtype=np.float64),
                    tilt_rotations_crystal=np.asarray(tilt_rotations, dtype=np.float64),
                    mean_axis_crystal=np.asarray(mean_axis, dtype=np.float64),
                    radial_edges_Ainv=radial_edges,
                    axial_edges_Ainv=axial_edges,
                )
            )
    family_mass = {
        family_m: np.zeros((settings.radial_bin_count, settings.axial_bin_count), dtype=np.float64)
        for family_m in PHYSICAL_FAMILIES
    }
    if settings.worker_count == 1:
        completed = map(_deposit_rod_histogram, tasks)
        for family_m, histogram in completed:
            family_mass[family_m] += histogram
    else:
        worker_count = min(settings.worker_count, len(tasks))
        with (
            _single_threaded_child_environment(),
            ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=get_context("spawn"),
            ) as executor,
        ):
            for family_m, histogram in executor.map(_deposit_rod_histogram, tasks):
                family_mass[family_m] += histogram
    family_display_density = np.stack(
        [
            _smoothed_annular_display_density_A4(
                family_mass[family_m],
                radial_edges,
                axial_edges,
            )
            for family_m in PHYSICAL_FAMILIES
        ],
        axis=0,
    )
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    axial_centers = 0.5 * (axial_edges[:-1] + axial_edges[1:])
    peaks = np.max(family_display_density, axis=(1, 2))
    return ReciprocalDisplayVolume(
        radial_centers_Ainv=radial_centers,
        axial_centers_Ainv=axial_centers,
        family_display_density_A4=family_display_density,
        family_m=np.asarray(PHYSICAL_FAMILIES, dtype=np.int64),
        family_display_peaks_A4=peaks,
    )


def _cache_payload(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
) -> dict[str, object]:
    rods_by_family = _physical_rods_by_family(inputs)
    payload: dict[str, object] = {
        "schema_version": CACHE_SCHEMA,
        "algorithm_id": VOLUME_ALGORITHM,
        "physics_revision": inputs.config.physics_revision,
        "cif_sha256": inputs.config.cif_sha256,
        "families": list(PHYSICAL_FAMILIES),
        "physical_rods": {
            str(family_m): [[rod.h, rod.k, rod.population] for rod in rods_by_family[family_m]]
            for family_m in PHYSICAL_FAMILIES
        },
        "settings": {
            "gaussian_sigma_deg": settings.gaussian_sigma_deg,
            "lorentzian_hwhm_deg": settings.lorentzian_hwhm_deg,
            "lorentzian_probability": settings.lorentzian_probability,
            "radial_max_Ainv": settings.radial_max_Ainv,
            "axial_max_Ainv": settings.axial_max_Ainv,
            "radial_bin_count": settings.radial_bin_count,
            "axial_bin_count": settings.axial_bin_count,
            "alpha_panel_count": settings.alpha_panel_count,
            "alpha_gauss_order": settings.alpha_gauss_order,
            "axial_trapezoid_point_count": AXIAL_QUADRATURE_POINT_COUNT,
            "display_smoothing_sigma_bins": list(DISPLAY_SMOOTHING_SIGMA_BINS),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["cache_key"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _load_cache_arrays(data: object) -> ReciprocalDisplayVolume:
    return ReciprocalDisplayVolume(
        radial_centers_Ainv=np.asarray(data["r_centers"], dtype=np.float64),
        axial_centers_Ainv=np.asarray(data["z_centers"], dtype=np.float64),
        family_display_density_A4=np.asarray(data["family_display_density"], dtype=np.float64),
        family_m=np.asarray(data["families"], dtype=np.int64),
        family_display_peaks_A4=np.asarray(data["display_peaks"], dtype=np.float64),
    )


def _load_volume_cache(
    cache_path: Path,
    expected_payload: dict[str, object],
) -> ReciprocalDisplayVolume:
    with np.load(cache_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("reciprocal display cache has no supported provenance metadata")
        observed = json.loads(str(np.asarray(data["metadata_json"]).item()))
        if observed != expected_payload:
            raise ValueError("reciprocal display cache provenance does not match this request")
        volume = _load_cache_arrays(data)
    settings = expected_payload["settings"]
    if not isinstance(settings, dict):
        raise ValueError("reciprocal display cache settings metadata is malformed")
    radial_count = int(settings["radial_bin_count"])
    axial_count = int(settings["axial_bin_count"])
    expected_radial = (np.arange(radial_count, dtype=np.float64) + 0.5) * (
        float(settings["radial_max_Ainv"]) / radial_count
    )
    expected_axial = (np.arange(axial_count, dtype=np.float64) + 0.5) * (
        float(settings["axial_max_Ainv"]) / axial_count
    )
    radial_tolerance = 8.0 * np.finfo(np.float64).eps * float(settings["radial_max_Ainv"])
    axial_tolerance = 8.0 * np.finfo(np.float64).eps * float(settings["axial_max_Ainv"])
    if not np.allclose(
        volume.radial_centers_Ainv,
        expected_radial,
        rtol=0.0,
        atol=radial_tolerance,
    ) or not np.allclose(
        volume.axial_centers_Ainv,
        expected_axial,
        rtol=0.0,
        atol=axial_tolerance,
    ):
        raise ValueError("reciprocal display cache grid does not match its declared settings")
    return volume


def _save_volume_cache(
    cache_path: Path,
    volume: ReciprocalDisplayVolume,
    payload: dict[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(json.dumps(payload, sort_keys=True, allow_nan=False)),
            r_centers=volume.radial_centers_Ainv,
            z_centers=volume.axial_centers_Ainv,
            family_display_density=volume.family_display_density_A4.astype(np.float32),
            families=volume.family_m,
            display_peaks=volume.family_display_peaks_A4,
        )
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_compute_display_volume(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
    cache_path: Path,
    *,
    recompute: bool,
) -> tuple[ReciprocalDisplayVolume, str]:
    payload = _cache_payload(inputs, settings)
    if cache_path.exists() and not recompute:
        return (
            _load_volume_cache(cache_path, payload),
            "reused",
        )
    volume = compute_axisymmetric_display_volume(inputs, settings)
    _save_volume_cache(cache_path, volume, payload)
    return volume, "computed"


def _configure_matplotlib() -> object:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _positive_log_bounds(values: Sequence[FloatArray], *, decades: float) -> tuple[float, float]:
    positive = np.concatenate(
        [np.asarray(value)[np.isfinite(value) & (np.asarray(value) > 0.0)] for value in values]
    )
    if not positive.size:
        raise RuntimeError("the requested reciprocal figure has no positive intensity")
    maximum = float(np.max(positive))
    return maximum * 10.0 ** (-decades), maximum


def _configured_structure_label(inputs: ConfiguredSimulationInputs) -> str:
    model_id = inputs.config.structure_factor.model_id
    model_label = "finite-2H" if model_id == "r3m_quintuple_finite_2h.v1" else model_id
    return f"{inputs.config.structure_factor.layers}-layer {model_label}"


def _save_figure(
    figure: object,
    output_directory: Path,
    stem: str,
    formats: Sequence[str],
    *,
    dpi: int,
    overwrite: bool,
) -> tuple[Path, ...]:
    outputs = tuple(output_directory / f"{stem}.{suffix}" for suffix in formats)
    existing = [path.name for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"outputs already exist ({', '.join(existing)}); pass --overwrite")
    for path, suffix in zip(outputs, formats, strict=True):
        figure.savefig(path, dpi=dpi, format=suffix, bbox_inches="tight")
    return outputs


def _requested_output_paths(
    output_directory: Path,
    mode: str,
    formats: Sequence[str],
) -> tuple[Path, ...]:
    format_tuple = tuple(formats)
    if len(set(format_tuple)) != len(format_tuple):
        raise ValueError("output formats must be unique")
    selected_modes = tuple(OUTPUT_STEMS) if mode == "all" else (mode,)
    if any(selected not in OUTPUT_STEMS for selected in selected_modes):
        raise ValueError("unsupported reciprocal figure mode")
    return tuple(
        output_directory / f"{OUTPUT_STEMS[selected]}.{suffix}"
        for selected in selected_modes
        for suffix in format_tuple
    )


def _preflight_output_paths(
    outputs: Sequence[Path],
    cache_path: Path,
    *,
    overwrite: bool,
) -> None:
    resolved_outputs = tuple(path.resolve() for path in outputs)
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("requested reciprocal figure outputs must be unique")
    if cache_path.resolve() in resolved_outputs:
        raise ValueError("cache path must not alias a requested figure output")
    existing = [path.name for path in resolved_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"outputs already exist ({', '.join(existing)}); pass --overwrite")


def render_zero_mosaic_figure(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
    output_directory: Path,
    formats: Sequence[str],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plt = _configure_matplotlib()
    profiles = _family_strength_profiles(inputs, settings)
    structure_label = _configured_structure_label(inputs)
    vmin, vmax = _positive_log_bounds([profile.strength_A2 for profile in profiles], decades=9.0)
    color_map = plt.colormaps["magma"]
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    figure = plt.figure(figsize=(11.8, 8.1))
    axis = figure.add_axes((0.07, 0.12, 0.68, 0.77), projection="3d")
    axis.set_proj_type("ortho")
    cutaway_angle = np.deg2rad(np.linspace(35.0, 215.0, settings.surface_angle_count))
    legend_colors = ("#1c1720", "#784080", "#d85c50")
    legend_handles: list[object] = []
    for shell_index, (profile, legend_color) in enumerate(
        zip(profiles, legend_colors, strict=True)
    ):
        if profile.family_m == 0:
            points = np.column_stack(
                (
                    np.zeros_like(profile.axial_Ainv),
                    np.zeros_like(profile.axial_Ainv),
                    profile.axial_Ainv,
                )
            )
            segments = np.stack((points[:-1], points[1:]), axis=1)
            segment_strength = 0.5 * (profile.strength_A2[:-1] + profile.strength_A2[1:])
            axis.add_collection3d(
                Line3DCollection(
                    segments,
                    colors=color_map(norm(segment_strength)),
                    linewidths=3.0,
                    rasterized=True,
                )
            )
        else:
            theta_grid, axial_grid = np.meshgrid(cutaway_angle, profile.axial_Ainv)
            x = profile.radial_Ainv * np.cos(theta_grid)
            y = profile.radial_Ainv * np.sin(theta_grid)
            colors = color_map(norm(np.broadcast_to(profile.strength_A2[:, None], x.shape)))
            axis.plot_surface(
                x,
                y,
                axial_grid,
                facecolors=colors,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                rcount=x.shape[0],
                ccount=x.shape[1],
                rasterized=True,
            )
        full_angle = np.linspace(0.0, 2.0 * math.pi, 361)
        axis.plot(
            profile.radial_Ainv * np.cos(full_angle),
            profile.radial_Ainv * np.sin(full_angle),
            np.zeros_like(full_angle),
            color="0.35",
            linewidth=0.7,
            alpha=0.65,
        )
        radial_text = "0" if profile.family_m == 0 else f"{profile.radial_Ainv:.3f} Å⁻¹"
        legend_handles.append(
            Line2D(
                (0,),
                (0,),
                color=legend_color,
                linewidth=5.0,
                label=(
                    f"shell {shell_index}:  m = {profile.family_m}   "
                    rf"$Q_r$ = {radial_text}"
                ),
            )
        )
    limit = settings.radial_max_Ainv
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(0.0, settings.axial_max_Ainv)
    axis.set_box_aspect((2.0 * limit, 2.0 * limit, settings.axial_max_Ainv))
    axis.view_init(elev=24.0, azim=-55.0)
    axis.set_xlabel(r"$Q_x$ (Å$^{-1}$)", labelpad=8)
    axis.set_ylabel(r"$Q_y$ (Å$^{-1}$)", labelpad=8)
    axis.set_zlabel(r"$Q_z$ (Å$^{-1}$)", labelpad=8)
    axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.03, 1.02))
    color_axis = figure.add_axes((0.79, 0.22, 0.025, 0.55))
    color_bar = figure.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=color_axis)
    color_bar.set_label(
        r"family-summed finite-stack SF strength  $\sum_{h,k} S_{hk}(L)$  (Å$^2$; log)"
    )
    figure.text(
        0.025,
        0.955,
        "Bi₂Se₃ reciprocal-space cylinders",
        fontsize=20,
        fontweight="bold",
    )
    figure.text(
        0.025,
        0.925,
        "First three physical radial shells  •  zero mosaic tilt  •  positive $Q_z$  •  "
        f"{structure_label} structure strength",
        fontsize=12,
        color="0.35",
    )
    figure.text(
        0.025,
        0.045,
        "Cutaway view: colors are the incoherent sum of the physical rods in each family. "
        f"Cylinders end at the configured $\\lambda={inputs.config.source.mean_wavelength_A:g}$ Å "
        "elastic-reach boundary; no mosaic, optics, Ewald, or detector factors are applied.\n"
        "Hexagonal indexing uses m = h² + hk + k², so no "
        "physical m = 2 family exists; shell index 2 is the next allowed family, m = 3.",
        fontsize=9.5,
        color="0.35",
    )
    outputs = _save_figure(
        figure,
        output_directory,
        OUTPUT_STEMS["zero-mosaic"],
        formats,
        dpi=settings.dpi,
        overwrite=overwrite,
    )
    plt.close(figure)
    return outputs


def _contour_segments(
    radial_centers: FloatArray,
    axial_centers: FloatArray,
    normalized_density: FloatArray,
    levels: Sequence[float],
) -> tuple[tuple[FloatArray, ...], ...]:
    plt = _configure_matplotlib()
    figure, axis = plt.subplots()
    contour = axis.contour(
        radial_centers,
        axial_centers,
        normalized_density.T,
        levels=levels,
    )
    segments = tuple(
        tuple(np.asarray(segment, dtype=np.float64) for segment in level_segments)
        for level_segments in contour.allsegs
    )
    plt.close(figure)
    return segments


def _add_mean_cylinder_guides(axis: object, radii_Ainv: dict[int, float], axial_max: float) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    for radius in radii_Ainv.values():
        axis.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros_like(theta),
            color="#29242e",
            linewidth=0.55,
            alpha=0.28,
            zorder=12,
        )
        for angle_deg in (35.0, 125.0, 215.0, 305.0):
            angle = math.radians(angle_deg)
            axis.plot(
                np.full(2, radius * math.cos(angle)),
                np.full(2, radius * math.sin(angle)),
                (0.0, axial_max),
                color="#29242e",
                linewidth=0.45,
                alpha=0.18,
            )
    axis.plot((0.0, 0.0), (0.0, 0.0), (0.0, axial_max), color="#17131b", linewidth=1.4)


def render_continuous_mosaic_figure(
    inputs: ConfiguredSimulationInputs,
    settings: ReciprocalFigureSettings,
    volume: ReciprocalDisplayVolume,
    output_directory: Path,
    formats: Sequence[str],
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    plt = _configure_matplotlib()
    structure_label = _configured_structure_label(inputs)
    total_density = np.sum(volume.family_display_density_A4, axis=0)
    maximum = float(np.max(total_density))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("continuous reciprocal volume has no positive intensity")
    normalized = total_density / maximum
    levels = (1.0e-5, 1.0e-3, 1.0e-2)
    level_segments = _contour_segments(
        volume.radial_centers_Ainv,
        volume.axial_centers_Ainv,
        normalized,
        levels,
    )
    color_map = plt.colormaps["magma"]
    log_norm = LogNorm(vmin=1.0e-7, vmax=1.0, clip=True)
    figure = plt.figure(figsize=(13.2, 7.1))
    axis_3d = figure.add_axes((0.04, 0.12, 0.55, 0.72), projection="3d")
    axis_3d.set_proj_type("ortho")
    theta = np.deg2rad(np.linspace(35.0, 215.0, settings.surface_angle_count))
    alphas = (0.10, 0.23, 0.52)
    legend_handles: list[object] = []
    for level, segments, alpha in zip(levels, level_segments, alphas, strict=True):
        color = color_map(log_norm(level))
        for segment in segments:
            finite = np.all(np.isfinite(segment), axis=1) & (segment[:, 0] >= 0.0)
            line = segment[finite]
            if line.shape[0] < 2:
                continue
            stride = max(1, math.ceil(line.shape[0] / 240))
            line = line[::stride]
            radial = line[:, 0, None]
            axial = line[:, 1, None]
            x = radial * np.cos(theta)[None, :]
            y = radial * np.sin(theta)[None, :]
            z = np.broadcast_to(axial, x.shape)
            axis_3d.plot_surface(
                x,
                y,
                z,
                color=color,
                alpha=alpha,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                rasterized=True,
            )
        legend_handles.append(
            Line2D(
                (0,),
                (0,),
                color=color,
                linewidth=5.0,
                label=rf"$I/I_{{max}} = 10^{{{int(math.log10(level))}}}$",
            )
        )
    rods_by_family = _physical_rods_by_family(inputs)
    radii = {
        family_m: float(
            inputs.reciprocal.qr_Ainv(
                np.array((rods_by_family[family_m][0].h, rods_by_family[family_m][0].k))
            )
        )
        for family_m in PHYSICAL_FAMILIES
    }
    _add_mean_cylinder_guides(axis_3d, radii, settings.axial_max_Ainv)
    axis_3d.set_xlim(-settings.radial_max_Ainv, settings.radial_max_Ainv)
    axis_3d.set_ylim(-settings.radial_max_Ainv, settings.radial_max_Ainv)
    axis_3d.set_zlim(0.0, settings.axial_max_Ainv)
    axis_3d.set_box_aspect((2.0 * settings.radial_max_Ainv,) * 2 + (settings.axial_max_Ainv,))
    axis_3d.view_init(elev=24.0, azim=-55.0)
    axis_3d.set_xlabel(r"$Q_x$ (Å$^{-1}$)", labelpad=7)
    axis_3d.set_ylabel(r"$Q_y$ (Å$^{-1}$)", labelpad=7)
    axis_3d.set_zlabel(r"$Q_z$ (Å$^{-1}$)", labelpad=7)
    axis_3d.set_title("(a) 3D iso-density envelope", loc="left", fontweight="bold")
    axis_3d.legend(handles=legend_handles, loc="upper left", framealpha=0.92)

    axis_2d = figure.add_axes((0.63, 0.16, 0.27, 0.68))
    section = np.concatenate((normalized[:0:-1, :], normalized), axis=0)
    image = axis_2d.imshow(
        section.T,
        origin="lower",
        extent=(
            -volume.radial_centers_Ainv[-1],
            volume.radial_centers_Ainv[-1],
            volume.axial_centers_Ainv[0],
            volume.axial_centers_Ainv[-1],
        ),
        aspect="auto",
        cmap=color_map,
        norm=log_norm,
        interpolation="bilinear",
    )
    for family_m in (1, 3):
        radius = radii[family_m]
        axis_2d.axvline(radius, color="white", linestyle="--", linewidth=0.7, alpha=0.5)
        axis_2d.axvline(-radius, color="white", linestyle="--", linewidth=0.7, alpha=0.5)
    axis_2d.axvline(0.0, color="white", linewidth=0.7, alpha=0.5)
    axis_2d.set_xlim(-settings.radial_max_Ainv, settings.radial_max_Ainv)
    axis_2d.set_ylim(0.0, settings.axial_max_Ainv)
    axis_2d.set_xlabel(r"axial-section $Q_x$ (Å$^{-1}$)")
    axis_2d.set_ylabel(r"$Q_z$ (Å$^{-1}$)")
    axis_2d.set_title(r"(b) continuous $Q_y=0$ section", loc="left", fontweight="bold")
    axis_2d.text(
        0.02,
        0.02,
        r"dashed: nominal $m=1,3$ radii",
        transform=axis_2d.transAxes,
        color="0.85",
        fontsize=8.5,
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 2.0},
    )
    color_axis = figure.add_axes((0.92, 0.23, 0.02, 0.52))
    color_bar = figure.colorbar(image, cax=color_axis, extend="min")
    color_bar.set_label(r"normalized smoothed display density  $I/I_{max}$  (log)")
    figure.text(
        0.02,
        0.96,
        "Bi₂Se₃ continuous mosaic-broadened reciprocal intensity",
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.925,
        f"Gaussian $\\sigma$ = {settings.gaussian_sigma_deg:g}°  •  Lorentzian HWHM = "
        f"{settings.lorentzian_hwhm_deg:g}°  •  $\\eta$ = {settings.lorentzian_probability:g}  •  "
        f"positive $Q_z$  •  m = 0, 1, 3  •  {structure_label} SF",
        fontsize=11.5,
        color="0.35",
    )
    figure.text(
        0.02,
        0.035,
        "The volume is the continuous mosaic $\\times$ SF measure deposited in reciprocal space; no "
        "discrete orientation copies are drawn. The 3D panel uses a camera-facing cutaway and "
        "nested iso-density surfaces; the axial section shows the full log-scaled field.\n"
        "Before annular-volume normalization, the displayed field applies mass-conserving "
        "Gaussian smoothing to cell masses with "
        f"$\\sigma_r={DISPLAY_SMOOTHING_SIGMA_BINS[0]:g}$ and "
        f"$\\sigma_z={DISPLAY_SMOOTHING_SIGMA_BINS[1]:g}$ bins; this is display-only. "
        "Thin dark guides mark the unrotated cylinders. Shell index 2 is physical family m=3; "
        "no hexagonal m=2 family exists.",
        fontsize=8.8,
        color="0.35",
    )
    outputs = _save_figure(
        figure,
        output_directory,
        OUTPUT_STEMS["continuous-mosaic"],
        formats,
        dpi=settings.dpi,
        overwrite=overwrite,
    )
    plt.close(figure)
    return outputs


def _external_path(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{name} must be outside the repository")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--mode", choices=("all", "zero-mosaic", "continuous-mosaic"), default="all"
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png",))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument(
        "--incidence-deg",
        type=float,
        default=10.0,
        help=(
            "sample incidence retained in configuration provenance; the intrinsic "
            "crystal-axis reciprocal plots are invariant under this rigid rotation"
        ),
    )
    parser.add_argument("--gaussian-sigma-deg", type=float, default=2.0)
    parser.add_argument("--lorentzian-hwhm-deg", type=float, default=0.2)
    parser.add_argument("--lorentzian-probability", type=float, default=0.1)
    parser.add_argument("--radial-bins", type=int, default=280)
    parser.add_argument("--axial-bins", type=int, default=380)
    parser.add_argument("--alpha-panels", type=int, default=20)
    parser.add_argument("--alpha-gauss-order", type=int, default=12)
    parser.add_argument("--worker-count", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--surface-angle-count", type=int, default=181)
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = ReciprocalFigureSettings(
        incidence_deg=arguments.incidence_deg,
        gaussian_sigma_deg=arguments.gaussian_sigma_deg,
        lorentzian_hwhm_deg=arguments.lorentzian_hwhm_deg,
        lorentzian_probability=arguments.lorentzian_probability,
        radial_bin_count=arguments.radial_bins,
        axial_bin_count=arguments.axial_bins,
        alpha_panel_count=arguments.alpha_panels,
        alpha_gauss_order=arguments.alpha_gauss_order,
        worker_count=arguments.worker_count,
        surface_angle_count=arguments.surface_angle_count,
        dpi=arguments.dpi,
    )
    output_directory = _external_path(arguments.output_directory, name="output directory")
    cache_path = _external_path(
        arguments.cache or output_directory / "bi2se3-continuous-mosaic-display-cache.npz",
        name="cache path",
    )
    formats = tuple(arguments.formats)
    requested_outputs = _requested_output_paths(output_directory, arguments.mode, formats)
    _preflight_output_paths(
        requested_outputs,
        cache_path,
        overwrite=arguments.overwrite,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    inputs = _configured_inputs(arguments.config.resolve(), settings)
    outputs: list[Path] = []
    cache_status: str | None = None
    volume: ReciprocalDisplayVolume | None = None
    if arguments.mode in {"all", "continuous-mosaic"}:
        volume, cache_status = load_or_compute_display_volume(
            inputs,
            settings,
            cache_path,
            recompute=arguments.recompute,
        )
    if arguments.mode in {"all", "zero-mosaic"}:
        outputs.extend(
            render_zero_mosaic_figure(
                inputs,
                settings,
                output_directory,
                formats,
                overwrite=arguments.overwrite,
            )
        )
    if arguments.mode in {"all", "continuous-mosaic"}:
        if volume is None:
            raise RuntimeError("continuous reciprocal display volume was not prepared")
        outputs.extend(
            render_continuous_mosaic_figure(
                inputs,
                settings,
                volume,
                output_directory,
                formats,
                overwrite=arguments.overwrite,
            )
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": arguments.mode,
                "outputs": [str(path) for path in outputs],
                "cache": str(cache_path) if cache_status is not None else None,
                "cache_status": cache_status,
                "families": list(PHYSICAL_FAMILIES),
                "mosaic": {
                    "gaussian_sigma_deg": settings.gaussian_sigma_deg,
                    "lorentzian_hwhm_deg": settings.lorentzian_hwhm_deg,
                    "lorentzian_probability": settings.lorentzian_probability,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
