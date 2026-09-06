"""Detector pullback of a continuous curvilinear Ewald coating."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from math import fsum, isfinite
from operator import index

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    EwaldLatentGeometry,
    Rod,
)
from painted_ewald.rotations import mosaic_axes
from painted_ewald.validation import positive_integer
from rasim_next.core.contracts import MaterialOptics
from rasim_next.core.scattering import (
    polarization_model_code,
    scattering_polarization_weight,
)
from rasim_next.core.validity import ValidityCode
from rasim_next.geometry.detector import (
    _DETECTOR_INCIDENCE_COSINE_TOL,
    _detector_coordinates_to_lab_points,
    _detector_incidence_cosine,
    _project_detector_rays,
)
from rasim_next.geometry.instrument import (
    CompiledInstrument,
    detector_path_linear_attenuation_at_wavelength_m_inv,
)
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.optics.attenuation import (
    external_path_attenuation,
    incident_illuminated_path_weight,
    mode_decay_constant,
    scalar_optical_weight,
    uniform_depth_attenuation,
)
from rasim_next.optics.refraction import _solve_exit_mode_arrays
from rasim_next.pipeline._continuous_detector_kernel import (
    CompiledDetectorEvaluator,
    CompiledDetectorState,
    pack_bi2se3_two_h_structure,
)
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.reflectivity import CompiledParrattStitch
from rasim_next.sampling.source import require_physical_intensity_source_model
from rasim_next.stacking import Parent

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SampleQIntensityEnvelope:
    """Directional event-intensity damping in the fixed sample frame."""

    u_radial_A2: float = 0.0
    u_normal_A2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("u_radial_A2", "u_normal_A2"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    def evaluate(self, q_sample_Ainv: ArrayLike) -> FloatArray:
        """Return ``exp(-U_r Q_r^2 - U_z Q_z^2)`` for sample-frame events."""

        supplied = np.asarray(q_sample_Ainv)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError("q_sample_Ainv must be real")
        q_sample = np.asarray(supplied.real, dtype=np.float64)
        if q_sample.ndim < 1 or q_sample.shape[-1] != 3 or not np.all(np.isfinite(q_sample)):
            raise ValueError("q_sample_Ainv must contain finite sample-frame vectors")
        return np.exp(
            -self.u_radial_A2 * (q_sample[..., 0] ** 2 + q_sample[..., 1] ** 2)
            - self.u_normal_A2 * q_sample[..., 2] ** 2
        )


class IntensityStatus(StrEnum):
    """Whether a mapped component carries a physical detector measure."""

    INCLUDED = "INCLUDED"
    SPECULAR_INTENSITY_EXCLUDED = "SPECULAR_INTENSITY_EXCLUDED"


class PixelIntegrationMethod(StrEnum):
    """Explicit detector-pixel integration implementations."""

    FIXED_NUMPY = "fixed_numpy"
    ADAPTIVE_COMPILED = "adaptive_compiled"


def _subdivided_legendre_rule(
    gauss_order: int, subdivision_count: int
) -> tuple[FloatArray, FloatArray]:
    """Return one positive tensor-rule axis on the exact ``[-0.5, 0.5]`` pixel interval."""

    gauss_node, gauss_weight = leggauss(gauss_order)
    subcell = np.arange(subdivision_count, dtype=np.float64)
    offset = (subcell[:, None] + 0.5 + 0.5 * gauss_node[None, :]).reshape(
        -1
    ) / subdivision_count - 0.5
    weight = np.tile(0.5 * gauss_weight / subdivision_count, subdivision_count)
    offset.setflags(write=False)
    weight.setflags(write=False)
    return offset, weight


@dataclass(frozen=True, slots=True)
class DetectorQuadrature:
    """Deterministic integration controls for exact native pixel rectangles.

    Fold candidates are detected from finite geometry probes. The refinement
    diagnostics establish convergence for a tested fixture; they are not a
    conservative topology certificate for arbitrary instruments.
    """

    pixel_gauss_order: int = 2
    fold_gauss_order: int = 4
    fold_subdivision_count: int = 8
    row_chunk_size: int = 8
    node_chunk_size: int = 65_536
    method: PixelIntegrationMethod = PixelIntegrationMethod.FIXED_NUMPY
    relative_tolerance: float = 1.0e-4
    absolute_tolerance_A2: float = 0.0
    max_depth: int = 4
    worker_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "pixel_gauss_order",
            "fold_gauss_order",
            "fold_subdivision_count",
            "row_chunk_size",
            "node_chunk_size",
            "max_depth",
            "worker_count",
        ):
            object.__setattr__(self, name, positive_integer(getattr(self, name), name))
        if self.fold_gauss_order < self.pixel_gauss_order:
            raise ValueError("fold_gauss_order must not be smaller than pixel_gauss_order")
        object.__setattr__(self, "method", PixelIntegrationMethod(self.method))
        for name in ("relative_tolerance", "absolute_tolerance_A2"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


def _float_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    supplied = np.asarray(value)
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
        raise ValueError(f"{name} must be real")
    array = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _complex_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> ComplexArray:
    array = np.array(value, dtype=np.complex128, copy=True, order="C")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _status_array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.str_]:
    supplied = np.asarray(value)
    if supplied.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result = np.array(supplied, dtype="U32", copy=True, order="C")
    allowed = np.asarray(tuple(code.value for code in ValidityCode), dtype="U32")
    if not np.all(np.isin(result, allowed)):
        raise ValueError(f"{name} contains an unknown validity code")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class DetectorMappedGeometry:
    """Continuous outgoing-wave and detector coordinates for one latent batch."""

    ewald_geometry: EwaldLatentGeometry
    kf_air_sample_Ainv: FloatArray
    kf_air_lab_Ainv: FloatArray
    column_px: FloatArray
    row_px: FloatArray
    ray_distance_m: FloatArray
    pixel_solid_angle_sr: FloatArray
    exit_status: NDArray[np.str_]
    detector_status: NDArray[np.str_]
    valid: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ewald_geometry, EwaldLatentGeometry):
            raise TypeError("ewald_geometry must be EwaldLatentGeometry")
        shape = self.ewald_geometry.alpha_rad.shape
        arrays = {
            "kf_air_sample_Ainv": _float_array(
                self.kf_air_sample_Ainv, (*shape, 3), "kf_air_sample_Ainv"
            ),
            "kf_air_lab_Ainv": _float_array(self.kf_air_lab_Ainv, (*shape, 3), "kf_air_lab_Ainv"),
            "column_px": _float_array(self.column_px, shape, "column_px"),
            "row_px": _float_array(self.row_px, shape, "row_px"),
            "ray_distance_m": _float_array(self.ray_distance_m, shape, "ray_distance_m"),
            "pixel_solid_angle_sr": _float_array(
                self.pixel_solid_angle_sr, shape, "pixel_solid_angle_sr"
            ),
        }
        if np.any(arrays["ray_distance_m"] < 0.0):
            raise ValueError("ray_distance_m must be nonnegative")
        if np.any(arrays["pixel_solid_angle_sr"] < 0.0):
            raise ValueError("pixel_solid_angle_sr must be nonnegative")
        exit_status = _status_array(self.exit_status, shape, "exit_status")
        detector_status = _status_array(self.detector_status, shape, "detector_status")
        valid = detector_status == ValidityCode.VALID
        valid.setflags(write=False)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "exit_status", exit_status)
        object.__setattr__(self, "detector_status", detector_status)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class DetectorVisibleEwaldCoating:
    """Intrinsic latent Ewald coating restricted only by active-panel visibility."""

    geometry: DetectorMappedGeometry
    coating_intensity_density_A2_rad2_inv: FloatArray
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, DetectorMappedGeometry):
            raise TypeError("geometry must be DetectorMappedGeometry")
        shape = self.geometry.ewald_geometry.alpha_rad.shape
        density = _float_array(
            self.coating_intensity_density_A2_rad2_inv,
            shape,
            "coating_intensity_density_A2_rad2_inv",
        )
        if np.any(density < 0.0) or np.any((~self.geometry.valid) & (density != 0.0)):
            raise ValueError("intrinsic Ewald coating must be nonnegative and detector-visible")
        rod = self.geometry.ewald_geometry.rod
        gap = self.detector_visible_m0_q_gap_Ainv
        if rod.family_m == 0:
            if gap is None or not isfinite(float(gap)) or float(gap) <= 0.0:
                raise ValueError("detector-visible m=0 requires a positive reciprocal support gap")
            gap = float(gap)
        elif gap is not None:
            raise ValueError("an m=0 reciprocal support gap requires the (0, 0) rod")
        if self.measure_id != "detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1":
            raise ValueError("unsupported detector-visible Ewald coating measure")
        object.__setattr__(self, "coating_intensity_density_A2_rad2_inv", density)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", gap)


@dataclass(frozen=True, slots=True)
class DetectorCoordinateGeometry:
    """Internal elastic wave selected by one continuous detector coordinate."""

    column_px: FloatArray
    row_px: FloatArray
    kf_air_sample_Ainv: FloatArray
    kf_film_sample_Ainv: FloatArray
    q_sample_Ainv: FloatArray
    ray_distance_m: FloatArray
    q_surface_jacobian_Ainv2_per_px2: FloatArray
    ewald_residual_Ainv: FloatArray
    status: NDArray[np.str_]
    valid: NDArray[np.bool_] = field(init=False)

    def __post_init__(self) -> None:
        column = np.array(self.column_px, dtype=np.float64, copy=True, order="C")
        shape = column.shape
        if not np.all(np.isfinite(column)):
            raise ValueError("column_px must be finite")
        arrays = {
            "row_px": _float_array(self.row_px, shape, "row_px"),
            "kf_air_sample_Ainv": _float_array(
                self.kf_air_sample_Ainv,
                (*shape, 3),
                "kf_air_sample_Ainv",
            ),
            "kf_film_sample_Ainv": _float_array(
                self.kf_film_sample_Ainv,
                (*shape, 3),
                "kf_film_sample_Ainv",
            ),
            "q_sample_Ainv": _float_array(
                self.q_sample_Ainv,
                (*shape, 3),
                "q_sample_Ainv",
            ),
            "ray_distance_m": _float_array(self.ray_distance_m, shape, "ray_distance_m"),
            "q_surface_jacobian_Ainv2_per_px2": _float_array(
                self.q_surface_jacobian_Ainv2_per_px2,
                shape,
                "q_surface_jacobian_Ainv2_per_px2",
            ),
            "ewald_residual_Ainv": _float_array(
                self.ewald_residual_Ainv,
                shape,
                "ewald_residual_Ainv",
            ),
        }
        if np.any(arrays["q_surface_jacobian_Ainv2_per_px2"] < 0.0):
            raise ValueError("Q-surface Jacobian must be nonnegative")
        if np.any(arrays["ray_distance_m"] < 0.0):
            raise ValueError("ray_distance_m must be nonnegative")
        if np.any(arrays["ewald_residual_Ainv"] < 0.0):
            raise ValueError("Ewald residual must be nonnegative")
        status = _status_array(self.status, shape, "status")
        valid = status == ValidityCode.VALID
        valid.setflags(write=False)
        column.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True, slots=True)
class DetectorCoordinateIntensity:
    """A.e. density of the raw detector pushforward in pixel coordinates.

    The zero-width rod model has integrable caustic curves. ``caustic`` marks
    their pointwise singular set; exact pixel masses remain finite.
    """

    geometry: DetectorCoordinateGeometry
    rods: tuple[Rod, ...]
    branch: int
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    per_rod_inverse_branch_count: NDArray[np.int64]
    caustic: NDArray[np.bool_]
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, DetectorCoordinateGeometry):
            raise TypeError("geometry must be DetectorCoordinateGeometry")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        if self.branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        shape = self.geometry.column_px.shape
        per_rod = np.array(
            self.per_rod_density_A2_per_px2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if per_rod.shape != (*shape, len(rods)) or np.any(np.isnan(per_rod)):
            raise ValueError("per-rod detector density has the wrong shape or contains NaN")
        if np.any(per_rod < 0.0):
            raise ValueError("per-rod detector density must be nonnegative")
        total = np.array(self.density_A2_per_px2, dtype=np.float64, copy=True, order="C")
        if total.shape != shape or np.any(np.isnan(total)) or np.any(total < 0.0):
            raise ValueError("detector density must be nonnegative and contain no NaN")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        finite = np.isfinite(expected)
        scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        if not np.all(
            np.abs(total[finite] - expected[finite]) <= 1024.0 * np.finfo(np.float64).eps * scale
        ) or not np.array_equal(np.isinf(total), np.isinf(expected)):
            raise ValueError("detector density must equal the physical rod sum")
        counts = np.array(
            self.per_rod_inverse_branch_count,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        if counts.shape != (*shape, len(rods)) or np.any(counts < 0):
            raise ValueError("per-rod inverse branch counts have the wrong shape")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != (*shape, len(rods)):
            raise ValueError("caustic must have one flag per detector coordinate and rod")
        if np.any((~self.geometry.valid)[..., None] & (per_rod != 0.0)):
            raise ValueError("invalid detector coordinates cannot carry intensity")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate measure")
        for value in (per_rod, total, counts, caustic):
            value.setflags(write=False)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_density_A2_per_px2", per_rod)
        object.__setattr__(self, "density_A2_per_px2", total)
        object.__setattr__(self, "per_rod_inverse_branch_count", counts)
        object.__setattr__(self, "caustic", caustic)


@dataclass(frozen=True, slots=True)
class EwaldDirectionIntensity:
    """A.e. intrinsic coating density on internal-film outgoing directions.

    The zero-width rod model has integrable caustic curves. Exact positive
    caustic directions are marked and carry ``+inf``; finite solid-angle
    integrals require a separate quadrature.
    """

    outgoing_direction_sample: FloatArray
    q_sample_Ainv: FloatArray
    rods: tuple[Rod, ...]
    branch: int | None
    per_rod_density_A2_per_sr: FloatArray
    density_A2_per_sr: FloatArray
    per_rod_inverse_branch_count: NDArray[np.int64]
    caustic: NDArray[np.bool_]
    measure_id: str = "intrinsic_ewald_direction_density_A2_per_sr.v1"

    def __post_init__(self) -> None:
        direction = np.array(
            self.outgoing_direction_sample,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if direction.ndim < 1 or direction.shape[-1] != 3 or not np.all(np.isfinite(direction)):
            raise ValueError(
                "outgoing directions must be finite vectors with final dimension three"
            )
        shape = direction.shape[:-1]
        norm = np.linalg.norm(direction, axis=-1)
        if not np.allclose(
            norm,
            1.0,
            rtol=0.0,
            atol=4096.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("outgoing directions must be unit vectors")
        q_sample = _float_array(self.q_sample_Ainv, (*shape, 3), "q_sample_Ainv")
        rods = tuple(self.rods)
        if (
            not rods
            or not all(isinstance(rod, Rod) and rod.family_m != 0 for rod in rods)
            or len({(rod.h, rod.k) for rod in rods}) != len(rods)
        ):
            raise ValueError("rods must contain distinct non-specular physical rods")
        if self.branch not in {None, 1, 2}:
            raise ValueError("branch must be None, 1, or 2")
        per_rod = np.array(
            self.per_rod_density_A2_per_sr,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            per_rod.shape != (*shape, len(rods))
            or np.any(np.isnan(per_rod))
            or np.any(per_rod < 0.0)
        ):
            raise ValueError("per-rod direction density has invalid shape or values")
        total = np.array(self.density_A2_per_sr, dtype=np.float64, copy=True, order="C")
        if total.shape != shape or np.any(np.isnan(total)) or np.any(total < 0.0):
            raise ValueError("direction density has invalid shape or values")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        finite = np.isfinite(expected)
        scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        if not np.all(
            np.abs(total[finite] - expected[finite]) <= 1024.0 * np.finfo(np.float64).eps * scale
        ) or not np.array_equal(np.isinf(total), np.isinf(expected)):
            raise ValueError("direction density must equal the physical rod sum")
        counts = np.array(
            self.per_rod_inverse_branch_count,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if counts.shape != (*shape, len(rods)) or np.any(counts < 0):
            raise ValueError("inverse branch counts must align with directions and rods")
        if caustic.shape != (*shape, len(rods)):
            raise ValueError("caustic flags must align with directions and rods")
        if np.any(np.isinf(per_rod) & ~caustic):
            raise ValueError("only declared caustics may carry infinite direction density")
        if self.measure_id != "intrinsic_ewald_direction_density_A2_per_sr.v1":
            raise ValueError("unsupported intrinsic Ewald direction measure")
        for value in (direction, per_rod, total, counts, caustic):
            value.setflags(write=False)
        object.__setattr__(self, "outgoing_direction_sample", direction)
        object.__setattr__(self, "q_sample_Ainv", q_sample)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_density_A2_per_sr", per_rod)
        object.__setattr__(self, "density_A2_per_sr", total)
        object.__setattr__(self, "per_rod_inverse_branch_count", counts)
        object.__setattr__(self, "caustic", caustic)


@dataclass(frozen=True, slots=True)
class DetectorVisibleEwaldDirectionIntensity:
    """Intrinsic Ewald-direction density restricted to one active detector panel.

    Detector coordinates parameterize the visible internal-film sphere patch.
    The result contains all regular inverse mosaic preimages, including the
    nonzero ``m=0`` support. The collapsed direct ``Q=0`` root is absent.
    """

    geometry: DetectorCoordinateGeometry
    outgoing_direction_sample: FloatArray
    rods: tuple[Rod, ...]
    per_rod_density_A2_per_sr: FloatArray
    density_A2_per_sr: FloatArray
    per_rod_inverse_branch_count: NDArray[np.int64]
    caustic: NDArray[np.bool_]
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, DetectorCoordinateGeometry):
            raise TypeError("geometry must be DetectorCoordinateGeometry")
        shape = self.geometry.column_px.shape
        direction = np.array(
            self.outgoing_direction_sample,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if direction.shape != (*shape, 3) or not np.all(np.isfinite(direction)):
            raise ValueError("outgoing directions must align with detector coordinates")
        if np.any(self.geometry.valid):
            norm = np.linalg.norm(direction[self.geometry.valid], axis=-1)
            if not np.allclose(
                norm,
                1.0,
                rtol=0.0,
                atol=4096.0 * np.finfo(np.float64).eps,
            ):
                raise ValueError("detector-visible outgoing directions must be unit vectors")
        if np.any(direction[~self.geometry.valid] != 0.0):
            raise ValueError("invalid detector coordinates must use zero outgoing directions")

        rods = tuple(self.rods)
        if (
            not rods
            or not all(isinstance(rod, Rod) for rod in rods)
            or len({(rod.h, rod.k) for rod in rods}) != len(rods)
        ):
            raise ValueError("rods must contain distinct physical rods")
        per_rod = np.array(
            self.per_rod_density_A2_per_sr,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        total = np.array(self.density_A2_per_sr, dtype=np.float64, copy=True, order="C")
        if (
            per_rod.shape != (*shape, len(rods))
            or total.shape != shape
            or np.any(np.isnan(per_rod))
            or np.any(np.isnan(total))
            or np.any(per_rod < 0.0)
            or np.any(total < 0.0)
        ):
            raise ValueError("detector-visible direction density has invalid shape or values")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        finite = np.isfinite(expected)
        scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        if not np.all(
            np.abs(total[finite] - expected[finite]) <= 1024.0 * np.finfo(np.float64).eps * scale
        ) or not np.array_equal(np.isinf(total), np.isinf(expected)):
            raise ValueError("direction density must equal the physical rod sum")
        if np.any((~self.geometry.valid)[..., None] & (per_rod != 0.0)):
            raise ValueError("invalid detector coordinates cannot carry direction density")

        counts = np.array(
            self.per_rod_inverse_branch_count,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if counts.shape != (*shape, len(rods)) or np.any(counts < 0):
            raise ValueError("inverse branch counts must align with detector coordinates and rods")
        if caustic.shape != (*shape, len(rods)):
            raise ValueError("caustic flags must align with detector coordinates and rods")
        if np.any(np.isinf(per_rod) & ~caustic):
            raise ValueError("only declared caustics may carry infinite direction density")

        has_m0 = any(rod.family_m == 0 for rod in rods)
        gap = self.detector_visible_m0_q_gap_Ainv
        if has_m0:
            if gap is None or not isfinite(float(gap)) or float(gap) <= 0.0:
                raise ValueError("detector-visible m=0 requires a positive reciprocal support gap")
            gap = float(gap)
            visible_q_norm = np.linalg.norm(
                self.geometry.q_sample_Ainv[self.geometry.valid],
                axis=-1,
            )
            if np.any(visible_q_norm <= gap):
                raise ValueError("detector-visible m=0 violated its reciprocal support gap")
        elif gap is not None:
            raise ValueError("an m=0 reciprocal support gap requires an m=0 rod")
        if self.measure_id != ("detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1"):
            raise ValueError("unsupported detector-visible Ewald direction measure")

        for value in (direction, per_rod, total, counts, caustic):
            value.setflags(write=False)
        object.__setattr__(self, "outgoing_direction_sample", direction)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_density_A2_per_sr", per_rod)
        object.__setattr__(self, "density_A2_per_sr", total)
        object.__setattr__(self, "per_rod_inverse_branch_count", counts)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", gap)

    @property
    def detector_visible(self) -> BoolArray:
        return self.geometry.valid

    @property
    def detector_status(self) -> NDArray[np.str_]:
        return self.geometry.status


@dataclass(frozen=True, slots=True)
class DetectorStructureResponse:
    """Sparse fixed-geometry coefficients multiplying candidate rod strengths."""

    rods: tuple[Rod, ...]
    coordinate_valid: BoolArray
    term_coordinate_index: IntArray
    term_rod_index: IntArray
    term_L: FloatArray
    term_alpha_rad: FloatArray
    term_q_radial_squared_Ainv2: FloatArray
    term_q_normal_squared_Ainv2: FloatArray
    term_fixed_density_per_mosaic_density_per_strength_px2_inv: FloatArray
    term_fixed_density_per_strength_px2_inv: FloatArray
    term_root_sign: NDArray[np.int8]
    per_rod_caustic: BoolArray
    k_norm_Ainv: float
    root_policy: str = "all_retained_roots.v1"
    measure_id: str = "fixed_detector_density_per_structure_strength_px2_inv.v1"

    def __post_init__(self) -> None:
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one physical Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical line")
        coordinate_valid = np.array(self.coordinate_valid, dtype=np.bool_, copy=True, order="C")
        if coordinate_valid.ndim != 1:
            raise ValueError("coordinate_valid must be one-dimensional")
        coordinate_count = coordinate_valid.size
        term_coordinate = np.array(
            self.term_coordinate_index,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        term_rod = np.array(self.term_rod_index, dtype=np.int64, copy=True, order="C")
        term_l = np.array(self.term_L, dtype=np.float64, copy=True, order="C")
        term_alpha = np.array(self.term_alpha_rad, dtype=np.float64, copy=True, order="C")
        term_q_radial_squared = np.array(
            self.term_q_radial_squared_Ainv2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        term_q_normal_squared = np.array(
            self.term_q_normal_squared_Ainv2,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        term_fixed_per_mosaic = np.array(
            self.term_fixed_density_per_mosaic_density_per_strength_px2_inv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        term_fixed = np.array(
            self.term_fixed_density_per_strength_px2_inv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        root_sign = np.array(self.term_root_sign, dtype=np.int8, copy=True, order="C")
        term_shape = term_coordinate.shape
        if (
            term_coordinate.ndim != 1
            or term_rod.shape != term_shape
            or term_l.shape != term_shape
            or term_alpha.shape != term_shape
            or term_q_radial_squared.shape != term_shape
            or term_q_normal_squared.shape != term_shape
            or term_fixed_per_mosaic.shape != term_shape
            or term_fixed.shape != term_shape
            or root_sign.shape != term_shape
        ):
            raise ValueError("all sparse structure-response term arrays must align")
        if (
            np.any((term_coordinate < 0) | (term_coordinate >= coordinate_count))
            or np.any((term_rod < 0) | (term_rod >= len(rods)))
            or np.any(~coordinate_valid[term_coordinate])
        ):
            raise ValueError("structure-response terms must reference valid coordinates and rods")
        if (
            not np.all(np.isfinite(term_l))
            or not np.all(np.isfinite(term_alpha))
            or np.any((term_alpha < 0.0) | (term_alpha > np.pi))
            or not np.all(np.isfinite(term_q_radial_squared))
            or np.any(term_q_radial_squared < 0.0)
            or not np.all(np.isfinite(term_q_normal_squared))
            or np.any(term_q_normal_squared < 0.0)
            or not np.all(np.isfinite(term_fixed_per_mosaic))
            or np.any(term_fixed_per_mosaic < 0.0)
            or not np.all(np.isfinite(term_fixed))
            or np.any(term_fixed < 0.0)
            or np.any(~np.isin(root_sign, (-1, 0, 1)))
        ):
            raise ValueError("structure-response terms must be finite and physically nonnegative")
        caustic = np.array(self.per_rod_caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != (coordinate_count, len(rods)):
            raise ValueError("per_rod_caustic must align with coordinates and rods")
        k_norm = float(self.k_norm_Ainv)
        if not isfinite(k_norm) or k_norm <= 0.0:
            raise ValueError("k_norm_Ainv must be finite and positive")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("structure response requires all retained inverse roots")
        if self.measure_id != "fixed_detector_density_per_structure_strength_px2_inv.v1":
            raise ValueError("unsupported structure-response measure")
        for value in (
            coordinate_valid,
            term_coordinate,
            term_rod,
            term_l,
            term_alpha,
            term_q_radial_squared,
            term_q_normal_squared,
            term_fixed_per_mosaic,
            term_fixed,
            root_sign,
            caustic,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "coordinate_valid", coordinate_valid)
        object.__setattr__(self, "term_coordinate_index", term_coordinate)
        object.__setattr__(self, "term_rod_index", term_rod)
        object.__setattr__(self, "term_L", term_l)
        object.__setattr__(self, "term_alpha_rad", term_alpha)
        object.__setattr__(self, "term_q_radial_squared_Ainv2", term_q_radial_squared)
        object.__setattr__(self, "term_q_normal_squared_Ainv2", term_q_normal_squared)
        object.__setattr__(
            self,
            "term_fixed_density_per_mosaic_density_per_strength_px2_inv",
            term_fixed_per_mosaic,
        )
        object.__setattr__(self, "term_fixed_density_per_strength_px2_inv", term_fixed)
        object.__setattr__(self, "term_root_sign", root_sign)
        object.__setattr__(self, "per_rod_caustic", caustic)
        object.__setattr__(self, "k_norm_Ainv", k_norm)


@dataclass(frozen=True, slots=True)
class DetectorLatentIntensity:
    """Post-optical latent density carried by the mapped detector measure."""

    geometry: DetectorMappedGeometry
    intensity_status: IntensityStatus
    exit_amplitude: ComplexArray
    attenuation_weight: FloatArray
    optical_weight: FloatArray
    coating_intensity_density_A2_rad2_inv: FloatArray
    event_intensity_envelope: FloatArray
    scattering_polarization_weight: FloatArray
    source_phase_weight: float
    postoptical_density_A2_rad2_inv: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, DetectorMappedGeometry):
            raise TypeError("geometry must be DetectorMappedGeometry")
        status = IntensityStatus(self.intensity_status)
        if status is not IntensityStatus.INCLUDED:
            raise ValueError("DetectorLatentIntensity must carry INCLUDED intensity")
        shape = self.geometry.ewald_geometry.alpha_rad.shape
        exit_amplitude = _complex_array(self.exit_amplitude, shape, "exit_amplitude")
        arrays = {
            "attenuation_weight": _float_array(
                self.attenuation_weight, shape, "attenuation_weight"
            ),
            "optical_weight": _float_array(self.optical_weight, shape, "optical_weight"),
            "coating_intensity_density_A2_rad2_inv": _float_array(
                self.coating_intensity_density_A2_rad2_inv,
                shape,
                "coating_intensity_density_A2_rad2_inv",
            ),
            "event_intensity_envelope": _float_array(
                self.event_intensity_envelope,
                shape,
                "event_intensity_envelope",
            ),
            "scattering_polarization_weight": _float_array(
                self.scattering_polarization_weight,
                shape,
                "scattering_polarization_weight",
            ),
            "postoptical_density_A2_rad2_inv": _float_array(
                self.postoptical_density_A2_rad2_inv,
                shape,
                "postoptical_density_A2_rad2_inv",
            ),
        }
        if any(np.any(value < 0.0) for value in arrays.values()):
            raise ValueError("detector intensity factors must be nonnegative")
        source_phase_weight = float(self.source_phase_weight)
        if not isfinite(source_phase_weight) or source_phase_weight < 0.0:
            raise ValueError("source_phase_weight must be finite and nonnegative")
        expected = (
            arrays["coating_intensity_density_A2_rad2_inv"]
            * arrays["event_intensity_envelope"]
            * arrays["scattering_polarization_weight"]
            * arrays["optical_weight"]
            * source_phase_weight
        )
        expected = np.where(self.geometry.valid, expected, 0.0)
        scale = max(
            float(np.max(expected, initial=0.0)),
            np.finfo(np.float64).tiny,
        )
        tolerance = 1024.0 * np.finfo(np.float64).eps * scale
        if not np.allclose(
            arrays["postoptical_density_A2_rad2_inv"],
            expected,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("post-optical density must apply each declared factor exactly once")
        object.__setattr__(self, "intensity_status", status)
        object.__setattr__(self, "exit_amplitude", exit_amplitude)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_phase_weight", source_phase_weight)


@dataclass(frozen=True, slots=True)
class SpecularDetectorGeometry:
    """Geometry-only m=0 marker; no divergent intensity field is present."""

    geometry: DetectorMappedGeometry
    intensity_status: IntensityStatus = IntensityStatus.SPECULAR_INTENSITY_EXCLUDED

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, DetectorMappedGeometry):
            raise TypeError("geometry must be DetectorMappedGeometry")
        status = IntensityStatus(self.intensity_status)
        if status is not IntensityStatus.SPECULAR_INTENSITY_EXCLUDED:
            raise ValueError("specular detector geometry cannot carry intensity")
        object.__setattr__(self, "intensity_status", status)


@dataclass(frozen=True, slots=True)
class DetectorPixelMass:
    """Raw detector-native pushforward mass in angstrom squared per pixel."""

    image_A2: FloatArray
    rods: tuple[Rod, ...]
    branch: int
    per_rod_detector_mass_A2: FloatArray
    total_detector_mass_A2: float
    quadrature: DetectorQuadrature
    fold_refined_pixel_count: int = 0
    fold_refinement_l1_A2: float = 0.0
    fold_refinement_centroid_shift_px: float = 0.0
    adaptive_refined_pixel_count: int = 0
    adaptive_unresolved_pixel_count: int = 0
    sampled_invalid_pixel_count: int = 0
    coordinate_evaluation_count: int = 0
    estimated_l1_error_A2: float = 0.0
    execution_backend: str = "numpy_vectorized.v1"
    measure_id: str = "raw_detector_pixel_mass_A2.v1"
    sampled_valid_pixel_center: BoolArray | None = None

    @property
    def adaptive_tolerance_satisfied(self) -> bool:
        """Whether every adaptively selected pixel met the requested tolerance."""

        return (
            self.quadrature.method is PixelIntegrationMethod.ADAPTIVE_COMPILED
            and self.adaptive_unresolved_pixel_count == 0
        )

    def __post_init__(self) -> None:
        image = np.array(self.image_A2, dtype=np.float64, copy=True, order="C")
        if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError("image_A2 must be a finite nonnegative detector image")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        if self.branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        per_rod = _float_array(
            self.per_rod_detector_mass_A2,
            (len(rods),),
            "per_rod_detector_mass_A2",
        )
        if np.any(per_rod < 0.0):
            raise ValueError("per-rod detector mass must be nonnegative")
        total = float(self.total_detector_mass_A2)
        expected = fsum(per_rod)
        tolerance = 1024.0 * np.finfo(np.float64).eps * max(expected, np.finfo(np.float64).tiny)
        if not isfinite(total) or total < 0.0 or abs(total - expected) > tolerance:
            raise ValueError("total detector mass must equal the physical rod sum")
        if abs(fsum(image.flat) - total) > tolerance:
            raise ValueError("detector image must conserve total detector mass")
        if not isinstance(self.quadrature, DetectorQuadrature):
            raise TypeError("quadrature must be DetectorQuadrature")
        center_valid: BoolArray | None = None
        if self.sampled_valid_pixel_center is not None:
            center_valid = np.array(
                self.sampled_valid_pixel_center,
                dtype=np.bool_,
                copy=True,
                order="C",
            )
            if center_valid.shape != image.shape:
                raise ValueError("sampled_valid_pixel_center must match the detector image")
            center_valid.setflags(write=False)
        refined_count = (
            positive_integer(
                self.fold_refined_pixel_count,
                "fold_refined_pixel_count",
            )
            if self.fold_refined_pixel_count != 0
            else 0
        )
        if refined_count > image.size:
            raise ValueError("fold_refined_pixel_count exceeds the detector size")
        integer_diagnostics: dict[str, int] = {}
        for name in (
            "adaptive_refined_pixel_count",
            "adaptive_unresolved_pixel_count",
            "sampled_invalid_pixel_count",
            "coordinate_evaluation_count",
        ):
            supplied = getattr(self, name)
            value = positive_integer(supplied, name) if supplied != 0 else 0
            integer_diagnostics[name] = value
        for name in (
            "adaptive_refined_pixel_count",
            "adaptive_unresolved_pixel_count",
            "sampled_invalid_pixel_count",
        ):
            if integer_diagnostics[name] > image.size:
                raise ValueError(f"{name} exceeds the detector size")
        if (
            integer_diagnostics["adaptive_unresolved_pixel_count"]
            > integer_diagnostics["adaptive_refined_pixel_count"]
        ):
            raise ValueError("unresolved adaptive pixels must be a subset of refined pixels")
        refinement_l1 = float(self.fold_refinement_l1_A2)
        if not isfinite(refinement_l1) or refinement_l1 < 0.0:
            raise ValueError("fold_refinement_l1_A2 must be finite and nonnegative")
        centroid_shift = float(self.fold_refinement_centroid_shift_px)
        if not isfinite(centroid_shift) or centroid_shift < 0.0:
            raise ValueError("fold_refinement_centroid_shift_px must be finite and nonnegative")
        estimated_l1 = float(self.estimated_l1_error_A2)
        if not isfinite(estimated_l1) or estimated_l1 < 0.0:
            raise ValueError("estimated_l1_error_A2 must be finite and nonnegative")
        if self.execution_backend not in {
            "numpy_vectorized.v1",
            "numba_nogil_thread_tiles.v1",
            "numba_source_averaged.v1",
        }:
            raise ValueError("unsupported detector integration backend")
        if self.measure_id != "raw_detector_pixel_mass_A2.v1":
            raise ValueError("unsupported detector-pixel measure")
        image.setflags(write=False)
        object.__setattr__(self, "image_A2", image)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_detector_mass_A2", per_rod)
        object.__setattr__(self, "total_detector_mass_A2", total)
        object.__setattr__(self, "sampled_valid_pixel_center", center_valid)
        object.__setattr__(self, "fold_refined_pixel_count", refined_count)
        object.__setattr__(self, "fold_refinement_l1_A2", refinement_l1)
        object.__setattr__(self, "fold_refinement_centroid_shift_px", centroid_shift)
        for name, value in integer_diagnostics.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "estimated_l1_error_A2", estimated_l1)


@dataclass(frozen=True, slots=True)
class _MappedArrays:
    geometry: DetectorMappedGeometry
    exit_amplitude: ComplexArray
    kz_film_Ainv: ComplexArray
    propagation_direction: NDArray[np.int8]


def _validate_geometry_mapping_context(
    *,
    incident: IncidentTransportResult,
    instrument: CompiledInstrument,
    ki_sample_Ainv: ArrayLike,
    material: MaterialOptics | None = None,
    incident_state_index: int | None = None,
) -> tuple[FloatArray, float, int]:
    if not isinstance(incident, IncidentTransportResult):
        raise TypeError("incident must be IncidentTransportResult")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    if material is not None and not isinstance(material, MaterialOptics):
        raise TypeError("material must be MaterialOptics")
    states = incident.states
    if incident_state_index is None:
        if states.incident_state_id.size != 1:
            raise ValueError(
                "detector geometry mapping requires exactly one incident state unless "
                "incident_state_index is explicit"
            )
        state_index = 0
    else:
        if isinstance(incident_state_index, (bool, np.bool_)):
            raise TypeError("incident_state_index must be an integer")
        state_index = index(incident_state_index)
        if state_index < 0 or state_index >= states.incident_state_id.size:
            raise ValueError("incident_state_index lies outside the incident batch")
    if not states.valid[state_index]:
        raise ValueError("the incident state must be valid")
    if states.sample_geometry_revision != instrument.sample_geometry_revision:
        raise ValueError("incident and detector geometry sample revisions disagree")
    if material is not None and states.material_revision != material.material_revision:
        raise ValueError("incident and material revisions disagree")
    ki_sample = np.asarray(ki_sample_Ainv, dtype=np.float64)
    if ki_sample.shape != (3,) or not np.all(np.isfinite(ki_sample)):
        raise ValueError("ki_sample_Ainv must be a finite three-vector")
    scale = max(float(np.linalg.norm(ki_sample)), 1.0)
    if not np.allclose(
        ki_sample,
        states.k_film_phase_sample_Ainv[state_index],
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * scale,
    ):
        raise ValueError("ki_sample_Ainv must match the canonical incident film-phase vector")
    return (
        ki_sample,
        2.0 * np.pi / float(states.wavelength_A[state_index]),
        state_index,
    )


def evaluate_detector_coordinates_geometry(
    column_px: ArrayLike,
    row_px: ArrayLike,
    *,
    incident: IncidentTransportResult,
    instrument: CompiledInstrument,
    ki_sample_Ainv: ArrayLike,
    include_surface_jacobian: bool = True,
    incident_state_index: int | None = None,
) -> DetectorCoordinateGeometry:
    """Map native detector coordinates to internal elastic Q without intensity work."""

    ki_sample, air_k0_Ainv, state_index = _validate_geometry_mapping_context(
        incident=incident,
        instrument=instrument,
        ki_sample_Ainv=ki_sample_Ainv,
        incident_state_index=incident_state_index,
    )
    supplied_column = np.asarray(column_px)
    supplied_row = np.asarray(row_px)
    if (np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0)) or (
        np.iscomplexobj(supplied_row) and np.any(supplied_row.imag != 0.0)
    ):
        raise ValueError("detector coordinates must be real")
    column, row = np.broadcast_arrays(
        np.asarray(supplied_column.real, dtype=np.float64),
        np.asarray(supplied_row.real, dtype=np.float64),
    )
    if not np.all(np.isfinite(column)) or not np.all(np.isfinite(row)):
        raise ValueError("detector coordinates must be finite")
    shape = column.shape
    flat_column = column.reshape(-1)
    flat_row = row.reshape(-1)
    size = flat_column.size
    rows, columns = instrument.detector_shape_rc
    inside = (
        (flat_column >= -0.5)
        & (flat_column <= columns - 0.5)
        & (flat_row >= -0.5)
        & (flat_row <= rows - 0.5)
    )
    status = np.full(size, ValidityCode.OUTSIDE_SUPPORT.value, dtype="U32")
    point_lab = _detector_coordinates_to_lab_points(flat_column, flat_row, instrument)
    origin_lab = incident.states.sample_intersection_lab_m[state_index]
    displacement_lab = point_lab - origin_lab
    distance_m = np.linalg.norm(displacement_lab, axis=1)
    nonzero = inside & (distance_m > 0.0)
    status[inside & ~nonzero] = ValidityCode.NO_SOLUTION.value
    direction_lab = np.zeros((size, 3), dtype=np.float64)
    direction_lab[nonzero] = displacement_lab[nonzero] / distance_m[nonzero, None]
    incidence_cosine = _detector_incidence_cosine(direction_lab, instrument)
    detector_parallel = nonzero & (np.abs(incidence_cosine) <= _DETECTOR_INCIDENCE_COSINE_TOL)
    status[detector_parallel] = ValidityCode.PARALLEL.value
    back_facing = nonzero & (incidence_cosine < -_DETECTOR_INCIDENCE_COSINE_TOL)
    status[back_facing] = ValidityCode.BACKWARD.value
    front_facing = nonzero & (incidence_cosine > _DETECTOR_INCIDENCE_COSINE_TOL)
    kf_air_sample = np.zeros((size, 3), dtype=np.float64)
    kf_air_sample[front_facing] = instrument.sample_from_lab.apply_vector(
        air_k0_Ainv * direction_lab[front_facing]
    )
    top_exit = front_facing & (kf_air_sample[:, 2] > 0.0)
    status[front_facing & (kf_air_sample[:, 2] < 0.0)] = ValidityCode.BACKWARD.value
    status[front_facing & (kf_air_sample[:, 2] == 0.0)] = ValidityCode.PARALLEL.value

    incident_norm = float(np.linalg.norm(ki_sample))
    parallel_squared = np.einsum(
        "ij,ij->i", kf_air_sample[:, :2], kf_air_sample[:, :2], optimize=True
    )
    normal_squared = incident_norm * incident_norm - parallel_squared
    reachable = top_exit & (normal_squared > 0.0)
    status[top_exit & (normal_squared == 0.0)] = ValidityCode.PARALLEL.value
    status[top_exit & (normal_squared < 0.0)] = ValidityCode.NON_PROPAGATING.value
    status[reachable] = ValidityCode.VALID.value
    kf_film = np.zeros((size, 3), dtype=np.float64)
    kf_film[reachable, :2] = kf_air_sample[reachable, :2]
    kf_film[reachable, 2] = np.sqrt(normal_squared[reachable])
    q_sample = np.zeros((size, 3), dtype=np.float64)
    q_sample[reachable] = kf_film[reachable] - ki_sample

    q_surface_jacobian = np.zeros(size, dtype=np.float64)
    if include_surface_jacobian and np.any(reachable):
        column_step_lab_m = (
            instrument.lab_from_detector.rotation[:, 0] * instrument.detector_column_pitch_m
        )
        row_step_lab_m = (
            instrument.lab_from_detector.rotation[:, 1] * instrument.detector_row_pitch_m
        )
        selected_direction = direction_lab[reachable]
        selected_distance = distance_m[reachable]
        d_direction_column_lab = (
            column_step_lab_m
            - selected_direction * (selected_direction @ column_step_lab_m)[:, None]
        ) / selected_distance[:, None]
        d_direction_row_lab = (
            row_step_lab_m - selected_direction * (selected_direction @ row_step_lab_m)[:, None]
        ) / selected_distance[:, None]
        d_kair_column_sample = instrument.sample_from_lab.apply_vector(
            air_k0_Ainv * d_direction_column_lab
        )
        d_kair_row_sample = instrument.sample_from_lab.apply_vector(
            air_k0_Ainv * d_direction_row_lab
        )
        selected_kf = kf_film[reachable]
        d_kfilm_column = d_kair_column_sample.copy()
        d_kfilm_row = d_kair_row_sample.copy()
        d_kfilm_column[:, 2] = (
            -np.einsum(
                "ij,ij->i",
                selected_kf[:, :2],
                d_kair_column_sample[:, :2],
                optimize=True,
            )
            / selected_kf[:, 2]
        )
        d_kfilm_row[:, 2] = (
            -np.einsum(
                "ij,ij->i",
                selected_kf[:, :2],
                d_kair_row_sample[:, :2],
                optimize=True,
            )
            / selected_kf[:, 2]
        )
        q_surface_jacobian[reachable] = np.linalg.norm(
            np.cross(d_kfilm_column, d_kfilm_row), axis=1
        )

    ewald_residual = np.zeros(size, dtype=np.float64)
    ewald_residual[reachable] = np.abs(
        np.linalg.norm(q_sample[reachable] + ki_sample, axis=1) - incident_norm
    )
    return DetectorCoordinateGeometry(
        column_px=column,
        row_px=row,
        kf_air_sample_Ainv=kf_air_sample.reshape((*shape, 3)),
        kf_film_sample_Ainv=kf_film.reshape((*shape, 3)),
        q_sample_Ainv=q_sample.reshape((*shape, 3)),
        ray_distance_m=distance_m.reshape(shape),
        q_surface_jacobian_Ainv2_per_px2=q_surface_jacobian.reshape(shape),
        ewald_residual_Ainv=ewald_residual.reshape(shape),
        status=status.reshape(shape),
    )


def _map_ewald_geometry_arrays(
    geometry: EwaldLatentGeometry,
    *,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    incident_state_index: int | None = None,
) -> _MappedArrays:
    """Apply canonical exit transport and native detector projection to Ewald geometry."""

    if not isinstance(geometry, EwaldLatentGeometry):
        raise TypeError("geometry must be EwaldLatentGeometry")
    if not isinstance(incident, IncidentTransportResult):
        raise TypeError("incident must be IncidentTransportResult")
    states = incident.states
    if isinstance(incident_state_index, (bool, np.bool_)):
        raise TypeError("incident_state_index must be an integer")
    state_index = 0 if incident_state_index is None else index(incident_state_index)
    ki_sample, air_k0_Ainv, state_index = _validate_geometry_mapping_context(
        incident=incident,
        material=material,
        instrument=instrument,
        ki_sample_Ainv=states.k_film_phase_sample_Ainv[state_index],
        incident_state_index=incident_state_index,
    )
    root_valid = geometry.valid.reshape(-1)
    q_sample = geometry.q_sample_Ainv.reshape(-1, 3)
    kf_sample = geometry.kf_sample_Ainv.reshape(-1, 3)
    scale = max(float(np.linalg.norm(ki_sample)), 1.0)
    if np.any(root_valid) and not np.allclose(
        kf_sample[root_valid],
        q_sample[root_valid] + ki_sample,
        rtol=0.0,
        atol=512.0 * np.finfo(np.float64).eps * scale,
    ):
        raise ValueError("Ewald geometry does not satisfy kf = ki + Q for this incident state")
    computed_residual = np.abs(np.linalg.norm(kf_sample, axis=1) - np.linalg.norm(ki_sample))
    supplied_residual = geometry.ewald_residual_Ainv.reshape(-1)
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    if np.any(root_valid) and (
        np.any(computed_residual[root_valid] > tolerance)
        or not np.allclose(
            supplied_residual[root_valid],
            computed_residual[root_valid],
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("regular Ewald geometry must satisfy the elastic residual tolerance")

    shape = geometry.alpha_rad.shape
    size = geometry.alpha_rad.size
    kf_film = kf_sample
    exit_status = np.full(size, ValidityCode.NO_SOLUTION.value, dtype="U32")
    negative = root_valid & (kf_film[:, 2] < 0.0)
    parallel = root_valid & (kf_film[:, 2] == 0.0)
    eligible = root_valid & (kf_film[:, 2] > 0.0)
    exit_status[negative] = ValidityCode.BACKWARD.value
    exit_status[parallel] = ValidityCode.PARALLEL.value

    kf_air_sample = np.zeros((size, 3), dtype=np.float64)
    exit_amplitude = np.zeros(size, dtype=np.complex128)
    kz_film = np.zeros(size, dtype=np.complex128)
    propagation_direction = np.zeros(size, dtype=np.int8)
    eligible_rows = np.flatnonzero(eligible)
    if eligible_rows.size:
        wavelengths = np.full(eligible_rows.size, states.wavelength_A[state_index])
        modes = _solve_exit_mode_arrays(kf_film[eligible_rows], wavelengths, material)
        exit_status[eligible_rows] = modes.status
        valid_mode = modes.status == ValidityCode.VALID
        valid_exit_rows = eligible_rows[valid_mode]
        kf_air_sample[valid_exit_rows] = modes.k_air_phase_sample_Ainv[valid_mode]
        exit_amplitude[valid_exit_rows] = modes.exit_amplitude[valid_mode]
        kz_film[valid_exit_rows] = modes.kz_film_Ainv[valid_mode]
        propagation_direction[valid_exit_rows] = modes.propagation_direction[valid_mode]

    exit_valid = exit_status == ValidityCode.VALID
    kf_air_lab = np.zeros((size, 3), dtype=np.float64)
    kf_air_lab[exit_valid] = instrument.lab_from_sample.apply_vector(kf_air_sample[exit_valid])
    detector_status = exit_status.copy()
    column_px = np.zeros(size, dtype=np.float64)
    row_px = np.zeros(size, dtype=np.float64)
    ray_distance_m = np.zeros(size, dtype=np.float64)
    pixel_solid_angle_sr = np.zeros(size, dtype=np.float64)
    exit_rows = np.flatnonzero(exit_valid)
    if exit_rows.size:
        origin = np.broadcast_to(
            states.sample_intersection_lab_m[state_index],
            (exit_rows.size, 3),
        )
        projection = _project_detector_rays(
            origin,
            kf_air_lab[exit_rows] / air_k0_Ainv,
            instrument,
        )
        detector_status[exit_rows] = projection.status
        column_px[exit_rows] = projection.column_px
        row_px[exit_rows] = projection.row_px
        ray_distance_m[exit_rows] = projection.ray_distance_m
        pixel_solid_angle_sr[exit_rows] = projection.pixel_solid_angle_sr
    mapped = DetectorMappedGeometry(
        ewald_geometry=geometry,
        kf_air_sample_Ainv=kf_air_sample.reshape((*shape, 3)),
        kf_air_lab_Ainv=kf_air_lab.reshape((*shape, 3)),
        column_px=column_px.reshape(shape),
        row_px=row_px.reshape(shape),
        ray_distance_m=ray_distance_m.reshape(shape),
        pixel_solid_angle_sr=pixel_solid_angle_sr.reshape(shape),
        exit_status=exit_status.reshape(shape),
        detector_status=detector_status.reshape(shape),
    )
    for value in (exit_amplitude, kz_film, propagation_direction):
        value.setflags(write=False)
    return _MappedArrays(
        geometry=mapped,
        exit_amplitude=exit_amplitude.reshape(shape),
        kz_film_Ainv=kz_film.reshape(shape),
        propagation_direction=propagation_direction.reshape(shape),
    )


def map_ewald_geometry_to_detector(
    geometry: EwaldLatentGeometry,
    *,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    incident_state_index: int | None = None,
) -> DetectorMappedGeometry:
    """Map exact Ewald geometry without intensity, mosaic, raster, or pixel work."""

    return _map_ewald_geometry_arrays(
        geometry,
        incident=incident,
        material=material,
        instrument=instrument,
        incident_state_index=incident_state_index,
    ).geometry


def _compile_detector_state(
    *,
    bragg_config: BraggSpaceConfig,
    strength_model: Bi2X3FiniteStackStrength,
    ki_sample_Ainv: ArrayLike,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    rods: tuple[Rod, ...],
    incident_state_index: int,
    source_phase_weight: float,
    intensity_envelope: SampleQIntensityEnvelope,
    specular_stitch: CompiledParrattStitch | None = None,
    packed_structure: tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        ComplexArray,
        int,
        float,
        float,
        float,
    ]
    | None = None,
) -> CompiledDetectorState:
    """Pack one canonical incident row for the shared compiled point kernel."""

    if isinstance(incident_state_index, (bool, np.bool_)):
        raise TypeError("incident_state_index must be an integer")
    state_index = index(incident_state_index)
    states = incident.states
    if state_index < 0 or state_index >= states.incident_state_id.size:
        raise ValueError("incident_state_index lies outside the incident batch")
    if not states.valid[state_index]:
        raise ValueError("the compiled incident state must be valid")
    if not isinstance(bragg_config, BraggSpaceConfig):
        raise TypeError("bragg_config must be BraggSpaceConfig")
    if not isinstance(strength_model, Bi2X3FiniteStackStrength):
        raise TypeError("compiled integration requires the accepted Bi2X3FiniteStackStrength model")
    if not isinstance(intensity_envelope, SampleQIntensityEnvelope):
        raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
    parent_codes = {Parent.TWO_H: 0, Parent.THREE_R: 1}
    try:
        stacking_parent_code = parent_codes[strength_model.parent]
    except KeyError as error:
        raise ValueError(
            f"compiled integration does not implement stacking parent {strength_model.parent.value}"
        ) from error
    basis_scale = max(float(np.linalg.norm(bragg_config.reciprocal_basis_Ainv)), 1.0)
    if not np.allclose(
        strength_model.reciprocal_basis_Ainv,
        bragg_config.reciprocal_basis_Ainv,
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * basis_scale,
    ):
        raise ValueError("strength-model and Bragg-space reciprocal bases do not match")
    mosaic = bragg_config.mosaic
    if mosaic.zero_tilt_probability_mass != 0.0:
        raise ValueError("compiled integration does not support zero-tilt atoms")
    wavelength_A = float(states.wavelength_A[state_index])
    air_k0_Ainv = 2.0 * np.pi / wavelength_A
    ki_sample = np.asarray(ki_sample_Ainv, dtype=np.float64)
    if ki_sample.shape != (3,) or not np.all(np.isfinite(ki_sample)):
        raise ValueError("ki_sample_Ainv must be finite with shape (3,)")
    scale = max(float(np.linalg.norm(ki_sample)), 1.0)
    if not np.allclose(
        ki_sample,
        states.k_film_phase_sample_Ainv[state_index],
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * scale,
    ):
        raise ValueError("ki_sample_Ainv must match the selected canonical incident state")
    if not np.isclose(
        bragg_config.k_norm_Ainv,
        air_k0_Ainv,
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * max(air_k0_Ainv, 1.0),
    ):
        raise ValueError("Bragg strength must use the selected incident air wavelength")
    if packed_structure is None:
        packed_structure = pack_bi2se3_two_h_structure(
            strength_model,
            wavelength_A=wavelength_A,
        )
    (
        atom_offsets,
        atom_properties,
        f0_parameters,
        anomalous,
        layers,
        normalization_divisor,
        u_radial_A2,
        u_normal_A2,
    ) = packed_structure

    detector_rotation = instrument.lab_from_detector.rotation
    column_step_lab = detector_rotation[:, 0] * instrument.detector_column_pitch_m
    row_step_lab = detector_rotation[:, 1] * instrument.detector_row_pitch_m
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    detector_zero_lab = (
        instrument.lab_from_detector.translation_m
        - reference_column * column_step_lab
        - reference_row * row_step_lab
    )
    basis = bragg_config.reciprocal_basis_Ainv
    mean_axis, tilt_axis = mosaic_axes(basis)
    reference_axis = np.cross(tilt_axis, mean_axis)
    crystal_from_local = np.column_stack((reference_axis, tilt_axis, mean_axis))
    crystal_to_sample = bragg_config.crystal_to_sample
    sample_from_local = crystal_to_sample @ crystal_from_local
    rod_hk_population = np.asarray(
        [(rod.h, rod.k, rod.population) for rod in rods],
        dtype=np.float64,
    )
    inplane_angle = 2.0 * np.pi * (rod_hk_population[:, :2] @ atom_offsets[:, :2].T)
    rod_atom_inplane_factor = np.cos(inplane_angle) + 1j * np.sin(inplane_angle)
    rod_parallel_crystal = np.asarray(
        [rod.h * basis[:, 0] + rod.k * basis[:, 1] for rod in rods],
        dtype=np.float64,
    )
    rod_parallel_local = rod_parallel_crystal @ crystal_from_local
    axial_offset = rod_parallel_crystal @ mean_axis
    perpendicular = rod_parallel_crystal - axial_offset[:, None] * mean_axis
    half_width = np.sqrt(
        np.maximum(
            0.0,
            (2.0 * bragg_config.k_norm_Ainv) ** 2
            - np.einsum("ij,ij->i", perpendicular, perpendicular),
        )
    )
    rod_u_bounds = np.column_stack((-axial_offset - half_width, -axial_offset + half_width))
    rod_inverse_constants = np.column_stack(
        (
            np.abs(rod_parallel_local[:, 1]),
            np.hypot(rod_parallel_local[:, 0], rod_parallel_local[:, 1]),
            np.einsum("ij,ij->i", rod_parallel_local, rod_parallel_local),
            1024.0
            * np.finfo(np.float64).eps
            * np.maximum.reduce(
                (
                    np.abs(rod_u_bounds[:, 0]),
                    np.abs(rod_u_bounds[:, 1]),
                    np.ones(len(rods), dtype=np.float64),
                )
            ),
        )
    )
    material_index = int(np.searchsorted(material.wavelength_A, wavelength_A))
    if (
        material_index >= material.wavelength_A.size
        or material.wavelength_A[material_index] != wavelength_A
    ):
        raise ValueError("material does not contain the exact incident wavelength")
    incident_direction = -1 if states.direction_sample[state_index, 2] < 0.0 else 1
    incident_decay = float(
        mode_decay_constant(
            states.kz_film_Ainv[state_index],
            incident_direction,
        )
    )
    if specular_stitch is not None and not isinstance(specular_stitch, CompiledParrattStitch):
        raise TypeError("specular_stitch must be CompiledParrattStitch")
    if specular_stitch is not None and (
        specular_stitch.film_refractive_index != complex(material.n_complex[material_index])
        or specular_stitch.film_thickness_A != instrument.film_thickness_A
    ):
        raise ValueError("compiled Parratt stitch disagrees with detector film optics or thickness")
    stitch_values = (
        {
            "specular_stitch_code": 0,
            "specular_substrate_refractive_index": 1.0 + 0.0j,
            "specular_top_roughness_A": 0.0,
            "specular_bottom_roughness_A": 0.0,
            "specular_qc_Ainv": 0.0,
            "specular_zero_strength_A2": 0.0,
            "specular_scale_factor": 0.0,
            "specular_blend_lower_q_over_qc": 0.0,
            "specular_blend_upper_q_over_qc": 0.0,
        }
        if specular_stitch is None
        else {
            "specular_stitch_code": specular_stitch.interface_code,
            "specular_substrate_refractive_index": (specular_stitch.substrate_refractive_index),
            "specular_top_roughness_A": specular_stitch.top_roughness_A,
            "specular_bottom_roughness_A": specular_stitch.bottom_roughness_A,
            "specular_qc_Ainv": specular_stitch.qc_Ainv,
            "specular_zero_strength_A2": specular_stitch.zero_strength_A2,
            "specular_scale_factor": specular_stitch.dimensionless_scale_factor,
            "specular_blend_lower_q_over_qc": specular_stitch.blend_bounds_q_over_qc[0],
            "specular_blend_upper_q_over_qc": specular_stitch.blend_bounds_q_over_qc[1],
        }
    )
    return CompiledDetectorState(
        detector_zero_lab_m=np.ascontiguousarray(detector_zero_lab),
        detector_column_step_lab_m=np.ascontiguousarray(column_step_lab),
        detector_row_step_lab_m=np.ascontiguousarray(row_step_lab),
        detector_pixel_area_vector_lab_m2=np.ascontiguousarray(
            np.cross(column_step_lab, row_step_lab)
        ),
        ray_origin_lab_m=np.ascontiguousarray(states.sample_intersection_lab_m[state_index]),
        sample_from_lab=np.ascontiguousarray(instrument.sample_from_lab.rotation),
        ki_film_sample_Ainv=np.ascontiguousarray(states.k_film_phase_sample_Ainv[state_index]),
        internal_k_Ainv=float(np.linalg.norm(states.k_film_phase_sample_Ainv[state_index])),
        air_k0_Ainv=air_k0_Ainv,
        refractive_index=complex(material.n_complex[material_index]),
        entrance_amplitude=complex(states.entrance_amplitude[state_index]),
        incident_decay_Ainv=incident_decay,
        film_thickness_A=instrument.film_thickness_A,
        detector_path_linear_attenuation_m_inv=(
            detector_path_linear_attenuation_at_wavelength_m_inv(
                instrument,
                wavelength_A,
            )
        ),
        **stitch_values,
        source_phase_weight=source_phase_weight,
        polarization_model_code=polarization_model_code(states.polarization_state_id[state_index]),
        sample_from_local=np.ascontiguousarray(sample_from_local),
        rod_hk_population=rod_hk_population,
        rod_parallel_local_Ainv=rod_parallel_local,
        rod_u_bounds_Ainv=rod_u_bounds,
        rod_inverse_constants=rod_inverse_constants,
        b3_norm_Ainv=float(np.linalg.norm(basis[:, 2])),
        gaussian_sigma_rad=mosaic.gaussian_sigma_rad,
        lorentzian_hwhm_rad=mosaic.lorentzian_half_width_rad,
        lorentzian_probability=mosaic.lorentzian_probability,
        atom_fractional_offset=atom_offsets,
        atom_occupancy_element=atom_properties,
        rod_atom_inplane_factor=rod_atom_inplane_factor,
        u_radial_A2=u_radial_A2,
        u_normal_A2=u_normal_A2,
        intensity_envelope_u_radial_A2=intensity_envelope.u_radial_A2,
        intensity_envelope_u_normal_A2=intensity_envelope.u_normal_A2,
        f0_parameters=f0_parameters,
        anomalous_factor_e=anomalous,
        layers=layers,
        stacking_parent_code=stacking_parent_code,
        shared_disorder_epsilon=strength_model.shared_disorder_epsilon,
        normalization_divisor=normalization_divisor,
    )


class DetectorEwaldMeasure:
    """Map one selected incident-state Ewald coating to the active detector panel.

    This class defines the pre-binned pushforward. Its detector-coordinate
    density contains the required coordinate Jacobian, but detector solid
    angle is never applied again as a physical acceptance factor.
    """

    __slots__ = (
        "_air_k0_Ainv",
        "_coating",
        "_crystal_from_local",
        "_crystal_to_sample",
        "_detector_path_linear_attenuation_m_inv",
        "_incident",
        "_incident_state_index",
        "_instrument",
        "_intensity_envelope",
        "_material",
        "_polarization_model_id",
        "_rod_catalog_revision",
        "_source_phase_weight",
    )

    def __init__(
        self,
        *,
        coating: ContinuousEwaldCoating,
        incident: IncidentTransportResult,
        material: MaterialOptics,
        instrument: CompiledInstrument,
        rod_catalog_revision: str | None = None,
        phase_population_weight: float = 1.0,
        polarization_weight: float = 1.0,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
        incident_state_index: int | None = None,
    ) -> None:
        if not isinstance(coating, ContinuousEwaldCoating):
            raise TypeError("coating must be ContinuousEwaldCoating")
        if not isinstance(incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        envelope = SampleQIntensityEnvelope() if intensity_envelope is None else intensity_envelope
        if not isinstance(envelope, SampleQIntensityEnvelope):
            raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
        if rod_catalog_revision is not None and (
            not isinstance(rod_catalog_revision, str) or not rod_catalog_revision
        ):
            raise ValueError("rod_catalog_revision must be nonempty when supplied")
        if coating.root_tolerance_rel != 0.0:
            raise ValueError(
                "detector measure requires strict Ewald root classification without a finite "
                "tangent dead band"
            )
        states = incident.states
        if incident_state_index is None:
            if states.incident_state_id.size != 1:
                raise ValueError(
                    "DetectorEwaldMeasure requires exactly one incident state unless "
                    "incident_state_index is explicit"
                )
            state_index = 0
        else:
            if isinstance(incident_state_index, (bool, np.bool_)):
                raise TypeError("incident_state_index must be an integer")
            state_index = index(incident_state_index)
            if state_index < 0 or state_index >= states.incident_state_id.size:
                raise ValueError("incident_state_index lies outside the incident batch")
        if not states.valid[state_index]:
            raise ValueError("the incident state must be valid")
        polarization_model_id = states.polarization_state_id[state_index]
        polarization_model_code(polarization_model_id)
        phase_weight = float(phase_population_weight)
        polarization = float(polarization_weight)
        if not isfinite(phase_weight) or phase_weight < 0.0:
            raise ValueError("phase_population_weight must be finite and nonnegative")
        if not isfinite(polarization) or polarization < 0.0:
            raise ValueError("polarization_weight must be finite and nonnegative")
        ki_scale = max(float(np.linalg.norm(coating.ki_sample_Ainv)), 1.0)
        if not np.allclose(
            coating.ki_sample_Ainv,
            states.k_film_phase_sample_Ainv[state_index],
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * ki_scale,
        ):
            raise ValueError("coating ki must be the canonical incident film-phase vector")
        air_k0_Ainv = 2.0 * np.pi / states.wavelength_A[state_index]
        if not np.isclose(
            coating.bragg_space.config.k_norm_Ainv,
            air_k0_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * max(air_k0_Ainv, 1.0),
        ):
            raise ValueError("Bragg strength must use the incident air wavelength")
        basis = coating.bragg_space.config.reciprocal_basis_Ainv
        mean_axis, tilt_axis = mosaic_axes(basis)
        reference_axis = np.cross(tilt_axis, mean_axis)
        crystal_from_local = np.column_stack((reference_axis, tilt_axis, mean_axis))
        crystal_from_local.setflags(write=False)
        crystal_to_sample = coating.bragg_space.config.crystal_to_sample
        source_phase_weight = float(
            states.source_weight[state_index]
            * states.footprint_acceptance[state_index]
            * incident_illuminated_path_weight(states.direction_sample[state_index])
            * (phase_weight * polarization)
        )
        detector_path_linear_attenuation_m_inv = (
            detector_path_linear_attenuation_at_wavelength_m_inv(
                instrument,
                states.wavelength_A[state_index],
            )
        )
        object.__setattr__(self, "_coating", coating)
        object.__setattr__(self, "_incident", incident)
        object.__setattr__(self, "_incident_state_index", state_index)
        object.__setattr__(self, "_material", material)
        object.__setattr__(self, "_polarization_model_id", polarization_model_id)
        object.__setattr__(self, "_instrument", instrument)
        object.__setattr__(self, "_intensity_envelope", envelope)
        object.__setattr__(self, "_rod_catalog_revision", rod_catalog_revision)
        object.__setattr__(self, "_air_k0_Ainv", air_k0_Ainv)
        object.__setattr__(self, "_crystal_from_local", crystal_from_local)
        object.__setattr__(self, "_crystal_to_sample", crystal_to_sample)
        object.__setattr__(
            self,
            "_detector_path_linear_attenuation_m_inv",
            detector_path_linear_attenuation_m_inv,
        )
        object.__setattr__(self, "_source_phase_weight", source_phase_weight)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("DetectorEwaldMeasure is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("DetectorEwaldMeasure is immutable")

    @property
    def coating(self) -> ContinuousEwaldCoating:
        return self._coating

    @property
    def incident(self) -> IncidentTransportResult:
        return self._incident

    @property
    def incident_state_index(self) -> int:
        return self._incident_state_index

    @property
    def instrument(self) -> CompiledInstrument:
        return self._instrument

    @property
    def intensity_envelope(self) -> SampleQIntensityEnvelope:
        return self._intensity_envelope

    @property
    def rod_catalog_revision(self) -> str | None:
        """Configured physical-rod authority, when this low-level measure has one."""

        return self._rod_catalog_revision

    def _require_physical_intensity_source(self) -> None:
        require_physical_intensity_source_model(self._incident.states.source_sampling_model_id)

    def map_ewald_geometry(self, geometry: EwaldLatentGeometry) -> DetectorMappedGeometry:
        """Map already constructed exact Ewald geometry to the active detector."""

        return self._map_geometry(geometry).geometry

    def _map_geometry(self, geometry: EwaldLatentGeometry) -> _MappedArrays:
        return _map_ewald_geometry_arrays(
            geometry,
            incident=self._incident,
            material=self._material,
            instrument=self._instrument,
            incident_state_index=self._incident_state_index,
        )

    def _event_scattering_polarization(
        self,
        kf_air_sample_Ainv: FloatArray,
        valid: BoolArray,
    ) -> FloatArray:
        result = np.zeros(valid.shape, dtype=np.float64)
        if np.any(valid):
            result[valid] = scattering_polarization_weight(
                self._incident.states.direction_sample[self._incident_state_index],
                kf_air_sample_Ainv[valid] / self._air_k0_Ainv,
                model_id=self._polarization_model_id,
            )
        result.setflags(write=False)
        return result

    def map_latent(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> DetectorLatentIntensity:
        """Map arbitrary continuous latent coordinates onto the active detector."""

        self._require_physical_intensity_source()

        intensity = self._coating.evaluate_latent(
            rod=rod,
            branch=branch,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        mapped = self._map_geometry(intensity.geometry)
        shape = intensity.geometry.alpha_rad.shape
        exit_valid = mapped.geometry.exit_status == ValidityCode.VALID
        attenuation = np.zeros(shape, dtype=np.float64)
        optical = np.zeros(shape, dtype=np.float64)
        if np.any(exit_valid):
            incident_direction = (
                -1
                if self._incident.states.direction_sample[self._incident_state_index, 2] < 0.0
                else 1
            )
            incident_kappa = mode_decay_constant(
                self._incident.states.kz_film_Ainv[self._incident_state_index],
                incident_direction,
            )
            exit_kappa = mode_decay_constant(
                mapped.kz_film_Ainv[exit_valid],
                mapped.propagation_direction[exit_valid],
            )
            attenuation[exit_valid] = uniform_depth_attenuation(
                incident_kappa,
                exit_kappa,
                self._instrument.film_thickness_A,
            )
            optical[exit_valid] = scalar_optical_weight(
                self._incident.states.entrance_amplitude[self._incident_state_index],
                mapped.exit_amplitude[exit_valid],
                attenuation[exit_valid],
            )
            optical[exit_valid] *= external_path_attenuation(
                self._detector_path_linear_attenuation_m_inv,
                mapped.geometry.ray_distance_m[exit_valid],
            )
        event_envelope = self._intensity_envelope.evaluate(intensity.geometry.q_sample_Ainv)
        event_polarization = self._event_scattering_polarization(
            mapped.geometry.kf_air_sample_Ainv,
            mapped.geometry.valid,
        )
        postoptical = (
            intensity.coating_intensity_density_A2_rad2_inv
            * event_envelope
            * event_polarization
            * optical
            * self._source_phase_weight
        )
        postoptical = np.where(mapped.geometry.valid, postoptical, 0.0)
        return DetectorLatentIntensity(
            geometry=mapped.geometry,
            intensity_status=IntensityStatus.INCLUDED,
            exit_amplitude=mapped.exit_amplitude,
            attenuation_weight=attenuation,
            optical_weight=optical,
            coating_intensity_density_A2_rad2_inv=(intensity.coating_intensity_density_A2_rad2_inv),
            event_intensity_envelope=event_envelope,
            scattering_polarization_weight=event_polarization,
            source_phase_weight=self._source_phase_weight,
            postoptical_density_A2_rad2_inv=postoptical,
        )

    def map_latent_geometry(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> DetectorMappedGeometry:
        """Map analytic non-specular Ewald geometry without evaluating intensity."""

        geometry = self._coating.evaluate_geometry(
            rod=rod,
            branch=branch,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        return self._map_geometry(geometry).geometry

    def map_detector_visible_coating(
        self,
        *,
        rod: Rod,
        branch: int,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> DetectorVisibleEwaldCoating:
        """Map the intrinsic coating while using exit geometry only as a visibility mask.

        No source weight, optical factor, attenuation, detector solid angle, or
        detector-coordinate Jacobian enters the returned latent density.
        """

        if not isinstance(rod, Rod):
            raise TypeError("rod must be a Rod")
        m0_gap: float | None = None
        if rod.family_m == 0:
            if branch != 0:
                raise ValueError("detector-visible m=0 requires branch 0")
            incident_normal = float(self._coating.ki_sample_Ainv[2])
            if incident_normal >= 0.0:
                raise ValueError("detector-visible m=0 requires negative incident sample-normal k")
            m0_gap = -incident_normal
            geometry, coarea = self._coating._evaluate_geometry(
                rod=rod,
                branch=0,
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
            )
            mapped = self._map_geometry(geometry)
            visible = mapped.geometry.valid
            density = np.zeros(geometry.alpha_rad.shape, dtype=np.float64)
            if np.any(visible):
                q_norm = np.linalg.norm(geometry.q_sample_Ainv[visible], axis=-1)
                if np.any(q_norm <= m0_gap):
                    raise FloatingPointError("detector-visible m=0 violated its reciprocal gap")
                latent = self._coating.bragg_space.evaluate_latent(
                    rod=rod,
                    alpha_rad=geometry.alpha_rad[visible],
                    beta_rad=geometry.beta_rad[visible],
                    u_Ainv=geometry.u_Ainv[visible],
                )
                density[visible] = latent.intensity_density_A2_rad2_inv * coarea[visible]
        else:
            if branch not in {1, 2}:
                raise ValueError("a nonzero rod requires branch 1 or 2")
            intensity = self._coating.evaluate_latent(
                rod=rod,
                branch=branch,
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
            )
            mapped = self._map_geometry(intensity.geometry)
            density = np.where(
                mapped.geometry.valid,
                intensity.coating_intensity_density_A2_rad2_inv,
                0.0,
            )
        return DetectorVisibleEwaldCoating(
            geometry=mapped.geometry,
            coating_intensity_density_A2_rad2_inv=density,
            detector_visible_m0_q_gap_Ainv=m0_gap,
        )

    def map_specular_geometry(
        self,
        *,
        rod: Rod,
        alpha_rad: ArrayLike,
        beta_rad: ArrayLike,
    ) -> SpecularDetectorGeometry:
        """Map m=0 branch geometry while explicitly assigning no intensity."""

        geometry = self._coating.evaluate_specular_geometry(
            rod=rod,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
        )
        return SpecularDetectorGeometry(self._map_geometry(geometry).geometry)

    def _detector_coordinate_state(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        include_optical: bool = True,
        include_surface_jacobian: bool = True,
    ) -> tuple[DetectorCoordinateGeometry, FloatArray]:
        geometry = evaluate_detector_coordinates_geometry(
            column_px,
            row_px,
            incident=self._incident,
            instrument=self._instrument,
            ki_sample_Ainv=self._coating.ki_sample_Ainv,
            include_surface_jacobian=include_surface_jacobian,
            incident_state_index=self._incident_state_index,
        )
        shape = geometry.column_px.shape
        size = geometry.column_px.size
        status = np.asarray(geometry.status).reshape(-1).copy()
        reachable = np.asarray(geometry.valid).reshape(-1)
        kf_air_sample = geometry.kf_air_sample_Ainv.reshape(-1, 3)
        kf_film = geometry.kf_film_sample_Ainv.reshape(-1, 3)
        optical = np.zeros(size, dtype=np.float64)
        reachable_rows = np.flatnonzero(reachable)
        if include_optical and reachable_rows.size:
            wavelengths = np.full(
                reachable_rows.size,
                self._incident.states.wavelength_A[self._incident_state_index],
            )
            modes = _solve_exit_mode_arrays(
                kf_film[reachable_rows],
                wavelengths,
                self._material,
            )
            status[reachable_rows] = modes.status
            mode_valid = modes.status == ValidityCode.VALID
            valid_rows = reachable_rows[mode_valid]
            if valid_rows.size:
                roundtrip_scale = max(self._air_k0_Ainv, 1.0)
                roundtrip_budget = 512.0 * np.finfo(np.float64).eps
                recovered = modes.k_air_phase_sample_Ainv[mode_valid]
                expected = kf_air_sample[valid_rows]
                # Near the horizon sqrt(k0²-k_parallel²) is ill-conditioned.
                # Certify the dispersion equation, not absolute error in its inverse.
                normal_squared_error = np.abs(
                    (recovered[:, 2] - expected[:, 2]) * (recovered[:, 2] + expected[:, 2])
                )
                if not (
                    np.all(
                        np.abs(recovered[:, :2] - expected[:, :2])
                        <= roundtrip_budget * roundtrip_scale
                    )
                    and np.all(np.signbit(recovered[:, 2]) == np.signbit(expected[:, 2]))
                    and np.all(normal_squared_error <= roundtrip_budget * roundtrip_scale**2)
                ):
                    raise FloatingPointError("detector ray failed the canonical exit round trip")
                incident_direction = (
                    -1
                    if self._incident.states.direction_sample[self._incident_state_index, 2] < 0.0
                    else 1
                )
                incident_kappa = mode_decay_constant(
                    self._incident.states.kz_film_Ainv[self._incident_state_index],
                    incident_direction,
                )
                exit_kappa = mode_decay_constant(
                    modes.kz_film_Ainv[mode_valid],
                    modes.propagation_direction[mode_valid],
                )
                attenuation = uniform_depth_attenuation(
                    incident_kappa,
                    exit_kappa,
                    self._instrument.film_thickness_A,
                )
                optical[valid_rows] = scalar_optical_weight(
                    self._incident.states.entrance_amplitude[self._incident_state_index],
                    modes.exit_amplitude[mode_valid],
                    attenuation,
                )
                optical[valid_rows] *= external_path_attenuation(
                    self._detector_path_linear_attenuation_m_inv,
                    geometry.ray_distance_m.reshape(-1)[valid_rows],
                )

        if include_optical:
            valid = status == ValidityCode.VALID
            ewald_residual = np.where(
                valid,
                geometry.ewald_residual_Ainv.reshape(-1),
                0.0,
            )
            geometry = DetectorCoordinateGeometry(
                column_px=geometry.column_px,
                row_px=geometry.row_px,
                kf_air_sample_Ainv=geometry.kf_air_sample_Ainv,
                kf_film_sample_Ainv=geometry.kf_film_sample_Ainv,
                q_sample_Ainv=geometry.q_sample_Ainv,
                ray_distance_m=geometry.ray_distance_m,
                q_surface_jacobian_Ainv2_per_px2=(geometry.q_surface_jacobian_Ainv2_per_px2),
                ewald_residual_Ainv=ewald_residual.reshape(shape),
                status=status.reshape(shape),
            )
        optical = optical.reshape(shape)
        optical.setflags(write=False)
        return geometry, optical

    def _inverse_rod_density(
        self,
        *,
        q_sample_Ainv: FloatArray,
        kf_sample_Ainv: FloatArray,
        surface_jacobian_Ainv2_per_output: FloatArray,
        coordinate_valid: BoolArray,
        optical_weight: FloatArray,
        source_phase_weight: float,
        rod: Rod,
        branch: int,
        response_blocks: list[
            tuple[
                IntArray,
                IntArray,
                FloatArray,
                FloatArray,
                FloatArray,
                FloatArray,
                NDArray[np.int8],
            ]
        ]
        | None = None,
        rod_index: int | None = None,
        scattering_polarization: FloatArray | None = None,
    ) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.bool_]]:
        if branch not in {0, 1, 2}:
            raise ValueError("branch must be 0, 1, or 2")
        if (response_blocks is None) != (rod_index is None):
            raise ValueError("response_blocks and rod_index must be supplied together")
        shape = q_sample_Ainv.shape[:-1]
        density = np.zeros(shape, dtype=np.float64)
        inverse_count = np.zeros(shape, dtype=np.int64)
        caustic = np.zeros(shape, dtype=np.bool_)
        valid_rows = np.flatnonzero(coordinate_valid.reshape(-1))
        if not valid_rows.size:
            return density, inverse_count, caustic

        q_sample = q_sample_Ainv.reshape(-1, 3)[valid_rows]
        event_envelope = self._intensity_envelope.evaluate(q_sample)
        kf_sample = kf_sample_Ainv.reshape(-1, 3)[valid_rows]
        q_crystal = q_sample @ self._crystal_to_sample
        q_local = q_crystal @ self._crystal_from_local
        basis = self._coating.bragg_space.config.reciprocal_basis_Ainv
        q_parallel_crystal = rod.h * basis[:, 0] + rod.k * basis[:, 1]
        q_parallel_local = q_parallel_crystal @ self._crystal_from_local
        a, b, c0 = q_parallel_local
        transverse_norm = np.hypot(q_local[:, 0], q_local[:, 1])
        x_squared = (transverse_norm - abs(b)) * (transverse_norm + abs(b))
        q_norm = np.linalg.norm(q_local, axis=1)
        q_norm_squared = q_norm * q_norm
        parallel_norm = float(np.hypot(a, b))
        w_squared = (q_norm - parallel_norm) * (q_norm + parallel_norm)
        inverse_scale = np.maximum.reduce(
            (
                q_norm_squared,
                np.full(q_norm_squared.shape, np.dot(q_parallel_local, q_parallel_local)),
                np.ones(q_norm_squared.shape),
            )
        )
        inverse_tolerance = 1024.0 * np.finfo(np.float64).eps * inverse_scale
        supported = (x_squared >= -inverse_tolerance) & (w_squared >= -inverse_tolerance)
        if not np.any(supported):
            return density, inverse_count, caustic
        x_magnitude = np.sqrt(np.maximum(x_squared, 0.0))
        w_magnitude = np.sqrt(np.maximum(w_squared, 0.0))
        azimuth_q = np.arctan2(q_local[:, 1], q_local[:, 0])
        two_pi = 2.0 * np.pi
        angular_tolerance = 2048.0 * np.finfo(np.float64).eps
        lower_u, upper_u = self._coating.bragg_space.rod_u_bounds_Ainv(rod)
        u_tolerance = 1024.0 * np.finfo(np.float64).eps * max(abs(lower_u), abs(upper_u), 1.0)
        flat_density = density.reshape(-1)
        flat_count = inverse_count.reshape(-1)
        flat_caustic = caustic.reshape(-1)
        infinite_density = np.zeros(flat_caustic.shape, dtype=np.bool_)
        area_jacobian = surface_jacobian_Ainv2_per_output.reshape(-1)[valid_rows]
        optical = optical_weight.reshape(-1)[valid_rows]
        polarization = (
            np.ones(valid_rows.size, dtype=np.float64)
            if scattering_polarization is None
            else scattering_polarization.reshape(-1)[valid_rows]
        )
        sample_from_local = self._crystal_to_sample @ self._crystal_from_local
        reconstruction_tolerance = (
            4096.0
            * np.finfo(np.float64).eps
            * max(float(np.linalg.norm(self._coating.ki_sample_Ainv)), 1.0)
        )

        for x_sign in (-1.0, 1.0):
            x_value = x_sign * x_magnitude
            beta = np.remainder(azimuth_q - np.arctan2(b, x_value), two_pi)
            beta[beta >= two_pi] = 0.0
            for w_sign in (-1.0, 1.0):
                w_value = w_sign * w_magnitude
                alpha = np.remainder(
                    np.arctan2(w_value, a) - np.arctan2(q_local[:, 2], x_value),
                    two_pi,
                )
                alpha = np.where(alpha >= two_pi - angular_tolerance, 0.0, alpha)
                folded = supported & (alpha <= np.pi + angular_tolerance)
                alpha = np.minimum(alpha, np.pi)
                u_value = w_value - c0
                folded &= (u_value >= lower_u - u_tolerance) & (u_value <= upper_u + u_tolerance)
                if not np.any(folded):
                    continue
                direction_local = np.column_stack(
                    (
                        np.sin(alpha) * np.cos(beta),
                        np.sin(alpha) * np.sin(beta),
                        np.cos(alpha),
                    )
                )
                direction_sample = direction_local @ sample_from_local.T
                root_sign = np.einsum(
                    "ij,ij->i",
                    kf_sample,
                    direction_sample,
                    optimize=True,
                )
                if branch == 2:
                    folded &= root_sign > 0.0
                elif branch == 1:
                    folded &= root_sign < 0.0
                if not np.any(folded):
                    continue
                jacobian = np.abs(w_value * x_value)
                singular_candidate = folded & (jacobian == 0.0)
                singular = np.zeros(folded.shape, dtype=np.bool_)
                if np.any(singular_candidate):
                    reconstructed = self._coating.bragg_space.map_latent(
                        rod=rod,
                        alpha_rad=alpha[singular_candidate],
                        beta_rad=beta[singular_candidate],
                        u_Ainv=u_value[singular_candidate],
                    )
                    reconstruction_error = np.linalg.norm(
                        reconstructed - q_sample[singular_candidate],
                        axis=1,
                    )
                    singular_rows = np.flatnonzero(singular_candidate)[
                        reconstruction_error <= reconstruction_tolerance
                    ]
                    singular[singular_rows] = True
                    flat_caustic[valid_rows[singular]] = True
                    if response_blocks is None and source_phase_weight > 0.0 and np.any(singular):
                        singular_latent = self._coating.bragg_space.evaluate_latent(
                            rod=rod,
                            alpha_rad=alpha[singular],
                            beta_rad=beta[singular],
                            u_Ainv=u_value[singular],
                        )
                        positive_numerator = (
                            (singular_latent.mosaic_probability_density_rad2_inv > 0.0)
                            & (rod.population > 0.0)
                            & (singular_latent.rod_strength_A2 > 0.0)
                            & (area_jacobian[singular] > 0.0)
                            & (optical[singular] > 0.0)
                            & (event_envelope[singular] > 0.0)
                            & (polarization[singular] > 0.0)
                        )
                        infinite_density[valid_rows[singular][positive_numerator]] = True
                regular = folded & (jacobian > 0.0)
                if not np.any(regular):
                    continue
                selected_rows = valid_rows[regular]
                if response_blocks is not None:
                    ell, mosaic_density = self._coating.bragg_space.evaluate_latent_mosaic_density(
                        rod=rod,
                        alpha_rad=alpha[regular],
                        beta_rad=beta[regular],
                        u_Ainv=u_value[regular],
                    )
                    fixed_density_per_mosaic = (
                        rod.population
                        * area_jacobian[regular]
                        * optical[regular]
                        * event_envelope[regular]
                        * polarization[regular]
                        * source_phase_weight
                        / jacobian[regular]
                    )
                    fixed_density = mosaic_density * fixed_density_per_mosaic
                    response_blocks.append(
                        (
                            np.array(selected_rows, dtype=np.int64, copy=True),
                            np.full(selected_rows.size, int(rod_index), dtype=np.int64),
                            np.array(ell, dtype=np.float64, copy=True),
                            np.array(alpha[regular], dtype=np.float64, copy=True),
                            np.array(
                                fixed_density_per_mosaic,
                                dtype=np.float64,
                                copy=True,
                            ),
                            np.array(fixed_density, dtype=np.float64, copy=True),
                            np.sign(root_sign[regular]).astype(np.int8, copy=False),
                        )
                    )
                    continue
                latent = self._coating.bragg_space.evaluate_latent(
                    rod=rod,
                    alpha_rad=alpha[regular],
                    beta_rad=beta[regular],
                    u_Ainv=u_value[regular],
                )
                fixed_density = (
                    latent.mosaic_probability_density_rad2_inv
                    * rod.population
                    * area_jacobian[regular]
                    * optical[regular]
                    * event_envelope[regular]
                    * polarization[regular]
                    * source_phase_weight
                    / jacobian[regular]
                )
                contribution = fixed_density * latent.rod_strength_A2
                flat_density[selected_rows] += contribution
                flat_count[selected_rows] += 1

        flat_density[infinite_density] = np.inf
        density.setflags(write=False)
        inverse_count.setflags(write=False)
        caustic.setflags(write=False)
        return density, inverse_count, caustic

    def evaluate_intrinsic_ewald_directions(
        self,
        outgoing_direction_sample: ArrayLike,
        *,
        rods: tuple[Rod, ...],
        branch: int | None = None,
    ) -> EwaldDirectionIntensity:
        """Evaluate the intrinsic non-specular coating per internal-film solid angle.

        ``outgoing_direction_sample`` is a unit vector (or array of unit vectors)
        in the sample frame. ``branch=None`` sums both retained analytic roots;
        branches ``1`` and ``2`` select the lower- and upper-``u`` roots.
        Detector visibility, exit optics, source weights, and detector solid
        angle are deliberately absent.
        """

        selected = self._validated_intensity_rods(rods)
        if branch not in {None, 1, 2}:
            raise ValueError("branch must be None, 1, or 2")
        supplied = np.asarray(outgoing_direction_sample)
        if np.iscomplexobj(supplied) and np.any(supplied.imag != 0.0):
            raise ValueError("outgoing directions must be real")
        direction = np.array(supplied.real, dtype=np.float64, copy=True, order="C")
        if direction.ndim < 1 or direction.shape[-1] != 3 or not np.all(np.isfinite(direction)):
            raise ValueError(
                "outgoing directions must be finite vectors with final dimension three"
            )
        norm = np.linalg.norm(direction, axis=-1)
        if not np.allclose(
            norm,
            1.0,
            rtol=0.0,
            atol=4096.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("outgoing directions must be unit vectors")
        direction /= norm[..., None]
        shape = direction.shape[:-1]
        k_magnitude_Ainv = float(np.linalg.norm(self._coating.ki_sample_Ainv))
        kf_sample_Ainv = k_magnitude_Ainv * direction
        q_sample_Ainv = kf_sample_Ainv - self._coating.ki_sample_Ainv
        surface_jacobian = np.full(shape, k_magnitude_Ainv**2, dtype=np.float64)
        valid = np.ones(shape, dtype=np.bool_)
        optical = np.ones(shape, dtype=np.float64)
        per_rod = np.zeros((*shape, len(selected)), dtype=np.float64)
        counts = np.zeros((*shape, len(selected)), dtype=np.int64)
        caustic = np.zeros((*shape, len(selected)), dtype=np.bool_)
        for rod_index, rod in enumerate(selected):
            rod_density, rod_count, rod_caustic = self._inverse_rod_density(
                q_sample_Ainv=q_sample_Ainv,
                kf_sample_Ainv=kf_sample_Ainv,
                surface_jacobian_Ainv2_per_output=surface_jacobian,
                coordinate_valid=valid,
                optical_weight=optical,
                source_phase_weight=1.0,
                rod=rod,
                branch=0 if branch is None else branch,
            )
            per_rod[..., rod_index] = rod_density
            counts[..., rod_index] = rod_count
            caustic[..., rod_index] = rod_caustic
        return EwaldDirectionIntensity(
            outgoing_direction_sample=direction,
            q_sample_Ainv=q_sample_Ainv,
            rods=selected,
            branch=branch,
            per_rod_density_A2_per_sr=per_rod,
            density_A2_per_sr=np.sum(per_rod, axis=-1, dtype=np.float64),
            per_rod_inverse_branch_count=counts,
            caustic=caustic,
        )

    def evaluate_detector_visible_ewald_directions(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        rods: tuple[Rod, ...],
    ) -> DetectorVisibleEwaldDirectionIntensity:
        """Evaluate intrinsic solid-angle density on the active-panel sphere patch.

        Native detector coordinates select internal-film outgoing directions
        through the canonical exit-refraction round trip. Detector validity is
        only a geometric support mask: source weights, optical factors,
        attenuation, detector Jacobians, and detector solid angle are absent.
        All regular inverse mosaic preimages are summed, including nonzero
        ``m=0`` support. Top exit supplies a strict positive ``m=0`` Q gap, so
        the collapsed direct root cannot enter this measure.
        """

        selected = self._validated_configured_rods(rods)
        geometry, _ = self._detector_coordinate_state(
            column_px,
            row_px,
            include_optical=True,
            include_surface_jacobian=False,
        )
        shape = geometry.column_px.shape
        k_magnitude_Ainv = float(np.linalg.norm(self._coating.ki_sample_Ainv))
        direction = np.zeros((*shape, 3), dtype=np.float64)
        direction[geometry.valid] = geometry.kf_film_sample_Ainv[geometry.valid] / k_magnitude_Ainv
        surface_jacobian = np.where(geometry.valid, k_magnitude_Ainv**2, 0.0)
        unit_optical = np.ones(shape, dtype=np.float64)
        per_rod = np.zeros((*shape, len(selected)), dtype=np.float64)
        counts = np.zeros((*shape, len(selected)), dtype=np.int64)
        caustic = np.zeros((*shape, len(selected)), dtype=np.bool_)
        for rod_index, rod in enumerate(selected):
            rod_density, rod_count, rod_caustic = self._inverse_rod_density(
                q_sample_Ainv=geometry.q_sample_Ainv,
                kf_sample_Ainv=geometry.kf_film_sample_Ainv,
                surface_jacobian_Ainv2_per_output=surface_jacobian,
                coordinate_valid=geometry.valid,
                optical_weight=unit_optical,
                source_phase_weight=1.0,
                rod=rod,
                branch=0,
            )
            per_rod[..., rod_index] = rod_density
            counts[..., rod_index] = rod_count
            caustic[..., rod_index] = rod_caustic

        m0_gap: float | None = None
        if any(rod.family_m == 0 for rod in selected):
            incident_normal_Ainv = float(self._coating.ki_sample_Ainv[2])
            if incident_normal_Ainv >= 0.0:
                raise ValueError("detector-visible m=0 requires negative incident sample-normal k")
            m0_gap = -incident_normal_Ainv

        return DetectorVisibleEwaldDirectionIntensity(
            geometry=geometry,
            outgoing_direction_sample=direction,
            rods=selected,
            per_rod_density_A2_per_sr=per_rod,
            density_A2_per_sr=np.sum(per_rod, axis=-1, dtype=np.float64),
            per_rod_inverse_branch_count=counts,
            caustic=caustic,
            detector_visible_m0_q_gap_Ainv=m0_gap,
        )

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        rods: tuple[Rod, ...],
        branch: int = 2,
    ) -> DetectorCoordinateIntensity:
        """Evaluate the detector pushforward density at arbitrary coordinates.

        This is the analytic ``detector point -> kf -> Q`` pullback. All
        inverse mosaic branches are summed as intensities. The returned
        density is defined almost everywhere. Exact caustic coordinates are
        marked and carry ``+inf`` only when a retained branch has a strictly
        positive physical numerator. A zero numerator uses the zero
        Radon--Nikodym representative at that measure-zero point; it is not a
        claim about the pointwise limit. Finite detector-pixel integrals are
        evaluated separately.
        """

        self._require_physical_intensity_source()

        selected = self._validated_intensity_rods(rods)
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        geometry, optical = self._detector_coordinate_state(column_px, row_px)
        polarization = self._event_scattering_polarization(
            geometry.kf_air_sample_Ainv,
            geometry.valid,
        )
        shape = geometry.column_px.shape
        per_rod = np.zeros((*shape, len(selected)), dtype=np.float64)
        counts = np.zeros((*shape, len(selected)), dtype=np.int64)
        caustic = np.zeros((*shape, len(selected)), dtype=np.bool_)
        for rod_index, rod in enumerate(selected):
            rod_density, rod_count, rod_caustic = self._inverse_rod_density(
                q_sample_Ainv=geometry.q_sample_Ainv,
                kf_sample_Ainv=geometry.kf_film_sample_Ainv,
                surface_jacobian_Ainv2_per_output=(geometry.q_surface_jacobian_Ainv2_per_px2),
                coordinate_valid=geometry.valid,
                optical_weight=optical,
                source_phase_weight=self._source_phase_weight,
                rod=rod,
                branch=branch,
                scattering_polarization=polarization,
            )
            per_rod[..., rod_index] = rod_density
            counts[..., rod_index] = rod_count
            caustic[..., rod_index] = rod_caustic
        return DetectorCoordinateIntensity(
            geometry=geometry,
            rods=selected,
            branch=branch,
            per_rod_density_A2_per_px2=per_rod,
            density_A2_per_px2=np.sum(per_rod, axis=-1, dtype=np.float64),
            per_rod_inverse_branch_count=counts,
            caustic=caustic,
        )

    def evaluate_detector_structure_response(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        rods: tuple[Rod, ...],
    ) -> DetectorStructureResponse:
        """Compile all regular inverse roots without dividing by a reference strength."""

        self._require_physical_intensity_source()

        selected = self._validated_configured_rods(rods)
        geometry, optical = self._detector_coordinate_state(column_px, row_px)
        polarization = self._event_scattering_polarization(
            geometry.kf_air_sample_Ainv,
            geometry.valid,
        )
        coordinate_count = geometry.column_px.size
        caustic = np.zeros((coordinate_count, len(selected)), dtype=np.bool_)
        blocks: list[
            tuple[
                IntArray,
                IntArray,
                FloatArray,
                FloatArray,
                FloatArray,
                FloatArray,
                NDArray[np.int8],
            ]
        ] = []
        for response_rod_index, rod in enumerate(selected):
            _, _, rod_caustic = self._inverse_rod_density(
                q_sample_Ainv=geometry.q_sample_Ainv,
                kf_sample_Ainv=geometry.kf_film_sample_Ainv,
                surface_jacobian_Ainv2_per_output=(geometry.q_surface_jacobian_Ainv2_per_px2),
                coordinate_valid=geometry.valid,
                optical_weight=optical,
                source_phase_weight=self._source_phase_weight,
                rod=rod,
                branch=0,
                response_blocks=blocks,
                rod_index=response_rod_index,
                scattering_polarization=polarization,
            )
            caustic[:, response_rod_index] = rod_caustic.reshape(-1)

        if blocks:
            term_coordinate = np.concatenate([block[0] for block in blocks])
            term_rod = np.concatenate([block[1] for block in blocks])
            term_l = np.concatenate([block[2] for block in blocks])
            term_alpha = np.concatenate([block[3] for block in blocks])
            term_fixed_per_mosaic = np.concatenate([block[4] for block in blocks])
            term_fixed = np.concatenate([block[5] for block in blocks])
            term_root_sign = np.concatenate([block[6] for block in blocks])
            positive = term_fixed_per_mosaic > 0.0
            term_coordinate = term_coordinate[positive]
            term_rod = term_rod[positive]
            term_l = term_l[positive]
            term_alpha = term_alpha[positive]
            term_fixed_per_mosaic = term_fixed_per_mosaic[positive]
            term_fixed = term_fixed[positive]
            term_root_sign = term_root_sign[positive]
        else:
            term_coordinate = np.empty(0, dtype=np.int64)
            term_rod = np.empty(0, dtype=np.int64)
            term_l = np.empty(0, dtype=np.float64)
            term_alpha = np.empty(0, dtype=np.float64)
            term_fixed_per_mosaic = np.empty(0, dtype=np.float64)
            term_fixed = np.empty(0, dtype=np.float64)
            term_root_sign = np.empty(0, dtype=np.int8)

        term_q_sample = geometry.q_sample_Ainv.reshape(-1, 3)[term_coordinate]
        term_q_radial_squared = np.sum(term_q_sample[:, :2] ** 2, axis=1, dtype=np.float64)
        term_q_normal_squared = term_q_sample[:, 2] ** 2

        m0_rod_index = np.asarray(
            [rod_index for rod_index, rod in enumerate(selected) if rod.h == 0 and rod.k == 0],
            dtype=np.int64,
        )
        if m0_rod_index.size:
            incident_normal = float(self._coating.ki_sample_Ainv[2])
            if incident_normal >= 0.0:
                raise ValueError("detector-visible m=0 requires negative incident sample-normal k")
            m0_term = np.isin(term_rod, m0_rod_index)
            if np.any(m0_term):
                q_norm = np.linalg.norm(geometry.q_sample_Ainv.reshape(-1, 3), axis=1)
                if np.any(q_norm[term_coordinate[m0_term]] <= -incident_normal):
                    raise FloatingPointError("detector-visible m=0 violated its reciprocal gap")

        return DetectorStructureResponse(
            rods=selected,
            coordinate_valid=geometry.valid.reshape(-1),
            term_coordinate_index=term_coordinate,
            term_rod_index=term_rod,
            term_L=term_l,
            term_alpha_rad=term_alpha,
            term_q_radial_squared_Ainv2=term_q_radial_squared,
            term_q_normal_squared_Ainv2=term_q_normal_squared,
            term_fixed_density_per_mosaic_density_per_strength_px2_inv=(term_fixed_per_mosaic),
            term_fixed_density_per_strength_px2_inv=term_fixed,
            term_root_sign=term_root_sign,
            per_rod_caustic=caustic,
            k_norm_Ainv=self._coating.bragg_space.config.k_norm_Ainv,
        )

    def evaluate_detector_geometry(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        include_surface_jacobian: bool = True,
    ) -> DetectorCoordinateGeometry:
        """Return canonical detector-ray, internal-kf, and Q geometry.

        Set ``include_surface_jacobian=False`` when only ray validity or Q is
        needed. The returned Jacobian is then identically zero and its
        derivative work is skipped.
        """

        return self._detector_coordinate_state(
            column_px,
            row_px,
            include_optical=False,
            include_surface_jacobian=include_surface_jacobian,
        )[0]

    def evaluate_detector_visible_geometry(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        include_surface_jacobian: bool = False,
    ) -> DetectorCoordinateGeometry:
        """Return detector geometry after the canonical exit-refraction round trip."""

        return self._detector_coordinate_state(
            column_px,
            row_px,
            include_optical=True,
            include_surface_jacobian=include_surface_jacobian,
        )[0]

    def _validated_intensity_rods(self, rods: tuple[Rod, ...]) -> tuple[Rod, ...]:
        selected = self._validated_configured_rods(rods)
        if any(rod.family_m == 0 for rod in selected):
            raise ValueError("m=0 intensity is excluded without physical direct-beam support")
        return selected

    def _validated_configured_rods(self, rods: tuple[Rod, ...]) -> tuple[Rod, ...]:
        selected = tuple(rods)
        if not selected or not all(isinstance(rod, Rod) for rod in selected):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in selected}) != len(selected):
            raise ValueError("rods must not repeat a physical rod")
        configured = {(rod.h, rod.k): rod for rod in self._coating.bragg_space.config.rods}
        result: list[Rod] = []
        for rod in selected:
            try:
                canonical = configured[(rod.h, rod.k)]
            except KeyError as error:
                raise ValueError(f"rod ({rod.h}, {rod.k}) is not configured") from error
            result.append(canonical)
        return tuple(result)

    def _compiled_evaluator(self, rods: tuple[Rod, ...]) -> CompiledDetectorEvaluator:
        """Pack one reusable evaluator for the accepted finite-parent model."""

        self._require_physical_intensity_source()

        strength = self._coating.bragg_space.strength_model
        if not isinstance(strength, Bi2X3FiniteStackStrength):
            raise TypeError(
                "adaptive_compiled integration requires the accepted Bi2X3FiniteStackStrength model"
            )
        state = _compile_detector_state(
            bragg_config=self._coating.bragg_space.config,
            strength_model=strength,
            ki_sample_Ainv=self._coating.ki_sample_Ainv,
            incident=self._incident,
            material=self._material,
            instrument=self._instrument,
            rods=rods,
            incident_state_index=self._incident_state_index,
            source_phase_weight=self._source_phase_weight,
            intensity_envelope=self._intensity_envelope,
        )
        return CompiledDetectorEvaluator(state, self._instrument.detector_shape_rc)

    def _fold_candidate_mask(
        self,
        *,
        rods: tuple[Rod, ...],
        row_chunk_size: int,
    ) -> NDArray[np.bool_]:
        """Flag folds or validity edges detected by corner/center probes."""

        rows, columns = self._instrument.detector_shape_rc
        candidate = np.zeros((rows, columns), dtype=np.bool_)
        offsets_column = np.array((-0.5, 0.5, -0.5, 0.5, 0.0))
        offsets_row = np.array((-0.5, -0.5, 0.5, 0.5, 0.0))
        column_center = np.arange(columns, dtype=np.float64)
        basis = self._coating.bragg_space.config.reciprocal_basis_Ainv
        local_from_sample = self._crystal_to_sample @ self._crystal_from_local
        for row_start in range(0, rows, row_chunk_size):
            row_stop = min(row_start + row_chunk_size, rows)
            row_center = np.arange(row_start, row_stop, dtype=np.float64)
            node_column, node_row = np.broadcast_arrays(
                column_center[None, :, None] + offsets_column[None, None, :],
                row_center[:, None, None] + offsets_row[None, None, :],
            )
            geometry = self._detector_coordinate_state(
                node_column,
                node_row,
                include_optical=False,
                include_surface_jacobian=False,
            )[0]
            valid = geometry.valid
            valid_count = np.sum(valid, axis=-1)
            local_candidate = (valid_count > 0) & (valid_count < offsets_column.size)
            all_valid = valid_count == offsets_column.size
            q_local = geometry.q_sample_Ainv @ local_from_sample
            transverse_norm = np.hypot(q_local[..., 0], q_local[..., 1])
            q_norm = np.linalg.norm(q_local, axis=-1)
            for rod in rods:
                q_parallel = rod.h * basis[:, 0] + rod.k * basis[:, 1]
                a, b, _ = q_parallel @ self._crystal_from_local
                parallel_norm = float(np.hypot(a, b))
                t_squared = (transverse_norm - abs(b)) * (transverse_norm + abs(b))
                w_squared = (q_norm - parallel_norm) * (q_norm + parallel_norm)
                local_candidate |= all_valid & (
                    (np.min(t_squared, axis=-1) <= 0.0) & (np.max(t_squared, axis=-1) >= 0.0)
                )
                local_candidate |= all_valid & (
                    (np.min(w_squared, axis=-1) <= 0.0) & (np.max(w_squared, axis=-1) >= 0.0)
                )
            candidate[row_start:row_stop] = local_candidate
        return candidate

    def _integrate_selected_pixels(
        self,
        *,
        flat_pixel_index: NDArray[np.int64],
        rods: tuple[Rod, ...],
        branch: int,
        gauss_order: int,
        subdivision_count: int,
        node_chunk_size: int,
    ) -> FloatArray:
        """Integrate selected exact pixel boxes without storing private nodes."""

        rows, columns = self._instrument.detector_shape_rc
        if np.any((flat_pixel_index < 0) | (flat_pixel_index >= rows * columns)):
            raise ValueError("flat_pixel_index lies outside the active detector")
        gauss_node, gauss_weight = leggauss(gauss_order)
        subcell = np.arange(subdivision_count, dtype=np.float64)
        offset = (subcell[:, None] + 0.5 + 0.5 * gauss_node[None, :]).reshape(
            -1
        ) / subdivision_count - 0.5
        one_dimensional_weight = np.tile(
            0.5 * gauss_weight / subdivision_count,
            subdivision_count,
        )
        node_weight = one_dimensional_weight[:, None] * one_dimensional_weight[None, :]
        result = np.zeros((flat_pixel_index.size, len(rods)), dtype=np.float64)
        node_count = offset.size
        pixel_chunk_size = max(1, node_chunk_size // (node_count * node_count))
        for start in range(0, flat_pixel_index.size, pixel_chunk_size):
            stop = min(start + pixel_chunk_size, flat_pixel_index.size)
            selected_index = flat_pixel_index[start:stop]
            pixel_row, pixel_column = np.divmod(selected_index, columns)
            node_column, node_row = np.broadcast_arrays(
                pixel_column[:, None, None] + offset[None, None, :],
                pixel_row[:, None, None] + offset[None, :, None],
            )
            evaluated = self.evaluate_detector_coordinates(
                node_column,
                node_row,
                rods=rods,
                branch=branch,
            )
            if np.any(evaluated.caustic):
                raise FloatingPointError(
                    "a pixel quadrature node lies on a caustic; choose another even order"
                )
            result[start:stop] = np.sum(
                evaluated.per_rod_density_A2_per_px2 * node_weight[None, :, :, None],
                axis=(1, 2),
                dtype=np.float64,
            )
        result.setflags(write=False)
        return result

    def _compiled_base_tile(
        self,
        *,
        evaluator: CompiledDetectorEvaluator,
        rods: tuple[Rod, ...],
        branch: int,
        quadrature: DetectorQuadrature,
        offset_px: FloatArray,
        one_dimensional_weight: FloatArray,
        row_start: int,
        row_stop: int,
    ) -> tuple[FloatArray, NDArray[np.bool_], BoolArray, int, int, float]:
        """Evaluate one independent detector-row tile and its embedded indicator."""

        _, columns = self._instrument.detector_shape_rc
        flat_pixel_index = np.arange(row_start * columns, row_stop * columns, dtype=np.int64)
        integrated = evaluator.integrate_pixel_boxes(
            flat_pixel_index,
            offset_px=offset_px,
            one_dimensional_weight=one_dimensional_weight,
            branch=branch,
            include_center_diagnostics=True,
        )
        tile_rows = row_stop - row_start
        pixel_rod_mass = integrated.per_rod_mass_A2.reshape((tile_rows, columns, len(rods)))
        center_density = integrated.center_per_rod_density_A2_per_px2.reshape(
            (tile_rows, columns, len(rods))
        )
        if np.any(~np.isfinite(pixel_rod_mass)) or np.any(~np.isfinite(center_density)):
            raise FloatingPointError("compiled pixel density is nonfinite away from a caustic")
        count_min = integrated.per_rod_inverse_count_min.reshape((tile_rows, columns, len(rods)))
        count_max = integrated.per_rod_inverse_count_max.reshape((tile_rows, columns, len(rods)))
        count_transition = np.any(count_min != count_max, axis=-1)
        valid_any = integrated.valid_any.reshape((tile_rows, columns))
        valid_all = integrated.valid_all.reshape((tile_rows, columns))
        center_valid = integrated.center_valid.reshape((tile_rows, columns))
        validity_transition = valid_any & ~valid_all
        exact_node_caustic = np.any(
            integrated.per_rod_caustic.reshape((tile_rows, columns, len(rods))),
            axis=-1,
        )
        base_total = np.sum(pixel_rod_mass, axis=-1, dtype=np.float64)
        center_total = np.sum(center_density, axis=-1, dtype=np.float64)
        embedded_difference = np.sum(
            np.abs(pixel_rod_mass - center_density),
            axis=-1,
            dtype=np.float64,
        )
        embedded_scale = np.sum(np.abs(pixel_rod_mass), axis=-1, dtype=np.float64)
        embedded_tolerance = (
            quadrature.absolute_tolerance_A2 + quadrature.relative_tolerance * embedded_scale
        )
        smooth_candidate = embedded_difference > embedded_tolerance
        unresolved_zero_support = valid_any & (base_total == 0.0) & (center_total == 0.0)
        candidate = valid_any & (
            count_transition
            | validity_transition
            | exact_node_caustic
            | smooth_candidate
            | unresolved_zero_support
        )
        invalid_count = int(np.count_nonzero(~valid_any))
        evaluation_count = int(tile_rows * columns * (offset_px.size**2 + 1))
        accepted_embedded_l1_A2 = float(np.sum(embedded_difference[~candidate], dtype=np.float64))
        return (
            pixel_rod_mass,
            candidate,
            center_valid,
            invalid_count,
            evaluation_count,
            accepted_embedded_l1_A2,
        )

    def _integrate_compiled_selected_pixels(
        self,
        *,
        evaluator: CompiledDetectorEvaluator,
        flat_pixel_index: NDArray[np.int64],
        rods: tuple[Rod, ...],
        branch: int,
        offset_px: FloatArray,
        one_dimensional_weight: FloatArray,
        node_chunk_size: int,
    ) -> FloatArray:
        """Integrate selected pixels in bounded fused chunks."""

        pixel_chunk_size = max(1, node_chunk_size // (offset_px.size * offset_px.size))
        result = np.empty((flat_pixel_index.size, len(rods)), dtype=np.float64)
        for start in range(0, flat_pixel_index.size, pixel_chunk_size):
            stop = min(start + pixel_chunk_size, flat_pixel_index.size)
            integrated = evaluator.integrate_pixel_boxes(
                flat_pixel_index[start:stop],
                offset_px=offset_px,
                one_dimensional_weight=one_dimensional_weight,
                branch=branch,
                include_center_diagnostics=False,
            )
            values = integrated.per_rod_mass_A2
            if np.any(~np.isfinite(values)):
                raise FloatingPointError("adaptive pixel density is nonfinite away from a caustic")
            result[start:stop] = values
        return result

    def _compiled_adaptive_tile(
        self,
        *,
        evaluator: CompiledDetectorEvaluator,
        rods: tuple[Rod, ...],
        branch: int,
        quadrature: DetectorQuadrature,
        quadrature_rules: tuple[tuple[FloatArray, FloatArray], ...],
        row_start: int,
        row_stop: int,
    ) -> tuple[FloatArray, BoolArray, int, int, int, int, float]:
        """Finish every adaptive depth for one bounded row tile."""

        base_offset, base_weight = quadrature_rules[0]
        (
            pixel_rod_mass,
            candidate,
            center_valid,
            invalid_count,
            evaluation_count,
            accepted_embedded_l1_A2,
        ) = self._compiled_base_tile(
            evaluator=evaluator,
            rods=rods,
            branch=branch,
            quadrature=quadrature,
            offset_px=base_offset,
            one_dimensional_weight=base_weight,
            row_start=row_start,
            row_stop=row_stop,
        )
        _, columns = self._instrument.detector_shape_rc
        local_candidate = np.flatnonzero(candidate).astype(np.int64)
        flat_pixel_rod_mass = pixel_rod_mass.reshape((-1, len(rods)))
        current = flat_pixel_rod_mass[local_candidate]
        last_difference = np.zeros(local_candidate.size, dtype=np.float64)
        active = np.arange(local_candidate.size, dtype=np.int64)
        global_candidate = local_candidate + row_start * columns
        for depth in range(1, quadrature.max_depth + 1):
            if active.size == 0:
                break
            offset_px, one_dimensional_weight = quadrature_rules[depth]
            evaluated = self._integrate_compiled_selected_pixels(
                evaluator=evaluator,
                flat_pixel_index=global_candidate[active],
                rods=rods,
                branch=branch,
                offset_px=offset_px,
                one_dimensional_weight=one_dimensional_weight,
                node_chunk_size=quadrature.node_chunk_size,
            )
            evaluation_count += int(active.size * offset_px.size**2)
            difference = np.sum(
                np.abs(evaluated - current[active]),
                axis=1,
                dtype=np.float64,
            )
            tolerance = quadrature.absolute_tolerance_A2 + quadrature.relative_tolerance * np.sum(
                np.abs(evaluated),
                axis=1,
                dtype=np.float64,
            )
            current[active] = evaluated
            last_difference[active] = difference
            active = active[difference > tolerance]
        if local_candidate.size:
            flat_pixel_rod_mass[local_candidate] = current
        estimated_l1_error_A2 = accepted_embedded_l1_A2 + float(
            np.sum(last_difference, dtype=np.float64)
        )
        return (
            pixel_rod_mass,
            center_valid,
            int(local_candidate.size),
            int(active.size),
            invalid_count,
            evaluation_count,
            estimated_l1_error_A2,
        )

    def _integrate_native_pixels_adaptive_compiled(
        self,
        *,
        rods: tuple[Rod, ...],
        branch: int,
        quadrature: DetectorQuadrature,
    ) -> DetectorPixelMass:
        """Compiled tiled integration with support/fold-aware adaptive refinement."""

        selected = self._validated_intensity_rods(rods)
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        evaluator = self._compiled_evaluator(selected)
        quadrature_rules = tuple(
            _subdivided_legendre_rule(quadrature.pixel_gauss_order, 2**depth)
            for depth in range(quadrature.max_depth + 1)
        )
        # Compile once on the calling thread before the evaluator is shared.
        evaluator.integrate_pixel_boxes(
            np.asarray([0], dtype=np.int64),
            offset_px=quadrature_rules[0][0],
            one_dimensional_weight=quadrature_rules[0][1],
            branch=branch,
            include_center_diagnostics=True,
        )
        rows, columns = self._instrument.detector_shape_rc
        image = np.zeros((rows, columns), dtype=np.float64)
        sampled_valid_pixel_center = np.zeros((rows, columns), dtype=np.bool_)
        per_rod = np.zeros(len(selected), dtype=np.float64)
        refined_count = 0
        unresolved_count = 0
        invalid_count = 0
        evaluation_count = 0
        estimated_l1_error_A2 = 0.0
        tiles = tuple(
            (row_start, min(row_start + quadrature.row_chunk_size, rows))
            for row_start in range(0, rows, quadrature.row_chunk_size)
        )
        with ThreadPoolExecutor(max_workers=quadrature.worker_count) as executor:
            pending = deque()
            next_tile = 0
            window_size = min(len(tiles), 2 * quadrature.worker_count)
            while next_tile < window_size:
                row_start, row_stop = tiles[next_tile]
                pending.append(
                    (
                        row_start,
                        row_stop,
                        executor.submit(
                            self._compiled_adaptive_tile,
                            evaluator=evaluator,
                            rods=selected,
                            branch=branch,
                            quadrature=quadrature,
                            quadrature_rules=quadrature_rules,
                            row_start=row_start,
                            row_stop=row_stop,
                        ),
                    )
                )
                next_tile += 1
            while pending:
                row_start, row_stop, future = pending.popleft()
                (
                    pixel_rod_mass,
                    local_center_valid,
                    local_refined,
                    local_unresolved,
                    local_invalid,
                    local_evaluations,
                    local_estimated_l1_error_A2,
                ) = future.result()
                image[row_start:row_stop] = np.sum(pixel_rod_mass, axis=-1, dtype=np.float64)
                sampled_valid_pixel_center[row_start:row_stop] = local_center_valid
                per_rod += np.sum(pixel_rod_mass, axis=(0, 1), dtype=np.float64)
                refined_count += local_refined
                unresolved_count += local_unresolved
                invalid_count += local_invalid
                evaluation_count += local_evaluations
                estimated_l1_error_A2 += local_estimated_l1_error_A2
                if next_tile < len(tiles):
                    next_row_start, next_row_stop = tiles[next_tile]
                    pending.append(
                        (
                            next_row_start,
                            next_row_stop,
                            executor.submit(
                                self._compiled_adaptive_tile,
                                evaluator=evaluator,
                                rods=selected,
                                branch=branch,
                                quadrature=quadrature,
                                quadrature_rules=quadrature_rules,
                                row_start=next_row_start,
                                row_stop=next_row_stop,
                            ),
                        )
                    )
                    next_tile += 1
        return DetectorPixelMass(
            image_A2=image,
            rods=selected,
            branch=branch,
            per_rod_detector_mass_A2=per_rod,
            total_detector_mass_A2=fsum(per_rod),
            quadrature=quadrature,
            sampled_valid_pixel_center=sampled_valid_pixel_center,
            adaptive_refined_pixel_count=refined_count,
            adaptive_unresolved_pixel_count=unresolved_count,
            sampled_invalid_pixel_count=invalid_count,
            coordinate_evaluation_count=evaluation_count,
            estimated_l1_error_A2=estimated_l1_error_A2,
            execution_backend="numba_nogil_thread_tiles.v1",
        )

    def _integrate_native_pixels_fixed(
        self,
        *,
        rods: tuple[Rod, ...],
        branch: int,
        quadrature: DetectorQuadrature,
    ) -> DetectorPixelMass:
        """Integrate the a.e. density over native boxes with declared quadrature."""

        selected = self._validated_intensity_rods(rods)
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        if not isinstance(quadrature, DetectorQuadrature):
            raise TypeError("quadrature must be DetectorQuadrature")
        rows, columns = self._instrument.detector_shape_rc
        image = np.zeros((rows, columns), dtype=np.float64)
        per_rod = np.zeros(len(selected), dtype=np.float64)
        refine_folds = (
            quadrature.fold_gauss_order > quadrature.pixel_gauss_order
            or quadrature.fold_subdivision_count > 1
        )
        fold_candidate = (
            self._fold_candidate_mask(
                rods=selected,
                row_chunk_size=quadrature.row_chunk_size,
            )
            if refine_folds
            else np.zeros((rows, columns), dtype=np.bool_)
        )
        gauss_node, gauss_weight = leggauss(quadrature.pixel_gauss_order)
        offset = 0.5 * gauss_node
        one_dimensional_weight = 0.5 * gauss_weight
        node_weight = one_dimensional_weight[:, None] * one_dimensional_weight[None, :]
        column_center = np.arange(columns, dtype=np.float64)
        for row_start in range(0, rows, quadrature.row_chunk_size):
            row_stop = min(row_start + quadrature.row_chunk_size, rows)
            row_center = np.arange(row_start, row_stop, dtype=np.float64)
            node_column, node_row = np.broadcast_arrays(
                column_center[None, :, None, None] + offset[None, None, None, :],
                row_center[:, None, None, None] + offset[None, None, :, None],
            )
            evaluated = self.evaluate_detector_coordinates(
                node_column,
                node_row,
                rods=selected,
                branch=branch,
            )
            if np.any(evaluated.caustic):
                raise FloatingPointError(
                    "a pixel quadrature node lies on a caustic; use a different even order"
                )
            weighted = evaluated.per_rod_density_A2_per_px2 * node_weight[None, None, :, :, None]
            pixel_rod_mass = np.sum(weighted, axis=(2, 3), dtype=np.float64)
            image[row_start:row_stop] = np.sum(pixel_rod_mass, axis=-1, dtype=np.float64)
            per_rod += np.sum(pixel_rod_mass, axis=(0, 1), dtype=np.float64)
            if refine_folds:
                counts = evaluated.per_rod_inverse_branch_count
                count_changes = np.any(
                    np.max(counts, axis=(2, 3)) != np.min(counts, axis=(2, 3)),
                    axis=-1,
                )
                fold_candidate[row_start:row_stop] |= count_changes

        refined_count = 0
        refinement_l1 = 0.0
        refinement_centroid_shift = 0.0
        if refine_folds:
            candidate_index = np.flatnonzero(fold_candidate).astype(np.int64)
            refined_count = int(candidate_index.size)
            if refined_count:
                base_candidate = self._integrate_selected_pixels(
                    flat_pixel_index=candidate_index,
                    rods=selected,
                    branch=branch,
                    gauss_order=quadrature.pixel_gauss_order,
                    subdivision_count=1,
                    node_chunk_size=quadrature.node_chunk_size,
                )
                comparison_subdivision = max(
                    1,
                    quadrature.fold_subdivision_count // 2,
                )
                comparison_candidate = self._integrate_selected_pixels(
                    flat_pixel_index=candidate_index,
                    rods=selected,
                    branch=branch,
                    gauss_order=quadrature.fold_gauss_order,
                    subdivision_count=comparison_subdivision,
                    node_chunk_size=quadrature.node_chunk_size,
                )
                refined_candidate = self._integrate_selected_pixels(
                    flat_pixel_index=candidate_index,
                    rods=selected,
                    branch=branch,
                    gauss_order=quadrature.fold_gauss_order,
                    subdivision_count=quadrature.fold_subdivision_count,
                    node_chunk_size=quadrature.node_chunk_size,
                )
                comparison_total = np.sum(comparison_candidate, axis=1, dtype=np.float64)
                refined_total = np.sum(refined_candidate, axis=1, dtype=np.float64)
                refinement_l1 = float(np.sum(np.abs(refined_total - comparison_total)))
                flat_image = image.reshape(-1)
                base_candidate_total = np.sum(base_candidate, axis=1, dtype=np.float64)
                base_total = float(np.sum(image, dtype=np.float64))
                pixel_row, pixel_column = np.divmod(candidate_index, columns)
                image_row_moment = float(
                    np.dot(
                        np.sum(image, axis=1, dtype=np.float64),
                        np.arange(rows, dtype=np.float64),
                    )
                )
                image_column_moment = float(
                    np.dot(
                        np.sum(image, axis=0, dtype=np.float64),
                        np.arange(columns, dtype=np.float64),
                    )
                )
                comparison_mass = base_total + float(
                    np.sum(comparison_total - base_candidate_total)
                )
                refined_mass = base_total + float(np.sum(refined_total - base_candidate_total))
                if comparison_mass < 0.0 or refined_mass < 0.0:
                    raise FloatingPointError("fold refinement produced negative detector mass")
                if comparison_mass == 0.0 and refined_mass == 0.0:
                    refinement_centroid_shift = 0.0
                elif comparison_mass == 0.0 or refined_mass == 0.0:
                    raise FloatingPointError(
                        "fold refinement changed a zero-mass field into a nonzero field"
                    )
                else:
                    comparison_centroid = np.array(
                        (
                            (
                                image_column_moment
                                + np.dot(pixel_column, comparison_total - base_candidate_total)
                            )
                            / comparison_mass,
                            (
                                image_row_moment
                                + np.dot(pixel_row, comparison_total - base_candidate_total)
                            )
                            / comparison_mass,
                        )
                    )
                    refined_centroid = np.array(
                        (
                            (
                                image_column_moment
                                + np.dot(pixel_column, refined_total - base_candidate_total)
                            )
                            / refined_mass,
                            (
                                image_row_moment
                                + np.dot(pixel_row, refined_total - base_candidate_total)
                            )
                            / refined_mass,
                        )
                    )
                    refinement_centroid_shift = float(
                        np.linalg.norm(refined_centroid - comparison_centroid)
                    )
                flat_image[candidate_index] = refined_total
                per_rod += np.sum(
                    refined_candidate - base_candidate,
                    axis=0,
                    dtype=np.float64,
                )
        return DetectorPixelMass(
            image_A2=image,
            rods=selected,
            branch=branch,
            per_rod_detector_mass_A2=per_rod,
            total_detector_mass_A2=fsum(per_rod),
            quadrature=quadrature,
            fold_refined_pixel_count=refined_count,
            fold_refinement_l1_A2=refinement_l1,
            fold_refinement_centroid_shift_px=refinement_centroid_shift,
        )

    def integrate_native_pixels(
        self,
        *,
        rods: tuple[Rod, ...],
        branch: int,
        quadrature: DetectorQuadrature,
    ) -> DetectorPixelMass:
        """Integrate continuous detector density over exact native pixel boxes."""

        self._require_physical_intensity_source()
        if not isinstance(quadrature, DetectorQuadrature):
            raise TypeError("quadrature must be DetectorQuadrature")
        if quadrature.method is PixelIntegrationMethod.FIXED_NUMPY:
            return self._integrate_native_pixels_fixed(
                rods=rods,
                branch=branch,
                quadrature=quadrature,
            )
        return self._integrate_native_pixels_adaptive_compiled(
            rods=rods,
            branch=branch,
            quadrature=quadrature,
        )


__all__ = [
    "DetectorCoordinateGeometry",
    "DetectorCoordinateIntensity",
    "DetectorEwaldMeasure",
    "DetectorLatentIntensity",
    "DetectorMappedGeometry",
    "DetectorPixelMass",
    "DetectorQuadrature",
    "DetectorVisibleEwaldCoating",
    "DetectorVisibleEwaldDirectionIntensity",
    "EwaldDirectionIntensity",
    "IntensityStatus",
    "PixelIntegrationMethod",
    "SampleQIntensityEnvelope",
    "SpecularDetectorGeometry",
    "evaluate_detector_coordinates_geometry",
    "map_ewald_geometry_to_detector",
]
