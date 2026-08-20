"""Incoherent incident-state average of the continuous detector field."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar, dataclass, field, replace
from math import fsum, isfinite, pi, tanh

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import (
    BraggSpaceConfig,
    MosaicParameters,
    Rod,
)
from painted_ewald.rotations import mosaic_axes
from painted_ewald.validation import integer, positive_integer
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.optics import (
    DETECTOR_PATH_ATTENUATION_MODEL_ID,
    INCIDENT_ILLUMINATED_PATH_MODEL_ID,
    incident_illuminated_path_weight,
    mode_decay_constant,
)
from rasim_next.pipeline._continuous_detector_kernel import (
    CompiledDetectorEvaluator,
    _compile_detector_projection,
    pack_bi2se3_two_h_structures,
)
from rasim_next.pipeline.bragg_space import Bi2X3FiniteStackStrength
from rasim_next.pipeline.continuous_detector import (
    DetectorPixelMass,
    DetectorQuadrature,
    PixelIntegrationMethod,
    SampleQIntensityEnvelope,
    _compile_detector_state,
    _float_array,
    _subdivided_legendre_rule,
)
from rasim_next.reflectivity import (
    LOCAL_LAMELLA_INTERFACE,
    CompiledParrattStitch,
    ParrattStitchStack,
    compile_parratt_stitch,
    parratt_stitch_interface_assumption,
)
from rasim_next.sampling.source import require_physical_intensity_source_model

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
_ARRAY_OWNERSHIP_TOKEN = object()


def _uses_local_lamella_stitch(stack: ParrattStitchStack | None) -> bool:
    return stack is not None and stack.interface_assumption == LOCAL_LAMELLA_INTERFACE


def _compile_source_parratt_stitch(
    stack: ParrattStitchStack | None,
    strength: Bi2X3FiniteStackStrength,
    rods: tuple[Rod, ...],
    *,
    wavelength_A: float,
    film_refractive_index: complex,
    film_thickness_A: float,
) -> CompiledParrattStitch | None:
    if stack is None:
        return None
    m0_rod = next((rod for rod in rods if rod.h == 0 and rod.k == 0), None)
    if m0_rod is None:
        return None
    k_norm = 2.0 * np.pi / float(wavelength_A)
    b3_norm = float(np.linalg.norm(strength.reciprocal_basis_Ainv[:, 2]))
    c_A = 2.0 * np.pi / b3_norm
    return compile_parratt_stitch(
        stack,
        lambda layer: strength.evaluate_profile(
            rod=m0_rod,
            L=layer,
            k_norm_Ainv=k_norm,
        ),
        wavelength_A=wavelength_A,
        film_refractive_index=film_refractive_index,
        film_thickness_A=film_thickness_A,
        c_A=c_A,
    )


def _retained_source_parratt_stitch(state: object) -> CompiledParrattStitch | None:
    """Keep the source-wavelength overlap normalization frozen during structure fitting."""

    if int(state.specular_stitch_code) == 0:
        return None
    return CompiledParrattStitch(
        film_refractive_index=state.refractive_index,
        substrate_refractive_index=state.specular_substrate_refractive_index,
        film_thickness_A=state.film_thickness_A,
        top_roughness_A=state.specular_top_roughness_A,
        bottom_roughness_A=state.specular_bottom_roughness_A,
        qc_Ainv=state.specular_qc_Ainv,
        zero_strength_A2=state.specular_zero_strength_A2,
        dimensionless_scale_factor=state.specular_scale_factor,
        blend_bounds_q_over_qc=(
            state.specular_blend_lower_q_over_qc,
            state.specular_blend_upper_q_over_qc,
        ),
        blend_selection=(
            "fallback"
            if (
                state.specular_blend_lower_q_over_qc == 3.0
                and state.specular_blend_upper_q_over_qc == 6.0
            )
            else "automatic"
        ),
        interface_assumption=parratt_stitch_interface_assumption(int(state.specular_stitch_code)),
    )


def _instrument_revision(instrument: CompiledInstrument) -> str:
    """Return the complete immutable detector/sample instrument identity."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    return canonical_revision_sha256(
        ("definition_id", "source_averaged_detector_instrument.v3"),
        ("lab_from_sample_rotation", instrument.lab_from_sample.rotation),
        ("lab_from_sample_translation_m", instrument.lab_from_sample.translation_m),
        ("sample_from_crystal_rotation", instrument.sample_from_crystal.rotation),
        ("sample_from_crystal_translation_m", instrument.sample_from_crystal.translation_m),
        ("lab_from_detector_rotation", instrument.lab_from_detector.rotation),
        ("lab_from_detector_translation_m", instrument.lab_from_detector.translation_m),
        ("detector_shape_rc", np.asarray(instrument.detector_shape_rc, dtype=np.int64)),
        ("detector_row_pitch_m", instrument.detector_row_pitch_m),
        ("detector_column_pitch_m", instrument.detector_column_pitch_m),
        (
            "detector_reference_coordinate_px",
            np.asarray(instrument.detector_reference_coordinate_px, dtype=np.float64),
        ),
        ("sample_support_model_id", instrument.sample_support_model_id),
        ("sample_width_is_unbounded", int(instrument.sample_width_m is None)),
        (
            "sample_width_m",
            0.0 if instrument.sample_width_m is None else instrument.sample_width_m,
        ),
        ("sample_length_is_unbounded", int(instrument.sample_length_m is None)),
        (
            "sample_length_m",
            0.0 if instrument.sample_length_m is None else instrument.sample_length_m,
        ),
        ("film_thickness_A", instrument.film_thickness_A),
        ("detector_path_medium_id", instrument.detector_path_medium_id),
        (
            "detector_path_linear_attenuation_m_inv",
            instrument.detector_path_linear_attenuation_m_inv,
        ),
        (
            "detector_path_wavelength_A",
            np.asarray(instrument.detector_path_wavelength_A, dtype=np.float64),
        ),
        (
            "detector_path_linear_attenuation_m_inv_by_wavelength",
            np.asarray(
                instrument.detector_path_linear_attenuation_m_inv_by_wavelength,
                dtype=np.float64,
            ),
        ),
    )


def _detector_native_chart_revision(instrument: CompiledInstrument) -> str:
    """Hash only the detector-native chart shared across sample poses."""

    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    return canonical_revision_sha256(
        ("definition_id", "detector_native_chart.v1"),
        ("lab_from_detector_rotation", instrument.lab_from_detector.rotation),
        ("lab_from_detector_translation_m", instrument.lab_from_detector.translation_m),
        ("detector_shape_rc", np.asarray(instrument.detector_shape_rc, dtype=np.int64)),
        ("detector_row_pitch_m", instrument.detector_row_pitch_m),
        ("detector_column_pitch_m", instrument.detector_column_pitch_m),
        (
            "detector_reference_coordinate_px",
            np.asarray(instrument.detector_reference_coordinate_px, dtype=np.float64),
        ),
    )


def source_averaged_detector_instrument_revision(
    detector: SourceAveragedDetectorEwaldMeasure,
) -> str:
    """Return the complete immutable instrument identity used by a detector."""

    return _instrument_revision(detector.instrument)


def source_averaged_detector_geometry_revision(
    detector: SourceAveragedDetectorEwaldMeasure,
) -> str:
    """Bind reusable geometry plans to their exact detector and active rods.

    Mosaic, structure strength, and the sample-Q intensity envelope are
    intentionally excluded because they are candidate physics rebound during
    fitting without changing detector geometry.
    """

    if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
        raise TypeError("detector must be SourceAveragedDetectorEwaldMeasure")
    rods = detector.rods
    return canonical_revision_sha256(
        ("definition_id", "source_averaged_detector_geometry.v1"),
        ("source_revision", detector.incident.states.source_revision),
        ("material_revision", detector.material.material_revision),
        ("incident_model_id", detector.incident.states.incident_model_id),
        ("instrument_revision", source_averaged_detector_instrument_revision(detector)),
        ("reciprocal_basis_Ainv", detector.strength_model.reciprocal_basis_Ainv),
        ("rod_catalog_revision", detector.rod_catalog_revision),
        ("active_rod_h", np.asarray([rod.h for rod in rods], dtype=np.int64)),
        ("active_rod_k", np.asarray([rod.k for rod in rods], dtype=np.int64)),
        ("active_rod_population", np.asarray([rod.population for rod in rods], dtype=np.float64)),
    )


def _incidence_angle_static_physics_revision(
    *,
    reciprocal_basis_Ainv: ArrayLike,
    strength_model_revision: str,
    mosaic: MosaicParameters,
    intensity_envelope: SampleQIntensityEnvelope,
    material_revision: str,
    incident_model_id: str,
    instrument: CompiledInstrument,
    phase_polarization_weight: float,
    specular_stitch_stack: ParrattStitchStack | None,
) -> str:
    """Hash detector physics that must remain fixed while incidence pose varies."""

    fields: list[tuple[str, object]] = [
        ("definition_id", "incidence_angle_static_detector_physics.v2"),
        ("reciprocal_basis_Ainv", np.asarray(reciprocal_basis_Ainv, dtype=np.float64)),
        ("strength_model_revision", strength_model_revision),
        ("material_revision", material_revision),
        ("incident_model_id", incident_model_id),
        ("mosaic_gaussian_sigma_rad", mosaic.gaussian_sigma_rad),
        ("mosaic_lorentzian_half_width_rad", mosaic.lorentzian_half_width_rad),
        ("mosaic_lorentzian_probability", mosaic.lorentzian_probability),
        ("mosaic_alpha_panel_count", mosaic.alpha_panel_count),
        ("mosaic_alpha_gauss_order", mosaic.alpha_gauss_order),
        ("mosaic_azimuth_count", mosaic.azimuth_count),
        ("mosaic_azimuth_phase_rad", mosaic.azimuth_phase_rad),
        ("intensity_envelope_u_radial_A2", intensity_envelope.u_radial_A2),
        ("intensity_envelope_u_normal_A2", intensity_envelope.u_normal_A2),
        ("phase_polarization_weight", phase_polarization_weight),
        ("incident_illuminated_path_model_id", INCIDENT_ILLUMINATED_PATH_MODEL_ID),
        ("detector_path_attenuation_model_id", DETECTOR_PATH_ATTENUATION_MODEL_ID),
        ("sample_from_crystal_rotation", instrument.sample_from_crystal.rotation),
        ("sample_from_crystal_translation_m", instrument.sample_from_crystal.translation_m),
        ("sample_support_model_id", instrument.sample_support_model_id),
        ("sample_width_is_unbounded", int(instrument.sample_width_m is None)),
        (
            "sample_width_m",
            0.0 if instrument.sample_width_m is None else instrument.sample_width_m,
        ),
        ("sample_length_is_unbounded", int(instrument.sample_length_m is None)),
        (
            "sample_length_m",
            0.0 if instrument.sample_length_m is None else instrument.sample_length_m,
        ),
        ("film_thickness_A", instrument.film_thickness_A),
        ("detector_path_medium_id", instrument.detector_path_medium_id),
        (
            "detector_path_linear_attenuation_m_inv",
            instrument.detector_path_linear_attenuation_m_inv,
        ),
        (
            "detector_path_wavelength_A",
            np.asarray(instrument.detector_path_wavelength_A, dtype=np.float64),
        ),
        (
            "detector_path_linear_attenuation_m_inv_by_wavelength",
            np.asarray(
                instrument.detector_path_linear_attenuation_m_inv_by_wavelength,
                dtype=np.float64,
            ),
        ),
        ("specular_stitch_present", int(specular_stitch_stack is not None)),
    ]
    if specular_stitch_stack is not None:
        fields.extend(
            (
                (
                    "specular_substrate_refractive_index",
                    specular_stitch_stack.substrate_refractive_index,
                ),
                ("specular_top_roughness_A", specular_stitch_stack.top_roughness_A),
                ("specular_bottom_roughness_A", specular_stitch_stack.bottom_roughness_A),
                ("specular_model_id", specular_stitch_stack.model_id),
                (
                    "specular_interface_assumption",
                    specular_stitch_stack.interface_assumption,
                ),
            )
        )
    return canonical_revision_sha256(*fields)


def _detector_pose_arrays(
    instrument: CompiledInstrument,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    detector_rotation = instrument.lab_from_detector.rotation
    column_step_lab = detector_rotation[:, 0] * instrument.detector_column_pitch_m
    row_step_lab = detector_rotation[:, 1] * instrument.detector_row_pitch_m
    reference_column, reference_row = instrument.detector_reference_coordinate_px
    detector_zero_lab = (
        instrument.lab_from_detector.translation_m
        - reference_column * column_step_lab
        - reference_row * row_step_lab
    )
    arrays = (
        np.ascontiguousarray(detector_zero_lab),
        np.ascontiguousarray(column_step_lab),
        np.ascontiguousarray(row_step_lab),
        np.ascontiguousarray(np.cross(column_step_lab, row_step_lab)),
        np.ascontiguousarray(instrument.sample_from_lab.rotation),
    )
    for value in arrays:
        value.setflags(write=False)
    return arrays


def _geometry_rebind_instrument_invariants(
    old: CompiledInstrument,
    new: CompiledInstrument,
) -> bool:
    return (
        new.detector_shape_rc == old.detector_shape_rc
        and new.detector_row_pitch_m == old.detector_row_pitch_m
        and new.detector_column_pitch_m == old.detector_column_pitch_m
        and new.detector_reference_coordinate_px == old.detector_reference_coordinate_px
        and new.sample_support_model_id == old.sample_support_model_id
        and new.sample_width_m == old.sample_width_m
        and new.sample_length_m == old.sample_length_m
        and new.film_thickness_A == old.film_thickness_A
        and new.detector_path_medium_id == old.detector_path_medium_id
        and new.detector_path_linear_attenuation_m_inv == old.detector_path_linear_attenuation_m_inv
        and new.detector_path_wavelength_A == old.detector_path_wavelength_A
        and new.detector_path_linear_attenuation_m_inv_by_wavelength
        == old.detector_path_linear_attenuation_m_inv_by_wavelength
        and np.array_equal(
            new.sample_from_crystal.rotation,
            old.sample_from_crystal.rotation,
        )
        and np.array_equal(
            new.sample_from_crystal.translation_m,
            old.sample_from_crystal.translation_m,
        )
    )


def _validated_cuda_coordinate_chunk_size(
    execution_backend: str,
    cuda_coordinate_chunk_size: int | None,
) -> int | None:
    if cuda_coordinate_chunk_size is None:
        return None
    chunk_size = positive_integer(cuda_coordinate_chunk_size, "cuda_coordinate_chunk_size")
    if execution_backend != "cuda":
        raise ValueError("cuda_coordinate_chunk_size requires the CUDA execution backend")
    return chunk_size


@dataclass(frozen=True, slots=True)
class SourceAveragedDetectorCoordinateIntensity:
    """Detector-coordinate density summed over independent source states.

    A coordinate selects a different outgoing ray for every sampled source
    position. There is therefore no representative ``kf`` or ``Q`` attached
    to this reduced result; those state-specific values are evaluated before
    the incoherent intensity sum.
    """

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    branch: int | None
    per_rod_density_A2_per_px2: FloatArray
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.int64]
    source_state_count: int
    source_revision: str
    root_policy: str = "single_nonzero_root.v1"
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0):
            raise ValueError("column_px must be real")
        column = np.array(
            supplied_column.real,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        shape = column.shape
        if not np.all(np.isfinite(column)):
            raise ValueError("column_px must be finite")
        row = _float_array(self.row_px, shape, "row_px")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        if not isinstance(self.rod_catalog_revision, str) or not self.rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        if self.branch is None:
            if self.root_policy != "all_retained_roots.v1":
                raise ValueError("all-root results require all_retained_roots.v1")
        elif self.branch not in {1, 2} or self.root_policy != "single_nonzero_root.v1":
            raise ValueError("a single-root result requires branch 1 or 2")
        supplied_per_rod = np.asarray(self.per_rod_density_A2_per_px2)
        if np.iscomplexobj(supplied_per_rod) and np.any(supplied_per_rod.imag != 0.0):
            raise ValueError("per-rod detector density must be real")
        per_rod = np.array(
            supplied_per_rod.real,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if per_rod.shape != (*shape, len(rods)) or np.any(np.isnan(per_rod)):
            raise ValueError("per-rod detector density has the wrong shape or contains NaN")
        if np.any(per_rod < 0.0):
            raise ValueError("per-rod detector density must be nonnegative")
        supplied_total = np.asarray(self.density_A2_per_px2)
        if np.iscomplexobj(supplied_total) and np.any(supplied_total.imag != 0.0):
            raise ValueError("detector density must be real")
        total = np.array(supplied_total.real, dtype=np.float64, copy=True, order="C")
        if total.shape != shape or np.any(np.isnan(total)) or np.any(total < 0.0):
            raise ValueError("detector density must be nonnegative and contain no NaN")
        expected = np.sum(per_rod, axis=-1, dtype=np.float64)
        finite = np.isfinite(expected)
        scale = np.maximum(np.abs(expected[finite]), np.finfo(np.float64).tiny)
        if not np.all(
            np.abs(total[finite] - expected[finite]) <= 1024.0 * np.finfo(np.float64).eps * scale
        ) or not np.array_equal(np.isinf(total), np.isinf(expected)):
            raise ValueError("detector density must equal the physical rod sum")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != (*shape, len(rods)):
            raise ValueError("caustic must have one flag per detector coordinate and rod")
        state_count = positive_integer(self.source_state_count, "source_state_count")
        valid_count = np.array(self.valid_source_count, dtype=np.int64, copy=True, order="C")
        if valid_count.shape != shape or np.any((valid_count < 0) | (valid_count > state_count)):
            raise ValueError("valid_source_count must lie within the source batch")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
            "hybrid_cuda_cpu_local_m0.v1",
            "numpy_cpu_sparse_source_averaged.v1",
        }:
            raise ValueError("unsupported detector-coordinate execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        device_backend = self.execution_backend in {
            "numba_cuda_source_averaged.v1",
            "hybrid_cuda_cpu_local_m0.v1",
        }
        if device_backend != (self.execution_device is not None):
            raise ValueError("execution_device must identify every CUDA-backed result")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        has_m0 = any(rod.h == 0 and rod.k == 0 for rod in rods)
        if has_m0:
            if self.branch is not None:
                raise ValueError("m=0 is available only in an all-root result")
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) < 0.0:
                raise ValueError("detector-visible m=0 requires a nonnegative support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("an m=0 support gap requires an m=0 rod")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate measure")
        for value in (column, row, per_rod, total, caustic, valid_count):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "per_rod_density_A2_per_px2", per_rod)
        object.__setattr__(self, "density_A2_per_px2", total)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "valid_source_count", valid_count)
        object.__setattr__(self, "source_state_count", state_count)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)


@dataclass(frozen=True, slots=True)
class SourceAveragedDetectorCoordinateDensity:
    """All-source, all-rod, all-root density on continuous detector coordinates.

    Physical rods remain explicit provenance, but no rod-valued detector array crosses this
    boundary. Every state-specific outgoing ray, inverse root, and physical-rod intensity is
    reduced before a pixel or macrobin integrator receives the result.
    """

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    density_A2_per_px2: FloatArray
    caustic: BoolArray
    valid_source_count: NDArray[np.int64]
    source_state_count: int
    source_revision: str
    rod_catalog_revision: str | None = field(default=None, kw_only=True)
    root_policy: str = "all_retained_roots.v1"
    detector_visible_m0_q_gap_Ainv: float | None = None
    measure_id: str = "raw_detector_coordinate_density_A2_per_px2.v1"
    execution_backend: str = "numba_cpu_source_averaged.v1"
    execution_device: str | None = None
    branch: None = field(init=False, default=None)

    def __post_init__(self) -> None:
        supplied_column = np.asarray(self.column_px)
        if np.iscomplexobj(supplied_column) and np.any(supplied_column.imag != 0.0):
            raise ValueError("column_px must be real")
        column = np.array(supplied_column.real, dtype=np.float64, copy=True, order="C")
        shape = column.shape
        if not np.all(np.isfinite(column)):
            raise ValueError("column_px must be finite")
        row = _float_array(self.row_px, shape, "row_px")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        if self.rod_catalog_revision is not None and (
            not isinstance(self.rod_catalog_revision, str) or not self.rod_catalog_revision
        ):
            raise ValueError("rod_catalog_revision must be None or nonempty")
        supplied_density = np.asarray(self.density_A2_per_px2)
        if np.iscomplexobj(supplied_density) and np.any(supplied_density.imag != 0.0):
            raise ValueError("detector density must be real")
        density = np.array(supplied_density.real, dtype=np.float64, copy=True, order="C")
        if density.shape != shape or np.any(np.isnan(density)) or np.any(density < 0.0):
            raise ValueError("detector density must be nonnegative and contain no NaN")
        caustic = np.array(self.caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != shape:
            raise ValueError("caustic must have one flag per detector coordinate")
        if np.any(np.isinf(density) & ~caustic):
            raise ValueError("infinite detector density requires a caustic")
        state_count = positive_integer(self.source_state_count, "source_state_count")
        valid_count = np.array(self.valid_source_count, dtype=np.int64, copy=True, order="C")
        if valid_count.shape != shape or np.any((valid_count < 0) | (valid_count > state_count)):
            raise ValueError("valid_source_count must lie within the source batch")
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("total detector density requires all_retained_roots.v1")
        if self.execution_backend not in {
            "numba_cpu_source_averaged.v1",
            "numba_cuda_source_averaged.v1",
            "hybrid_cuda_cpu_local_m0.v1",
        }:
            raise ValueError("unsupported detector-coordinate execution backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be None or a nonempty string")
        device_backend = self.execution_backend in {
            "numba_cuda_source_averaged.v1",
            "hybrid_cuda_cpu_local_m0.v1",
        }
        if device_backend != (self.execution_device is not None):
            raise ValueError("execution_device must identify every CUDA-backed result")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        has_m0 = any(rod.family_m == 0 for rod in rods)
        if has_m0:
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) < 0.0:
                raise ValueError("detector-visible m=0 requires a nonnegative support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("an m=0 support gap requires an m=0 rod")
        if self.measure_id != "raw_detector_coordinate_density_A2_per_px2.v1":
            raise ValueError("unsupported detector-coordinate measure")
        for value in (column, row, density, caustic, valid_count):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "density_A2_per_px2", density)
        object.__setattr__(self, "caustic", caustic)
        object.__setattr__(self, "valid_source_count", valid_count)
        object.__setattr__(self, "source_state_count", state_count)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)


@dataclass(frozen=True, slots=True)
class MonteCarloDetectorPixelMass:
    """Weighted forward-sampling estimate of native detector-pixel mass."""

    image_A2: FloatArray
    replicate_total_mass_A2: FloatArray
    total_detector_mass_A2: float
    draws_per_source_state: int
    source_state_count: int
    active_source_state_count: int
    attempted_root_count: int
    visible_hit_count: int
    maximum_root_deposit_A2: float
    seed: int
    rods: tuple[Rod, ...]
    source_revision: str
    rod_catalog_revision: str
    detector_visible_m0_q_gap_Ainv: float | None
    execution_backend: str = "numba_cpu_forward_monte_carlo.v2"
    execution_device: str | None = None
    execution_worker_count: int | None = 1
    _array_ownership_token: InitVar[object | None] = None
    measure_id: str = field(
        init=False,
        default="raw_detector_pixel_mass_monte_carlo_estimate_A2.v1",
    )
    proposal_id: str = field(init=False, default="folded_wrapped_mosaic_full_beta.v1")
    rng_model_id: str = field(
        init=False,
        default="numpy.philox.fixed_width_source_draw.v1",
    )

    def __post_init__(self, _array_ownership_token: object | None) -> None:
        if _array_ownership_token is _ARRAY_OWNERSHIP_TOKEN:
            image = np.asarray(self.image_A2, dtype=np.float64, order="C")
            if image.base is not None or not image.flags.owndata:
                raise RuntimeError("internally owned Monte Carlo images must own their storage")
        else:
            image = np.array(self.image_A2, dtype=np.float64, copy=True, order="C")
        if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError("image_A2 must be a finite nonnegative detector array")
        draws = positive_integer(self.draws_per_source_state, "draws_per_source_state")
        replicate = _float_array(
            self.replicate_total_mass_A2,
            (draws,),
            "replicate_total_mass_A2",
        )
        if np.any(replicate < 0.0):
            raise ValueError("replicate_total_mass_A2 must be nonnegative")
        total = float(self.total_detector_mass_A2)
        maximum = float(self.maximum_root_deposit_A2)
        if not isfinite(total) or total < 0.0 or not isfinite(maximum) or maximum < 0.0:
            raise ValueError("Monte Carlo detector masses must be finite and nonnegative")
        attempted = positive_integer(self.attempted_root_count, "attempted_root_count")
        expected_total = fsum(replicate) / draws
        rounding_ratio = max(attempted, image.size, draws) * np.finfo(np.float64).eps
        if rounding_ratio >= 0.01:
            raise ValueError("Monte Carlo work count exceeds the conservation roundoff budget")
        tolerance = (
            rounding_ratio / (1.0 - rounding_ratio) + 64.0 * np.finfo(np.float64).eps
        ) * max(expected_total, np.finfo(np.float64).tiny)
        if abs(total - expected_total) > tolerance or abs(float(np.sum(image)) - total) > tolerance:
            raise ValueError("image, replicate, and total Monte Carlo masses disagree")
        source_count = positive_integer(self.source_state_count, "source_state_count")
        active_count = positive_integer(self.active_source_state_count, "active_source_state_count")
        if active_count > source_count:
            raise ValueError("active_source_state_count exceeds source_state_count")
        visible = integer(self.visible_hit_count, "visible_hit_count")
        if visible < 0 or visible > attempted:
            raise ValueError("visible_hit_count exceeds the attempted-root ledger")
        seed = integer(self.seed, "seed")
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        rods = tuple(self.rods)
        if not rods or not all(isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical rod")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        if any(rod.family_m == 0 for rod in rods):
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) <= 0.0:
                raise ValueError("detector-visible m=0 requires a positive reciprocal support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("an m=0 support gap requires an m=0 rod")
        for name in ("source_revision", "rod_catalog_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be nonempty")
        if self.execution_backend not in {
            "numba_cpu_forward_monte_carlo.v2",
            "numba_cuda_forward_monte_carlo.v1",
        }:
            raise ValueError("unsupported forward Monte Carlo execution_backend")
        if (self.execution_backend == "numba_cuda_forward_monte_carlo.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("execution_device must identify exactly the CUDA backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("execution_device must be a nonempty string when supplied")
        if self.execution_worker_count is None:
            if self.execution_backend != "numba_cuda_forward_monte_carlo.v1":
                raise ValueError("only the CUDA backend omits a CPU execution worker count")
        else:
            workers = positive_integer(self.execution_worker_count, "execution_worker_count")
            if self.execution_backend != "numba_cpu_forward_monte_carlo.v2" or workers > 4:
                raise ValueError("CPU forward Monte Carlo uses between one and four workers")
            object.__setattr__(self, "execution_worker_count", workers)
        image.setflags(write=False)
        object.__setattr__(self, "image_A2", image)
        object.__setattr__(self, "replicate_total_mass_A2", replicate)
        object.__setattr__(self, "total_detector_mass_A2", total)
        object.__setattr__(self, "draws_per_source_state", draws)
        object.__setattr__(self, "source_state_count", source_count)
        object.__setattr__(self, "active_source_state_count", active_count)
        object.__setattr__(self, "attempted_root_count", attempted)
        object.__setattr__(self, "visible_hit_count", visible)
        object.__setattr__(self, "maximum_root_deposit_A2", maximum)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)


@dataclass(frozen=True, slots=True)
class MonteCarloDetectorPresentation:
    """Full-native float32 display lease valid until the sampler's next operation."""

    image_A2: Float32Array
    total_detector_mass_A2: float
    draws_per_source_state: int
    source_state_count: int
    active_source_state_count: int
    attempted_root_count: int
    visible_hit_count: int
    maximum_root_deposit_A2: float
    seed: int
    rod_count: int
    execution_backend: str
    execution_device: str | None

    def __post_init__(self) -> None:
        image = np.asarray(self.image_A2)
        if image.dtype != np.float32 or image.ndim != 2 or not image.flags.c_contiguous:
            raise ValueError(
                "presentation image must be a contiguous two-dimensional float32 array"
            )
        draws = positive_integer(self.draws_per_source_state, "draws_per_source_state")
        source_count = positive_integer(self.source_state_count, "source_state_count")
        active_count = positive_integer(
            self.active_source_state_count,
            "active_source_state_count",
        )
        attempted = positive_integer(self.attempted_root_count, "attempted_root_count")
        visible = integer(self.visible_hit_count, "visible_hit_count")
        rods = positive_integer(self.rod_count, "rod_count")
        seed = integer(self.seed, "seed")
        total = float(self.total_detector_mass_A2)
        maximum = float(self.maximum_root_deposit_A2)
        if active_count > source_count or visible < 0 or visible > attempted:
            raise ValueError("presentation work ledgers are inconsistent")
        if not isfinite(total) or total < 0.0 or not isfinite(maximum) or maximum < 0.0:
            raise ValueError("presentation detector masses must be finite and nonnegative")
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if self.execution_backend not in {
            "numba_cpu_forward_monte_carlo.v2",
            "numba_cuda_forward_monte_carlo.v1",
        }:
            raise ValueError("unsupported presentation execution backend")
        if (self.execution_backend == "numba_cuda_forward_monte_carlo.v1") != (
            self.execution_device is not None
        ):
            raise ValueError("presentation device must identify exactly the CUDA backend")
        if self.execution_device is not None and (
            not isinstance(self.execution_device, str) or not self.execution_device
        ):
            raise ValueError("presentation device must be a nonempty string when supplied")
        object.__setattr__(self, "draws_per_source_state", draws)
        object.__setattr__(self, "source_state_count", source_count)
        object.__setattr__(self, "active_source_state_count", active_count)
        object.__setattr__(self, "attempted_root_count", attempted)
        object.__setattr__(self, "visible_hit_count", visible)
        object.__setattr__(self, "rod_count", rods)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "total_detector_mass_A2", total)
        object.__setattr__(self, "maximum_root_deposit_A2", maximum)


class MonteCarloSamplingCancelled(RuntimeError):
    """Raised when a caller supersedes forward Monte Carlo work."""


def _sample_mosaic_orientation_matrix(
    mosaic: MosaicParameters,
    draw_count: int,
    *,
    seed: int,
    source_state_count: int,
    draw_start: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Draw source-count-independent fixed-width Philox lanes for every stratum."""

    start = integer(draw_start, "draw_start")
    if start < 0 or start >= draw_count:
        raise ValueError("draw_start must precede draw_count")
    if source_state_count * 2 >= 2**64 or draw_count >= 2**128:
        raise ValueError("Philox source or draw count exceeds the reserved counter layout")
    latent_uniform = np.empty((draw_count - start, source_state_count, 5), dtype=np.float64)
    philox_key = np.random.SeedSequence(seed).generate_state(2, dtype=np.uint64)
    for local_draw_index, draw_index in enumerate(range(start, draw_count)):
        fixed_width = np.random.Generator(
            np.random.Philox(key=philox_key, counter=draw_index << 64)
        ).random((source_state_count, 8))
        latent_uniform[local_draw_index] = fixed_width[:, :5]
    eta = mosaic.lorentzian_probability
    radius = np.sqrt(
        -2.0
        * np.log(
            np.maximum(
                latent_uniform[:, :, 1],
                np.finfo(np.float64).tiny,
            )
        )
    )
    gaussian_tilt = mosaic.gaussian_sigma_rad * radius * np.cos(2.0 * pi * latent_uniform[:, :, 2])
    scale = tanh(0.5 * mosaic.lorentzian_half_width_rad)
    lorentzian_tilt = 2.0 * np.arctan(scale * np.tan(pi * (latent_uniform[:, :, 3] - 0.5)))
    signed_tilt = np.where(
        latent_uniform[:, :, 0] < eta,
        lorentzian_tilt,
        gaussian_tilt,
    )
    wrapped = np.remainder(signed_tilt + pi, 2.0 * pi) - pi
    alpha = np.ascontiguousarray(np.abs(wrapped).T)
    beta = np.ascontiguousarray((2.0 * pi * latent_uniform[:, :, 4]).T)
    return alpha, beta


@dataclass(frozen=True, slots=True)
class _IndexedCompiledEvaluator:
    evaluator: CompiledDetectorEvaluator
    master_rod_index: NDArray[np.int64]
    incident_state_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, CompiledDetectorEvaluator):
            raise TypeError("evaluator must be CompiledDetectorEvaluator")
        indices = np.array(self.master_rod_index, dtype=np.int64, copy=True, order="C")
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0):
            raise ValueError("master_rod_index must contain nonnegative indices")
        if np.unique(indices).size != indices.size:
            raise ValueError("master_rod_index must not repeat an index")
        state_index = self.incident_state_index
        if isinstance(state_index, bool) or not isinstance(state_index, (int, np.integer)):
            raise TypeError("incident_state_index must be an integer")
        if state_index < 0:
            raise ValueError("incident_state_index must be nonnegative")
        indices.setflags(write=False)
        object.__setattr__(self, "master_rod_index", indices)
        object.__setattr__(self, "incident_state_index", int(state_index))

    def with_evaluator(self, evaluator: CompiledDetectorEvaluator) -> _IndexedCompiledEvaluator:
        """Reuse the already validated immutable rod-index map."""

        if not isinstance(evaluator, CompiledDetectorEvaluator):
            raise TypeError("evaluator must be CompiledDetectorEvaluator")
        rebound = object.__new__(type(self))
        object.__setattr__(rebound, "evaluator", evaluator)
        object.__setattr__(rebound, "master_rod_index", self.master_rod_index)
        object.__setattr__(rebound, "incident_state_index", self.incident_state_index)
        return rebound


def _flatten_evaluators(
    evaluator_blocks: tuple[tuple[_IndexedCompiledEvaluator, ...], ...],
) -> tuple[_IndexedCompiledEvaluator, ...]:
    return tuple(indexed for block in evaluator_blocks for indexed in block)


def _inverse_fold_squared_geometry(
    evaluator: CompiledDetectorEvaluator,
    column_px: FloatArray,
    row_px: FloatArray,
    fold_radius_Ainv: float,
    parallel_norm_Ainv: float,
    *,
    normal_is_column: BoolArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    """Return the two inverse-root squares and an optional x-fold derivative."""

    state = evaluator.state
    displacement = (
        state.detector_zero_lab_m[None, :]
        + column_px[:, None] * state.detector_column_step_lab_m[None, :]
        + row_px[:, None] * state.detector_row_step_lab_m[None, :]
        - state.ray_origin_lab_m[None, :]
    )
    distance = np.linalg.norm(displacement, axis=1)
    direction = displacement / distance[:, None]
    kf_air_sample = (state.sample_from_lab @ (state.air_k0_Ainv * direction).T).T
    parallel_squared = np.sum(kf_air_sample[:, :2] ** 2, axis=1)
    film_normal_squared = state.internal_k_Ainv**2 - parallel_squared
    if np.any(film_normal_squared <= 0.0):
        raise FloatingPointError("fold plan encountered an unreachable exit direction")
    kf_film_z = np.sqrt(film_normal_squared)
    q_sample = np.column_stack((kf_air_sample[:, :2], kf_film_z))
    q_sample -= state.ki_film_sample_Ainv[None, :]
    q_local = q_sample @ state.sample_from_local
    qr_squared = q_local[:, 0] ** 2 + q_local[:, 1] ** 2
    fold_squared = qr_squared - fold_radius_Ainv**2
    w_squared = np.sum(q_local**2, axis=1) - parallel_norm_Ainv**2
    if normal_is_column is None:
        return fold_squared, w_squared, None

    detector_step = np.where(
        normal_is_column[:, None],
        state.detector_column_step_lab_m[None, :],
        state.detector_row_step_lab_m[None, :],
    )
    direction_step = (
        detector_step - direction * np.einsum("ij,ij->i", direction, detector_step)[:, None]
    ) / distance[:, None]
    kf_air_step = (state.sample_from_lab @ (state.air_k0_Ainv * direction_step).T).T
    kf_film_z_step = (
        -np.sum(
            kf_air_sample[:, :2] * kf_air_step[:, :2],
            axis=1,
        )
        / kf_film_z
    )
    q_local_step = np.column_stack((kf_air_step[:, :2], kf_film_z_step))
    q_local_step = q_local_step @ state.sample_from_local
    derivative = 2.0 * np.sum(q_local[:, :2] * q_local_step[:, :2], axis=1)
    return fold_squared, w_squared, derivative


@dataclass(frozen=True, slots=True)
class _ForwardMonteCarloBlockResult:
    raw_image_A2: FloatArray
    replicate_total_mass_A2: FloatArray
    attempted_root_count: int
    visible_hit_count: int
    maximum_root_weight_A2: float


def _forward_monte_carlo_blocks(
    evaluator_blocks: tuple[tuple[_IndexedCompiledEvaluator, ...], ...],
) -> tuple[tuple[_IndexedCompiledEvaluator, ...], ...]:
    evaluators = tuple(indexed for block in evaluator_blocks for indexed in block)
    block_size = max(1, (len(evaluators) + 3) // 4)
    return tuple(
        tuple(evaluators[start : start + block_size])
        for start in range(0, len(evaluators), block_size)
    )


def _raise_if_monte_carlo_cancelled(
    cancel_requested: Callable[[], bool] | None,
) -> None:
    if cancel_requested is not None and cancel_requested():
        raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")


def _accumulate_forward_monte_carlo_block(
    evaluators: tuple[_IndexedCompiledEvaluator, ...],
    *,
    alpha_rad: FloatArray,
    beta_rad: FloatArray,
    detector_pixel_count: int,
    cancel_requested: Callable[[], bool] | None,
) -> _ForwardMonteCarloBlockResult:
    """Accumulate one fixed source block into private full-native scratch."""

    _raise_if_monte_carlo_cancelled(cancel_requested)
    draw_count = alpha_rad.shape[1]
    raw_image = np.zeros(detector_pixel_count, dtype=np.float64)
    replicate_total = np.zeros(draw_count, dtype=np.float64)
    attempted_root_count = 0
    visible_hit_count = 0
    maximum_root_weight = 0.0
    for indexed in evaluators:
        _raise_if_monte_carlo_cancelled(cancel_requested)
        state_visible, state_maximum = indexed.evaluator.accumulate_latent_pixel_mass(
            alpha_rad[indexed.incident_state_index],
            beta_rad[indexed.incident_state_index],
            raw_image,
            replicate_total,
            image_weight_scale=1.0,
        )
        state_rods = indexed.evaluator.state.rod_hk_population
        m0_count = int(np.count_nonzero((state_rods[:, 0] == 0.0) & (state_rods[:, 1] == 0.0)))
        attempted_root_count += draw_count * (2 * state_rods.shape[0] - m0_count)
        visible_hit_count += state_visible
        maximum_root_weight = max(maximum_root_weight, state_maximum)
    _raise_if_monte_carlo_cancelled(cancel_requested)
    return _ForwardMonteCarloBlockResult(
        raw_image_A2=raw_image,
        replicate_total_mass_A2=replicate_total,
        attempted_root_count=attempted_root_count,
        visible_hit_count=visible_hit_count,
        maximum_root_weight_A2=maximum_root_weight,
    )


def _sum_compiled_evaluator_block(
    evaluators: tuple[_IndexedCompiledEvaluator, ...],
    column_px: FloatArray,
    row_px: FloatArray,
    branch: int | None,
    master_rod_count: int,
) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
    """Sum one fixed source-state block without materializing a state axis."""

    per_rod = np.zeros((column_px.size, master_rod_count), dtype=np.float64)
    caustic = np.zeros(per_rod.shape, dtype=np.bool_)
    valid_source_count = np.zeros(column_px.size, dtype=np.int64)
    for indexed in evaluators:
        if branch is None:
            density, _, state_caustic, state_valid = indexed.evaluator.evaluate_all_roots(
                column_px,
                row_px,
            )
        else:
            density, _, state_caustic, state_valid = indexed.evaluator.evaluate(
                column_px,
                row_px,
                branch=branch,
            )
        per_rod[:, indexed.master_rod_index] += density
        caustic[:, indexed.master_rod_index] |= state_caustic
        valid_source_count += state_valid
    if not evaluators:
        raise ValueError("a source-state block must contain at least one evaluator")
    return per_rod, caustic, valid_source_count


def _warm_compiled_evaluator_blocks(
    evaluator_blocks: tuple[tuple[_IndexedCompiledEvaluator, ...], ...],
    column_px: FloatArray,
    row_px: FloatArray,
    branch: int | None,
) -> None:
    """Compile the exact coordinate signature before any evaluator enters a worker thread."""

    first = next(
        (indexed.evaluator for block in evaluator_blocks for indexed in block),
        None,
    )
    if first is None:
        return
    empty_column = column_px[:0]
    empty_row = row_px[:0]
    if branch is None:
        first.evaluate_all_roots(empty_column, empty_row)
    else:
        first.evaluate(empty_column, empty_row, branch=branch)


def _reachable_master_rod_indices(
    rods: tuple[Rod, ...],
    reciprocal_basis_Ainv: FloatArray,
    k_norm_Ainv: float,
) -> NDArray[np.int64]:
    """Return stable master indices whose rod lines enter one elastic ball."""

    mean_axis, _ = mosaic_axes(reciprocal_basis_Ainv)
    q_parallel = np.asarray(
        [rod.h * reciprocal_basis_Ainv[:, 0] + rod.k * reciprocal_basis_Ainv[:, 1] for rod in rods],
        dtype=np.float64,
    )
    perpendicular = q_parallel - (q_parallel @ mean_axis)[:, None] * mean_axis
    distance = np.linalg.norm(perpendicular, axis=1)
    maximum_distance = 2.0 * float(k_norm_Ainv)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(maximum_distance, 1.0)
    result = np.flatnonzero(distance <= maximum_distance + tolerance).astype(np.int64)
    result.setflags(write=False)
    return result


class SourceAveragedDetectorEwaldMeasure:
    """Continuous detector density after an incoherent incident-state sum.

    Every valid state retains its source position, wavelength, refracted
    ``ki``, outgoing detector ray, exit refraction, attenuation, and
    Ewald-constrained ``Q``. The detector-coordinate densities are summed with
    their empirical source masses before pixel quadrature. This is the exact
    common-domain reduction when sampled source positions differ; no
    representative ``kf`` or per-state detector image is formed.
    """

    __slots__ = (
        "_detector_visible_m0_q_gap_Ainv",
        "_evaluator_blocks",
        "_incidence_angle_static_physics_revision",
        "_incidence_axis_angle_rad",
        "_incident",
        "_instrument",
        "_intensity_envelope",
        "_material",
        "_mosaic",
        "_phase_polarization_weight",
        "_reachable_rod_count_per_source_state",
        "_rod_catalog_revision",
        "_rods",
        "_scan_calibration_binding_revision",
        "_specular_stitch_stack",
        "_strength_model",
        "_valid_state_count",
        "_worker_count",
    )

    _MAX_STATE_BLOCK_COUNT = 32
    _COORDINATE_CHUNK_SIZE = 16_384
    _TOTAL_DENSITY_COORDINATE_CHUNK_SIZE = 4_096

    def __init__(
        self,
        *,
        reciprocal_basis_Ainv: ArrayLike,
        crystal_to_sample: ArrayLike,
        rods: tuple[Rod, ...],
        rod_catalog_revision: str,
        mosaic: MosaicParameters,
        strength_model: Bi2X3FiniteStackStrength,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
        incident: IncidentTransportResult,
        material: MaterialOptics,
        instrument: CompiledInstrument,
        phase_population_weight: float = 1.0,
        polarization_weight: float = 1.0,
        worker_count: int = 1,
        specular_stitch_stack: ParrattStitchStack | None = None,
        incidence_axis_angle_rad: float | None = None,
        scan_calibration_binding_revision: str | None = None,
    ) -> None:
        if not isinstance(incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        if not isinstance(mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(strength_model, Bi2X3FiniteStackStrength):
            raise TypeError("strength_model must be Bi2X3FiniteStackStrength")
        if specular_stitch_stack is not None and not isinstance(
            specular_stitch_stack, ParrattStitchStack
        ):
            raise TypeError("specular_stitch_stack must be ParrattStitchStack")
        envelope = SampleQIntensityEnvelope() if intensity_envelope is None else intensity_envelope
        if not isinstance(envelope, SampleQIntensityEnvelope):
            raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
        if mosaic.zero_tilt_probability_mass != 0.0:
            raise ValueError("source-averaged integration does not support zero-tilt atoms")
        selected = tuple(rods)
        if not selected or not all(isinstance(rod, Rod) for rod in selected):
            raise ValueError("rods must contain at least one Rod")
        if len({(rod.h, rod.k) for rod in selected}) != len(selected):
            raise ValueError("rods must not repeat a physical rod")
        if not isinstance(rod_catalog_revision, str) or not rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        phase_weight = float(phase_population_weight)
        polarization = float(polarization_weight)
        if not isfinite(phase_weight) or phase_weight < 0.0:
            raise ValueError("phase_population_weight must be finite and nonnegative")
        if not isfinite(polarization) or polarization < 0.0:
            raise ValueError("polarization_weight must be finite and nonnegative")
        workers = positive_integer(worker_count, "worker_count")
        axis_angle = None if incidence_axis_angle_rad is None else float(incidence_axis_angle_rad)
        if axis_angle is not None and not isfinite(axis_angle):
            raise ValueError("incidence_axis_angle_rad must be finite when provided")
        scan_binding = scan_calibration_binding_revision
        if scan_binding is not None and (not isinstance(scan_binding, str) or not scan_binding):
            raise ValueError("scan_calibration_binding_revision must be nonempty when provided")
        states = incident.states
        require_physical_intensity_source_model(states.source_sampling_model_id)
        if states.material_revision != material.material_revision:
            raise ValueError("incident states and material must have the same revision")
        if states.sample_geometry_revision != instrument.sample_geometry_revision:
            raise ValueError("incident states and instrument must have the same sample geometry")
        supplied_rotation = np.asarray(crystal_to_sample)
        if np.iscomplexobj(supplied_rotation) and np.any(supplied_rotation.imag != 0.0):
            raise ValueError("crystal_to_sample must be real")
        supplied_crystal_to_sample = np.asarray(supplied_rotation.real, dtype=np.float64)
        if supplied_crystal_to_sample.shape != (3, 3) or not np.all(
            np.isfinite(supplied_crystal_to_sample)
        ):
            raise ValueError("crystal_to_sample must be finite with shape (3, 3)")
        valid_state_index = np.flatnonzero(states.valid)
        if not valid_state_index.size:
            raise ValueError("source-averaged detector requires at least one valid incident state")
        maximum_air_k_Ainv = 2.0 * np.pi / float(np.min(states.wavelength_A[valid_state_index]))
        reference_config = BraggSpaceConfig(
            reciprocal_basis_Ainv=reciprocal_basis_Ainv,
            crystal_to_sample=supplied_crystal_to_sample,
            rods=selected,
            mosaic=mosaic,
            k_norm_Ainv=maximum_air_k_Ainv,
        )
        rotation_tolerance = 512.0 * np.finfo(np.float64).eps
        if not np.allclose(
            reference_config.crystal_to_sample,
            instrument.sample_from_crystal.rotation,
            rtol=0.0,
            atol=rotation_tolerance,
        ):
            raise ValueError("crystal_to_sample must be the canonical instrument rotation")
        basis_scale = max(float(np.linalg.norm(reference_config.reciprocal_basis_Ainv)), 1.0)
        if not np.allclose(
            strength_model.reciprocal_basis_Ainv,
            reference_config.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * basis_scale,
        ):
            raise ValueError("strength-model and Bragg-space reciprocal bases do not match")

        m0_gap: float | None = None
        if any(rod.family_m == 0 for rod in selected):
            if not valid_state_index.size:
                raise ValueError("detector-visible m=0 requires at least one valid incident state")
            incident_normal = states.k_film_phase_sample_Ainv[valid_state_index, 2]
            if np.any(incident_normal >= 0.0):
                raise ValueError(
                    "detector-visible m=0 requires every valid incident state to enter "
                    "through the negative sample-normal half-space"
                )
            m0_gap = (
                0.0
                if _uses_local_lamella_stitch(specular_stitch_stack)
                else float(np.min(-incident_normal))
            )
            if not isfinite(m0_gap) or m0_gap < 0.0:
                raise ValueError("detector-visible m=0 reciprocal support gap is invalid")

        reachable_count = np.zeros(states.incident_state_id.size, dtype=np.int64)
        evaluators: list[_IndexedCompiledEvaluator] = []
        stitch_by_wavelength: dict[float, CompiledParrattStitch | None] = {}
        (
            atom_offsets,
            atom_properties,
            f0_parameters,
            anomalous_factors,
            layers,
            normalization_divisor,
            u_radial_A2,
            u_normal_A2,
        ) = pack_bi2se3_two_h_structures(
            strength_model,
            wavelength_A=states.wavelength_A[valid_state_index],
        )
        for valid_position, state_index in enumerate(valid_state_index):
            wavelength_A = float(states.wavelength_A[state_index])
            active_index = _reachable_master_rod_indices(
                selected,
                reference_config.reciprocal_basis_Ainv,
                2.0 * np.pi / wavelength_A,
            )
            if not active_index.size:
                continue
            reachable_count[state_index] = active_index.size
            active_rods = tuple(selected[int(position)] for position in active_index)
            bragg_config = BraggSpaceConfig(
                reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
                crystal_to_sample=reference_config.crystal_to_sample,
                rods=active_rods,
                mosaic=mosaic,
                k_norm_Ainv=2.0 * np.pi / wavelength_A,
            )
            source_phase_weight = float(
                states.source_weight[state_index]
                * states.footprint_acceptance[state_index]
                * incident_illuminated_path_weight(states.direction_sample[state_index])
                * (phase_weight * polarization)
            )
            packed = (
                atom_offsets,
                atom_properties,
                f0_parameters,
                anomalous_factors[valid_position],
                layers,
                normalization_divisor,
                u_radial_A2,
                u_normal_A2,
            )
            material_index = int(np.searchsorted(material.wavelength_A, wavelength_A))
            if (
                material_index >= material.wavelength_A.size
                or material.wavelength_A[material_index] != wavelength_A
            ):
                raise ValueError("material does not contain the exact incident wavelength")
            if wavelength_A not in stitch_by_wavelength:
                stitch_by_wavelength[wavelength_A] = _compile_source_parratt_stitch(
                    specular_stitch_stack,
                    strength_model,
                    active_rods,
                    wavelength_A=wavelength_A,
                    film_refractive_index=complex(material.n_complex[material_index]),
                    film_thickness_A=instrument.film_thickness_A,
                )
            compiled_stitch = stitch_by_wavelength[wavelength_A]
            evaluators.append(
                _IndexedCompiledEvaluator(
                    evaluator=CompiledDetectorEvaluator(
                        _compile_detector_state(
                            bragg_config=bragg_config,
                            strength_model=strength_model,
                            ki_sample_Ainv=states.k_film_phase_sample_Ainv[state_index],
                            incident=incident,
                            material=material,
                            instrument=instrument,
                            rods=active_rods,
                            incident_state_index=int(state_index),
                            source_phase_weight=source_phase_weight,
                            intensity_envelope=envelope,
                            specular_stitch=compiled_stitch,
                            packed_structure=packed,
                        ),
                        instrument.detector_shape_rc,
                    ),
                    master_rod_index=active_index,
                    incident_state_index=int(state_index),
                )
            )
        block_size = max(
            1,
            (len(evaluators) + self._MAX_STATE_BLOCK_COUNT - 1) // self._MAX_STATE_BLOCK_COUNT,
        )
        blocks = tuple(
            tuple(evaluators[start : start + block_size])
            for start in range(0, len(evaluators), block_size)
        )
        reachable_count.setflags(write=False)
        object.__setattr__(self, "_evaluator_blocks", blocks)
        object.__setattr__(self, "_detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(self, "_incident", incident)
        object.__setattr__(self, "_instrument", instrument)
        object.__setattr__(self, "_intensity_envelope", envelope)
        object.__setattr__(self, "_incidence_axis_angle_rad", axis_angle)
        object.__setattr__(self, "_scan_calibration_binding_revision", scan_binding)
        object.__setattr__(
            self,
            "_incidence_angle_static_physics_revision",
            _incidence_angle_static_physics_revision(
                reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
                strength_model_revision=strength_model.structure_model_revision,
                mosaic=mosaic,
                intensity_envelope=envelope,
                material_revision=material.material_revision,
                incident_model_id=states.incident_model_id,
                instrument=instrument,
                phase_polarization_weight=phase_weight * polarization,
                specular_stitch_stack=specular_stitch_stack,
            ),
        )
        object.__setattr__(self, "_material", material)
        object.__setattr__(self, "_mosaic", mosaic)
        object.__setattr__(self, "_phase_polarization_weight", phase_weight * polarization)
        object.__setattr__(self, "_reachable_rod_count_per_source_state", reachable_count)
        object.__setattr__(self, "_rod_catalog_revision", rod_catalog_revision)
        object.__setattr__(self, "_rods", selected)
        object.__setattr__(self, "_strength_model", strength_model)
        object.__setattr__(self, "_specular_stitch_stack", specular_stitch_stack)
        object.__setattr__(self, "_valid_state_count", len(evaluators))
        object.__setattr__(self, "_worker_count", workers)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SourceAveragedDetectorEwaldMeasure is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SourceAveragedDetectorEwaldMeasure is immutable")

    @property
    def incident(self) -> IncidentTransportResult:
        return self._incident

    @property
    def instrument(self) -> CompiledInstrument:
        return self._instrument

    @property
    def material(self) -> MaterialOptics:
        return self._material

    @property
    def mosaic(self) -> MosaicParameters:
        return self._mosaic

    @property
    def intensity_envelope(self) -> SampleQIntensityEnvelope:
        return self._intensity_envelope

    @property
    def strength_model(self) -> Bi2X3FiniteStackStrength:
        return self._strength_model

    @property
    def specular_stitch_stack(self) -> ParrattStitchStack | None:
        return self._specular_stitch_stack

    @property
    def rods(self) -> tuple[Rod, ...]:
        return self._rods

    @property
    def rod_catalog_revision(self) -> str:
        return self._rod_catalog_revision

    @property
    def source_state_count(self) -> int:
        return int(self._incident.states.incident_state_id.size)

    @property
    def source_revision(self) -> str:
        return self._incident.states.source_revision

    @property
    def sample_geometry_revision(self) -> str:
        return self._incident.states.sample_geometry_revision

    @property
    def detector_shape_rc(self) -> tuple[int, int]:
        return self._instrument.detector_shape_rc

    @property
    def detector_panel_revision(self) -> str:
        return _detector_native_chart_revision(self._instrument)

    @property
    def valid_source_state_count(self) -> int:
        return self._valid_state_count

    @property
    def reachable_rod_count_per_source_state(self) -> NDArray[np.int64]:
        """Stable master-catalog reach count for each aligned incident state."""

        return self._reachable_rod_count_per_source_state

    @property
    def detector_visible_m0_q_gap_Ainv(self) -> float | None:
        """Infimum of supported ``|Q|``; zero only for local-lamella stitched m=0."""

        return self._detector_visible_m0_q_gap_Ainv

    @property
    def incidence_angle_static_physics_revision(self) -> str:
        """Physics identity required to be common to every incidence node."""

        return self._incidence_angle_static_physics_revision

    @property
    def incidence_axis_angle_rad(self) -> float | None:
        """Calibrated scalar incidence-axis setting, when configured."""

        return self._incidence_axis_angle_rad

    @property
    def scan_calibration_binding_revision(self) -> str | None:
        """Calibrated scan lineage carried by a canonical scan component."""

        return self._scan_calibration_binding_revision

    def sample_native_pixel_mass(
        self,
        *,
        draws_per_source_state: int,
        seed: int,
        execution_backend: str = "cpu",
        cancel_requested: Callable[[], bool] | None = None,
    ) -> MonteCarloDetectorPixelMass:
        """Estimate native pixel mass by forward-sampling mosaic orientation.

        Each valid canonical incident state is a source stratum. Known-zero
        states retain fixed latent lanes for prefix identity but perform no root
        or physics work; every active state enumerates all reachable physical
        rods and retained roots. The root ledger is numerical work evidence,
        not a calibrated detector count.
        """

        return self.compile_monte_carlo_sampler(
            execution_backend=execution_backend,
            seed=seed,
        ).advance_to(
            draws_per_source_state,
            cancel_requested=cancel_requested,
        )

    def compile_monte_carlo_sampler(
        self,
        *,
        execution_backend: str,
        seed: int,
    ) -> CompiledMonteCarloDetectorSampler:
        """Compile a mutable progressive execution resource for this immutable detector."""

        if _uses_local_lamella_stitch(self._specular_stitch_stack) and any(
            rod.family_m == 0 for rod in self._rods
        ):
            raise ValueError("forward pixel sampling does not implement local-lamella stitched m=0")

        return CompiledMonteCarloDetectorSampler(
            self,
            execution_backend=execution_backend,
            seed=seed,
        )

    def restrict_rods(
        self,
        rods: tuple[Rod, ...],
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Return an exact view of the compiled source physics over a rod subset."""

        requested = tuple(rods)
        if not requested or any(not isinstance(rod, Rod) for rod in requested):
            raise ValueError("rods must contain at least one Rod")
        configured_by_hk = {(rod.h, rod.k): rod for rod in self._rods}
        configured_index_by_hk = {(rod.h, rod.k): index for index, rod in enumerate(self._rods)}
        requested_hk = tuple((rod.h, rod.k) for rod in requested)
        if len(set(requested_hk)) != len(requested_hk):
            raise ValueError("rods must not repeat a physical rod")
        try:
            selected = tuple(configured_by_hk[rod_hk] for rod_hk in requested_hk)
        except KeyError as error:
            raise ValueError(f"rod subset contains unconfigured rod {error.args[0]}") from error
        if selected == self._rods:
            return self
        contains_m0 = any(rod.family_m == 0 for rod in selected)
        selected_master_index = tuple(configured_index_by_hk[key] for key in requested_hk)
        new_index_by_master = {
            master_index: new_index for new_index, master_index in enumerate(selected_master_index)
        }
        reachable_count = np.zeros_like(self._reachable_rod_count_per_source_state)
        restricted_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        for block in self._evaluator_blocks:
            restricted_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                local_index_by_master = {
                    int(master_index): local_index
                    for local_index, master_index in enumerate(indexed.master_rod_index)
                }
                retained_master_index = tuple(
                    master_index
                    for master_index in selected_master_index
                    if master_index in local_index_by_master
                )
                if not retained_master_index:
                    continue
                local_index = np.asarray(
                    [local_index_by_master[index] for index in retained_master_index],
                    dtype=np.int64,
                )
                new_master_index = np.asarray(
                    [new_index_by_master[index] for index in retained_master_index],
                    dtype=np.int64,
                )
                state = indexed.evaluator.state
                if not contains_m0 and state.specular_stitch_code:
                    state = replace(
                        state,
                        specular_stitch_code=0,
                        specular_substrate_refractive_index=1.0 + 0.0j,
                        specular_top_roughness_A=0.0,
                        specular_bottom_roughness_A=0.0,
                        specular_qc_Ainv=0.0,
                        specular_zero_strength_A2=0.0,
                        specular_scale_factor=0.0,
                        specular_blend_lower_q_over_qc=0.0,
                        specular_blend_upper_q_over_qc=0.0,
                    )
                restricted_state = replace(
                    state,
                    rod_hk_population=state.rod_hk_population[local_index],
                    rod_parallel_local_Ainv=state.rod_parallel_local_Ainv[local_index],
                    rod_u_bounds_Ainv=state.rod_u_bounds_Ainv[local_index],
                    rod_inverse_constants=state.rod_inverse_constants[local_index],
                    rod_atom_inplane_factor=state.rod_atom_inplane_factor[local_index],
                )
                restricted_block.append(
                    _IndexedCompiledEvaluator(
                        evaluator=CompiledDetectorEvaluator(
                            restricted_state,
                            self._instrument.detector_shape_rc,
                        ),
                        master_rod_index=new_master_index,
                        incident_state_index=indexed.incident_state_index,
                    )
                )
                reachable_count[indexed.incident_state_index] = local_index.size
            if restricted_block:
                restricted_blocks.append(tuple(restricted_block))
        reachable_count.setflags(write=False)
        restricted = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(restricted, slot, getattr(self, slot))
        object.__setattr__(restricted, "_evaluator_blocks", tuple(restricted_blocks))
        object.__setattr__(
            restricted,
            "_detector_visible_m0_q_gap_Ainv",
            (self._detector_visible_m0_q_gap_Ainv if contains_m0 else None),
        )
        object.__setattr__(
            restricted,
            "_specular_stitch_stack",
            self._specular_stitch_stack if contains_m0 else None,
        )
        if not contains_m0 and self._specular_stitch_stack is not None:
            object.__setattr__(
                restricted,
                "_incidence_angle_static_physics_revision",
                _incidence_angle_static_physics_revision(
                    reciprocal_basis_Ainv=self._strength_model.reciprocal_basis_Ainv,
                    strength_model_revision=self._strength_model.structure_model_revision,
                    mosaic=self._mosaic,
                    intensity_envelope=self._intensity_envelope,
                    material_revision=self._material.material_revision,
                    incident_model_id=self._incident.states.incident_model_id,
                    instrument=self._instrument,
                    phase_polarization_weight=self._phase_polarization_weight,
                    specular_stitch_stack=None,
                ),
            )
        object.__setattr__(
            restricted,
            "_reachable_rod_count_per_source_state",
            reachable_count,
        )
        object.__setattr__(restricted, "_rods", selected)
        object.__setattr__(
            restricted,
            "_valid_state_count",
            sum(len(block) for block in restricted_blocks),
        )
        return restricted

    def rebind_physics(
        self,
        *,
        mosaic: MosaicParameters | None = None,
        strength_model: Bi2X3FiniteStackStrength | None = None,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Replace mosaic/structure arrays while retaining the combined source geometry."""

        rebound_mosaic = self._mosaic if mosaic is None else mosaic
        rebound_strength = self._strength_model if strength_model is None else strength_model
        rebound_envelope = (
            self._intensity_envelope if intensity_envelope is None else intensity_envelope
        )
        if not isinstance(rebound_mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(rebound_strength, Bi2X3FiniteStackStrength):
            raise TypeError("strength_model must be Bi2X3FiniteStackStrength")
        if not isinstance(rebound_envelope, SampleQIntensityEnvelope):
            raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
        reference = self._strength_model
        if (
            rebound_strength.crystal is not reference.crystal
            or rebound_strength.layers != reference.layers
            or rebound_strength.parent is not reference.parent
            or rebound_strength.normalization != reference.normalization
        ):
            raise ValueError(
                "physics rebinding requires unchanged crystal, layer count, parent, and "
                "normalization"
            )
        valid_state_index = np.flatnonzero(self._incident.states.valid)
        (
            atom_offsets,
            atom_properties,
            f0_parameters,
            anomalous_factors,
            layers,
            normalization_divisor,
            u_radial_A2,
            u_normal_A2,
        ) = pack_bi2se3_two_h_structures(
            rebound_strength,
            wavelength_A=self._incident.states.wavelength_A[valid_state_index],
        )
        valid_position_by_state = {
            int(state_index): position for position, state_index in enumerate(valid_state_index)
        }
        rebound_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        for block in self._evaluator_blocks:
            rebound_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                state = indexed.evaluator.state.rebind_physics(
                    mosaic=rebound_mosaic,
                    atom_fractional_offset=atom_offsets,
                    atom_occupancy_element=atom_properties,
                    f0_parameters=f0_parameters,
                    anomalous_factor_e=anomalous_factors[
                        valid_position_by_state[indexed.incident_state_index]
                    ],
                    layers=layers,
                    normalization_divisor=normalization_divisor,
                    u_radial_A2=u_radial_A2,
                    u_normal_A2=u_normal_A2,
                    intensity_envelope_u_radial_A2=rebound_envelope.u_radial_A2,
                    intensity_envelope_u_normal_A2=rebound_envelope.u_normal_A2,
                    shared_disorder_epsilon=rebound_strength.shared_disorder_epsilon,
                    specular_stitch=_retained_source_parratt_stitch(indexed.evaluator.state),
                )
                rebound_block.append(
                    indexed.with_evaluator(
                        CompiledDetectorEvaluator(
                            state,
                            self._instrument.detector_shape_rc,
                        )
                    )
                )
            rebound_blocks.append(tuple(rebound_block))

        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", tuple(rebound_blocks))
        object.__setattr__(rebound, "_mosaic", rebound_mosaic)
        object.__setattr__(rebound, "_strength_model", rebound_strength)
        object.__setattr__(rebound, "_intensity_envelope", rebound_envelope)
        object.__setattr__(
            rebound,
            "_incidence_angle_static_physics_revision",
            _incidence_angle_static_physics_revision(
                reciprocal_basis_Ainv=rebound_strength.reciprocal_basis_Ainv,
                strength_model_revision=rebound_strength.structure_model_revision,
                mosaic=rebound_mosaic,
                intensity_envelope=rebound_envelope,
                material_revision=self._material.material_revision,
                incident_model_id=self._incident.states.incident_model_id,
                instrument=self._instrument,
                phase_polarization_weight=self._phase_polarization_weight,
                specular_stitch_stack=self._specular_stitch_stack,
            ),
        )
        return rebound

    def with_specular_stitch(
        self,
        stack: ParrattStitchStack | None,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Return a detector with an explicit empirical `m=0` optical stack."""

        if stack is self._specular_stitch_stack or stack == self._specular_stitch_stack:
            return self
        if stack is not None and not isinstance(stack, ParrattStitchStack):
            raise TypeError("stack must be ParrattStitchStack")
        rebuilt = type(self)(
            reciprocal_basis_Ainv=self._strength_model.reciprocal_basis_Ainv,
            crystal_to_sample=self._instrument.sample_from_crystal.rotation,
            rods=self._rods,
            rod_catalog_revision=self._rod_catalog_revision,
            mosaic=self._mosaic,
            strength_model=self._strength_model,
            intensity_envelope=self._intensity_envelope,
            incident=self._incident,
            material=self._material,
            instrument=self._instrument,
            phase_population_weight=self._phase_polarization_weight,
            polarization_weight=1.0,
            worker_count=self._worker_count,
            specular_stitch_stack=stack,
            incidence_axis_angle_rad=self._incidence_axis_angle_rad,
            scan_calibration_binding_revision=self._scan_calibration_binding_revision,
        )
        return rebuilt.with_maximum_state_block_count(len(self._evaluator_blocks))

    def with_maximum_state_block_count(
        self,
        maximum_state_block_count: int,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Return an immutable execution view with at most the requested state blocks."""

        maximum_blocks = positive_integer(
            maximum_state_block_count,
            "maximum_state_block_count",
        )
        evaluators = tuple(item for block in self._evaluator_blocks for item in block)
        block_size = max(1, (len(evaluators) + maximum_blocks - 1) // maximum_blocks)
        blocks = tuple(
            tuple(evaluators[start : start + block_size])
            for start in range(0, len(evaluators), block_size)
        )
        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", blocks)
        return rebound

    def rebind_geometry(
        self,
        *,
        incident: IncidentTransportResult,
        instrument: CompiledInstrument,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Bind new rigid geometry while reusing immutable rod, mosaic, and SF state.

        Source identities, wavelengths, material, detector calibration, crystal mounting, and
        the valid-state topology are frozen. Only detector/sample rigid geometry and the incident
        quantities causally derived from that geometry are replaced.
        """

        if not isinstance(incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        old_states = self._incident.states
        new_states = incident.states
        if new_states.sample_geometry_revision != instrument.sample_geometry_revision:
            raise ValueError(
                "geometry rebind requires incident transport and instrument to share one pose"
            )
        invariant_arrays = (
            ("incident_state_id", old_states.incident_state_id, new_states.incident_state_id),
            ("incident_sample_id", old_states.incident_sample_id, new_states.incident_sample_id),
            ("source_weight", old_states.source_weight, new_states.source_weight),
            ("wavelength_A", old_states.wavelength_A, new_states.wavelength_A),
            ("valid", old_states.valid, new_states.valid),
        )
        for name, old, new in invariant_arrays:
            if not np.array_equal(old, new):
                raise ValueError(f"geometry rebind requires unchanged {name}")
        if (
            old_states.source_revision != new_states.source_revision
            or old_states.material_revision != new_states.material_revision
            or old_states.polarization_state_id != new_states.polarization_state_id
            or old_states.incident_model_id != new_states.incident_model_id
        ):
            raise ValueError(
                "geometry rebind requires unchanged source, material, and transport identity"
            )
        old_instrument = self._instrument
        if not _geometry_rebind_instrument_invariants(old_instrument, instrument):
            raise ValueError(
                "geometry rebind requires unchanged detector calibration, sample support, "
                "film, and crystal mounting"
            )

        (
            detector_zero_lab,
            column_step_lab,
            row_step_lab,
            detector_area_vector,
            sample_from_lab,
        ) = _detector_pose_arrays(instrument)
        propagation_direction = np.where(new_states.direction_sample[:, 2] < 0.0, -1, 1)
        incident_decay_Ainv = np.asarray(
            mode_decay_constant(new_states.kz_film_Ainv, propagation_direction),
            dtype=np.float64,
        )
        source_phase_weight = np.zeros_like(new_states.source_weight)
        valid_state = np.asarray(new_states.valid, dtype=np.bool_)
        source_phase_weight[valid_state] = (
            new_states.source_weight[valid_state]
            * new_states.footprint_acceptance[valid_state]
            * incident_illuminated_path_weight(new_states.direction_sample[valid_state])
            * self._phase_polarization_weight
        )
        internal_k_Ainv = np.linalg.norm(new_states.k_film_phase_sample_Ainv, axis=1)
        evaluator_state_indices = tuple(
            indexed.incident_state_index for block in self._evaluator_blocks for indexed in block
        )
        projection = _compile_detector_projection(
            detector_zero_lab_m=detector_zero_lab,
            detector_column_step_lab_m=column_step_lab,
            detector_row_step_lab_m=row_step_lab,
            detector_pixel_area_vector_lab_m2=detector_area_vector,
            sample_from_lab=sample_from_lab,
            ray_origin_lab_m=np.ascontiguousarray(
                new_states.sample_intersection_lab_m[
                    np.asarray(evaluator_state_indices, dtype=np.int64)
                ]
            ),
        )
        rebound_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        projection_index = 0
        for block in self._evaluator_blocks:
            rebound_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                state_index = indexed.incident_state_index
                rebound_state = indexed.evaluator.state._rebind_validated_geometry(
                    detector_zero_lab_m=detector_zero_lab,
                    detector_column_step_lab_m=column_step_lab,
                    detector_row_step_lab_m=row_step_lab,
                    detector_pixel_area_vector_lab_m2=detector_area_vector,
                    ray_origin_lab_m=new_states.sample_intersection_lab_m[state_index],
                    sample_from_lab=sample_from_lab,
                    ki_film_sample_Ainv=new_states.k_film_phase_sample_Ainv[state_index],
                    internal_k_Ainv=float(internal_k_Ainv[state_index]),
                    entrance_amplitude=complex(new_states.entrance_amplitude[state_index]),
                    incident_decay_Ainv=float(incident_decay_Ainv[state_index]),
                    source_phase_weight=float(source_phase_weight[state_index]),
                )
                rebound_evaluator = indexed.evaluator._rebind_validated_projection(
                    rebound_state,
                    projection,
                    state_index=projection_index,
                )
                projection_index += 1
                rebound_block.append(indexed.with_evaluator(rebound_evaluator))
            rebound_blocks.append(tuple(rebound_block))

        m0_gap: float | None = None
        if any(rod.family_m == 0 for rod in self._rods):
            incident_normal = new_states.k_film_phase_sample_Ainv[new_states.valid, 2]
            if not incident_normal.size or np.any(incident_normal >= 0.0):
                raise ValueError("detector-visible m=0 requires negative incident sample-normal k")
            m0_gap = (
                0.0
                if _uses_local_lamella_stitch(self._specular_stitch_stack)
                else float(np.min(-incident_normal))
            )
        rebound = object.__new__(type(self))
        for slot in self.__slots__:
            object.__setattr__(rebound, slot, getattr(self, slot))
        object.__setattr__(rebound, "_evaluator_blocks", tuple(rebound_blocks))
        object.__setattr__(rebound, "_detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(rebound, "_incident", incident)
        object.__setattr__(rebound, "_instrument", instrument)
        object.__setattr__(rebound, "_incidence_axis_angle_rad", None)
        object.__setattr__(rebound, "_scan_calibration_binding_revision", None)
        object.__setattr__(
            rebound,
            "_incidence_angle_static_physics_revision",
            _incidence_angle_static_physics_revision(
                reciprocal_basis_Ainv=self._strength_model.reciprocal_basis_Ainv,
                strength_model_revision=self._strength_model.structure_model_revision,
                mosaic=self._mosaic,
                intensity_envelope=self._intensity_envelope,
                material_revision=self._material.material_revision,
                incident_model_id=incident.states.incident_model_id,
                instrument=instrument,
                phase_polarization_weight=self._phase_polarization_weight,
                specular_stitch_stack=self._specular_stitch_stack,
            ),
        )
        return rebound

    def _rebind_calibrated_incidence_scan_geometry(
        self,
        *,
        incident: IncidentTransportResult,
        instrument: CompiledInstrument,
        incidence_axis_angle_rad: float,
        scan_calibration_binding_revision: str,
    ) -> SourceAveragedDetectorEwaldMeasure:
        """Internal calibrated-scan rebind used only by the configured builder."""

        angle = float(incidence_axis_angle_rad)
        if not isfinite(angle):
            raise ValueError("incidence_axis_angle_rad must be finite")
        binding = scan_calibration_binding_revision
        if not isinstance(binding, str) or not binding:
            raise ValueError("scan_calibration_binding_revision must be nonempty")
        rebound = self.rebind_geometry(incident=incident, instrument=instrument)
        object.__setattr__(rebound, "_incidence_axis_angle_rad", angle)
        object.__setattr__(rebound, "_scan_calibration_binding_revision", binding)
        return rebound

    def _thread_pool(self) -> ThreadPoolExecutor | None:
        if self._worker_count <= 1 or len(self._evaluator_blocks) <= 1:
            return None
        return ThreadPoolExecutor(max_workers=min(self._worker_count, len(self._evaluator_blocks)))

    def _evaluate_flat_coordinates(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        *,
        branch: int | None,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
        per_rod = np.zeros((column_px.size, len(self._rods)), dtype=np.float64)
        caustic = np.zeros(per_rod.shape, dtype=np.bool_)
        valid_source_count = np.zeros(column_px.size, dtype=np.int64)
        for start in range(0, column_px.size, self._COORDINATE_CHUNK_SIZE):
            stop = min(start + self._COORDINATE_CHUNK_SIZE, column_px.size)
            chunk_column = column_px[start:stop]
            chunk_row = row_px[start:stop]
            if executor is None:
                block_results = (
                    _sum_compiled_evaluator_block(
                        block,
                        chunk_column,
                        chunk_row,
                        branch,
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
            else:
                futures = tuple(
                    executor.submit(
                        _sum_compiled_evaluator_block,
                        block,
                        chunk_column,
                        chunk_row,
                        branch,
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
                block_results = (future.result() for future in futures)
            for block_density, block_caustic, block_valid_count in block_results:
                per_rod[start:stop] += block_density
                caustic[start:stop] |= block_caustic
                valid_source_count[start:stop] += block_valid_count
        return per_rod, caustic, valid_source_count

    def _evaluate_flat_density_all_roots(
        self,
        column_px: FloatArray,
        row_px: FloatArray,
        *,
        executor: ThreadPoolExecutor | None,
    ) -> tuple[FloatArray, BoolArray, NDArray[np.int64]]:
        density = np.zeros(column_px.size, dtype=np.float64)
        caustic = np.zeros(column_px.size, dtype=np.bool_)
        valid_source_count = np.zeros(column_px.size, dtype=np.int64)
        chunk_size = self._TOTAL_DENSITY_COORDINATE_CHUNK_SIZE
        for start in range(0, column_px.size, chunk_size):
            stop = min(start + chunk_size, column_px.size)
            chunk_column = column_px[start:stop]
            chunk_row = row_px[start:stop]
            chunk_per_rod = np.zeros((stop - start, len(self._rods)), dtype=np.float64)
            chunk_per_rod_caustic = np.zeros(chunk_per_rod.shape, dtype=np.bool_)
            if executor is None:
                block_results = (
                    _sum_compiled_evaluator_block(
                        block,
                        chunk_column,
                        chunk_row,
                        None,
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
            else:
                futures = tuple(
                    executor.submit(
                        _sum_compiled_evaluator_block,
                        block,
                        chunk_column,
                        chunk_row,
                        None,
                        len(self._rods),
                    )
                    for block in self._evaluator_blocks
                )
                block_results = (future.result() for future in futures)
            for block_density, block_caustic, block_valid_count in block_results:
                chunk_per_rod += block_density
                chunk_per_rod_caustic |= block_caustic
                valid_source_count[start:stop] += block_valid_count
            density[start:stop] = np.sum(chunk_per_rod, axis=1, dtype=np.float64)
            caustic[start:stop] = np.any(chunk_per_rod_caustic, axis=1)
        return density, caustic, valid_source_count

    def _evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        branch: int | None,
        execution_backend: str,
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
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
        flat_column = np.ascontiguousarray(column.reshape(-1))
        flat_row = np.ascontiguousarray(row.reshape(-1))
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be 'cpu' or 'cuda'")
        cuda_chunk_size = _validated_cuda_coordinate_chunk_size(
            execution_backend,
            cuda_coordinate_chunk_size,
        )
        if execution_backend == "cuda":
            if branch is not None:
                raise ValueError("the CUDA backend currently supports all retained roots only")
            from rasim_next.pipeline._continuous_detector_cuda import (
                evaluate_source_averaged_all_roots_cuda,
            )

            per_rod, caustic, valid_source_count, execution_device = (
                evaluate_source_averaged_all_roots_cuda(
                    self._evaluator_blocks,
                    flat_column,
                    flat_row,
                    detector_shape_rc=self._instrument.detector_shape_rc,
                    master_rod_count=len(self._rods),
                    **(
                        {}
                        if cuda_chunk_size is None
                        else {"coordinate_chunk_size": cuda_chunk_size}
                    ),
                )
            )
            if _uses_local_lamella_stitch(self._specular_stitch_stack) and any(
                rod.family_m == 0 for rod in self._rods
            ):
                m0_index, m0_rod = next(
                    (index, rod) for index, rod in enumerate(self._rods) if rod.family_m == 0
                )
                local_m0 = (
                    self.restrict_rods((m0_rod,))
                    .with_maximum_state_block_count(1)
                    .evaluate_detector_coordinates_all_roots(
                        flat_column,
                        flat_row,
                        execution_backend="cpu",
                    )
                )
                per_rod[:, m0_index] = np.asarray(local_m0.per_rod_density_A2_per_px2).reshape(-1)
                caustic[:, m0_index] = np.asarray(local_m0.caustic).reshape(-1)
                valid_source_count = np.asarray(local_m0.valid_source_count).reshape(-1)
                backend_id = "hybrid_cuda_cpu_local_m0.v1"
            else:
                backend_id = "numba_cuda_source_averaged.v1"
        else:
            executor = self._thread_pool()
            try:
                if executor is not None and self._evaluator_blocks and flat_column.size:
                    _warm_compiled_evaluator_blocks(
                        self._evaluator_blocks,
                        flat_column,
                        flat_row,
                        branch,
                    )
                per_rod, caustic, valid_source_count = self._evaluate_flat_coordinates(
                    flat_column,
                    flat_row,
                    branch=branch,
                    executor=executor,
                )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
            execution_device = None
            backend_id = "numba_cpu_source_averaged.v1"
        reshaped_per_rod = per_rod.reshape((*shape, len(self._rods)))
        return SourceAveragedDetectorCoordinateIntensity(
            column_px=column,
            row_px=row,
            rods=self._rods,
            rod_catalog_revision=self._rod_catalog_revision,
            branch=branch,
            per_rod_density_A2_per_px2=reshaped_per_rod,
            density_A2_per_px2=np.sum(reshaped_per_rod, axis=-1, dtype=np.float64),
            caustic=caustic.reshape((*shape, len(self._rods))),
            valid_source_count=valid_source_count.reshape(shape),
            source_state_count=self.source_state_count,
            source_revision=self._incident.states.source_revision,
            execution_backend=backend_id,
            execution_device=execution_device,
            root_policy=("all_retained_roots.v1" if branch is None else "single_nonzero_root.v1"),
            detector_visible_m0_q_gap_Ainv=(
                self._detector_visible_m0_q_gap_Ainv if branch is None else None
            ),
        )

    def evaluate_detector_coordinates(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        branch: int = 2,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate one nonzero-rod root at arbitrary detector coordinates."""

        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        if any(rod.family_m == 0 for rod in self._rods):
            raise ValueError("branch-specific evaluation cannot include m=0")
        return self._evaluate_detector_coordinates(
            column_px,
            row_px,
            branch=branch,
            execution_backend="cpu",
        )

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Evaluate every retained physical root, including supported kinematic m=0."""

        return self._evaluate_detector_coordinates(
            column_px,
            row_px,
            branch=None,
            execution_backend=execution_backend,
            cuda_coordinate_chunk_size=cuda_coordinate_chunk_size,
        )

    def evaluate_selected_source_rod_groups_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        evaluator_index_by_coordinate: ArrayLike,
        rod_group_index_by_coordinate: ArrayLike,
        group_master_rod_mask: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> tuple[FloatArray, BoolArray, BoolArray, str, str | None]:
        """Evaluate one source evaluator and master-rod group at each coordinate.

        This narrow reduction is used to replace an affected contribution after
        a continuous fold change of variables.  Coordinates remain continuous;
        no native-pixel identity enters this API.
        """

        column = np.ascontiguousarray(column_px, dtype=np.float64).reshape(-1)
        row = np.ascontiguousarray(row_px, dtype=np.float64).reshape(-1)
        evaluator_index = np.ascontiguousarray(
            evaluator_index_by_coordinate,
            dtype=np.int64,
        ).reshape(-1)
        group_index = np.ascontiguousarray(
            rod_group_index_by_coordinate,
            dtype=np.int64,
        ).reshape(-1)
        group_mask = np.ascontiguousarray(group_master_rod_mask, dtype=np.bool_)
        indexed_evaluators = _flatten_evaluators(self._evaluator_blocks)
        if (
            column.shape != row.shape
            or evaluator_index.shape != column.shape
            or group_index.shape != column.shape
            or not np.all(np.isfinite(column))
            or not np.all(np.isfinite(row))
            or group_mask.ndim != 2
            or group_mask.shape[1] != len(self._rods)
            or not group_mask.shape[0]
            or np.any((evaluator_index < 0) | (evaluator_index >= len(indexed_evaluators)))
            or np.any((group_index < 0) | (group_index >= group_mask.shape[0]))
            or np.any(~np.any(group_mask, axis=1))
        ):
            raise ValueError("selected source/rod-group coordinates are inconsistent")
        for selected_evaluator in np.unique(evaluator_index):
            indexed = indexed_evaluators[int(selected_evaluator)]
            selected = evaluator_index == selected_evaluator
            active_masks = group_mask[group_index[selected]][:, indexed.master_rod_index]
            if np.any(~np.any(active_masks, axis=1)):
                raise ValueError("a selected rod group is inactive for its source evaluator")
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be 'cpu' or 'cuda'")
        if execution_backend == "cuda" and _uses_local_lamella_stitch(self._specular_stitch_stack):
            m0_index = next(
                (index for index, rod in enumerate(self._rods) if rod.family_m == 0),
                None,
            )
            if m0_index is not None and np.any(group_mask[np.unique(group_index), m0_index]):
                raise ValueError(
                    "CUDA selected source/rod-group evaluation does not implement "
                    "local-lamella stitched m=0"
                )
        cuda_chunk_size = _validated_cuda_coordinate_chunk_size(
            execution_backend,
            cuda_coordinate_chunk_size,
        )
        if execution_backend == "cuda":
            from rasim_next.pipeline._continuous_detector_cuda import (
                evaluate_selected_source_rod_groups_all_roots_cuda,
            )

            density, caustic, valid, device = evaluate_selected_source_rod_groups_all_roots_cuda(
                self._evaluator_blocks,
                column,
                row,
                evaluator_index,
                group_index,
                group_mask,
                detector_shape_rc=self._instrument.detector_shape_rc,
                master_rod_count=len(self._rods),
                **({} if cuda_chunk_size is None else {"coordinate_chunk_size": cuda_chunk_size}),
            )
            backend_id = "numba_cuda_selected_source_rod_group.v1"
        else:
            density = np.zeros(column.size, dtype=np.float64)
            caustic = np.zeros(column.size, dtype=np.bool_)
            valid = np.zeros(column.size, dtype=np.bool_)
            for selected_evaluator in np.unique(evaluator_index):
                indexed = indexed_evaluators[int(selected_evaluator)]
                selected_position = np.flatnonzero(evaluator_index == selected_evaluator)
                evaluated, _, evaluated_caustic, evaluated_valid = (
                    indexed.evaluator.evaluate_all_roots(
                        column[selected_position],
                        row[selected_position],
                    )
                )
                local_masks = group_mask[group_index[selected_position]][
                    :, indexed.master_rod_index
                ]
                density[selected_position] = np.sum(
                    np.where(local_masks, evaluated, 0.0),
                    axis=1,
                    dtype=np.float64,
                )
                caustic[selected_position] = np.any(
                    local_masks & evaluated_caustic,
                    axis=1,
                )
                valid[selected_position] = evaluated_valid
            device = None
            backend_id = "numba_cpu_selected_source_rod_group.v1"
        for value in (density, caustic, valid):
            value.setflags(write=False)
        return density, caustic, valid, backend_id, device

    def evaluate_detector_density_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateDensity:
        """Reduce every source, physical rod, and retained root at each coordinate."""

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
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be 'cpu' or 'cuda'")
        cuda_chunk_size = _validated_cuda_coordinate_chunk_size(
            execution_backend,
            cuda_coordinate_chunk_size,
        )
        shape = column.shape
        flat_column = np.ascontiguousarray(column.reshape(-1))
        flat_row = np.ascontiguousarray(row.reshape(-1))
        if execution_backend == "cuda":
            if _uses_local_lamella_stitch(self._specular_stitch_stack) and any(
                rod.family_m == 0 for rod in self._rods
            ):
                detailed = self._evaluate_detector_coordinates(
                    column,
                    row,
                    branch=None,
                    execution_backend="cuda",
                    cuda_coordinate_chunk_size=cuda_chunk_size,
                )
                per_rod = np.asarray(detailed.per_rod_density_A2_per_px2)
                density = np.sum(per_rod, axis=-1, dtype=np.float64).reshape(-1)
                caustic = np.any(np.asarray(detailed.caustic), axis=-1).reshape(-1)
                valid_source_count = np.asarray(detailed.valid_source_count).reshape(-1)
                execution_device = detailed.execution_device
                backend_id = detailed.execution_backend
            else:
                from rasim_next.pipeline._continuous_detector_cuda import (
                    evaluate_source_averaged_density_all_roots_cuda,
                )

                density, caustic, valid_source_count, execution_device = (
                    evaluate_source_averaged_density_all_roots_cuda(
                        self._evaluator_blocks,
                        flat_column,
                        flat_row,
                        detector_shape_rc=self._instrument.detector_shape_rc,
                        master_rod_count=len(self._rods),
                        **(
                            {}
                            if cuda_chunk_size is None
                            else {"coordinate_chunk_size": cuda_chunk_size}
                        ),
                    )
                )
                backend_id = "numba_cuda_source_averaged.v1"
        else:
            executor = self._thread_pool()
            try:
                if executor is not None and self._evaluator_blocks and flat_column.size:
                    _warm_compiled_evaluator_blocks(
                        self._evaluator_blocks,
                        flat_column,
                        flat_row,
                        None,
                    )
                density, caustic, valid_source_count = self._evaluate_flat_density_all_roots(
                    flat_column,
                    flat_row,
                    executor=executor,
                )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)
            execution_device = None
            backend_id = "numba_cpu_source_averaged.v1"
        return SourceAveragedDetectorCoordinateDensity(
            column_px=column,
            row_px=row,
            rods=self._rods,
            density_A2_per_px2=density.reshape(shape),
            caustic=caustic.reshape(shape),
            valid_source_count=valid_source_count.reshape(shape),
            source_state_count=self.source_state_count,
            source_revision=self._incident.states.source_revision,
            detector_visible_m0_q_gap_Ainv=self._detector_visible_m0_q_gap_Ainv,
            execution_backend=backend_id,
            execution_device=execution_device,
            rod_catalog_revision=self._rod_catalog_revision,
        )

    def integrate_native_pixels(
        self,
        *,
        branch: int,
        quadrature: DetectorQuadrature,
        include_per_rod_evidence: bool = False,
    ) -> DetectorPixelMass:
        """Integrate detailed per-rod pixel evidence after explicit opt-in."""

        if include_per_rod_evidence is not True:
            raise ValueError(
                "per-rod pixel evidence is disabled by default; pass "
                "include_per_rod_evidence=True explicitly"
            )

        if not isinstance(quadrature, DetectorQuadrature):
            raise TypeError("quadrature must be DetectorQuadrature")
        if quadrature.method is not PixelIntegrationMethod.FIXED_NUMPY:
            raise ValueError("source-averaged pixel integration requires fixed_numpy")
        if (
            quadrature.fold_gauss_order != quadrature.pixel_gauss_order
            or quadrature.fold_subdivision_count != 1
        ):
            raise ValueError("source-averaged integration does not apply a second fold rule")
        if branch not in {1, 2}:
            raise ValueError("branch must be 1 or 2")
        if any(rod.family_m == 0 for rod in self._rods):
            raise ValueError("branch-specific pixel integration cannot include m=0")
        rows, columns = self._instrument.detector_shape_rc
        offset, one_dimensional_weight = _subdivided_legendre_rule(
            quadrature.pixel_gauss_order,
            1,
        )
        node_weight = one_dimensional_weight[:, None] * one_dimensional_weight[None, :]
        image = np.zeros((rows, columns), dtype=np.float64)
        per_rod_mass = np.zeros(len(self._rods), dtype=np.float64)
        column_center = np.arange(columns, dtype=np.float64)
        if self._evaluator_blocks:
            self._evaluator_blocks[0][0].evaluator.evaluate(
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                branch=branch,
            )
        executor = self._thread_pool()
        try:
            for row_start in range(0, rows, quadrature.row_chunk_size):
                row_stop = min(row_start + quadrature.row_chunk_size, rows)
                row_center = np.arange(row_start, row_stop, dtype=np.float64)
                node_column, node_row = np.broadcast_arrays(
                    column_center[None, :, None, None] + offset[None, None, None, :],
                    row_center[:, None, None, None] + offset[None, None, :, None],
                )
                node_shape = node_column.shape
                flat_density, flat_caustic, _ = self._evaluate_flat_coordinates(
                    np.ascontiguousarray(node_column.reshape(-1)),
                    np.ascontiguousarray(node_row.reshape(-1)),
                    branch=branch,
                    executor=executor,
                )
                if np.any(flat_caustic):
                    raise FloatingPointError(
                        "a pixel quadrature node lies on a caustic; choose another even order"
                    )
                tile_per_rod = np.sum(
                    flat_density.reshape((*node_shape, len(self._rods)))
                    * node_weight[None, None, :, :, None],
                    axis=(2, 3),
                    dtype=np.float64,
                )
                image[row_start:row_stop] = np.sum(
                    tile_per_rod,
                    axis=-1,
                    dtype=np.float64,
                )
                per_rod_mass += np.sum(tile_per_rod, axis=(0, 1), dtype=np.float64)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        return DetectorPixelMass(
            image_A2=image,
            rods=self._rods,
            branch=branch,
            per_rod_detector_mass_A2=per_rod_mass,
            total_detector_mass_A2=fsum(per_rod_mass),
            quadrature=quadrature,
            coordinate_evaluation_count=rows * columns * quadrature.pixel_gauss_order**2,
            execution_backend="numba_source_averaged.v1",
        )


class CompiledMonteCarloDetectorSampler:
    """Mutable progressive execution state bound to one immutable detector model."""

    __slots__ = (
        "_attempted_root_count",
        "_attempted_root_count_per_draw",
        "_bound_instrument",
        "_cuda_workspace",
        "_detector",
        "_draw_count",
        "_execution_backend",
        "_execution_device",
        "_execution_worker_count",
        "_forward_blocks",
        "_maximum_root_weight_A2",
        "_poisoned",
        "_presentation_image_A2",
        "_raw_image_A2",
        "_replicate_total_mass_A2",
        "_seed",
        "_visible_hit_count",
    )

    def __init__(
        self,
        detector: SourceAveragedDetectorEwaldMeasure,
        *,
        execution_backend: str,
        seed: int,
    ) -> None:
        if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
            raise TypeError("detector must be a SourceAveragedDetectorEwaldMeasure")
        if execution_backend not in {"cpu", "cuda"}:
            raise ValueError("execution_backend must be cpu or cuda")
        detector_seed = integer(seed, "seed")
        if not 0 <= detector_seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        forward_blocks = _forward_monte_carlo_blocks(detector._evaluator_blocks)
        rows, columns = detector.instrument.detector_shape_rc
        self._detector = detector
        self._bound_instrument = detector.instrument
        self._seed = detector_seed
        self._cuda_workspace: object | None = None
        if execution_backend == "cuda":
            from rasim_next.pipeline._forward_detector_cuda import (
                CudaForwardMonteCarloWorkspace,
            )

            workspace = CudaForwardMonteCarloWorkspace(
                detector._evaluator_blocks,
                detector_shape_rc=detector.instrument.detector_shape_rc,
                master_rod_count=len(detector.rods),
            )
            self._cuda_workspace = workspace
            self._execution_backend = "numba_cuda_forward_monte_carlo.v1"
            self._execution_device = workspace.device_name
            self._execution_worker_count = None
        else:
            self._execution_backend = "numba_cpu_forward_monte_carlo.v2"
            self._execution_device = None
            self._execution_worker_count = min(
                detector._worker_count,
                len(forward_blocks),
                4,
            )
        self._forward_blocks = forward_blocks
        self._poisoned = False
        self._attempted_root_count_per_draw = sum(
            2 * indexed.evaluator.state.rod_hk_population.shape[0]
            - int(
                np.count_nonzero(
                    (indexed.evaluator.state.rod_hk_population[:, 0] == 0.0)
                    & (indexed.evaluator.state.rod_hk_population[:, 1] == 0.0)
                )
            )
            for block in forward_blocks
            for indexed in block
        )
        self._raw_image_A2 = (
            np.empty(0, dtype=np.float64)
            if self._cuda_workspace is not None
            else np.zeros(rows * columns, dtype=np.float64)
        )
        self._presentation_image_A2 = (
            np.empty(0, dtype=np.float32)
            if self._cuda_workspace is not None
            else np.empty(rows * columns, dtype=np.float32)
        )
        self._replicate_total_mass_A2 = np.empty(0, dtype=np.float64)
        self._draw_count = 0
        self._attempted_root_count = 0
        self._visible_hit_count = 0
        self._maximum_root_weight_A2 = 0.0

    @property
    def draws_completed(self) -> int:
        return self._draw_count

    def _require_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError("the failed CUDA sampler must be discarded")

    def _reset_host_state(self) -> None:
        self._raw_image_A2.fill(0.0)
        self._replicate_total_mass_A2 = np.empty(0, dtype=np.float64)
        self._draw_count = 0
        self._attempted_root_count = 0
        self._visible_hit_count = 0
        self._maximum_root_weight_A2 = 0.0

    def reset(self, *, seed: int | None = None) -> None:
        """Discard accumulated draws while retaining compiled execution state."""

        self._require_usable()
        detector_seed = self._seed
        if seed is not None:
            detector_seed = integer(seed, "seed")
            if not 0 <= detector_seed < 2**64:
                raise ValueError("seed must be an integer in [0, 2**64)")
        if self._cuda_workspace is not None:
            try:
                self._cuda_workspace.reset()
            except Exception:
                self._poisoned = True
                raise
        self._seed = detector_seed
        self._reset_host_state()

    def rebind_detector_pose(self, instrument: CompiledInstrument) -> None:
        """Reset sampling after changing only the detector's rigid lab pose."""

        if not isinstance(instrument, CompiledInstrument):
            raise TypeError("instrument must be a CompiledInstrument")
        self._require_usable()
        old_instrument = self._bound_instrument
        unchanged_sample_pose = (
            instrument.sample_geometry_revision == old_instrument.sample_geometry_revision
            and np.array_equal(
                instrument.sample_from_lab.rotation,
                old_instrument.sample_from_lab.rotation,
            )
            and np.array_equal(
                instrument.sample_from_lab.translation_m,
                old_instrument.sample_from_lab.translation_m,
            )
        )
        if (
            not _geometry_rebind_instrument_invariants(old_instrument, instrument)
            or not unchanged_sample_pose
        ):
            raise ValueError(
                "detector-pose rebind requires unchanged detector calibration and sample pose"
            )
        (
            detector_zero_lab,
            column_step_lab,
            row_step_lab,
            detector_area_vector,
            sample_from_lab,
        ) = _detector_pose_arrays(instrument)
        indexed_evaluators = tuple(indexed for block in self._forward_blocks for indexed in block)
        state_indices = np.fromiter(
            (indexed.incident_state_index for indexed in indexed_evaluators),
            dtype=np.int64,
            count=len(indexed_evaluators),
        )
        projection = _compile_detector_projection(
            detector_zero_lab_m=detector_zero_lab,
            detector_column_step_lab_m=column_step_lab,
            detector_row_step_lab_m=row_step_lab,
            detector_pixel_area_vector_lab_m2=detector_area_vector,
            sample_from_lab=sample_from_lab,
            ray_origin_lab_m=np.ascontiguousarray(
                self._detector.incident.states.sample_intersection_lab_m[state_indices]
            ),
        )
        rebound_blocks: list[tuple[_IndexedCompiledEvaluator, ...]] = []
        projection_index = 0
        for block in self._forward_blocks:
            rebound_block: list[_IndexedCompiledEvaluator] = []
            for indexed in block:
                rebound_block.append(
                    indexed.with_evaluator(
                        indexed.evaluator._rebind_validated_projection(
                            indexed.evaluator.state,
                            projection,
                            state_index=projection_index,
                        )
                    )
                )
                projection_index += 1
            rebound_blocks.append(tuple(rebound_block))
        if self._cuda_workspace is not None:
            try:
                self._cuda_workspace.rebind_detector_projection(projection)
            except Exception:
                self._poisoned = True
                raise
        self._forward_blocks = tuple(rebound_blocks)
        self._bound_instrument = instrument
        if self._cuda_workspace is None:
            self.reset()
        else:
            self._reset_host_state()

    def rebind_geometry(self, detector: SourceAveragedDetectorEwaldMeasure) -> None:
        """Bind an already validated geometry view and reset its sampled estimate."""

        if not isinstance(detector, SourceAveragedDetectorEwaldMeasure):
            raise TypeError("detector must be a SourceAveragedDetectorEwaldMeasure")
        self._require_usable()
        old_states = self._detector.incident.states
        new_states = detector.incident.states
        invariant_source_arrays = (
            np.array_equal(old_states.incident_state_id, new_states.incident_state_id)
            and np.array_equal(old_states.incident_sample_id, new_states.incident_sample_id)
            and np.array_equal(old_states.source_weight, new_states.source_weight)
            and np.array_equal(old_states.wavelength_A, new_states.wavelength_A)
            and np.array_equal(old_states.valid, new_states.valid)
        )
        old_instrument = self._bound_instrument
        new_instrument = detector.instrument
        invariant_instrument = _geometry_rebind_instrument_invariants(
            old_instrument,
            new_instrument,
        )
        old_evaluators = tuple(
            indexed for block in self._detector._evaluator_blocks for indexed in block
        )
        new_evaluators = tuple(indexed for block in detector._evaluator_blocks for indexed in block)
        invariant_topology = len(old_evaluators) == len(new_evaluators) and all(
            old.incident_state_index == new.incident_state_index
            and np.array_equal(old.master_rod_index, new.master_rod_index)
            for old, new in zip(old_evaluators, new_evaluators, strict=True)
        )
        if (
            not invariant_source_arrays
            or not invariant_instrument
            or not invariant_topology
            or detector.source_state_count != self._detector.source_state_count
            or detector.rods != self._detector.rods
            or detector.rod_catalog_revision != self._detector.rod_catalog_revision
            or detector.mosaic is not self._detector.mosaic
            or detector.strength_model is not self._detector.strength_model
            or detector._specular_stitch_stack != self._detector._specular_stitch_stack
            or detector.material.material_revision != self._detector.material.material_revision
            or detector.incident.states.source_revision
            != self._detector.incident.states.source_revision
            or detector._phase_polarization_weight != self._detector._phase_polarization_weight
            or detector.instrument.detector_shape_rc != self._detector.instrument.detector_shape_rc
        ):
            raise ValueError(
                "geometry rebind requires unchanged source, rods, physics, and detector shape"
            )
        forward_blocks = _forward_monte_carlo_blocks(detector._evaluator_blocks)
        if self._cuda_workspace is not None:
            try:
                self._cuda_workspace.rebind_geometry(detector._evaluator_blocks)
            except Exception:
                self._poisoned = True
                raise
        self._detector = detector
        self._bound_instrument = detector.instrument
        self._forward_blocks = forward_blocks
        if self._cuda_workspace is None:
            self._execution_worker_count = min(
                detector._worker_count,
                len(self._forward_blocks),
                4,
            )
        if self._cuda_workspace is None:
            self.reset()
        else:
            self._reset_host_state()

    def advance_to(
        self,
        draws_per_source_state: int,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> MonteCarloDetectorPixelMass:
        """Add only the missing Philox draw prefix and return an immutable snapshot."""

        self._advance_to_count(
            draws_per_source_state,
            cancel_requested=cancel_requested,
        )
        return self._snapshot()

    def advance_preview_to(
        self,
        draws_per_source_state: int,
        *,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> MonteCarloDetectorPresentation:
        """Advance and lease a float32 frame until this sampler's next operation."""

        self._advance_to_count(
            draws_per_source_state,
            cancel_requested=cancel_requested,
        )
        return self._presentation_snapshot()

    def _advance_to_count(
        self,
        draws_per_source_state: int,
        *,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        self._require_usable()
        target_draw_count = positive_integer(
            draws_per_source_state,
            "draws_per_source_state",
        )
        if target_draw_count < self._draw_count:
            raise ValueError("progressive draw count cannot decrease; reset the sampler first")
        try:
            _raise_if_monte_carlo_cancelled(cancel_requested)
            if target_draw_count > self._draw_count:
                if self._cuda_workspace is None:
                    self._advance_cpu(
                        target_draw_count,
                        cancel_requested=cancel_requested,
                    )
                else:
                    self._advance_cuda(
                        target_draw_count,
                        cancel_requested=cancel_requested,
                    )
        except Exception:
            try:
                self.reset()
            except Exception as reset_error:
                self._poisoned = True
                raise RuntimeError(
                    "forward Monte Carlo cleanup failed; discard the sampler"
                ) from reset_error
            raise

    def _advance_cuda(
        self,
        target_draw_count: int,
        *,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        draw_start = self._draw_count
        evaluators = tuple(indexed for block in self._forward_blocks for indexed in block)
        draw_count = target_draw_count - draw_start
        all_alpha, all_beta = _sample_mosaic_orientation_matrix(
            self._detector.mosaic,
            target_draw_count,
            seed=self._seed,
            source_state_count=self._detector.source_state_count,
            draw_start=draw_start,
        )
        state_index = np.asarray(
            [indexed.incident_state_index for indexed in evaluators],
            dtype=np.int64,
        )
        alpha = np.ascontiguousarray(all_alpha[state_index])
        beta = np.ascontiguousarray(all_beta[state_index])
        _raise_if_monte_carlo_cancelled(cancel_requested)
        completed = self._cuda_workspace.advance(
            alpha,
            beta,
            draw_start=draw_start,
            draw_stop=target_draw_count,
            cancel_requested=cancel_requested,
        )
        if not completed:
            raise MonteCarloSamplingCancelled("forward Monte Carlo sampling was cancelled")
        self._draw_count = target_draw_count
        self._attempted_root_count += draw_count * self._attempted_root_count_per_draw

    def _advance_cpu(
        self,
        target_draw_count: int,
        *,
        cancel_requested: Callable[[], bool] | None,
    ) -> None:
        draw_start = self._draw_count
        rows, columns = self._detector.instrument.detector_shape_rc
        alpha, beta = _sample_mosaic_orientation_matrix(
            self._detector.mosaic,
            target_draw_count,
            seed=self._seed,
            source_state_count=self._detector.source_state_count,
            draw_start=draw_start,
        )

        def accumulate(
            block: tuple[_IndexedCompiledEvaluator, ...],
        ) -> _ForwardMonteCarloBlockResult:
            return _accumulate_forward_monte_carlo_block(
                block,
                alpha_rad=alpha,
                beta_rad=beta,
                detector_pixel_count=rows * columns,
                cancel_requested=cancel_requested,
            )

        if self._execution_worker_count == 1:
            block_results = tuple(accumulate(block) for block in self._forward_blocks)
        else:
            with ThreadPoolExecutor(max_workers=self._execution_worker_count) as executor:
                futures = tuple(
                    executor.submit(accumulate, block) for block in self._forward_blocks
                )
                block_results = tuple(future.result() for future in futures)
        _raise_if_monte_carlo_cancelled(cancel_requested)

        replicate_total = np.zeros(target_draw_count, dtype=np.float64)
        replicate_total[:draw_start] = self._replicate_total_mass_A2
        attempted = self._attempted_root_count
        visible = self._visible_hit_count
        maximum = self._maximum_root_weight_A2
        for result in block_results:
            self._raw_image_A2 += result.raw_image_A2
            replicate_total[draw_start:] += result.replicate_total_mass_A2
            attempted += result.attempted_root_count
            visible += result.visible_hit_count
            maximum = max(maximum, result.maximum_root_weight_A2)
        self._replicate_total_mass_A2 = replicate_total
        self._draw_count = target_draw_count
        self._attempted_root_count = attempted
        self._visible_hit_count = visible
        self._maximum_root_weight_A2 = maximum

    def _snapshot(self) -> MonteCarloDetectorPixelMass:
        if self._draw_count < 1:
            raise RuntimeError("a Monte Carlo snapshot requires at least one completed draw")
        rows, columns = self._detector.instrument.detector_shape_rc
        if self._cuda_workspace is not None:
            (
                self._raw_image_A2,
                self._replicate_total_mass_A2,
                self._visible_hit_count,
                self._maximum_root_weight_A2,
            ) = self._cuda_workspace.snapshot()
        normalized_image_A2 = np.empty((rows, columns), dtype=np.float64)
        np.divide(
            self._raw_image_A2.reshape(rows, columns),
            self._draw_count,
            out=normalized_image_A2,
        )
        return MonteCarloDetectorPixelMass(
            image_A2=normalized_image_A2,
            replicate_total_mass_A2=self._replicate_total_mass_A2,
            total_detector_mass_A2=fsum(self._replicate_total_mass_A2) / self._draw_count,
            draws_per_source_state=self._draw_count,
            source_state_count=self._detector.source_state_count,
            active_source_state_count=self._detector.valid_source_state_count,
            attempted_root_count=self._attempted_root_count,
            visible_hit_count=self._visible_hit_count,
            maximum_root_deposit_A2=self._maximum_root_weight_A2 / self._draw_count,
            seed=self._seed,
            rods=self._detector.rods,
            source_revision=self._detector.incident.states.source_revision,
            rod_catalog_revision=self._detector.rod_catalog_revision,
            detector_visible_m0_q_gap_Ainv=(self._detector.detector_visible_m0_q_gap_Ainv),
            execution_backend=self._execution_backend,
            execution_device=self._execution_device,
            execution_worker_count=self._execution_worker_count,
            _array_ownership_token=_ARRAY_OWNERSHIP_TOKEN,
        )

    def _presentation_snapshot(self) -> MonteCarloDetectorPresentation:
        if self._draw_count < 1:
            raise RuntimeError("a Monte Carlo presentation requires at least one completed draw")
        rows, columns = self._detector.instrument.detector_shape_rc
        if self._cuda_workspace is None:
            np.multiply(
                self._raw_image_A2,
                1.0 / self._draw_count,
                out=self._presentation_image_A2,
                casting="unsafe",
            )
            presentation_image = self._presentation_image_A2.reshape(rows, columns)
        else:
            (
                presentation_image,
                self._replicate_total_mass_A2,
                self._visible_hit_count,
                self._maximum_root_weight_A2,
            ) = self._cuda_workspace.presentation_snapshot()
        return MonteCarloDetectorPresentation(
            image_A2=presentation_image,
            total_detector_mass_A2=fsum(self._replicate_total_mass_A2) / self._draw_count,
            draws_per_source_state=self._draw_count,
            source_state_count=self._detector.source_state_count,
            active_source_state_count=self._detector.valid_source_state_count,
            attempted_root_count=self._attempted_root_count,
            visible_hit_count=self._visible_hit_count,
            maximum_root_deposit_A2=self._maximum_root_weight_A2 / self._draw_count,
            seed=self._seed,
            rod_count=len(self._detector.rods),
            execution_backend=self._execution_backend,
            execution_device=self._execution_device,
        )


__all__ = [
    "CompiledMonteCarloDetectorSampler",
    "MonteCarloDetectorPixelMass",
    "MonteCarloDetectorPresentation",
    "MonteCarloSamplingCancelled",
    "SourceAveragedDetectorCoordinateDensity",
    "SourceAveragedDetectorCoordinateIntensity",
    "SourceAveragedDetectorEwaldMeasure",
    "source_averaged_detector_geometry_revision",
    "source_averaged_detector_instrument_revision",
]
