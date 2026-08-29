"""Sparse source-averaged detector transfer for structure fitting."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from painted_ewald import (
    BraggSpaceConfig,
    ContinuousEwaldCoating,
    MosaicBraggSpace,
    MosaicParameters,
    Rod,
    wrapped_mosaic_line_density_rad_inv,
)
from painted_ewald.validation import positive_integer, reject_complex
from rasim_next.core.contracts import MaterialOptics, canonical_revision_sha256
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.geometry.transport import IncidentTransportResult
from rasim_next.optics import (
    DETECTOR_PATH_ATTENUATION_MODEL_ID,
    INCIDENT_ILLUMINATED_PATH_MODEL_ID,
)
from rasim_next.pipeline.bragg_space import RevisionedStructureStrengthModel
from rasim_next.pipeline.continuous_detector import (
    DetectorEwaldMeasure,
    DetectorStructureResponse,
    SampleQIntensityEnvelope,
)
from rasim_next.pipeline.source_averaged_detector import (
    SourceAveragedDetectorCoordinateIntensity,
    _detector_native_chart_revision,
    _incidence_angle_static_physics_revision,
    _instrument_revision,
    _reachable_master_rod_indices,
)
from rasim_next.sampling.source import require_physical_intensity_source_model

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "SourceAveragedDetectorStructureResponse",
    "SourceAveragedStructureDetector",
    "compile_source_averaged_detector_structure_response",
]


def _structure_model_revision(strength_model: RevisionedStructureStrengthModel) -> str:
    revision = getattr(strength_model, "structure_model_revision", None)
    if (
        not isinstance(revision, str)
        or len(revision) != 64
        or revision != revision.lower()
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("strength model must expose a lowercase SHA-256 structure-model revision")
    return revision


@dataclass(frozen=True, slots=True)
class SourceAveragedDetectorStructureResponse:
    """Sparse source-averaged detector transfer independent of structure strength.

    Each term is one regular inverse root. Source states, physical rods, and
    roots remain explicit until :meth:`apply_strength` performs the declared
    incoherent intensity sum.
    """

    column_px: FloatArray
    row_px: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    reciprocal_basis_Ainv: FloatArray
    term_coordinate_index: NDArray[np.int64]
    term_rod_index: NDArray[np.int64]
    term_source_state_index: NDArray[np.int64]
    term_L: FloatArray
    term_alpha_rad: FloatArray
    term_k_norm_Ainv: FloatArray
    term_q_radial_squared_Ainv2: FloatArray
    term_q_normal_squared_Ainv2: FloatArray
    term_fixed_density_per_mosaic_density_per_strength_px2_inv: FloatArray
    term_fixed_density_per_strength_px2_inv: FloatArray
    term_root_sign: NDArray[np.int8]
    source_state_k_norm_Ainv: FloatArray
    valid_source_count: NDArray[np.int64]
    per_rod_caustic: BoolArray
    source_state_count: int
    source_revision: str
    material_revision: str
    sample_geometry_revision: str
    instrument_revision: str
    fixed_physics_revision: str
    reference_structure_model_revision: str
    detector_visible_m0_q_gap_Ainv: float | None
    reference_intensity_envelope_u_radial_A2: float = 0.0
    reference_intensity_envelope_u_normal_A2: float = 0.0
    root_policy: str = "all_retained_roots.v1"
    measure_id: str = "fixed_source_averaged_detector_density_per_structure_strength_px2_inv.v1"
    response_revision: str = field(init=False)

    def __post_init__(self) -> None:
        column = np.array(self.column_px, dtype=np.float64, copy=True, order="C")
        row = np.array(self.row_px, dtype=np.float64, copy=True, order="C")
        if (
            column.shape != row.shape
            or not np.all(np.isfinite(column))
            or not np.all(np.isfinite(row))
        ):
            raise ValueError("detector coordinates must be finite aligned arrays")
        coordinate_count = column.size
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one physical Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical line")
        if not isinstance(self.rod_catalog_revision, str) or not self.rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        basis = np.array(
            self.reciprocal_basis_Ainv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            basis.shape != (3, 3)
            or not np.all(np.isfinite(basis))
            or np.isclose(np.linalg.det(basis), 0.0)
        ):
            raise ValueError("reciprocal_basis_Ainv must be finite and nonsingular")

        term_coordinate = np.array(
            self.term_coordinate_index,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        term_rod = np.array(self.term_rod_index, dtype=np.int64, copy=True, order="C")
        term_source = np.array(
            self.term_source_state_index,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        term_l = np.array(self.term_L, dtype=np.float64, copy=True, order="C")
        term_alpha = np.array(self.term_alpha_rad, dtype=np.float64, copy=True, order="C")
        term_k = np.array(self.term_k_norm_Ainv, dtype=np.float64, copy=True, order="C")
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
        term_root = np.array(self.term_root_sign, dtype=np.int8, copy=True, order="C")
        term_shape = term_coordinate.shape
        if (
            term_coordinate.ndim != 1
            or term_rod.shape != term_shape
            or term_source.shape != term_shape
            or term_l.shape != term_shape
            or term_alpha.shape != term_shape
            or term_k.shape != term_shape
            or term_q_radial_squared.shape != term_shape
            or term_q_normal_squared.shape != term_shape
            or term_fixed_per_mosaic.shape != term_shape
            or term_fixed.shape != term_shape
            or term_root.shape != term_shape
        ):
            raise ValueError("all sparse source-response term arrays must align")
        state_count = positive_integer(self.source_state_count, "source_state_count")
        state_k = np.array(
            self.source_state_k_norm_Ainv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            state_k.shape != (state_count,)
            or np.any(~np.isfinite(state_k))
            or np.any(state_k <= 0.0)
        ):
            raise ValueError("source_state_k_norm_Ainv must be finite and positive")
        if (
            np.any((term_coordinate < 0) | (term_coordinate >= coordinate_count))
            or np.any((term_rod < 0) | (term_rod >= len(rods)))
            or np.any((term_source < 0) | (term_source >= state_count))
            or np.any(~np.isfinite(term_l))
            or np.any(~np.isfinite(term_alpha))
            or np.any((term_alpha < 0.0) | (term_alpha > np.pi))
            or np.any(~np.isfinite(term_k))
            or np.any(term_k <= 0.0)
            or np.any(~np.isfinite(term_q_radial_squared))
            or np.any(term_q_radial_squared < 0.0)
            or np.any(~np.isfinite(term_q_normal_squared))
            or np.any(term_q_normal_squared < 0.0)
            or np.any(~np.isfinite(term_fixed_per_mosaic))
            or np.any(term_fixed_per_mosaic < 0.0)
            or np.any(~np.isfinite(term_fixed))
            or np.any(term_fixed < 0.0)
            or np.any(~np.isin(term_root, (-1, 0, 1)))
        ):
            raise ValueError("sparse source-response terms are invalid")
        if term_source.size and not np.array_equal(term_k, state_k[term_source]):
            raise ValueError("term k norms must match their exact source states")
        reference_u_radial = float(self.reference_intensity_envelope_u_radial_A2)
        reference_u_normal = float(self.reference_intensity_envelope_u_normal_A2)
        if (
            not isfinite(reference_u_radial)
            or reference_u_radial < 0.0
            or not isfinite(reference_u_normal)
            or reference_u_normal < 0.0
        ):
            raise ValueError("reference intensity-envelope coefficients must be nonnegative")

        valid_count = np.array(
            self.valid_source_count,
            dtype=np.int64,
            copy=True,
            order="C",
        )
        if valid_count.shape != column.shape or np.any(
            (valid_count < 0) | (valid_count > state_count)
        ):
            raise ValueError("valid_source_count must align and lie within the source batch")
        caustic = np.array(self.per_rod_caustic, dtype=np.bool_, copy=True, order="C")
        if caustic.shape != (*column.shape, len(rods)):
            raise ValueError("per_rod_caustic must align with coordinates and rods")
        for name in (
            "source_revision",
            "material_revision",
            "sample_geometry_revision",
            "instrument_revision",
            "fixed_physics_revision",
            "reference_structure_model_revision",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a sha256 revision")
        if self.root_policy != "all_retained_roots.v1":
            raise ValueError("sparse structure response requires all retained roots")
        if self.measure_id != (
            "fixed_source_averaged_detector_density_per_structure_strength_px2_inv.v1"
        ):
            raise ValueError("unsupported sparse source-response measure")
        m0_gap = self.detector_visible_m0_q_gap_Ainv
        has_m0 = any(rod.h == 0 and rod.k == 0 for rod in rods)
        if has_m0:
            if m0_gap is None or not isfinite(float(m0_gap)) or float(m0_gap) < 0.0:
                raise ValueError("detector-visible 00L requires a nonnegative support gap")
            m0_gap = float(m0_gap)
        elif m0_gap is not None:
            raise ValueError("a 00L support gap requires the (0, 0) rod")

        response_revision = canonical_revision_sha256(
            ("definition_id", "source_averaged_sparse_structure_response.v3"),
            ("fixed_physics_revision", self.fixed_physics_revision),
            ("reference_structure_model_revision", self.reference_structure_model_revision),
            ("column_px", column),
            ("row_px", row),
            ("term_coordinate_index", term_coordinate),
            ("term_rod_index", term_rod),
            ("term_source_state_index", term_source),
            ("term_L", term_l),
            ("term_alpha_rad", term_alpha),
            ("term_k_norm_Ainv", term_k),
            ("term_q_radial_squared_Ainv2", term_q_radial_squared),
            ("term_q_normal_squared_Ainv2", term_q_normal_squared),
            (
                "term_fixed_density_per_mosaic_density_per_strength_px2_inv",
                term_fixed_per_mosaic,
            ),
            ("term_fixed_density_per_strength_px2_inv", term_fixed),
            ("reference_intensity_envelope_u_radial_A2", reference_u_radial),
            ("reference_intensity_envelope_u_normal_A2", reference_u_normal),
            ("term_root_sign", term_root),
            ("valid_source_count", valid_count),
            ("per_rod_caustic", caustic),
        )
        for value in (
            column,
            row,
            basis,
            term_coordinate,
            term_rod,
            term_source,
            term_l,
            term_alpha,
            term_k,
            term_q_radial_squared,
            term_q_normal_squared,
            term_fixed_per_mosaic,
            term_fixed,
            term_root,
            state_k,
            valid_count,
            caustic,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "column_px", column)
        object.__setattr__(self, "row_px", row)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "term_coordinate_index", term_coordinate)
        object.__setattr__(self, "term_rod_index", term_rod)
        object.__setattr__(self, "term_source_state_index", term_source)
        object.__setattr__(self, "term_L", term_l)
        object.__setattr__(self, "term_alpha_rad", term_alpha)
        object.__setattr__(self, "term_k_norm_Ainv", term_k)
        object.__setattr__(self, "term_q_radial_squared_Ainv2", term_q_radial_squared)
        object.__setattr__(self, "term_q_normal_squared_Ainv2", term_q_normal_squared)
        object.__setattr__(
            self,
            "term_fixed_density_per_mosaic_density_per_strength_px2_inv",
            term_fixed_per_mosaic,
        )
        object.__setattr__(self, "term_fixed_density_per_strength_px2_inv", term_fixed)
        object.__setattr__(self, "term_root_sign", term_root)
        object.__setattr__(self, "source_state_k_norm_Ainv", state_k)
        object.__setattr__(self, "valid_source_count", valid_count)
        object.__setattr__(self, "per_rod_caustic", caustic)
        object.__setattr__(self, "source_state_count", state_count)
        object.__setattr__(
            self,
            "reference_intensity_envelope_u_radial_A2",
            reference_u_radial,
        )
        object.__setattr__(
            self,
            "reference_intensity_envelope_u_normal_A2",
            reference_u_normal,
        )
        object.__setattr__(self, "detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(self, "response_revision", response_revision)

    def _weighted_term_strength(
        self,
        strength_model: RevisionedStructureStrengthModel,
        *,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
    ) -> FloatArray:
        _structure_model_revision(strength_model)
        basis = np.asarray(getattr(strength_model, "reciprocal_basis_Ainv", None))
        basis_scale = max(float(np.linalg.norm(self.reciprocal_basis_Ainv)), 1.0)
        if basis.shape != (3, 3) or not np.allclose(
            basis,
            self.reciprocal_basis_Ainv,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * basis_scale,
        ):
            raise ValueError("strength-model and detector-response reciprocal bases disagree")
        if np.any(self.per_rod_caustic):
            raise FloatingPointError(
                "sparse structure response cannot apply strength at an exact detector caustic"
            )
        supplied = strength_model.evaluate_hkl(
            h=np.asarray([self.rods[index].h for index in self.term_rod_index]),
            k=np.asarray([self.rods[index].k for index in self.term_rod_index]),
            L=self.term_L,
            k_norm_Ainv=self.term_k_norm_Ainv,
        )
        reject_complex(supplied, "evaluate_hkl result")
        term_strength = np.asarray(supplied, dtype=np.float64)
        if term_strength.shape != self.term_L.shape:
            raise ValueError("evaluate_hkl returned the wrong shape")
        if np.any(~np.isfinite(term_strength)) or np.any(term_strength < 0.0):
            raise ValueError("structure strength must be finite and nonnegative")

        envelope_factor: FloatArray | float = 1.0
        if intensity_envelope is not None:
            if not isinstance(intensity_envelope, SampleQIntensityEnvelope):
                raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
            delta_radial = (
                intensity_envelope.u_radial_A2 - self.reference_intensity_envelope_u_radial_A2
            )
            delta_normal = (
                intensity_envelope.u_normal_A2 - self.reference_intensity_envelope_u_normal_A2
            )
            with np.errstate(over="ignore", under="ignore"):
                envelope_factor = np.exp(
                    -delta_radial * self.term_q_radial_squared_Ainv2
                    - delta_normal * self.term_q_normal_squared_Ainv2
                )
            if np.any(~np.isfinite(envelope_factor)):
                raise ValueError("requested intensity envelope overflows the sparse response")
        return np.asarray(term_strength * envelope_factor, dtype=np.float64)

    def _coordinate_intensity(
        self, term_density: FloatArray
    ) -> SourceAveragedDetectorCoordinateIntensity:
        coordinate_count = self.column_px.size
        rod_count = len(self.rods)
        flat_index = self.term_coordinate_index * rod_count + self.term_rod_index
        flat_per_rod = np.bincount(
            flat_index,
            weights=term_density,
            minlength=coordinate_count * rod_count,
        )
        per_rod = flat_per_rod.reshape((*self.column_px.shape, rod_count))
        total = np.sum(per_rod, axis=-1, dtype=np.float64)
        return SourceAveragedDetectorCoordinateIntensity(
            column_px=self.column_px,
            row_px=self.row_px,
            rods=self.rods,
            rod_catalog_revision=self.rod_catalog_revision,
            branch=None,
            per_rod_density_A2_per_px2=per_rod,
            density_A2_per_px2=total,
            caustic=self.per_rod_caustic,
            valid_source_count=self.valid_source_count,
            source_state_count=self.source_state_count,
            source_revision=self.source_revision,
            root_policy=self.root_policy,
            detector_visible_m0_q_gap_Ainv=self.detector_visible_m0_q_gap_Ainv,
            execution_backend="numpy_cpu_sparse_source_averaged.v1",
        )

    def apply_strength(
        self,
        strength_model: RevisionedStructureStrengthModel,
        *,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        """Apply one revisioned structure provider to the frozen detector transfer."""

        weighted_strength = self._weighted_term_strength(
            strength_model,
            intensity_envelope=intensity_envelope,
        )
        return self._coordinate_intensity(
            self.term_fixed_density_per_strength_px2_inv * weighted_strength
        )

    def apply_strength_for_mosaics(
        self,
        strength_model: RevisionedStructureStrengthModel,
        mosaics: tuple[MosaicParameters, ...],
        *,
        intensity_envelope: SampleQIntensityEnvelope | None = None,
    ) -> tuple[SourceAveragedDetectorCoordinateIntensity, ...]:
        """Apply one structure evaluation to several continuous mosaic densities."""

        requested = tuple(mosaics)
        if not requested or any(not isinstance(item, MosaicParameters) for item in requested):
            raise ValueError("mosaics must contain at least one MosaicParameters value")
        if any(item.zero_tilt_probability_mass != 0.0 for item in requested):
            raise ValueError("sparse mosaic rebinding does not support zero-tilt atoms")
        weighted_strength = self._weighted_term_strength(
            strength_model,
            intensity_envelope=intensity_envelope,
        )
        fixed = self.term_fixed_density_per_mosaic_density_per_strength_px2_inv * weighted_strength
        return tuple(
            self._coordinate_intensity(
                fixed * (wrapped_mosaic_line_density_rad_inv(self.term_alpha_rad, mosaic) / np.pi)
            )
            for mosaic in requested
        )


@dataclass(frozen=True, slots=True)
class SourceAveragedStructureDetector:
    """Material-neutral detector function used by mosaic and SF fitting.

    The sparse transfer is compiled at requested coordinates and the bound
    strength provider is then applied. Optimized Bi2X3 renderers remain a
    separate execution path but must reproduce this fitting authority.
    """

    reciprocal_basis_Ainv: FloatArray
    crystal_to_sample: FloatArray
    rods: tuple[Rod, ...]
    rod_catalog_revision: str
    mosaic: MosaicParameters
    strength_model: RevisionedStructureStrengthModel
    incident: IncidentTransportResult
    material: MaterialOptics
    instrument: CompiledInstrument
    intensity_envelope: SampleQIntensityEnvelope = field(default_factory=SampleQIntensityEnvelope)
    phase_population_weight: float = 1.0
    polarization_weight: float = 1.0
    incidence_axis_angle_rad: float | None = None
    scan_calibration_binding_revision: str | None = None
    _detector_visible_m0_q_gap_Ainv: float | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _fixed_physics_revision: str = field(init=False, repr=False, compare=False)
    _incidence_angle_static_physics_revision: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        basis = np.array(
            self.reciprocal_basis_Ainv,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if (
            basis.shape != (3, 3)
            or not np.all(np.isfinite(basis))
            or np.isclose(np.linalg.det(basis), 0.0)
        ):
            raise ValueError("reciprocal_basis_Ainv must be finite and nonsingular")
        strength_basis = np.asarray(getattr(self.strength_model, "reciprocal_basis_Ainv", None))
        basis_scale = max(float(np.linalg.norm(basis)), 1.0)
        if strength_basis.shape != (3, 3) or not np.allclose(
            strength_basis,
            basis,
            rtol=0.0,
            atol=256.0 * np.finfo(np.float64).eps * basis_scale,
        ):
            raise ValueError("strength-model and detector reciprocal bases disagree")
        _structure_model_revision(self.strength_model)
        rods = tuple(self.rods)
        if not rods or any(not isinstance(rod, Rod) for rod in rods):
            raise ValueError("rods must contain at least one physical Rod")
        if len({(rod.h, rod.k) for rod in rods}) != len(rods):
            raise ValueError("rods must not repeat a physical line")
        if not isinstance(self.rod_catalog_revision, str) or not self.rod_catalog_revision:
            raise ValueError("rod_catalog_revision must be nonempty")
        if not isinstance(self.mosaic, MosaicParameters):
            raise TypeError("mosaic must be MosaicParameters")
        if not isinstance(self.incident, IncidentTransportResult):
            raise TypeError("incident must be IncidentTransportResult")
        if not isinstance(self.material, MaterialOptics):
            raise TypeError("material must be MaterialOptics")
        if not isinstance(self.instrument, CompiledInstrument):
            raise TypeError("instrument must be CompiledInstrument")
        envelope = self.intensity_envelope
        if not isinstance(envelope, SampleQIntensityEnvelope):
            raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
        phase_weight = float(self.phase_population_weight)
        polarization = float(self.polarization_weight)
        if not isfinite(phase_weight) or phase_weight < 0.0:
            raise ValueError("phase_population_weight must be finite and nonnegative")
        if not isfinite(polarization) or polarization < 0.0:
            raise ValueError("polarization_weight must be finite and nonnegative")
        axis_angle = (
            None if self.incidence_axis_angle_rad is None else float(self.incidence_axis_angle_rad)
        )
        if axis_angle is not None and not isfinite(axis_angle):
            raise ValueError("incidence_axis_angle_rad must be finite when provided")
        scan_binding = self.scan_calibration_binding_revision
        if scan_binding is not None and (not isinstance(scan_binding, str) or not scan_binding):
            raise ValueError("scan_calibration_binding_revision must be nonempty when provided")
        crystal_to_sample = np.array(
            self.crystal_to_sample,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if crystal_to_sample.shape != (3, 3) or not np.all(np.isfinite(crystal_to_sample)):
            raise ValueError("crystal_to_sample must be a finite 3x3 rotation")
        rotation_tolerance = 512.0 * np.finfo(np.float64).eps
        if not np.allclose(
            crystal_to_sample,
            self.instrument.sample_from_crystal.rotation,
            rtol=0.0,
            atol=rotation_tolerance,
        ):
            raise ValueError("crystal_to_sample must be the canonical instrument rotation")
        states = self.incident.states
        require_physical_intensity_source_model(states.source_sampling_model_id)
        if states.material_revision != self.material.material_revision:
            raise ValueError("incident states and material revisions disagree")
        if states.sample_geometry_revision != self.instrument.sample_geometry_revision:
            raise ValueError("incident states and instrument sample revisions disagree")
        fixed_revision = _source_averaged_sparse_fixed_physics_revision(
            reciprocal_basis_Ainv=basis,
            crystal_to_sample=crystal_to_sample,
            rods=rods,
            rod_catalog_revision=self.rod_catalog_revision,
            mosaic=self.mosaic,
            incident=self.incident,
            material=self.material,
            instrument=self.instrument,
            intensity_envelope=envelope,
            phase_population_weight=phase_weight,
            polarization_weight=polarization,
        )
        m0_gap: float | None = None
        if any(rod.h == 0 and rod.k == 0 for rod in rods):
            valid_state_index = np.flatnonzero(states.valid)
            if not valid_state_index.size:
                raise ValueError("detector-visible 00L requires a valid incident state")
            incident_normal = states.k_film_phase_sample_Ainv[valid_state_index, 2]
            if np.any(incident_normal >= 0.0):
                raise ValueError(
                    "detector-visible 00L requires every valid source state to enter "
                    "through the negative sample-normal half-space"
                )
            m0_gap = float(np.min(-incident_normal))
        incidence_static_revision = _incidence_angle_static_physics_revision(
            reciprocal_basis_Ainv=basis,
            strength_model_revision=_structure_model_revision(self.strength_model),
            mosaic=self.mosaic,
            intensity_envelope=envelope,
            material_revision=self.material.material_revision,
            incident_model_id=states.incident_model_id,
            instrument=self.instrument,
            phase_polarization_weight=phase_weight * polarization,
            specular_stitch_stack=None,
        )
        basis.setflags(write=False)
        crystal_to_sample.setflags(write=False)
        object.__setattr__(self, "reciprocal_basis_Ainv", basis)
        object.__setattr__(self, "crystal_to_sample", crystal_to_sample)
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "phase_population_weight", phase_weight)
        object.__setattr__(self, "polarization_weight", polarization)
        object.__setattr__(self, "incidence_axis_angle_rad", axis_angle)
        object.__setattr__(self, "scan_calibration_binding_revision", scan_binding)
        object.__setattr__(self, "_detector_visible_m0_q_gap_Ainv", m0_gap)
        object.__setattr__(self, "_fixed_physics_revision", fixed_revision)
        object.__setattr__(
            self,
            "_incidence_angle_static_physics_revision",
            incidence_static_revision,
        )

    def restrict_rods(self, rods: tuple[Rod, ...]) -> SourceAveragedStructureDetector:
        """Return the same physical detector over an explicit signed-rod subset."""

        requested = tuple(rods)
        if not requested or any(not isinstance(rod, Rod) for rod in requested):
            raise ValueError("rods must contain at least one Rod")
        requested_hk = tuple((rod.h, rod.k) for rod in requested)
        if len(set(requested_hk)) != len(requested_hk):
            raise ValueError("rods must not repeat a physical rod")
        configured_by_hk = {(rod.h, rod.k): rod for rod in self.rods}
        try:
            selected = tuple(configured_by_hk[key] for key in requested_hk)
        except KeyError as error:
            raise ValueError(f"rod subset contains unconfigured rod {error.args[0]}") from error
        return self if selected == self.rods else replace(self, rods=selected)

    @property
    def fixed_physics_revision(self) -> str:
        return self._fixed_physics_revision

    @property
    def detector_visible_m0_q_gap_Ainv(self) -> float | None:
        """Infimum of supported ``|Q|`` for an included detector-visible 00L rod."""

        return self._detector_visible_m0_q_gap_Ainv

    @property
    def source_revision(self) -> str:
        return self.incident.states.source_revision

    @property
    def source_state_count(self) -> int:
        return int(self.incident.states.incident_state_id.size)

    @property
    def sample_geometry_revision(self) -> str:
        return self.incident.states.sample_geometry_revision

    @property
    def detector_shape_rc(self) -> tuple[int, int]:
        return self.instrument.detector_shape_rc

    @property
    def detector_panel_revision(self) -> str:
        return _detector_native_chart_revision(self.instrument)

    @property
    def incidence_angle_static_physics_revision(self) -> str:
        """Physics identity required to be common to every incidence node."""

        return self._incidence_angle_static_physics_revision

    def compile_structure_response(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
    ) -> SourceAveragedDetectorStructureResponse:
        """Freeze source, optics, mosaic, and detector transfer at fit coordinates."""

        response = compile_source_averaged_detector_structure_response(
            column_px,
            row_px,
            reciprocal_basis_Ainv=self.reciprocal_basis_Ainv,
            crystal_to_sample=self.crystal_to_sample,
            rods=self.rods,
            rod_catalog_revision=self.rod_catalog_revision,
            mosaic=self.mosaic,
            reference_strength_model=self.strength_model,
            incident=self.incident,
            material=self.material,
            instrument=self.instrument,
            intensity_envelope=self.intensity_envelope,
            phase_population_weight=self.phase_population_weight,
            polarization_weight=self.polarization_weight,
        )
        if response.fixed_physics_revision != self.fixed_physics_revision:
            raise RuntimeError("sparse detector fixed physics changed during compilation")
        return response

    def evaluate_detector_coordinates_all_roots(
        self,
        column_px: ArrayLike,
        row_px: ArrayLike,
        *,
        execution_backend: str = "cpu",
        cuda_coordinate_chunk_size: int | None = None,
    ) -> SourceAveragedDetectorCoordinateIntensity:
        if execution_backend != "cpu":
            raise ValueError("the sparse structure detector currently requires the CPU backend")
        if cuda_coordinate_chunk_size is not None:
            raise ValueError("the sparse structure detector does not use CUDA coordinate chunks")
        return self.compile_structure_response(column_px, row_px).apply_strength(
            self.strength_model
        )


def _source_averaged_sparse_fixed_physics_revision(
    *,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    rods: tuple[Rod, ...],
    rod_catalog_revision: str,
    mosaic: MosaicParameters,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    intensity_envelope: SampleQIntensityEnvelope,
    phase_population_weight: float,
    polarization_weight: float,
) -> str:
    """Hash the structure-independent state frozen by one sparse response."""

    return canonical_revision_sha256(
        ("definition_id", "source_averaged_sparse_fixed_physics.v2"),
        ("detector_path_attenuation_model_id", DETECTOR_PATH_ATTENUATION_MODEL_ID),
        ("incident_illuminated_path_model_id", INCIDENT_ILLUMINATED_PATH_MODEL_ID),
        ("source_revision", incident.states.source_revision),
        ("material_revision", material.material_revision),
        ("sample_geometry_revision", incident.states.sample_geometry_revision),
        ("instrument_revision", _instrument_revision(instrument)),
        ("reciprocal_basis_Ainv", np.asarray(reciprocal_basis_Ainv, dtype=np.float64)),
        ("crystal_to_sample", np.asarray(crystal_to_sample, dtype=np.float64)),
        ("rod_catalog_revision", rod_catalog_revision),
        ("rod_h", np.asarray([rod.h for rod in rods], dtype=np.int64)),
        ("rod_k", np.asarray([rod.k for rod in rods], dtype=np.int64)),
        ("rod_population", np.asarray([rod.population for rod in rods], dtype=np.float64)),
        ("mosaic_gaussian_sigma_rad", mosaic.gaussian_sigma_rad),
        ("mosaic_lorentzian_half_width_rad", mosaic.lorentzian_half_width_rad),
        ("mosaic_lorentzian_probability", mosaic.lorentzian_probability),
        ("mosaic_alpha_panel_count", mosaic.alpha_panel_count),
        ("mosaic_alpha_gauss_order", mosaic.alpha_gauss_order),
        ("mosaic_azimuth_count", mosaic.azimuth_count),
        ("mosaic_azimuth_phase_rad", mosaic.azimuth_phase_rad),
        ("intensity_envelope_u_radial_A2", intensity_envelope.u_radial_A2),
        ("intensity_envelope_u_normal_A2", intensity_envelope.u_normal_A2),
        ("phase_population_weight", phase_population_weight),
        ("polarization_weight", polarization_weight),
    )


def compile_source_averaged_detector_structure_response(
    column_px: ArrayLike,
    row_px: ArrayLike,
    *,
    reciprocal_basis_Ainv: ArrayLike,
    crystal_to_sample: ArrayLike,
    rods: tuple[Rod, ...],
    rod_catalog_revision: str,
    mosaic: MosaicParameters,
    reference_strength_model: RevisionedStructureStrengthModel,
    incident: IncidentTransportResult,
    material: MaterialOptics,
    instrument: CompiledInstrument,
    intensity_envelope: SampleQIntensityEnvelope | None = None,
    phase_population_weight: float = 1.0,
    polarization_weight: float = 1.0,
) -> SourceAveragedDetectorStructureResponse:
    """Compile a material-neutral sparse detector transfer for structure fitting.

    The reference strength supplies only the reciprocal-basis contract required
    by :class:`MosaicBraggSpace`; no reference strength is evaluated or used to
    prune terms. Candidate structures are applied later through the returned
    response.
    """

    if not isinstance(incident, IncidentTransportResult):
        raise TypeError("incident must be IncidentTransportResult")
    require_physical_intensity_source_model(incident.states.source_sampling_model_id)
    if not isinstance(material, MaterialOptics):
        raise TypeError("material must be MaterialOptics")
    if not isinstance(instrument, CompiledInstrument):
        raise TypeError("instrument must be CompiledInstrument")
    if not isinstance(mosaic, MosaicParameters):
        raise TypeError("mosaic must be MosaicParameters")
    if mosaic.zero_tilt_probability_mass != 0.0:
        raise ValueError("sparse source averaging does not support zero-tilt atoms")
    envelope = SampleQIntensityEnvelope() if intensity_envelope is None else intensity_envelope
    if not isinstance(envelope, SampleQIntensityEnvelope):
        raise TypeError("intensity_envelope must be SampleQIntensityEnvelope")
    selected = tuple(rods)
    if not selected or any(not isinstance(rod, Rod) for rod in selected):
        raise ValueError("rods must contain at least one Rod")
    if len({(rod.h, rod.k) for rod in selected}) != len(selected):
        raise ValueError("rods must not repeat a physical line")
    if not isinstance(rod_catalog_revision, str) or not rod_catalog_revision:
        raise ValueError("rod_catalog_revision must be nonempty")
    phase_weight = float(phase_population_weight)
    polarization = float(polarization_weight)
    if not isfinite(phase_weight) or phase_weight < 0.0:
        raise ValueError("phase_population_weight must be finite and nonnegative")
    if not isfinite(polarization) or polarization < 0.0:
        raise ValueError("polarization_weight must be finite and nonnegative")

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
    column = np.array(column, dtype=np.float64, copy=True, order="C")
    row = np.array(row, dtype=np.float64, copy=True, order="C")

    states = incident.states
    if states.material_revision != material.material_revision:
        raise ValueError("incident states and material revisions disagree")
    if states.sample_geometry_revision != instrument.sample_geometry_revision:
        raise ValueError("incident states and instrument sample revisions disagree")
    valid_state_index = np.flatnonzero(states.valid)
    if not valid_state_index.size:
        raise ValueError("source-averaged response requires a valid incident state")
    source_state_k = 2.0 * np.pi / np.asarray(states.wavelength_A, dtype=np.float64)
    maximum_k = float(np.max(source_state_k[valid_state_index]))
    reference_config = BraggSpaceConfig(
        reciprocal_basis_Ainv=reciprocal_basis_Ainv,
        crystal_to_sample=crystal_to_sample,
        rods=selected,
        mosaic=mosaic,
        k_norm_Ainv=maximum_k,
    )
    rotation_tolerance = 512.0 * np.finfo(np.float64).eps
    if not np.allclose(
        reference_config.crystal_to_sample,
        instrument.sample_from_crystal.rotation,
        rtol=0.0,
        atol=rotation_tolerance,
    ):
        raise ValueError("crystal_to_sample must be the canonical instrument rotation")
    strength_basis = np.asarray(getattr(reference_strength_model, "reciprocal_basis_Ainv", None))
    basis_scale = max(float(np.linalg.norm(reference_config.reciprocal_basis_Ainv)), 1.0)
    if strength_basis.shape != (3, 3) or not np.allclose(
        strength_basis,
        reference_config.reciprocal_basis_Ainv,
        rtol=0.0,
        atol=256.0 * np.finfo(np.float64).eps * basis_scale,
    ):
        raise ValueError("reference strength and Bragg-space reciprocal bases disagree")
    reference_structure_revision = _structure_model_revision(reference_strength_model)

    rod_count = len(selected)
    valid_count = np.zeros(column.shape, dtype=np.int64)
    caustic = np.zeros((*column.shape, rod_count), dtype=np.bool_)
    term_coordinates: list[NDArray[np.int64]] = []
    term_rods: list[NDArray[np.int64]] = []
    term_sources: list[NDArray[np.int64]] = []
    term_l_values: list[FloatArray] = []
    term_alpha_values: list[FloatArray] = []
    term_k_values: list[FloatArray] = []
    term_q_radial_squared_values: list[FloatArray] = []
    term_q_normal_squared_values: list[FloatArray] = []
    term_fixed_per_mosaic_values: list[FloatArray] = []
    term_fixed_values: list[FloatArray] = []
    term_root_signs: list[NDArray[np.int8]] = []
    for supplied_state_index in valid_state_index:
        state_index = int(supplied_state_index)
        k_norm = float(source_state_k[state_index])
        active_index = _reachable_master_rod_indices(
            selected,
            reference_config.reciprocal_basis_Ainv,
            k_norm,
        )
        if not active_index.size:
            continue
        active_rods = tuple(selected[int(position)] for position in active_index)
        bragg = MosaicBraggSpace(
            BraggSpaceConfig(
                reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
                crystal_to_sample=reference_config.crystal_to_sample,
                rods=active_rods,
                mosaic=mosaic,
                k_norm_Ainv=k_norm,
            ),
            reference_strength_model,
        )
        detector = DetectorEwaldMeasure(
            coating=ContinuousEwaldCoating(
                bragg,
                ki_sample_Ainv=states.k_film_phase_sample_Ainv[state_index],
            ),
            incident=incident,
            incident_state_index=state_index,
            material=material,
            instrument=instrument,
            rod_catalog_revision=rod_catalog_revision,
            phase_population_weight=phase_weight,
            polarization_weight=polarization,
            intensity_envelope=envelope,
        )
        state_response: DetectorStructureResponse = detector.evaluate_detector_structure_response(
            column,
            row,
            rods=active_rods,
        )
        valid_count += state_response.coordinate_valid.reshape(column.shape)
        caustic[..., active_index] |= state_response.per_rod_caustic.reshape(
            (*column.shape, active_index.size)
        )
        if not state_response.term_coordinate_index.size:
            continue
        local_rod_index = state_response.term_rod_index
        term_count = local_rod_index.size
        term_coordinates.append(state_response.term_coordinate_index)
        term_rods.append(active_index[local_rod_index])
        term_sources.append(np.full(term_count, state_index, dtype=np.int64))
        term_l_values.append(state_response.term_L)
        term_alpha_values.append(state_response.term_alpha_rad)
        term_k_values.append(np.full(term_count, k_norm, dtype=np.float64))
        term_q_radial_squared_values.append(state_response.term_q_radial_squared_Ainv2)
        term_q_normal_squared_values.append(state_response.term_q_normal_squared_Ainv2)
        term_fixed_per_mosaic_values.append(
            state_response.term_fixed_density_per_mosaic_density_per_strength_px2_inv
        )
        term_fixed_values.append(state_response.term_fixed_density_per_strength_px2_inv)
        term_root_signs.append(state_response.term_root_sign)

    def concatenate_or_empty(
        blocks: list[NDArray[np.generic]],
        dtype: np.dtype[np.generic] | type[np.generic],
    ) -> NDArray[np.generic]:
        if blocks:
            return np.asarray(np.concatenate(blocks), dtype=dtype)
        return np.empty(0, dtype=dtype)

    term_coordinate = concatenate_or_empty(term_coordinates, np.int64)
    term_rod = concatenate_or_empty(term_rods, np.int64)
    term_source = concatenate_or_empty(term_sources, np.int64)
    term_l = concatenate_or_empty(term_l_values, np.float64)
    term_alpha = concatenate_or_empty(term_alpha_values, np.float64)
    term_k = concatenate_or_empty(term_k_values, np.float64)
    term_q_radial_squared = concatenate_or_empty(
        term_q_radial_squared_values,
        np.float64,
    )
    term_q_normal_squared = concatenate_or_empty(
        term_q_normal_squared_values,
        np.float64,
    )
    term_fixed_per_mosaic = concatenate_or_empty(
        term_fixed_per_mosaic_values,
        np.float64,
    )
    term_fixed = concatenate_or_empty(term_fixed_values, np.float64)
    term_root = concatenate_or_empty(term_root_signs, np.int8)
    m0_gap: float | None = None
    if any(rod.h == 0 and rod.k == 0 for rod in selected):
        incident_normal = states.k_film_phase_sample_Ainv[valid_state_index, 2]
        if np.any(incident_normal >= 0.0):
            raise ValueError(
                "detector-visible 00L requires every valid source state to enter "
                "through the negative sample-normal half-space"
            )
        m0_gap = float(np.min(-incident_normal))

    instrument_revision = _instrument_revision(instrument)
    fixed_physics_revision = _source_averaged_sparse_fixed_physics_revision(
        reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
        crystal_to_sample=reference_config.crystal_to_sample,
        rods=selected,
        rod_catalog_revision=rod_catalog_revision,
        mosaic=mosaic,
        incident=incident,
        material=material,
        instrument=instrument,
        intensity_envelope=envelope,
        phase_population_weight=phase_weight,
        polarization_weight=polarization,
    )
    return SourceAveragedDetectorStructureResponse(
        column_px=column,
        row_px=row,
        rods=selected,
        rod_catalog_revision=rod_catalog_revision,
        reciprocal_basis_Ainv=reference_config.reciprocal_basis_Ainv,
        term_coordinate_index=term_coordinate,
        term_rod_index=term_rod,
        term_source_state_index=term_source,
        term_L=term_l,
        term_alpha_rad=term_alpha,
        term_k_norm_Ainv=term_k,
        term_q_radial_squared_Ainv2=term_q_radial_squared,
        term_q_normal_squared_Ainv2=term_q_normal_squared,
        term_fixed_density_per_mosaic_density_per_strength_px2_inv=(term_fixed_per_mosaic),
        term_fixed_density_per_strength_px2_inv=term_fixed,
        term_root_sign=term_root,
        source_state_k_norm_Ainv=source_state_k,
        valid_source_count=valid_count,
        per_rod_caustic=caustic,
        source_state_count=states.incident_state_id.size,
        source_revision=states.source_revision,
        material_revision=material.material_revision,
        sample_geometry_revision=states.sample_geometry_revision,
        instrument_revision=instrument_revision,
        fixed_physics_revision=fixed_physics_revision,
        reference_structure_model_revision=reference_structure_revision,
        detector_visible_m0_q_gap_Ainv=m0_gap,
        reference_intensity_envelope_u_radial_A2=envelope.u_radial_A2,
        reference_intensity_envelope_u_normal_A2=envelope.u_normal_A2,
    )
