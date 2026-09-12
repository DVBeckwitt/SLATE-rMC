"""Immutable detector identities shared by native responses and geometry scans."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from painted_ewald import MosaicParameters
from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.geometry.instrument import CompiledInstrument
from rasim_next.optics import DETECTOR_PATH_ATTENUATION_MODEL_ID, INCIDENT_ILLUMINATED_PATH_MODEL_ID
from rasim_next.pipeline.continuous_detector import SampleQIntensityEnvelope
from rasim_next.reflectivity import ParrattStitchStack


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
                ("specular_overlap_measure", specular_stitch_stack.overlap_measure),
                (
                    "specular_interface_assumption",
                    specular_stitch_stack.interface_assumption,
                ),
            )
        )
    return canonical_revision_sha256(*fields)
