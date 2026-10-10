"""Explicit acquisition-owned source and instrument corrections for native fits."""

from dataclasses import dataclass, replace

import numpy as np

from rasim_next.core.contracts import canonical_revision_sha256
from rasim_next.fitting.native_input import NativeFitPhysics
from rasim_next.geometry.instrument import compose_intrinsic_xy_rotation
from rasim_next.materials.optics import material_optics

NATIVE_INSTRUMENT_PARAMETER_NAMES = (
    "detector_center_column_delta_px",
    "detector_center_row_delta_px",
    "detector_normal_distance_delta_m",
    "detector_local_x_tilt_rad",
    "detector_local_y_tilt_rad",
    "sample_local_x_tilt_rad",
    "sample_local_y_tilt_rad",
    "sample_normal_offset_m",
    "source_spatial_sigma_0_m",
    "source_spatial_sigma_1_m",
    "source_divergence_sigma_0_rad",
    "source_divergence_sigma_1_rad",
    "source_position_divergence_correlation_0",
    "source_position_divergence_correlation_1",
    "source_line_0_probability",
    "source_common_wavelength_sigma_A",
    "source_line_0_wavelength_A",
    "source_line_1_wavelength_A",
)


@dataclass(frozen=True, slots=True)
class NativeInstrumentModel:
    """Eighteen corrections at a fixed declared source plane and fixed panel chart.

    Rotations are active intrinsic local x then current local y. Distance and
    sample offset use their respective nominal normals. A separate incidence
    offset would duplicate the sample-tilt gauge and is deliberately absent.
    The effective Gaussian source is not separated into intrinsic beam and PSF.
    """

    reference: NativeFitPhysics
    acquisition_id: str
    parameter_units = (
        "pixel",
        "pixel",
        "metre",
        "radian",
        "radian",
        "radian",
        "radian",
        "metre",
        "metre",
        "metre",
        "radian",
        "radian",
        "1",
        "1",
        "1",
        "angstrom",
        "angstrom",
        "angstrom",
    )

    def __post_init__(self):
        definition = self.reference.source_definition
        if not self.acquisition_id or definition is None or len(definition.line_wavelength_A) != 2:
            raise ValueError(
                "instrument fitting requires acquisition ownership and two explicit source lines"
            )

    @property
    def initial_values(self):
        source = self.reference.source_definition
        return np.r_[
            np.zeros(8),
            source.spatial_sigma_m,
            source.divergence_sigma_rad,
            source.position_divergence_correlation,
            source.line_probability[0],
            source.common_wavelength_sigma_A,
            source.line_wavelength_A,
        ]

    def bind(
        self, physics: NativeFitPhysics, values, *, optical_factor_cache=None
    ) -> NativeFitPhysics:
        values = np.asarray(values)
        if values.shape != (18,) or np.iscomplexobj(values) or np.any(~np.isfinite(values)):
            raise ValueError("instrument correction requires eighteen finite real coordinates")
        base = self.reference.instrument
        panel, sample = base.lab_from_detector, base.lab_from_sample
        instrument = replace(
            base,
            detector_reference_coordinate_px=tuple(
                np.asarray(base.detector_reference_coordinate_px) + values[:2]
            ),
            lab_from_detector=replace(
                panel,
                rotation=compose_intrinsic_xy_rotation(panel.rotation, *values[3:5]),
                translation_m=panel.translation_m + values[2] * panel.rotation[:, 2],
            ),
            lab_from_sample=replace(
                sample,
                rotation=compose_intrinsic_xy_rotation(sample.rotation, *values[5:7]),
                translation_m=sample.translation_m + values[7] * sample.rotation[:, 2],
            ),
        )
        definition = replace(
            physics.source_definition,
            spatial_sigma_m=values[8:10],
            divergence_sigma_rad=values[10:12],
            position_divergence_correlation=values[12:14],
            line_probability=(values[14], 1 - values[14]),
            common_wavelength_sigma_A=values[15],
            line_wavelength_A=values[16:18],
        )
        source = definition.sample()
        material = material_optics(
            physics.structure.crystals[0],
            source.mean_rays.wavelength_A,
            factor_cache=optical_factor_cache,
        )
        revision = canonical_revision_sha256(
            ("definition_id", "native_instrument_candidate.v1"),
            ("structure_input", physics.input_revision),
            ("acquisition_id", self.acquisition_id),
            ("values", values),
            ("source", source.revision),
        )
        return replace(
            physics,
            instrument=instrument,
            source=source,
            material=material,
            source_definition=definition,
            input_revision=revision,
        )
