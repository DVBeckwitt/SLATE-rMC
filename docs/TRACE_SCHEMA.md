# Stage-trace schema

Every proof trace uses stable stage IDs. A result may omit non-applicable stages, but it may not invent branch-specific names for shared quantities.

## Stage IDs

```text
osc.raw_header
osc.raw_array
osc.detector_native_array
osc.beam_center_raw
osc.beam_center_native

geometry.instrument_transforms
geometry.lab_ray
geometry.sample_intersection
geometry.footprint_acceptance
geometry.detector_frame

optics.ki_air_sample
optics.ki_parallel_sample
optics.kz_incident_film
optics.entrance_amplitude
optics.kf_film_sample
optics.kz_exit_air
optics.exit_amplitude
optics.kappa_incident
optics.kappa_exit
optics.uniform_depth_attenuation

reciprocal.rod_id
reciprocal.family_id
reciprocal.intersection_support
reciprocal.quadrature_coordinate
reciprocal.event_q_internal
reciprocal.ewald_residual
reciprocal.event_weight

ordered.atomic_amplitude
ordered.unit_cell_amplitude
ordered.layer_amplitude
ordered.finite_stack_amplitude
ordered.event_intensity

reflectivity.layer_kz
reflectivity.interface_amplitude
reflectivity.recursion_amplitude
reflectivity.parratt_intensity
reflectivity.kinematic_intensity
reflectivity.composite_intensity

stacking.registry_phase
stacking.transition_matrix_6
stacking.transition_matrix_reduced
stacking.pair_kernel
stacking.finite_intensity
stacking.population_intensity

geometry.kf_air_lab
geometry.detector_intersection
geometry.detector_column_px
geometry.detector_row_px

measurement.source_weight
measurement.reciprocal_weight
measurement.model_intensity
measurement.population_weight
measurement.optical_weight
measurement.footprint_weight
measurement.polarization_weight
measurement.pixel_solid_angle
measurement.deposition_indices
measurement.deposition_weights
measurement.total_detector_mass

selection.radial_family_key
selection.reflection_group_key
selection.branch_azimuth
selection.branch_id
selection.candidate_residual
selection.association_status
selection.manifest_hash

fitting.source_parameters
fitting.detector_parameters
fitting.sample_geometry_parameters
fitting.selection_revision
fitting.mosaic_parameters
fitting.ordered_parameters
fitting.stacking_parameters
fitting.objective_value
fitting.held_out_metric
fitting.invalidation_summary
```

## Required metadata

Each record includes:

```text
case_id
stage_id
value or array key
shape
dtype
unit
coordinate frame
amplitude, intensity, density, or mass measure
model version
source provenance
```

## Comparator

The common comparator:

1. verifies schema and metadata
2. aligns records by case and stage
3. applies stage-specific tolerances
4. reports the first failing stage
5. accepts downstream disagreement only for a declared `CORRECTED` case with an independent proof record

The comparator must not hide missing stages by comparing only final outputs.
