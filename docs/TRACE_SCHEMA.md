# Stage-trace schema

Every proof trace uses stable stage IDs. A result may omit non-applicable stages, but it may not
invent branch-specific names for shared quantities. The frozen trace schema remains v4; contract
API v11 does not renumber historical evidence.

The registry preserves source/incident, reciprocal-root, ordered/stacking, optical,
detector-coordinate, and total-detector-mass identifiers for reference comparison. Current
`TraceRecord` producers cover transport, proof, and reference paths; a listed identifier is not a
claim that every continuous result emits a trace row. Candidate, selection, sampled-event,
deposition-index, deposition-weight, and clipped-point identifiers below are frozen historical
names only; the live runtime never emits them. New continuous proofs prefer typed result measure
IDs over inventing replacement trace aliases. The API-v11 stochastic result likewise records its
measure, proposal, RNG, source, rod, and work ledgers in the typed result; it does not revive event
or deposition trace rows.

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

mosaic.wrapped_line_density

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
geometry.detector_pixel_solid_angle

sampling.source_empirical_mass
measurement.reciprocal_weight
measurement.scattering_strength
measurement.population_weight
measurement.optical_weight
measurement.footprint_weight
measurement.polarization_weight
measurement.deposition_indices
measurement.deposition_weights
measurement.total_detector_mass

render.candidate_mass
render.selection_probability
render.selected_event_mass
render.raw_detector_image
render.clipped_mass

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

The additive `rasim-osc-geometry-fit-result-v4` typed proof envelope carries its series manifest,
image IDs and commanded angles, parameterization, canonical fitted/fixed coordinate names, shared
corrections, selected-coordinate scaled singular spectrum and weakest direction, active bounds,
per-image native errors, the direct-root and frozen-candidate acceptance audits, the
frozen-reindexing lineage, optional data-specific qualification profile, and the separate
global-rediscovery diagnostic. Expected input or numerical rejection in JSON mode uses
the additive `rasim-osc-geometry-fit-rejection-v1` envelope rather than a traceback. These
map to the existing `selection.manifest_hash`, `fitting.detector_parameters`,
`fitting.sample_geometry_parameters`, `fitting.selection_revision`, `fitting.objective_value`,
`fitting.held_out_metric`, and `fitting.invalidation_summary` stages; they do not extend or renumber
frozen trace schema v4.

The additive `rasim-staged-fit-replay-certificate-v1` envelope is an orchestration proof, not a new
physics trace stage. It carries the exact case SHA-256, material, requested backend and terminal
stage, a `verification_runtime` with the case-bound lock hash and actual
interpreter/platform/package identity, the ordered
`geometry -> mosaic -> ordered_intensity -> render` scientific revisions, and the verified compact
summaries. Each `rasim-staged-fit-replay-stage-v1` member records the execution runtime and binds its
immediate upstream revision, case/source identity, and compact scientific state. Runtime identity is
operational provenance and does not enter a stage scientific revision, but a resumed stage must
match it exactly so the verifier cannot claim a reused result was executed by its own runtime.
Geometry records the canonical one-state source batch; later stages record the common 250-state
batch. Fresh and resumed envelopes use the same strict validation, and a stage's compact summary is
verified before that stage is persisted or supplied downstream. Volatile artifact paths and
path-dependent JSON container hashes, timings, device labels,
and memory measurements do not enter a scientific revision. Decoded render identities do. Missing
stages cannot be hidden: verification always covers every stage through the requested terminal
boundary.

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

Incident trace provenance is diagnostic-only and allocated only when tracing is requested. Its
canonical JSON envelope retains the T02 scientific provenance and adds sampling-model ID, RNG-model
ID, seed, source-parameter revision, complete source revision, sample-geometry revision, material
revision, and incident-model ID. Geometry-failure stages emit zero geometry/optical values; optical
failures preserve accepted intersection, direction, air-side `ki`, and footprint evidence while
zeroing film-side values. `optics.ki_parallel_sample` follows the same canonical row order without a
trace-only full-batch scientific scratch array.
Every source, sample, and material revision in this envelope is copied unchanged from its owning
immutable object. Trace construction never rejoins raw source rows or recomputes a revision. The
v2 material/sample digest rebaseline changes provenance identifiers only; trace stage IDs and
accepted numeric values remain unchanged.

## Comparator

The common comparator:

1. verifies schema and metadata
2. aligns records by case and stage
3. applies stage-specific tolerances
4. reports the first failing stage
5. accepts downstream disagreement only for a declared `CORRECTED` case with an independent proof record

The comparator must not hide missing stages by comparing only final outputs.
`mosaic.wrapped_line_density` has unit `rad^-1`, no coordinate frame, and `PROBABILITY_DENSITY` measure.
