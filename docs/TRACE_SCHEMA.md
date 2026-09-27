# Stage-trace schema

Native refinement result v2 additionally records `prediction_store`, completed raw
vectors/N/predictions, pending work, public optimizer termination metadata and execution
status. Revision identity binds plan, inputs, source files, runner and numerical dependency
versions. Parent compilation metrics explicitly exclude isolated worker-group builds.
Native render v1 records the candidate/target kind, implementation identity, whole-panel
batch count, completion and separate fit/image numerical status. Only consistent batch
boundaries are atomically published; interruption retains the last published checkpoint.
These are execution manifests, not new physical trace stages.

Transport records and archived proof traces use stable stage IDs. A result may omit non-applicable stages, but it may not
invent branch-specific names for shared quantities. The frozen trace schema remains v4; contract
API v17 does not renumber historical evidence.

The registry preserves source/incident, reciprocal-root, ordered/stacking, optical,
detector-coordinate, and total-detector-mass identifiers for reference comparison. Current
`TraceRecord` producers cover transport; proof/reference runners are retired; a listed identifier is not a
claim that every continuous result emits a trace row. Candidate, selection, sampled-event,
deposition-index, deposition-weight, and clipped-point identifiers below are frozen historical
names only; the live runtime never emits them. Continuous results use typed result measure
IDs rather than replacement trace aliases. The stochastic result introduced in API v11 and
execution-extended in API v12 records its measure, proposal, RNG, source, rod, backend/device, and
work ledgers in typed results; it does not revive event or deposition trace rows. Progressive
execution and the leased presentation frame are execution state, not new trace stages.

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

The additive `rasim-osc-geometry-fit-result-v6` typed proof envelope carries its series manifest,
image IDs and commanded angles, parameterization, canonical fitted/fixed coordinate names, shared
corrections, the optional one common incidence-angle delta, canonical zero-sum trim contrasts,
per-image trims and effective angles, combined
Jacobian parameter names, scaled singular spectrum and weakest direction, active bounds,
per-image native errors, the direct-root and frozen-candidate acceptance audits, the
frozen-reindexing lineage, optional data-specific qualification profile, and the separate
global-rediscovery diagnostic. The retired historical qualification records
`status=historical_qualification_retired`, `accepted=false`; its old benchmark fields
are null with `status=retired`. Actual fitting elapsed times remain measured. Expected input or numerical rejection in JSON mode uses
the additive `rasim-osc-geometry-fit-rejection-v1` envelope rather than a traceback. These
map to the existing `selection.manifest_hash`, `fitting.detector_parameters`,
`fitting.sample_geometry_parameters`, `fitting.selection_revision`, `fitting.objective_value`,
`fitting.held_out_metric`, and `fitting.invalidation_summary` stages; they do not extend or renumber
frozen trace schema v4.

The historical `rasim-staged-fit-replay-certificate-v2` envelope is an orchestration proof, not a new
physics trace stage. It carries the exact case SHA-256, material, requested backend and terminal
stage, a `verification_runtime` with the case-bound lock hash and actual
interpreter/platform/package identity, the ordered
`geometry -> mosaic -> ordered_intensity -> render` scientific revisions, and the verified compact
summaries. Each `rasim-staged-fit-replay-stage-v2` member records the execution runtime and binds its
stage-scoped case hash, which covers the declared consumed-file records, together with its immediate
upstream revision, case/source identity, and compact scientific state. Runtime identity is
operational provenance and does not enter a stage scientific revision, but a resumed stage must
match it exactly so the verifier cannot claim a reused result was executed by its own runtime.
Geometry records the canonical one-state source batch plus commanded/effective incidence vectors,
one common delta, fixed gauge, and the full position proof evidence; later stages record the common
250-state batch and the consumed upstream revision. The Bi2Te3 mosaic member additionally binds the
geometry-stage fixed position, implicit CIF lattice, and case simulation-config identity; its
ordered member must preserve those records exactly. Fresh and resumed envelopes use the same strict
validation, and a stage's compact summary is
verified before that stage is persisted or supplied downstream. Volatile artifact paths and
path-dependent JSON container hashes, timings, device labels,
and memory measurements do not enter a scientific revision. Decoded render identities do. Missing
stages cannot be hidden: verification always covers every stage through the requested terminal
boundary.

The retired outer layered-fit runner used `rasim-layered-fit-workflow-progress-v2` as an exclusive attempt
marker and `rasim-layered-fit-workflow-stage-v3` as the terminal stage manifest. Progress v2 binds
the stage, `status=started`, plan revision, live repository/runtime scientific revision, upstream
revision, and completed-command count. Any surviving progress marker, including count zero,
rejects reuse and requires a new output directory; partial commands are never resumed. Stage v3
binds material/model identity, the same plan/repository/upstream revisions, ordered completion
artifact paths and hashes, and the derived stage revision. Reuse rehashes inputs and artifacts and
recomputes that revision. These orchestration records are not physics trace stages and do not
renumber trace schema v4.

The following identifiers remain useful when reading archived matched-region evidence and do
not change frozen trace-stage numbering. Lattice sensitivity and fixed-state contracts remain
live. The layered-quintuple workflow, background/fit/profile and figure envelopes below are
historical; they do not imply a current runner or a reusable native-search result.

- `rasim-osc-lattice-sensitivity-v2` binds one verified completed position artifact, preserving
  whether its status is qualified or model-limited, and compares tightly
  regularized hexagonal in-plane and normal log strains with the reference CIF basis. It serializes
  the constrained result as a full direct basis and records separate data-only and
  penalized sensitivities and declares either `RETAIN_CIF_LATTICE` or
  `ACCEPT_FITTED_LATTICE`; only a data-only full-rank, well-conditioned, in-bounds result with the
  required improvement may promote the fitted basis.
- `rasim-fixed-mosaic-state-v1` records a provided Gaussian/Lorentzian mixture and nonempty
  provenance. `rasim-fixed-experiment-state-v2` composes that strict mosaic state with the complete
  v6 position state and `rasim-fixed-lattice-state-v1`; it binds ordered image IDs, commanded/shared/
  trim/effective angles, source-state count, detector shape, and every input hash.
- `rasim-layered-quintuple-matched-regions-v4` freezes preparation/display row discovery state,
  mixed phi/2theta and Qr/L row identities, selected native-pixel ordering, exact fixed-experiment
  state, chart rectangles, recipe hash, and every input hash. Its center-selected count masses and
  support are not the fit observable; fit-v14 reprojects the verified raw OSC continuously.
- `rasim-shared-radial-background-v4` freezes one independently calibrated rise-decay radial halo
  with shared shape, per-OSC amplitude and pedestal, covariance, held-out cells, and exact input
  identities. It binds the fit plan and beam center and excludes every native pixel touched by the
  oracle continuous projection. Its empirical-native-pixel method flags, adapter, and complete
  numerical implementation are hash-bound and reverified on load.
- `rasim-layered-quintuple-matched-fit-progress-v11` atomically records the last complete stage-specific model
  evaluation and binds diagnostic, recipe, fit plan, full numerical implementation, backend,
  cubature, block, rod-scope, optimizer-start, and stopping-budget identities.
  `rasim-layered-quintuple-matched-region-fit-v14` records the declared execution policy: either
  the v7 five-coordinate A -> B -> C -> joint chain with exact active/frozen names, child starts,
  and direct/recursive predecessor hashes, or v8 `seeded_joint_only.v1` with an explicit numeric
  start/hash (or verified same-stage restart) and no fabricated predecessors. Fit-v14 does not
  claim a seed-file identity; the outer workflow separately hashes seed-v2 into its plan. It also records one finite
  positive scale per recipe-ordered OSC, the exact fitted rod roster, frozen radial-background
  identity, residuals, convergence, and separate parameter-scaled
  data-only and penalized sensitivity. Practical data-only rank and condition determine
  admissibility; priors cannot create rank. Its objective uses full projected count covariance and
  records its hash and projection revisions. `COMPLETE` projection/cubature status, tolerance, and
  fit/oracle orders are bound to the hash-verified recipe. Joint model cubature gates the actual
  anchor-conditioned `(I-A)m` observable by family plus raw anchor rows. Every stage must share the
  same background, data projection, and rod roster. Only the joint artifact is eligible downstream,
  and its qualification state is preserved.
- `rasim-layered-quintuple-matched-profile-progress-v7` reports each completed continuous-profile pass for
  monitoring but is not a resume source. `rasim-layered-quintuple-matched-figure-profiles-v14` is the atomic
  completed diagnostic with
  full detector-visible profiles in both L and physical Qz, continuous-support edge exclusions,
  measured/model profiles, the exact fit-bound rod roster, content hashes for every render-critical
  array, and the full verified upstream/predecessor hash chain. Its
  only current evidence level is `FIT_CONDITIONED`; it sets `publication_ready=false` and makes no
  all-configured-rod or full-profile-oracle claim.
- Typed source and configured-simulation provenance carry the spectral line wavelengths and
  probability masses, position/divergence correlations, illuminated-path model ID, and external
  detector-path attenuation model. The matched-fit chain additionally binds the declared dark
  scale/basis/covariance and the vacancy occupancy rule: outer-chalcogen occupancy `1-v`, outer-site
  Bi substitution zero. A retained 3R record declares its exact epsilon value; epsilon zero is a
  gate-retained nested model, not an omitted field or an aggregate-grid optimum claim. These
  additions use existing typed metadata and do
  not change trace schema v4 stage numbering.
- `rasim-layered-quintuple-peak-alignment-v3` reports m=0 offsets in two-theta and nonzero-family offsets in L.
  `rasim-layered-quintuple-matched-figure-output-v1` is promoted last and binds the PNG, PDF, alignment JSON,
  renderer, and profile diagnostic hashes; its presence is the transaction commit marker.

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

## Historical comparator (retired)

The former executable comparator is archived at Git revision `358362e`; no comparator
runner is retained. Its historical record format used the following sequence:

1. verifies schema and metadata
2. aligns records by case and stage
3. applies stage-specific tolerances
4. reports the first failing stage
5. accepts downstream disagreement only for a declared `CORRECTED` case with an independent proof record

The historical comparator rejected missing stages rather than comparing final outputs alone.
`mosaic.wrapped_line_density` has unit `rad^-1`, no coordinate frame, and `PROBABILITY_DENSITY` measure.

Optional geometry calibration names/values, configured detector reference coordinates, and the
fixed-position calibration-provenance flag belong to the existing position envelope. Sparse
response, strength-parameterization, region-model, and fitted-structure SHA-256 revisions map to
the existing fitting invalidation/provenance fields; they do not create new trace stage IDs.

Acquisition-bound quadrature and averaged-detector records carry acquisition, calibration,
quadrature, scan, and component revisions. Adaptive records carry source/evaluator revisions,
covariance, convergence gate, and refinement tree; staged scores carry immutable SHA-256 comparison
revisions and block shapes. These are typed records or external proof artifacts, not new trace
stages, so trace-v4 numbering remains unchanged.

Native-refinement v16 diagnostics retain candidate values/N, actual source/material/
thickness-bound instrument revisions, raw-acquisition and observation hashes, code
hashes, numerical prediction/contrast arrays, optimizer/guard statuses, profile
policy and conditional-validation groups. These remain external numeric NPZ records
with one embedded manifest; they add no trace stage or trace-version increment.

Contract-v17 native result schema `rasim-native-refinement-result-v2` additionally
retains the stitch overlap measure, frozen Q domains, quadrature kind, physical panel cap
and angular-support policy through the
effective plan and dependency identities. `optimizer_candidate` is separate from
`selected`. Initial, fitted, discrete-ranking, profile and conditional-validation
numerical statuses accompany the actual candidate arrays and target revisions.
Profile records map numerical candidates to sorted grid indices and preserve better
admissible points requiring a joint refit. Failed initial qualification checkpoints
`workflow_complete=false`; execution-only results cannot inherit scientific qualification.
Source hashes attest the startup filesystem snapshot, not loaded bytecode.

Optional `rasim-native-axial-mesh-v1` preparation diagnostics retain the complete
plan, acquisition/physics/observation/source identities, fixed scale, one-N stencil,
prepared physical meshes, parent/refined predictions, comparison results and build
costs. Acceptance is only `empirical_mesh_agreement_only`. Normal native result v2
may retain `reference_acceleration` records: fixed high/low rules, reference/candidate
vectors, exact objective checks, rejection/fallback evidence and low-rule work counts.
These are proposal records (`exact_checked_warm_start_only`), never exact prediction
cache entries or selected estimates. Numeric comparison vectors may appear as JSON
lists in these compact manifests. No new physical trace stage/version is introduced.
