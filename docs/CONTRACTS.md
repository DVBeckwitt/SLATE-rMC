# Contracts

> Test/proof references record historical evidence. Executable harnesses are retired;
> current assessment policy is [VALIDATION.md](VALIDATION.md). Runtime physical and
> numerical contracts below remain binding.

## Native preparation and execution (v18)

The sole regular conditional engine is positive strength-weighted Gaussian axial
quadrature with native-pixel-error angular refinement. See
[NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md) and [ENGINE_MIGRATION.md](ENGINE_MIGRATION.md).
Every distinct candidate rebuilds physical panels, canonical signed strengths,
W-dependent nodes/masses and actual-mosaic angular acceptance. Exact completed
predictions may be reused within their immutable candidate/observable owner.
`FiberScatteringCache` retains bounded pre-projection state keyed by actual nodes
and upstream optics. Source weights, origins, visibility and footprints apply
afresh; no survivor mask determines support. Frozen domains require candidate
coverage checks and do not freeze strength-dependent rules. The required local-m0
endpoint chart remains explicit.

`NativeRefinementModel` supplies names, units, a physical binding and exact inactivity rules.
Built-in Bi/Pb symmetry rules live in those models. Search, observations, instrument binding,
transport, probability integration and execution checkpoints are shared. Raw prediction
recovery is keyed by full float64 candidate, integer N and declared forward identity;
optimizer state, residuals and Jacobians are not replayed.

Contract API version: **18**. Trace schema version: **4**. Reference pack version: **1**.

T44 adds native rectangular-window storage to `DetectorSpatialKernels.integrate_native_pixels`.
`row_offset` and `column_offset` locate the window without changing pixel coordinates or Gaussian
probability arithmetic. Optional `out` explicitly adds mass to caller-owned writable contiguous
float64 storage and returns that same array. Invalid inputs fail before deposition; a numerical
failure during deposition can leave partial contents, which the caller must discard.
`native_pixel_bounds` encloses the existing column-conditioned traversal and returns half-open
row/column bounds, or `None` when no pixel is visited. It uses no intensity threshold and retains
off-panel centers. These are storage controls, not angular error bounds or new physical measures.
The research and full-image qualification gates are in `tasks/44_ewald_integration_research.md`.

Near-unit-correlation Gaussian corners use complementary residuals from the signed
unit-correlation limit, following Genz (2004). Cache representation is fixed per
Gaussian; rectangles outside the standardized-endpoint guard use direct conditional
integration without accessing that cache. Native pixels retain two rolling corner
edges and marginal tails. Native region rectangles use the same arithmetic and
preserve their frozen membership weights. Requested order, tail clipping and
cancellation fallback remain explicit; the spatial default stays 16.

Fixed-parameter controls retain the complete physical vector and fixed-value
provenance; ordinary fits release every admitted coordinate. Local-m0 has its own
named controls and keeps the physical endpoint transform. Neither controls nor
deterministic preparation qualify a fit.

An optional Gauss-Hermite `NativeSourceDefinition.local_m0_divergence_order` uses
a separate normalized numerical source rule for stitched `(0,0)`. Bind physical
source parameters first, then resolve disjoint `NativeFitPhysics.integration_parts()`.
Regular and local intensities sum before one scale; each partition retains its own
source/response identity and cache. Equal orders preserve the unsplit path.

The explicitly selected `FiberIntegrationRule.local_m0_angular_rule=
"cdf_stratified_importance.v1"` is a nominal local-lamella endpoint estimator.
It uses one scrambled two-dimensional Sobol net of 4096 axial nodes and 32
conditional angular strata per node; `local_m0_seed` is the direct scramble seed
and `local_m0_replica` is explicit response identity. Original source rows share
that net within a replica. Disjoint physical arcs retain their original proposal
mass Z and weight `Z/(4096*32*pq*pphi)`, including the original attempted divisor.
Nonempty unresolved CDF arcs fail; empty support contributes zero without
resampling. It reuses canonical signed strengths, transfer and native Gaussian
probabilities. It does not qualify accuracy, shared regular error or a fit.
The default remains `resolved_cdf_gl8.v1`; regular quadrature is unchanged.

Production contracts are frozen dataclasses or immutable model objects. Numeric arrays are copied to
contiguous, read-only storage at public boundaries. Shapes, units, frames, measure IDs, validity,
and ordering are validated eagerly.
The sole presentation exception is the explicitly leased `MonteCarloDetectorPresentation` buffer,
whose lifetime ends at the sampler's next operation and which is never retained scientific state.

## Global conventions

- Column vectors and active rotations.
- Radians internally.
- Metres for instrument positions; angstroms for wavelength and crystal lengths; inverse angstroms
  for wavevectors.
- Array access is `[row, column]`; continuous detector coordinates are `(column_px, row_px)`.
- `Q = kf - ki` in the declared frame.
- Invalid rows have explicit status and cannot carry fabricated nonzero intensity.
- IDs prove alignment and provenance; they are not numerical weights or sorting keys.

## Stable core data contracts

| Contract | Owner | Essential payload |
|---|---|---|
| `SourceConfiguration` / `sample_gaussian_source_rays` / `sample_discrete_gaussian_line_source_rays` | configured pipeline / sampling | validated Gaussian phase-space parameters, optional per-axis position--divergence correlation, legacy Gaussian or weighted discrete-line spectrum, and immutable sampled source rows |
| `IncidentSampleBatch` | sampling | complete source rows, arbitrary finite nonnegative probability masses summing to one, wavelength, polarization, provenance |
| `InstrumentConfiguration` / `CompiledInstrument` | geometry | canonical transforms, sample support, detector shape/pitch/reference coordinate, detector-path medium plus mutually exclusive scalar or exact-wavelength attenuation table, revisions |
| `IncidentStateBatch` | geometry | entrance-intersected air and film-phase `ki`, Fresnel amplitude, decay, footprint, source identity |
| `MaterialOptics` | materials | wavelength-aligned complex refractive index and material revision |
| `RodCatalog` | reciprocal | every physical `(h,k)` rod and exact family metadata |
| `SphericalMosaicDensity` | painted_ewald | explicitly named directed spherical-area law, component mass normalizers, latent and antipodal densities; shared CPU raw-count refit measure described in [RESULT_MEASURE](RESULT_MEASURE.md) |
| `RodQueryBatch` | ordered/stacking | rod-aligned `L` queries with stable IDs |
| `EventIntensityResult` | ordered/stacking | query-aligned amplitude, intensity, normalization, and model revision |
| `ParrattResult` / `SpecularResult` / `KinematicScaleSpecularResult` | reflectivity | separately named pure and unit-preserving composite specular outputs |
| `ParrattStitchStack` / `CompiledParrattStitch` | reflectivity / detector pipeline | immutable substrate/interface inputs and source-wavelength overlap state; explicit local-lamella inverse-map or fixed-external-Qz regular-map `(0,0)` convention |
| `MeasuredPeakDiscovery` / `MeasuredIndexingResult` | selection | hashed image/mask/calibration provenance, native coordinates, reciprocal labels, decisions, and replicated branch tracks |
| `ConfiguredGeometryInputs` / `GeometryOnlyEwaldContext` | configured pipeline | one nominal ray, material optics, reciprocal basis, rods, and instrument; no strength or mosaic object |
| `EwaldDirectionIntensity` | continuous detector pipeline | sample-frame internal-film outgoing directions and `Q`; exact a.e. total/per-rod `A2/sr` density, inverse counts, caustics, rods, branch selection, and measure identity |
| `OscGeometrySeriesConfiguration` / `OscGeometryIndexingRun` | selection | strict IDs/paths/commanded angles plus one provenance-bound frozen selection and retained fit-ready models |
| `CommensurateLayerOrder` / `LayerLMarkerDefinition` / `LayerLMarkerObservations` | geometry/fitting | exact reduced layer coordinate; rod-free physical marker identity plus explicit signed-rod provenance; one frozen coordinate/covariance row per physical detector locus |
| `Pbi2PolytypeLandmarkCatalogue` | fitting | ideal-parent 2H/4H/6H support in one declared single-trilayer PbI2 metric, exact overlap deduplication, signed-rod/parent provenance, source-CIF and full geometry-context revisions; no intensity or measured-centroid claim |
| `Pbi2ParentMixtureStrength` / `Pbi2ParentLogRatioParameterization` | structure binding | fixed near-parent fractions, signed rods, finite-layer normalization and provider revision |
| `IndexedGeometryImage` / `IndexedGeometryFitResult` | fitting | one frozen integer- or rational-layer image block and one selected subset of the shared geometry pack, optionally augmented by one common incidence-angle delta and zero-sum Helmert trims, with commanded/trim/effective-angle provenance, fixed-coordinate provenance, per-image metrics, and combined rank diagnostics |
| `MosaicProfileDefinition` / `MosaicProfileSet` | measurement boundary | frozen reflection identity, finite-bin S, N, validity, angle layout and source/observation provenance |
| `LayeredReciprocalFrame` / `ReciprocalProfileRegion` | measurement | one explicit reciprocal basis, active sample-from-crystal rotation, declared axial basis vector, radial band, axial bin edges, detector-side interval, and sidebands; no material-specific family equation or detector raster |
| `ReciprocalProfileMembership` / `BinnedSampleIntegral` | measurement | immutable sample-to-bin identities and separately accumulated signal/measure vectors; division occurs only after finite-bin integration |

`axis_rotation_transform(rotation)` is the authoritative conversion of one `AxisRotation` into an
active LAB-to-LAB rigid transform about its declared LAB pivot. `compile_instrument` uses the same
function for every commanded goniometer motion in tuple order.

`EventIntensityResult` retains “event” in its historical type name, but it is an ordered query result;
it is not a sampled scattering-event runtime. Its strength excludes source probability, rod
population, mosaic probability, optics, polarization, detector Jacobians, and pixel integration.

## Continuous reciprocal contracts

### `MosaicParameters`

Owns Gaussian sigma, Lorentzian HWHM, mixture probability, and deterministic proof quadrature. A
zero width is permitted only for an inactive mixture component. The accepted continuous law is
folded alpha/full beta and integrates to one.

### `Rod`

Owns exact `(h,k)`, family metadata, and population. Rod identity is never a floating radial value.

### Reciprocal profile regions

`LayeredReciprocalFrame.coordinates` maps arbitrary continuous sample-frame wavevectors into radial
distance from a declared reciprocal axis and its fractional axial coordinate. A
`ReciprocalProfileRegion` then owns the finite `L` edges, radial signal width, explicit detector
column interval, and radial sidebands. `reciprocal_profile_membership` applies that same immutable
region to detector-pixel centers or continuous cubature nodes. `accumulate_binned_samples` always
returns signal and measure separately; its `mean` is the only division. These functions do not own
background policy, material family labels, intensity scaling, rendering, or detector rasterization.

### `MosaicBraggSpace`

- `map_latent(rod, alpha_rad, beta_rad, u_Ainv)` maps arbitrary broadcast coordinates to sample-frame
  `Q`.
- `evaluate_latent(...)` returns per-rod mosaic density, finite-stack strength, and their product.
- `rod_u_bounds_Ainv(rod)` returns the full elastic-reach axial interval.

The map and density are functions; a quadrature node set is never the model.

`map_tied_rotation_latent(...)` is the public geometry-only authority for the same tied rotation,
without constructing `MosaicBraggSpace`. `evaluate_infinite_rod_ewald_geometry(...)` is the matching
geometry-only continuous Ewald-section authority.

### `ContinuousEwaldCoating`

- `evaluate_geometry(rod, branch, alpha_rad, beta_rad)` returns only the analytic Ewald geometry for
  a non-specular rod/root and evaluates no mosaic or structure-factor intensity.
- `evaluate_latent(rod, branch, alpha_rad, beta_rad)` solves the analytic elastic root and returns
  geometry, strength, mosaic density, and the once-only Ewald coarea-weighted coating.
- `evaluate_specular_geometry(...)` exposes retained nonzero branch-0 `m=0` geometry without
  inventing finite direct-beam intensity. The algebraic `Q=0` direct root is suppressed. The
  separate detector-visible all-roots path may include regular nonzero `m=0` support only with a
  positive reciprocal-gap certificate.

## Detector contracts

### `DetectorEwaldMeasure`

Immutable one-incident-state model. It requires the canonical transported film-phase `ki`, matching
air wavelength, strict root classification, material optics, and compiled instrument.

- `evaluate_intrinsic_ewald_directions(outgoing_direction_sample, rods, branch=None)` evaluates
  arbitrary broadcast sample-frame unit internal-film `kf` directions. It excludes `m=0`, sums both
  analytic roots when `branch=None`, and returns the exact almost-everywhere
  `intrinsic_ewald_direction_density_A2_per_sr.v1` with per-rod inverse counts and caustic flags.
  Detector visibility, exit optics, source/phase/polarization weights, and detector solid angle are
  absent. All returned arrays are copied to read-only storage.
- `evaluate_detector_visible_ewald_directions(column_px, row_px, rods)` uses the configured active
  detector coordinates only to select internal-film outgoing directions, then returns
  `detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1`. It sums every regular inverse
  preimage, including nonzero `m=0`, with the same intrinsic `k_film^2 / |J_latent|` density. It
  applies no source, optical, attenuation, detector Jacobian, or solid-angle factor and carries the
  detector-visible `m=0` Q-gap certificate. The ordinary mean-plane chart has a strictly positive
  gap; a compiled local-lamella Parratt stitch declares zero because its accepted detector chart
  reaches the specular origin.
- `evaluate_detector_geometry(column_px, row_px, *, include_surface_jacobian=True)` performs detector
  point → outgoing ray → exit refraction → film `kf` → sample-frame `Q` and reports
  validity and elastic residual. The ray is valid only on the active front face,
  `n_D dot kf_hat > 1e-14`; back-side and tangent approaches are rejected before optical or
  reciprocal work. Passing `False` skips derivative work and returns a zero Jacobian when only ray
  geometry is needed. `evaluate_detector_visible_geometry(...)` additionally performs the canonical
  exit-refraction round-trip validation without attaching optical intensity.
- `evaluate_detector_coordinates(column_px, row_px, rods, branch)` returns the almost-everywhere
  `raw_detector_coordinate_density_A2_per_px2.v1`, per-rod contributions, inverse-branch counts, and
  caustic flags.
- `map_latent(...)` is the independent forward route used for proof and diagnostics.
- `map_latent_geometry(...)` is the same canonical forward geometry, exit refraction, and detector
  intersection without mosaic, structure strength, attenuation, or intensity work.
- `integrate_native_pixels(...)` returns deterministic `raw_detector_pixel_mass_A2.v1` with
  convergence, work-count, validity, and per-rod evidence.

### `SourceAveragedDetectorEwaldMeasure`

Owns an ordered tuple of complete incident-state measures. Its arbitrary-coordinate evaluator sums
independent source intensities over all retained roots. Detector-visible `m=0` is admitted only when
every contributing top-exit state proves the positive direct-root gap. Its quantitative native-pixel
integrator is deliberately branch-specific and rejects any model containing `m=0`; the configured
all-root macrobin path is an explicitly nonquantitative display preview. Source state order and
weights are preserved; wavelength-dependent evaluators are never collapsed geometrically.

`sample_native_pixel_mass(*, draws_per_source_state, seed, execution_backend,
cancel_requested=None)` is the optional all-root stochastic terminal. Every valid canonical source
state is a mathematical stratum. A fixed-width NumPy Philox counter layout reserves eight raw lanes
and consumes five for every `(seed, draw index, canonical source-state index)`, including known-zero
states; changing the requested draw or source count therefore cannot change an existing latent
prefix. Its identity is `numpy.philox.fixed_width_source_draw.v1`. Known-zero states perform no root
or physics work. Every active state maps every reachable
physical rod and retained root through the canonical exit optics and detector geometry. The natural
mosaic proposal cancels its probability density, so each visible root deposits
`source * rod population * structure * Ewald coarea * optics / draws` directly into its native
pixel. Internal pixel boundaries use the half-open owner `floor(coordinate + 0.5)`; an exact closed
outer-panel edge belongs to the final pixel. Invalid sources, no roots, nonpropagating exits, and
off-panel roots contribute zero without resampling or renormalization. Nonfinite numerical state
fails closed.

The immutable `MonteCarloDetectorPixelMass` returns `image_A2`, per-draw complete-source total-mass
replicates, total mass, total/active-source/root work ledgers, the maximum individual root deposit, seed,
physical rods, source and rod-catalog revisions, and fixed proposal/RNG/backend identities. Its
detector-visible m=0 support gap is retained whenever the rod set contains m=0. Its measure is
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`. It returns no event or hit table;
`visible_hit_count` counts deposited roots and is not a detector count. Replicate totals expose seed
stability but do not assert finite variance or a Gaussian confidence interval near Ewald folds.
It also records the explicit CPU/CUDA backend, CUDA device when applicable, and bounded CPU worker
count. CUDA selection fails closed and never changes to CPU implicitly.

`compile_monte_carlo_sampler(*, execution_backend, seed, beam_position=None)` creates the explicit mutable,
thread-confined `CompiledMonteCarloDetectorSampler`. `advance_to(...)` returns the authoritative
float64 result; `advance_preview_to(...)` leases a full-native contiguous float32 presentation frame
until the sampler's next operation. `reset(...)` retains compiled state.
`rebind_detector_pose(...)` requires unchanged detector calibration and sample pose, resets the
accumulator, and changes only the four ray-to-pixel projection arrays.
`rebind_geometry(...)` accepts only unchanged source rows, topology, rods, physics, detector
calibration, sample support, film, and crystal mounting. A superseded request raises
`MonteCarloSamplingCancelled`; no partial result is returned. The CPU backend uses at most four
private full-native accumulators and stable block-order reduction. The CUDA backend owns persistent
packed state and raw/presentation buffers, deposits roots directly without an event table, and
uses separate failure-atomic projection and transport buffers on a valid rebind.

`beam_position=ConditionalBeamPosition.from_source(config.source, source_revision=...)` enables
conditional Gaussian pixel integration for source inputs built with
`build_configured_simulation_inputs(..., conditional_source_position=True)`. The source revision,
conditional-source model and residual covariance must agree. Unsupported finite sample support,
external-path absorption or insufficient forward-flight margin fail explicitly. Conditional-mean
inputs cannot enter the sampled-position or continuous-density terminals without their residual
position integral. The result records `position_integration_model_id` and the distinct measure
defined in `RESULT_MEASURE.md`; hit counts count roots with positive panel mass, and the maximum
root deposit is its total panel mass. Geometry rebinds also update the small spatial projection.

`evaluate_detector_density_all_roots(...)` completes this source reduction and returns one detector
function. Downstream peak-center comparison, display sampling, and pixel integration consume that
combined function; none may fit, rescale, normalize, recenter, or form a residual for an individual
source row. `restrict_rods(...)` is a linear selected-group view of the same reduction, and
`rebind_physics(...)` replaces mosaic or strength without changing the immutable source realization.
The source count and realization revision are provenance. The tracked Bi2Se3 proof uses 250 rows,
but the core contract permits any positive count.

`with_maximum_state_block_count(...)` returns an immutable execution view; it does not mutate the
detector or its source/rod identity. CUDA coordinate chunk size is an explicit positive argument and
is rejected for CPU execution. Both controls may change work partitioning but never the declared
measure, source order, or accepted observable beyond its frozen backend tolerance.

The source-averaged `integrate_native_pixels(...)` method is a detailed per-rod proof path, not a
production renderer. It fails closed unless the caller explicitly passes
`include_per_rod_evidence=True`. The detailed arbitrary-coordinate evaluator remains available
without that flag because it performs no pixel integration.

`evaluate_detector_coordinates_geometry(...)` owns the geometry-only native-coordinate inverse
map, and `map_ewald_geometry_to_detector(...)` owns the corresponding forward exit/refraction and
detector intersection. Intensity-bearing detector measures delegate to these authorities.

### `ContinuousNormalizedAngleFunction`

Owns one pose-bound `ContinuousDetectorFunction` and one fixed `AngleFrame`. Callers provide only
broadcastable `two_theta_rad` and `phi_rad`; the wrapper obtains the corrected
`CompiledInstrument` from the detector function, so a stale or mismatched detector pose cannot be
supplied separately. `phi` is canonicalized into `[-pi, pi)`.

The immutable `ContinuousNormalizedAngleValues` payload retains the inverse detector coordinates,
the detector-area Jacobian `N`, angular signal density `S`, normalized detector density `I=S/N`,
caustic flags, and pointwise validity. It applies no detector solid-angle acceptance correction.
The exact pole and every invalid/off-panel direction have `S=N=I=0`. Finite bins are a separate
consumer and must integrate `S` and `N` before division.

### Finite-bin mosaic-profile evaluation

`evaluate_continuous_mosaic_profiles(...)` is the narrow fitting-boundary adapter over the
canonical all-root angle pullback. For every frozen profile bin it integrates angular signal `S`
and detector-area normalization `N` independently and exposes `I=sum(S)/sum(N)`. It evaluates
continuous coordinates only; no detector raster, pixel deposition, interpolation, or pointwise
average of `S/N` enters the fit.

`MosaicReflectionGroupKey` owns physical-rod membership and reflection identity; it does not imply
that peak amplitudes are known or shared. `EXPLICIT_NONZERO` profiles retain each frozen indexed
branch with independent analytic-branch and root-side identities. `COLLAPSED_00L` has
`branch_id=None` and `analytic_branch_id=0`, because the regular `m=0` inverse preimages are summed
by one all-root profile. Profile, angle-frame, physical source, backend/device, rod, layout, and
any predeclared bin exclusions are immutable and must agree throughout a profile evaluation.
`MosaicProfileSet.source_revision` is the detector result's actual source realization revision; a
profile-layout hash cannot substitute for it.

Layered nonzero groups may carry either the retained integer `L` metadata or an exact
`CommensurateLayerOrder` with its reciprocal-basis revision, never both.
`build_layer_l_mosaic_profile_definitions(...)` maps one already frozen
`LayerLMarkerObservations` pack to one profile per rod-free physical key, copies the complete
contributing signed-rod tuple, and converts its detector centroid into the canonical angular frame.
It rejects a stale basis or invalid/nonpolar center. `None` returns no profiles: a missing optional
half-/third-order landmark is not represented by a zero-valued residual. Parent support remains
upstream provenance and is neither expanded into duplicate profiles nor interpreted as a mosaic
weight.

`compile_detector_profile_projector(...)` is the detector-native observation boundary for measured
images. It clips each physical pixel polygon directly into independent local `(2theta, phi)` bins,
accumulates detector signal `S` and pixel-area normalization `N` separately, and never renormalizes
clipped support. Its accepted topology is a connected local angular window that is fully contained
by the finite panel. Boundary samples must project to the panel, the expanded pixel crop must clear
all retained support, and any retained support touching a physical panel edge fails closed. Detector
and profile masks, angle frame, instrument fingerprint, polygon/unwrap policies, and topology
contract all enter the immutable cache key. This projector pixelizes only the already pixelated OSC
observation; simulated component responses remain continuous angular integrals.

Measured profile sets carry an `observation_revision` instead of a physical `source_revision`.
`MosaicProfileDefinition.excluded_phi_bin_indices` records frozen numerical-support
exclusions. Exclusions are provenance-bound and cannot depend on measured intensity.
The nuisance projection, component bank and shape-only search are retired.

### Retired fixed-position ordered-intensity fitting

Selected-component point-density/ROI-mass fitting APIs are archived at `1f65a09`.
Their conditional objective is distinct from native-count fitting. Use the shared
native refinement workflow for current inference.

### Configured simulation

- `load_simulation_config(path, repository_root=...)` accepts one strict
  `rasim-simulation-v2` YAML document. Unknown, duplicate, aliased, or missing fields fail.
- `build_configured_simulation_inputs(config)` creates source rows, canonical incident states,
  material, rods, the configured finite ordered-parent strength including R-centered 3R, and Bragg
  space once. The RichEpsilon transition law supports the native 3R parent in proof, compiled CPU,
  and CUDA paths; `epsilon=0` retains its exact native-sequence fast path.
- `build_configured_geometry_inputs(config)` creates only source rows, material, reciprocal basis,
  rods, and compiled instrument. `build_geometry_only_ewald_context(...)` adds one nominal incident
  state. Neither boundary constructs structure strength or mosaic probability.
- `sample_configured_source(source, sample_count=...)` is the one mapping from validated configured
  source parameters to either `gaussian.v1` or `discrete_gaussian_lines.v1`, including optional
  position--divergence correlations and exact declared line masses. A physical discrete source
  requires at least one row per line. The separate nominal-geometry source API returns one centroid
  row at the mean wavelength and that model ID is rejected by every source-weighted detector-
  measure intensity boundary. Source-free intrinsic coating/locus density is permitted only for
  geometry displays and reference landmarks, never as source intensity or likelihood evidence. Equal
  per-line row counts use a shared geometry grid. A declared nonzero correlation is moment-matched
  to the full four-dimensional covariance only when each line has at least eight rows; zero-correlation
  LHS grids and smaller correlated grids are finite quadrature approximations, and non-divisible
  counts carry an explicit unequal-grid model ID. Unknown or mixed declarations fail; no field is
  silently ignored.
- `build_source_averaged_detector(inputs)` builds the all-state detector model.
- `SourceAveragedDetectorEwaldMeasure.sample_native_pixel_mass(...)` returns the optional weighted
  native-pixel Monte Carlo estimate without creating an event table or count calibration.
- `evaluate_detector_density_all_roots(column_px, row_px, ...)` returns one continuous-coordinate
  density after every source state, physical rod, and retained inverse root has been reduced. Its
  coordinate-shaped caustic flag is the logical OR of the detailed rod flags. The existing
  per-rod coordinate evaluator remains the proof and diagnostic interface.
- `evaluate_nominal_integer_l_markers(context)` solves exact integer-L intersections on the
  peak-mosaic (`alpha=0`) manifold, applies the canonical nominal-state exit/refraction and active
  detector visibility path, and retains every physical rod before grouping only coincident display
  labels. The seam-safe `root_sign = sign(sin(beta - phase))` distinguishes the analytic
  `phase-delta` and `phase+delta` sites even when `(m,L,branch)` labels coincide. These are
  `peak_mosaic_alpha0_integer_L_center.v3` nominal-source references, not source-averaged raster
  maxima.
- `sample_reciprocal_space`, `evaluate_nominal_ewald_surface`, and
  `integrate_detector_macrobins` generate optional display data without becoming model authority.
  Macrobin integration consumes only the completed total coordinate density; its result has no
  per-rod pixel axis.
- `sample_detector_pixel_center_density(...)` samples the completed all-source, all-rod, all-root
  detector function once at each native pixel center and returns
  `raw_detector_coordinate_density_A2_per_px2.v1`. It is display-only point sampling, not
  `raw_detector_pixel_mass_A2.v1`, a box/macrobin integral, count calibration, or raw OSC counts.

## Measured selection and indexing

`discover_measured_cake_peaks(...)` accepts a detector-native image, mask, `CompiledInstrument`,
`AngleFrame`, and immutable policy. It builds a tiled angle chart without accepting a marker
catalogue or predicted marker coordinates, then refines each proposal once on the native raster.
Image, mask, calibration, and policy identities are retained in `MeasuredPeakDiscovery`.

`index_discovered_integer_l_peaks(...)` converts only discovered native coordinates to sample-frame
`Q`, infers family `m`, integer `L`, analytic Ewald branch, and beta-root sign through the canonical
geometry/root authorities, and deterministically drops tangent, conflicting, noncoincident, or
ambiguously owned labels. Exact alpha-zero anchors are generated only after the discrete label is
frozen. `select_confident_branch_tracks(...)`
requires distinct-incidence replication with shared `L` identities and returns an immutable hashed
`MeasuredIndexingResult`. Non-detection is never treated as a physical extinction, and fitting may
not change an identity inside an optimization.

## Exact integer-L tagged detector-function fitting

`ContinuousDetectorGeometryModel.bind(...)` returns an immutable callable detector field
`D(column_px,row_px)`. Reference and trial fields remain functions; fitting does not rasterize,
pixel-integrate, or evaluate a detector-domain quadrature. Exact tags are associated landmarks on
each field's continuous detector-coordinate domain. They are not scalar-density maxima or
intensity centroids.

All tags use exactly one deterministic incident companion state with source-center origin, zero
divergence, and mean wavelength (`nominal_source_center.zero_divergence.mean_wavelength.v1`). The
Monte Carlo source index is not part of tag identity, so an N-state detector field still has
one tag per visible user-facing `(m,L,tag_branch)`. The companion state traverses the same
incident transport, Ewald, exit-refraction, and detector-intersection code, but is not inserted as
an artificial finite-mass delta into the empirical Gaussian source sum.

For nonzero `m`, `IntegerLMarkerKey` freezes `(m, L, Ewald branch, beta-root sign,
representative physical rod)` and the nominal-source `alpha=0` policy. The two `root_sign` values
are the two analytic beta-root sides; they are not two Ewald branches. User-facing `tag_branch=1`
is `root_sign=-1`, and `tag_branch=2` is `root_sign=+1`; the analytic Ewald branch remains separate
provenance. Exactly one tag is allowed for each `(m,L,tag_branch)`. Missing, tangent, or
branch-changing roots fail instead of being reassigned. For `m=0`, exact `L` alone leaves a
continuous orientation curve. The tagged geometry-fit policy selects the unique exact-L
reciprocal-axis direction with minimum mosaic tilt relative to the unmosaicked axis. It is a
reproducible geometry landmark, not an `m=0` intensity maximum; its user-facing branch is
`tag_branch=0`.

An exact landmark may lie on an integrable caustic of the zero-transverse-width detector density.
The tagged-coordinate observable is still well defined there, but its pointwise scalar intensity
is not used as a residual. Pixel mass would require the separate native-pixel integration API and
is deliberately outside this fit.

`fit_tagged_detector_function_geometry(...)` evaluates reference tags from one bound callable and
trial tags from the same prepared detector-function family. It simultaneously minimizes
covariance-whitened exact-tag coordinate residuals, fixed-span half-angle residuals between each
nonzero-m root-side pair, and the total-least-squares m=0 exact-L line angle. There is no centroid
term. `evaluate_tagged_geometry_objective_residual(...)` exposes the same immutable whitened
site-plus-line vector for scientific diagnostics and error injection.
Its ordering is nonzero coordinate pairs, sorted paired-branch half-angle terms, m=0 coordinate
pairs, then the m=0 TLS-line half-angle term; m=0 L identities and wavelength must match exactly.
The fitter uses bounded TRF least squares. The accepted single-5-degree parameterization
`detector_xy_plus_pivoted_effective_sample_normal_xy.v2` applies active intrinsic local-x then
current-local-y rotations relative to the frozen base poses:

```text
R_detector = R_detector,0 Rx(delta_detector,column) Ry(delta_detector,row)
R_sample   = R_sample,0   Rx(delta_sample-normal,x) Ry(delta_sample-normal,y)
```

Detector corrections keep the declared detector reference point fixed. Sample corrections use one
fixed mechanical LAB pivot `p`. With `A = R_sample R_sample,0^T`, the sample origin changes as

```text
t_sample = p + A (t_sample,0 - p).
```

The model infers `p` only when every configured goniometer axis declares the same pivot. A
configuration with no axis or unequal pivots must supply an explicit
`sample_correction_pivot_lab_m`; it never silently substitutes the sample origin or one arbitrary
axis pivot.

Every trial tag rebuilds canonical incident transport, solves the exact Ewald constraint, applies
exit refraction, and intersects the corrected detector plane. No reciprocal-space point is
geometrically projected onto an image surrogate.

TRF accepts steps by decrease of the complete coordinate-plus-angle least-squares objective. It
does not require every individual line angle to decrease at every accepted step. The angle-only
Jacobian has rank two for this four-parameter fixture, and different tagged lines can request
conflicting local directions; an angle-monotone projection would therefore discard identifiable
coordinate information or stall. Exact coordinates provide the missing directions, while the
reported chord and m=0 angles verify the shared line geometry at the solution.

The detector limits use RA-SIM's direct `(+/-10 deg, +/-10 deg)` pitch/yaw hard shells. The two
effective sample-normal components conservatively use its `+/-5 deg` angular-correction shell;
RA-SIM has no separately bounded second reduced-normal coordinate. They share the declared fixed
pivot but are not separately recovered sample-mount and named goniometer-axis errors. A preflight
bound-scaled Jacobian must have rank four and acceptable condition.
`audit_integer_l_marker_selection(...)` independently re-enumerates the full visible
non-specular catalog after fitting and reports `SAME`, `MISSING`, `CHANGED`, or `AMBIGUOUS` without
reassignment.

One image at one commanded goniometer angle cannot distinguish raw sample-zero tilt, goniometer-axis
yaw/pitch, crystal rotation about the rod axis, detector tangent translation/beam-center shifts, or
sample-normal offset/detector-distance gauges. Those parameters require additional constrained data,
especially multiple commanded goniometer angles; they are not exposed by this fit contract.

### Optional exact rational-layer landmarks

`CommensurateLayerOrder(numerator, denominator)` is the exact identity for a coordinate along one
declared layer repeat. It reduces by the greatest common divisor, keeps a positive denominator,
orders by exact cross multiplication, and converts to `float64` only at the fixed-L Ewald equation.
`solve_layer_l_ewald_roots(...)` owns that equation. The retained integer solver delegates to the
same kernel with denominator one; integer marker/catalogue/selection records and their hashes are
unchanged.

`LayerLMarkerKey` freezes `(m, reduced L, analytic branch, beta-root sign,
reciprocal-basis revision)` and deliberately contains no representative rod. A
`LayerLMarkerDefinition` carries every signed `(h,k)` rod that realizes the detector locus. The
predictor independently verifies that all such rods map to the same Cartesian `Q`, detector
coordinate, status, and root identity; a split locus fails closed. `LayerLMarkerObservations`
is a nonzero-`m` contract and stores one coordinate/covariance row per key, so exact parent or rod overlaps never duplicate a
geometry residual. `merge_layer_l_marker_observations(...)` joins already-qualified optional rows,
rejects conflicting duplicates, and returns the original baseline object when no optional pack is
supplied.

For PbI2, `build_ideal_pbi2_polytype_landmark_catalogue(...)` requires the one-trilayer PbI2 metric
supplied by its geometry model; T22 validates the tracked `PbI2_2H.cif` as that declared reference.
It admits reduced denominators one, two, and three and evaluates pure-parent support with exact modular arithmetic in the corrected
`h+2k` registry gauge. Integer loci may be shared; half-order loci can carry both 4H hands; and
third-order support swaps 6H hand between the two signed registry sectors. The catalogue retains
the supporting parent and signed rod as provenance only, not as an intensity or population weight;
geometry uses only the qualified coordinate and covariance. It never evaluates a structure amplitude, population
fraction, stacking recurrence, mosaic distribution, detector raster, or pixel integral.

`IndexedGeometryImage` accepts either the retained integer observations or one rational-layer pack
that contains the integer baseline plus any observed half-/third-order rows. Missing, weak,
off-panel, unresolved, or centroid-unqualified optional peaks simply add no rows. The same
covariance-whitened site-plus-chord residual and nine-coordinate multi-incidence optimizer are
used. `audit_exact_layer_l_geometry_roots(...)` independently brackets the fixed-L elastic
equation for every contributing rod and detects root/order swaps without calling the production
rational solver.

`admit_discovered_pbi2_layer_l_peaks(...)` is the material-specific measured admission boundary.
It runs only after catalogue-free `MeasuredPeakDiscovery` under the same frozen detector geometry
and angle frame. It requires the complete five-parent catalogue, considers only active half- and
third-order sites, and admits a group only when its native centroids pass the discovery significance,
hard-distance, covariance-aware uncertainty, competing-site, competing-peak, and complete-visible-
root gates. It never searches the image, evaluates a predicted structure factor or parent
population intensity, or turns a missing candidate into an extinction; measured discovery
significance is still required. Success returns `Pbi2LayerLPeakAdmission`, which retains the complete hashed blind
discovery, complete parent catalogue, exact catalogue-to-peak row join, and the frozen
`LayerLMarkerObservations`; callers pass its `observations` member to
`merge_layer_l_marker_observations(...)`. No admitted group returns `None`, preserving the integer
observation object and legacy path at the caller boundary.

The general OSC manifest and integer indexing schemas remain unchanged; measured rational rows are
an additive post-discovery pack. T25 exposed this auditable admission boundary. Contract v13 now
supplies a shared numerical PbI2 detector response, but no accepted measured observation/background
recipe or per-material staged runner is inferred from the CIF. The separately relaxed native 4H/6H
cells do not share the exact 2H layer repeat, so this admission is valid only for the declared
common-layer transition model. Broad, asymmetric, or population-dependent blends belong to the
stacking-aware intensity fit. The common-incidence-delta/sample-normal-x gauge is unchanged by the
additional landmarks.

### Fixed-parent PbI2 strength

`Pbi2ParentMixtureStrength` and `Pbi2ParentLogRatioParameterization` retain five fixed
near-parent templates, explicit fractions, signed rods and shared finite-layer response.
The intrinsic-landmark population compiler/fitter is retired. This provider does not
infer an arbitrary transition law.

### Indexed multi-OSC geometry series

`load_osc_geometry_series(...)` accepts one strict `rasim-osc-geometry-fit-v1` manifest with a
simulation configuration, one commanded-axis index, and a nonempty list of unique image IDs, OSC
paths, and complete numeric axis-angle tuples. Paths are manifest-relative. Motor angles are
explicit declarations: neither filenames nor OSC headers are an authority. Version 1 requires one
configured axis at index zero, allows repeated commanded angles, and requires one material/mount
per shared fit group. The same contract is reusable for different materials in separate groups;
its first discrete marker identity remains layered-hexagonal `(m,L,branch,root_sign,rod)`.
The selection `AngleFrame` origin is the transported nominal incident/sample intersection, never
the sample-transform translation; its column/row directions come from the active corrected detector
pose rather than stale nominal detector axes.

`index_osc_geometry_series(...)` compiles shared material/reciprocal state once, reads each OSC,
performs the one clockwise I/O conversion, runs position-free discovery and reciprocal indexing
with a geometry-only context, and freezes the cross-image selection once. Its
`OscGeometryIndexingRun` retains the source discoveries, geometry inputs, geometry contexts, and
selection from that provenance-bound run. It exposes exact fit-ready models only when every image
has accepted visible integer-marker observations; otherwise `indexed_images` is `None` and the run
is a retained-discovery preflight that cannot enter fitting or frozen-key reindexing. Instrument-
override global-rediscovery runs likewise expose no fit-ready images. Model, discovery, data, mask,
policy, commanded-angle, and indexing-context hashes are checked together.
Missing, extra, duplicate, angle-mismatched, or image-provenance-mismatched IDs fail before fitting.
Geometry-only inputs recompute and require the exact configured one-row nominal source revision;
each fit-ready image also recompiles and compares the complete detector, sample, axis, and pivot
state before the residual hot path is admitted.

`fit_indexed_geometry_series(...)` owns one complete nine-coordinate vector shared by every image,
concatenates the existing canonical per-image residual blocks in sorted image-ID order, and uses
bounded TRF least squares. `SHARED_GEOMETRY_PARAMETER_NAMES` declares the only accepted coordinate
names and their canonical order. `fitted_parameter_names` selects any subset; it may be empty only
when the common incidence-angle delta remains active. Names are canonicalized before optimization
and every omitted coordinate is copied bit-exactly from `initial` into every residual evaluation
and the final correction. The default selects all nine shared coordinates.
The complete shared pack is, in order:

```text
detector local-column tilt, detector current-local-row tilt,
sample local-x tilt, sample current-local-y tilt,
goniometer-axis pitch, goniometer-axis yaw,
signed sample-plane normal offset,
goniometer-pivot pitch-tangent offset, goniometer-pivot yaw-tangent offset
```

The corrected axis owns a transported orthonormal pitch/yaw tangent basis. Pivot offsets are
applied in that basis; axis-parallel pivot motion is a gauge. The ordered rigid transformation is
corrected axis/pivot, commanded motion, sample intrinsic x/current-y correction about that pivot,
signed displacement along the final sample normal, and detector intrinsic x/current-y correction.
Detector roll is an exact gauge with sample-y/axis-pitch coordinates, crystal roll is an
axial-powder gauge, and detector center/distance/pitch remain calibration-owned.
The hard half-spans are `(10 deg, 10 deg, 5 deg, 5 deg, 5 deg, 5 deg, 0.1 mm, 0.1 mm, 0.1 mm)` in
the declared parameter order.

An optional `IncidenceAngleDeltaBounds` activates one common series-level coordinate,
`incidence_angle_delta_rad`, and may pair it with zero-sum per-image trims represented in a
canonical Helmert basis. Thus
`effective_angle[j] = commanded_angle[j] + incidence_angle_delta_rad + trim[j]`, with
`sum(trim)=0`. The common delta remains the mean calibration correction; the trims can absorb small
commanded-angle errors without letting three independent offsets replace it. Both are applied once
by rebinding each image's configured axis angle before the remaining shared corrections. At the
nominal x incidence axis the common delta occupies the same gauge as
`sample_normal_x_tilt_rad`, so activating both fails before optimization. The accepted Bi2X3
parameterization fixes that sample-x coordinate at zero, bounds the common delta to `+/-0.5 deg`,
and tightly bounds the two independent trim contrasts.

The actual bound-scaled Jacobian must have rank equal to the selected coordinate count with
condition at most `1e8` before optimization. Singular values, weakest direction, and active-bound
flags have that same combined-coordinate length; the result explicitly reports canonical shared
fitted/fixed names and `jacobian_parameter_names`, whose last entry is the common delta when active.
Full rank does not imply precise pivot recovery; the weakest direction must be reported. Beam
center and lattice constants are not members of this pack and cannot be activated accidentally. A
separate lattice-sensitivity command
may consume only a qualified completed position artifact. It fits near-CIF hexagonal in-plane and
normal log strains, serializes their constrained full basis, reports data-only and penalized
sensitivities separately, and promotes the candidate only
when the data-only scaled rank, condition, improvement, bounds, prior-pull, selection, and root gates
all pass. Otherwise downstream construction receives no basis override and follows the CIF path.

`audit_indexed_geometry_series_roots(...)` dispatches by observation contract. For integer records
it brackets `F(beta)=q(beta) dot (q(beta)+2 ki)` on its two monotone arcs without calling the
production integer-`L` root solver. For rational records it applies the same independent bracketed
solve to every contributing signed rod without calling the production rational solver. Both paths
assign root sign from the oriented crossing and branch from the signed axial derivative, then
compare independent and production detector coordinates per physical key. A swapped beta/root
assignment therefore fails even when the key set is unchanged.

`reindex_frozen_osc_geometry_series(...)` returns a typed `FrozenOscGeometryReindexing` bound to the
source manifest and discovery hashes. `audit_frozen_osc_geometry_reindexing(...)` independently
recomputes the exact typed relabeling from the source run, requires manifest equality, reconstructs
each corrected geometry context, and verifies the context digest, which also owns the frozen keys,
native coordinates, and covariance hashes, before comparing visibility. The operation
preserves exactly the selected position-free native coordinates, support matrices, scores,
image/mask hashes, and policy, recomputes only their corrected angles and reciprocal labels, and
requires the same full keys with coherent tracks. It performs no OSC I/O, cake search, or
native-coordinate refinement.
A complete corrected-geometry `index_osc_geometry_series(...)` pass is a separate operational
diagnostic because its cake grid and same-key candidate ownership change with geometry. Newly
visible or differently selected unfitted lobes are reported but never censor or replace frozen data.

### Historical portable staged-fit replay

This contract documents the retained baseline replay and is not the current layered-quintuple
workflow. New fits compose the strict fixed position, optional accepted lattice, and provided
mosaic checkpoints described below, then use the mixed-chart matched-region pipeline.

`rasim-staged-fit-replay-v2` is a strict, material-case manifest for geometry, mosaic,
ordered-intensity, and optional render stages. Every path is case-relative and must remain inside
the repository. Every file has an exact SHA-256; OSC records additionally bind decoded native
shape, dtype, and `[row,column]` byte content. Paths nested inside geometry-series, simulation,
mosaic, ordered-intensity, and policy files must resolve to their corresponding declared roles;
hash-complete decoy roles are rejected. Bi2Se3 and Bi2Te3 cases require the declared
5/10/15-degree triplet, source seed 1729, one ideal source state for geometry, and the identical
250-state realization for mosaic and ordered intensity.

Each stage emits `rasim-staged-fit-replay-stage-v2` with case/material identity, a stage-scoped case
hash, backend, its actual execution runtime, source identity, compact scientific state, and the
immediately preceding scientific revision. Every stage declares its consumed file roles; its case
hash includes only those file records, that stage's configuration and expected result, and that
stage's tolerances. Downstream-only case edits therefore do not invalidate a verified position
checkpoint. `rasim-staged-fit-replay-certificate-v2` records the complete current case hash, the
ordered revision chain, and the verified summary. Before creating output, the runner reloads and
exact-compares the case, all role mappings,
all file hashes, decoded OSC identities, and nested consumed paths. It reads and hashes one
case-bound `uv.lock` byte snapshot, then
requires the complete transitive numerical dependency closure to equal that lock; a render replay
also requires the locked Pillow version and a successful `PIL.Image` import. The certificate's
`verification_runtime` records the verifier's lock hash, interpreter, platform, and installed
versions. Runtime is operational provenance excluded from scientific revisions, but resume requires
the stored stage runtime to match the current stage runtime exactly. Fresh and resumed stage results
receive the same exact envelope, recomputed-revision, artifact-contract, and upstream checks. A
verified stage must pass its partial scientific summary before its JSON is written or it can become
an upstream input. Resume rejects a changed stage-scoped case, backend, source, compact state, or
canonical scientific projection of an externally referenced artifact. A caller may stop after any
stage. The replay CLI resumes from
the complete verified predecessor chain in the same output directory; direct external-artifact
ingestion is not yet a replay CLI option. A later fit never infers, silently recomputes, or
substitutes an earlier result. The measured mosaic CLI requires one verified position artifact and
may additionally consume its bound lattice-decision artifact. It atomically extracts the position
revision, all nine corrections, and one common incidence delta, and records complete fixed-position
and fixed-lattice states. The ordered/SF consumer accepts the lattice only through that mosaic
checkpoint and exact-checks both states, including source and rod-catalogue revisions.
`RETAIN_CIF_LATTICE` passes no explicit matrix override; an accepted full basis rebuilds the nominal
and source-averaged material, optical, reciprocal, rod, and detector state.
The Bi2Se3 geometry state records the commanded image angles, the one common fitted delta, all three
effective angles, and the fixed sample-x gauge so mosaic and ordered/SF construction reproduce the
same transforms exactly.
The Bi2Te3 mosaic state records the exact geometry-stage position projection, the implicit
hash-verified CIF lattice, and the case-declared simulation-config path and SHA-256. Resume
reconstructs and compares all three; ordered intensity must carry those records byte-for-byte from
the qualified mosaic stage.

This is a nominal scientific-replay contract. It assumes trusted stage implementations and that the
declared repository inputs are not edited during an active stage; it is not an adversarial
tamper-proof execution boundary.

The geometry-stage envelope binds the canonical one-state configured source batch, including its
seed and source revision. Mosaic, ordered-intensity, and render envelopes bind the case-wide
250-state batch. Each nested scientific summary separately retains that case-wide batch identity so
the combined fit certificate has one explicit downstream realization.

Input identities, profile identities including every admitted `m=0`, source revisions,
fitted/fixed coordinates, active-bound masks, rank, classification, and stage order compare
exactly. Declared floating fit outputs compare by case-specific absolute tolerance. Absolute paths,
artifact destinations, wall time, peak memory, device labels, and PNG encoding are excluded from
scientific revisions. Decoded display pixels may be certified separately; they are not fit inputs.
Path-dependent JSON artifact container hashes are operational resume checks and are excluded from
the scientific revision; the consumed compact state remains revision-bound. Render artifact mode,
size, and decoded-pixel hashes are revision-bound and rechecked from every PNG during resume.

The Bi2Se3 and Bi2Te3 ordered replays transfer identity-matched measured-mosaic nuisance
amplitudes onto baseline 250-source continuous peak-center signals. Both are model-limited
`NO_ORACLE` estimates, not direct count-calibrated raw-OSC structure recovery. The Bi2Se3 stage
fixes Bi as the occupancy gauge and reports Se1/Bi and Se2/Bi ratios; its admissible normalized
representative is not an absolute-occupancy result. Optional named active-parameter bounds narrow
the canonical ordered-intensity optimizer bounds without activating frozen coordinates.

## Once-only factor ownership

| Factor | Owner |
|---|---|
| empirical source mass | source sampler |
| sample-footprint acceptance and entrance amplitude | incident transport |
| flat-film illuminated-path `1/|direction_sample,z|` | detector source-phase construction |
| per-rod population | `MosaicBraggSpace` |
| CIF/finite-stack structure strength | ordered strength model |
| wrapped mosaic probability | `MosaicBraggSpace` |
| Ewald restriction / inverse-map determinant | Ewald or detector pushforward, by declared route |
| internal-film Ewald solid-angle `k^2` | intrinsic direction pushforward only |
| exit amplitude and uniform-depth attenuation | detector optical mapping |
| external detector-path Beer--Lambert factor resolved per sampled wavelength | detector optical mapping |
| phase and polarization weights | detector measure construction |
| source-state sum | source-averaged detector measure |
| detector box integration or weighted hard-bin estimator | selected terminal pixel estimator |

The two Ewald routes are equivalent proof views and are never multiplied together.

## Removed API

Contract version 12 retains no compatibility shims for mosaic-orientation batch APIs, candidate
selection, scattering-event batches, outgoing-wave batches, detector-hit batches, event transport,
general point deposition, discrete Ewald painters, sphere textures, or raster grids. The optional
terminal stochastic estimator is not one of those retired runtimes: it exposes no intermediate
sample/event object and streams weighted roots into its declared final pixel-mass estimate. Callers
must use the continuous reciprocal and detector contracts above.

## Shared native objective for generic-CIF matched regions

`MatchedRegionObservations` declares acquisition IDs, complete signal/background blocks,
native integrated count mass, physical support and full count covariance.
`FixedMatchedRegionBackground` declares a frozen baseline with uncertainty; adjacent
anchors may condition it once. `IntegratedPeakAreaProjection` optionally sums complete
signal bins without changing the detector measure.

`prepare_matched_region_objective` applies the same anchor operator H to model and data,
then signal selection/peak aggregation S. It preserves
`S (C + Cbackground - H C - C H.T) S.T`; no cross-acquisition covariance is dropped.
H and peak groups cannot mix acquisitions. Each acquisition retains its own nonnegative
scale, fitted jointly by covariance-whitened NNLS in `NativeFitObservations`.
Background-corrected observations and model may be signed; underlying raw model masses
remain nonnegative. No detector area or solid-angle correction is silently added.

`ParameterizedStructureRegionModel` binds exact sparse response blocks to one compatible
`StructureStrengthParameterization`. `AffineCifFiniteStackParameterization` retains explicit
fixed-cell site/occupancy/isotropic-`U` modes and conventional-cell repeat semantics; it does not
invent disorder from a CIF. Bi2X3 and the fixed-parent Pb provider share this strength seam.
Reciprocal basis, source, optics, mosaic, rods and geometry must match the response revisions.

`fit_structure_regions` accepts declared `FitParameter` names/units/bounds, one shared
specimen owner, starts and optional `GaussianCalibration` blocks. It delegates to
`fit_native_parameters`; there is no second least-squares implementation. Data-only
profiled-scale sensitivity must pass numerical rank, practical rank and conditioning
limits. An unresolved optimizer or rank failure raises `StructureRegionIdentifiabilityError`
carrying the shared result. Calibration cannot supply missing data rank. Successful local
identification is explicitly not numerical quadrature qualification or global uniqueness.

The returned shared search result retains acquisition IDs, fitted native counts,
parameterization and fitted-strength revisions and sensitivity. The former
`fit_matched_regions`, `fit_parameterized_matched_regions`, their result wrappers and
arbitrary prior callbacks are retired at revision `b8dba2b`. The shared objective reports
full squared residual sums; the old optimizer reported half. See [STAGED_FITTING.md](STAGED_FITTING.md).

## Continuous incidence-exposure contract

`ContinuousIncidenceAcquisition` owns the canonical physical support and normalized motor-exposure
law. Reversing the declared direction preserves its prediction identity while remaining distinct
serialized provenance. Duration, traversal and cycle counts, `ScanImageStep`, and measured
endpoint/speed diagnostics are provenance only and are excluded from numerical angle nodes,
weights, and the prediction revision. An acquisition-bound quadrature binds the acquisition
prediction revision, calibration revision, nodes, normalized masses, panel partition, and effective
support; a rule compiled for 5--20 degrees must fail closed for the authoritative 5--25-degree
request. The adaptive layer does not own a detector-region operator: its caller-supplied evaluation
revision owns that identity.

`compile_uniform_incidence_angle_legendre_rule(...)` first compiles immutable commanded-angle
nodes and normalized masses without inventing a calibration identity. Those commanded nodes build
`FixedIncidenceScanSeries`; its effective nodes and `scan_calibration_revision` then construct
`IncidenceAngleQuadrature` with
`exposure_density_id="uniform_normalized_incidence_angle_density.v1"` for a normalized uniform
motor-exposure law and both effective support bounds (commanded bounds plus the exact calibrated
delta).
`IncidenceAngleQuadrature.uniform_legendre(...)` is a standalone analytic-rule constructor. Even
with an explicit calibration revision it has no acquisition prediction revision. Acquisition-bound
rules use `ContinuousIncidenceAcquisition.uniform_panel_quadrature(...)` or explicit nodes from
`FixedIncidenceScanSeries`. There is no default calibration identity. `FixedIncidenceScanSeries`
binds the position artifact, calibration model,
commanded nodes, effective nodes, shared source revision, and ordered sample-geometry revisions.
The current calibration model applies only the fitted common incidence delta. Fitted image-specific
trims have no continuous interpolation and are never extrapolated into a scan.

`IncidenceAngleAveragedDetector` owns one complete fixed-angle detector view per effective
quadrature node. The optimized Bi2X3 path may compile one static-physics template and call
`rebind_source_averaged_detector_incidence_scan(...)` to derive every calibrated node. That exact
rebind consumes each node's canonical precomputed incident transport and recomputes the engine's
derived attenuation and rigid detector/sample projection while retaining packed
structure, rods, mosaic, envelope, Parratt stitch, and execution blocking; it does not interpolate
an angle field. It requires unchanged source validity topology and fails closed otherwise. The same
template may serve different calibrated coarse/fine rules.

Construction requires identical source realization, detector-native
chart, rods and order, material/structure/mosaic/envelope/stitch physics, and requires each engine's
declared incidence setting to equal its aligned quadrature node. The calibrated builder stamps one
binding revision onto each configured input and fixed-angle engine. The wrapper derives the
ordered component identity and rejects missing, differently calibrated, or relabeled components;
callers do not supply loose expected revisions. It intentionally exposes no single
pose-bound `instrument`. It composes with the optimized Bi2X3 renderer and generic structure
detector through their existing all-root detector-coordinate evaluations; the rod-reduced method
streams each optimized child's reduced kernel without retaining per-angle images.

The returned validity diagnostic is the exposure-weighted fraction
`sum_j p_j valid_source_count_j/N`, not an integer source count. Invalid contributions stay zero and
are not survivor-renormalized. The v1 reusable optimized-engine builder requires the source
validity mask to be identical at every angle node. Exact
quadrature-node caustics raise `FloatingPointError`. The v1 wrapper is detector-native:
outgoing-angle adapters, native-center culling based on one sample
normal, and geometry-bound continuous-fold correction plans are not admitted consumers.

`evaluate_adaptive_scan_oracle(...)` is a separate finite-ROI contract. Every physical panel gets
a mandatory coarse/fine comparison; only numerically failing or fold/topology/window-event leaves
are bisected further. All batches share one calibration delta, source revision, and evaluator
revision. Signed ROI contrast is preserved and convergence requires every terminal leaf plus the
absolute covariance-whitened total to pass the declared gate.

`run_staged_delayed_acceptance(...)` is material-neutral. Baseline failure stops before proposal
callbacks, and every candidate passes the exact fixed-dataset gate before an exact scan call.
Fixed and scan comparison revisions remain frozen. With accepted-iteration limit `K`, the exact
scan score count is at most `1 + 2K`.

## Native Bi cell, site and morphology refinement (v15)

This additive contract leaves the historical fixed-cell affine-CIF and five-coordinate
sample-Q-envelope fits unchanged. `fitting.bi_native.BiCellSiteParameters` declares a,c (A),
the signed Bi and outer-chalcogen 6c z coordinates, all three orbit occupancies, and one
radial/normal displacement pair (A2) for each 6c/3a/6c orbit. `BiNativeStructureModel`
preserves the complete expanded conventional-cell orbit rows and fixed integer termination
lifts. It rebuilds the reciprocal basis and composition/volume-derived optics; individual
physical rods retain their identities. `validate_rod_coverage` proves the supplied roster
complete over the declared cell box using an absorption-aware elastic bound and exhaustive
integer enumeration. Expanded search bounds require a new coverage check.

`CifFiniteStackStrength.site_displacement_tensors_A2` is optional immutable per-site crystal-frame
state. It uses the existing unit-cell amplitude's `exp(-Q.T U Q/2)` and coherent finite-repeat
equation. Tensors are real, symmetric, positive semidefinite and exclusive with unknown-U
substitution. Isotropic tensors reproduce the original path. The per-orbit transverse-isotropic
projector has one implementation in `SiteDisplacementProfile.tensors_A2`.

`pipeline.conditional_detector.ConditionalStructureDetector` integrates individual rods, both
signed axial sheets, independent uniform crystal azimuth, spherical mosaic, wavelength/source
mass, scalar optics and conditional Gaussian source position on fixed detector-native pixel
support. The local m0 output is the named empirical Parratt/kinematic composite. Its phase-Q
conversion and local geometry share the existing detector arithmetic. Regular
panels carry physical W du masses and canonical S+/W, S-/W fractions through the
same compiler and factor contraction. Accepted angular child patches are reused;
compact native windows accumulate into caller images or literal fractional
observation memberships. Source mass applies once without survivor renormalization.
Continuous `density_at` uses the same prepared transfers and Gaussian kernels,
with a separate pointwise/continuous-region qualification obligation.

`fitting.native_observations.NativeFitObservations` freezes net native counts, supported rows and
the full count covariance plus one background-mode covariance. GLS uses a Cholesky factor on
supported rows. Historical F/G losses and guards remain separate measures. Unguarded
nonnegative scale is profiled analytically. Guarded SLSQP fits one literal nonnegative scale
under the original inequalities; exact quadratic scale intervals remain available for
fixed-shape profiling and diagnostics. An empty intersection denotes an infeasible shape,
never a relaxed guard. The versioned numeric loader verifies array/projection hashes and
unchanged net observations. No executable checkpoint is admitted.

`fitting.bi_joint.BiJointCandidate` has the 13 atomic coordinates plus Gaussian width,
Lorentzian width/mass, two bounded surface-mixture coordinates, extra film thickness and two
interface roughnesses. Surface fractions are `(s0,(1-s0)s1,(1-s0)(1-s1))`; physical film
thickness is `N*c + extra`, so a coherent stack cannot exceed the film. N is an integer
conventional-cell repeat count, never a rounded continuous parameter. `BiJointModel` now
binds this candidate through the common `NativeRefinementModel` contract. The duplicate
Bi evaluator/search/CLI has retired. `NativeJointEvaluator`, `fit_native_parameters` and
`scripts/refine_native.py` own execution for all built-in specimen models. Output remains
one external diagnostic with separate candidate, convergence and numerical-selection states.

The opt-in component Python API returns complete normalized Gaussian/Lorentzian
raw columns in that order. Shared single-exposure GLS can profile their two
nonnegative amplitudes while retaining the full physical vector and explicit
zero-exposure/nonuniqueness diagnostics. Its beta derivative is conditional on a
stable active face; it is not a globally smooth or jointly identified result.
Component and mixed raw replay identities remain distinct. See
[NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md#conditional-gaussianlorentzian-amplitudes)
for API scope and qualification limitations.

Local sensitivity uses range-scaled, covariance-whitened prediction derivatives with profiled
scale. The CLI records SVD directions and each perturbation's empirical stitch selection,
interval and normalization; changed intervals flag a possible nonsmooth derivative. These are
diagnostics, not posterior uncertainties or numerical qualification. Bounds do not establish
identification. Full occupancy scale remains active through density-derived optics, and may
still be weakly constrained. Source/mounting calibration, PSF, strain distributions, additional
off-specular channels and stacking disorder are outside this first ordered Bi milestone.

## Joint native Bi/Pb and acquisition refinement (v16)

`PbNativeStructureModel` preserves signed iodine integer lifts while varying a,c,z,
both occupancies and independent radial/normal Pb/I ADPs. `PbJointModel` adds all
declared phase/parent simplex coordinates, per-phase epsilon, initial orientation,
mosaic and finite terminations. `Pbi2FiniteSurfaceStrength.site_displacement_profile`
uses the existing unit-cell tensor amplitude for every finite whole/lower/upper motif
and both orientations. N retains its declared single-layer Pb or conventional-cell Bi unit.
`native_structure.rebind_native_structure` owns common cell/material/rod rebinding.

`NativeSourceDefinition` retains the explicit source input at the loader boundary.
`NativeInstrumentModel` binds eighteen acquisition coordinates with fixed frames and units,
resamples the conditional source and recomputes candidate optics at actual wavelengths.
`NativeJointEvaluator` retains at most 64 exact completed predictions and one
bounded upstream scattering cache. Strength, source, geometry, material and actual
mosaic changes rebuild dependent preparation. Complete G/L columns prepare both
pure laws independently, including eta endpoints. A mixed accepted mesh cannot
qualify an inactive component. `FiberIntegrationRule` declares inner cone order
separately from axial preparation and outgoing-angle acceptance.
The same rule declares `stitch_grid_size` (default 513, minimum 257); the fitting
trace uses that exact grid. `SpecularResult` retains immutable internal-phase and
zero-phase strengths in the input evaluator's units, and `KinematicScaleSpecularResult`
retains `zero_strength_A2`. Downstream handoff calculations consume those existing
values instead of invoking the same structure evaluator repeatedly. Other declared
observables and handoff-selection equations are unchanged.
Native projection preserves its immutable observation membership owner.

`native_search` profiles unguarded nonnegative scale in the full supported covariance and
fits guarded scale as a literal SLSQP nuisance coordinate. It accepts
independently owned Gaussian calibration blocks, separates assumptions, refits every
continuous nuisance at each discrete N and sweeps each profile in both directions.
Best evaluated, feasible and converged points are distinct. An unfinished lower admissible
point prevents a resolved-minimum claim. Training/synthetic observations prohibit historical
guard constraints. Conditional validation uses the original cross-covariance and verifies
declared disjoint groups; marginal group scores are correlated and cannot be summed.

`native_accuracy` compares fixed-scale predictions, parameter contrasts and profiled
objective contrasts against the actual objective observations. Guarded-objective checks
are required for guarded fits. The runner checks fitted centers for every N and compares
their numerical score offsets before reporting discrete-ranking agreement. All such
agreement is empirical at the declared probes, not a certified error bound. Exact mixture
boundary inactivity is reported without freezing coordinates or claiming identification.

The interface and measurement limitations are in `NATIVE_REFINEMENT.md`.
`scripts/refine_native.py` binds numeric plans to raw-acquisition, source-code and observation
hashes and writes one external NPZ with embedded manifest. It never promotes a candidate
automatically or turns an optimizer success, bound or raw profile into physical certainty.

## Explicit numerical measures and acceptance gates (v17)

`ParrattStitchStack.overlap_measure` and `CompiledParrattStitch.overlap_measure` default
to `sampled_log_median`, preserving the manuscript's sampled empirical prescription.
The opt-in `continuous_q_median` is a named extension, not a legacy match. Its scale is
the uniform-Q median of `A/B` on the complete interval `[5Qc,10Qc]`, where
`A=R_Parratt*zero_strength*Q^2` and `B=internal_phase_strength`. The public pure
`continuous_overlap_scale` interpolates A and B separately linearly and solves for the
level whose sublevel set occupies half the interval. Exact endpoints are included;
isolated zeros are permitted, but zero numerator/denominator segments are rejected.
The first divergence from compatibility is overlap normalization, after identical
Parratt and structure strengths. Automatic/fallback blend selection remains empirical
and potentially nonsmooth. Identity includes the overlap measure in addition to model ID.
Both public result types, `SpecularResult` and `KinematicScaleSpecularResult`, retain
the overlap measure independently of their originating stack or caller.

`FiberIntegrationRule.regular_q_bounds_Ainv` and `local_m0_q_bounds_Ainv` may
freeze conservative domains enclosing actual pooled source/region support.
Insufficient coverage raises. Regular integration has no ordinary-engine selector,
manual axial mesh or fallback. `strength_gauss_order` defaults to four; discrete
moment checks do not establish continuous axial accuracy. `pixel_error_rtol` and
`pixel_error_atol` control tiled native L1 parent/child indicators, allocating
absolute tolerance over all source/group/response-panel slots. Work, depth and
live patch-memory guards raise without claiming a qualified prefix. Spatial order
stays 16. Local-m0 retains explicit axial/Sobol and native-resolution angular
controls. Physical inputs, actual/proposal mosaic, strength revision and numerical
controls participate in detector identity. Schema v1 requires explicit migration;
old qualifications/checkpoints cannot carry forward.

Native result schema v2 separates `optimizer_candidate` from `selected`. By default,
`require_initial_qualification=True` stops requested fitting/profiling after a failed
initial screen, checkpoints `selected=None`, and exits with status 2. An explicit false
value permits execution after initial failure; that failure still prevents selection.
Every fitted N,
profile center and cross-N objective contrast requires its own numerical checks. Conditional
validation compares refined predictions with scales fitted on training alone. Any unresolved
alternative or better admissible profile point blocks selection pending joint refitting of
that N. Numerical agreement remains empirical at declared probes; it is not a certified
error bound or an identified physical estimate.

The native spatial projector may combine adjacent pixel columns into a wider rectangle
only when observation owner, exact weight and vertical bounds agree. Rectangle geometry
can be shared between observations, but each original membership weight applies once.
The same continuous Gaussian rectangle integral supplies the probability. This is an
internal partition change with no new result measure, raster or public API.

The optional reference-correction warm start remains specified in
`NATIVE_REFINEMENT.md`. Low/high rules both use the sole current engine.
Approximate proposals cannot select fits or supply exact recovery rows. Existing
physical acceptance and independent source/angular/axial checks remain binding.

The conditional stream may omit the local composite with explicit
`include_local_m0=False`. It still groups all original rods and uses the
original combined group/source count for the regular per-panel absolute
error allowance. Default full-detector behavior and physical populations
remain unchanged; a rod-subset detector has its own subset allowance.
