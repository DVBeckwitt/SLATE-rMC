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
The preserved research/full-image qualification gates are in
[VALIDATION.md](VALIDATION.md#historical-full-image-comparison-gates). Historical T44
implementation records are archived as described in [REPOSITORY_HISTORY.md](REPOSITORY_HISTORY.md).

Native spatial execution defaults to `auto`, with explicit `cpu` and `cuda` overrides.
`NativeSpatialExecutor` owns process/thread-local calibration and selection accounting;
it is an explicit performance resource, not physical model state. Auto uses unchanged
float64 Gaussian terminals and preserves support, weights, requested rules and adaptive
acceptance. CPU admission is recorded before execution; a selected CUDA failure never
retries on CPU. Calibration changes no scientific tolerances or qualification. See
[native execution](NATIVE_REFINEMENT.md#automatic-cpucuda-spatial-execution) for admission,
setup costs, checkpoint scope and the separate desktop CPU reservation.

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
It uses one scrambled two-dimensional Sobol net with at least 4096 axial nodes
and 32 conditional angular strata per node. The counts are
`N=2**local_m0_axial_power` and `S=2**local_m0_angular_power`; refinements retain
whole strata per batch and respect the declared total-node budget.
`local_m0_seed` is the direct scramble seed and `local_m0_replica` is explicit
response identity. Original source rows share that net within a replica.
Disjoint physical arcs retain their original proposal mass Z and weight
`Z/(N*S*pq*pphi)`, including the original attempted divisor.
Nonempty unresolved CDF arcs fail; empty support contributes zero without
resampling. It reuses canonical signed strengths, transfer and native Gaussian
probabilities. It does not qualify accuracy, shared regular error or a fit.
The default remains `resolved_cdf_gl8.v1`; regular quadrature is unchanged.

Importance inversion retains the 2e-15 CDF residual requirement. A selected
quantile above the half-turn is solved in a signed periodic chart, with compensated
2-pi translation; the original arc masses and Sobol strata are unchanged.
Wrapped-Cauchy evaluation leaves trigonometric range reduction to libm rather
than subtracting a rounded period near its peak. Node azimuths may therefore be
negative; their sine and cosine define the same physical direction.

Resolved angular preparation is compiled without fast-math and streamed in
16-axial-node slices. Global axial masses and the cumulative angular-node budget
are retained across slices. A prefix emitted before an exception is not a
complete integral.

`ConditionalStructureDetector.integrate_native_regions(..., source_state_indices=...)`
can project selected original source rows for caller-managed CPU parallelism.
The shared support calculation and original source probabilities are unchanged.
The caller must sum every disjoint source partition exactly once and reject an
incomplete set; a partial source integral is not a normalized prediction.

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

## Desktop native exclusions (U04/U04a)

`interactive/mask_state.py` owns immutable `NativeMask` state: decoded OSC SHA-256, native
`(rows, columns)` shape, monotonic signed-64-bit revision, sorted disjoint half-open row spans
and bounded provenance. Reason zero includes; reasons 1–4 annotate user exclusion, beamstop,
detector gap and saturation. Adjacent equal-reason spans are canonicalized. Acquisition UUID
ownership is supplied by the project; an identical source in another acquisition has independent
mask/history state. Original counts and source bytes are never edited.

Rectangle membership includes native pixel centers inside its bounds; polygon uses even-odd
crossings with half-open edges; brush is the union of closed native-radius capsules. Boolean
C-order `.npy` import requires exact native shape, uses True to retain existing membership and
False to assign the chosen nonzero reason, and records the imported bytes' SHA-256. No OSC
orientation conversion or inferred saturation rule is applied by this boundary.

The existing `exact_band_profiles` reducer owns masked sums, means and valid-pixel support.
Exclusions and nonfinite samples contribute neither signal nor support. Signed counts remain
signed; zero support is missing (sum zero, mean NaN). Display visibility cannot change these
values. Full-image reductions share one valid/safe plane; narrow reductions keep their declared
band/ROI bounds. Published native arrays, reason planes, inclusion and profile snapshots are
read-only. History checkpoints are compact immutable span arrays; undo/redo advances revision.

The desktop job identity binds project/acquisition UUID, data and mask revisions plus the global
generation. Publication also checks the current source, base mask and pending edit prefix.
Mask changes invalidate the panel's full/ROI/query caches; reasons, inclusion and exact profiles
are published together for the matching acquisition. Prior arrays survive unchanged. Preparation
keeps prior values explicitly labeled; exported inspection metadata names the mask revision and
provenance. Future observation/result binding must consume a frozen mask identity; this UI slice
does not create or modify those scientific products.

Mask state is persisted in desktop schema 6 through the existing bounded atomic writer. Schema
1–5 reads default to no exclusions. Runtime limits and user interaction are documented in
[interactive/README.md](../interactive/README.md#native-exclusion-editing); these desktop contracts
do not change the numerical package API version or qualify a physical fit.

## Desktop hBN qualification, review and inspection consistency

Saved hBN records must agree with the existing `fitting.hbn` qualification predicate: solver
success, rank five, scaled condition below 1e8, no active bounds, each of five ring counts at least
eight, each angular coverage at least 0.15, and maximum ring RMS at most 2.5 px. These are existing
thresholds. The solver and admission use one shared rule and shared residual/support/contact
statistics. Recorded values must obey original admitted seeds/bounds/units and native ring/sector
support. Import/Open checks derivable counts, coverage, ring/aggregate residuals and contacts against
the exact pack and canonical residual owner without refitting or calculating a new Jacobian.
Contradictory flags/statistics fail before publication; genuine unqualified records and unavailable
uncertainty stay honest. Unknown condition cannot support a qualified claim.

Pending visible hBN inclusion/reason edits block Fit before dispatch. Commit invalidates dependent
frozen readiness/selection, and explicit freeze retains discovered coordinates and identities.
Restore/undo-to-committed decisions restores corresponding readiness; displayed edits cannot be
silently reset by Show draft/control/history actions. Save/autosave/Open persists committed state.
Readonly result points carry `(result_id,row)` and use one immutable displayed record for inspection.
Changing the combo choice clears previous table/curves/overlays until explicit worker presentation;
result removal/reopen respects the same identity. Inspection never implies selection or geometry
adoption. These desktop corrections supply no new scientific/performance qualification.

## Desktop reviewed sample geometry (U05c/U08c/U10a)

A bounded sample session admits the complete 2-8-image manifest with hash-bound configuration/CIF,
raw OSC and decoded source identities, native shapes, commanded angles and per-image fitting masks.
Canonical discovery/indexing intersects user masks with its existing all-zero edge validity. Review
may exclude discovered observations with reasons; it cannot invent positions/reflections or relax
confident-track admission. A frozen pack retains exact native coordinates, covariance, wavelength,
rod/branch/track/image identities, ordering, source/mask revisions and review/hash provenance.

Fit calls the existing indexed-series owner with those frozen observations and no reindexing. The
launch snapshot contains actual shared/calibration correction seeds/bounds/fixed scopes, common delta,
complete zero-sum Helmert contrasts/prior and fixed solver defaults. Degrees convert once to radians;
metres and native pixels retain their named units. The existing delta/sample-normal-x gauge and
owner constraints apply. Pending visible edits block work until committed; input/mask/review edits
invalidate readiness. One global worker owns execution; cancellation clears obsolete pending work,
checks phase/residual boundaries and rejects late/stale publication. No GUI-thread solver join or
forced termination is introduced.

All point/table/inspection/plot bindings refer to one immutable result ID. Solver termination remains
separate from rank, conditioning, bound contacts and scientific qualification. Sample-only candidates
retain dataset/precision/downstream flags false when root/outer/heldout audits are absent. Existing
scaled singular/weak-direction diagnostics are retained; calibrated covariance/standard errors remain
unavailable. Explicit selection grants inspection, with no experiment-geometry mutation or downstream
admission. Import/Open reconstruct the existing result dataclass predicate and canonical coordinates,
residuals/metrics without fitting; a self-consistent hash alone is insufficient.

Desktop schema 12 adds the session while retaining schemas 1-11. Four results, 16 export references,
512 reasoned exclusions and per-document expansion caps supplement the unchanged prospective 1 MiB
project cap. Prepared/frozen/result JSON is losslessly packed with bounded strict expansion; logical
memory reservations include expanded data. Exact result/observation JSON and PNG/value exports require
new external destinations. Project writes protect all retained current/historical input and export
paths. Missing/changed historical sources cause explicit validation failure; general recovery and
scientific/performance acceptance are separate work.

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
every contributing top-exit state proves the positive direct-root gap. The configured all-root
macrobin path is an explicitly nonquantitative display preview. Source state order and weights are
preserved; wavelength-dependent evaluators are never collapsed geometrically.

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

The unused source-averaged per-rod `integrate_native_pixels(...)` proof terminal retired in the
October 8 cleanup; its implementation is preserved in Git. The detailed arbitrary-coordinate
evaluator, one-incident-state pixel integrator, native conditional integrator and stochastic
terminal retain their distinct live contracts.

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

## Desktop joint geometry (U08d/U10b)

The supported capture roster is exactly frozen hBN plus Bi2Se3 and Bi2Te3. Captures retain their
original source/config/CIF hashes, native coordinates, mask revisions, rod/branch/track identities
and review exclusions. Replacing a group is explicit; opening another sample editor never replaces
an earlier capture. Missing groups or unsupported materials block Fit. A genuine retained hBN
calibration candidate is the current owner's seed input; it need not be successful or newly fitted.

Controls use the canonical 21-coordinate order and units. Unchanged displayed values retain their
exact stored floats. Reduced-gauge references and absent PbI2 coordinates remain zero and readonly;
adjustable starts are never silently overwritten. Bounds are strictly ordered, starts are finite
and within bounds, and private hBN distance stays positive. The owner keeps existing solver,
conditioning, covariance and scientific qualification work. Cancellation is cooperative at phase,
residual, qualification and per-image boundaries; one existing worker handles all jobs.

Joint report import checks recorded structure/roles, rank/condition, bound/confidence consistency,
per-image/pooled metrics and the existing qualification verdict. Current-launch records additionally
bind exact frozen observations, initial/bound vectors and saved site/ring diagnostics. No import
recalculates a fit or Jacobian. Historical reports without desktop launch lineage cannot claim
starting values, frozen coordinates or current selection, and their views do not require live files.

Schema 13 persists bounded captures, controls, pending edits, four immutable results and 16 protected
file references. Strict bounded lossless expansion supplements the unchanged 1 MiB prospective
project limit. Expanded data participates in shared memory admission. Qualified joint handoff
save/reload uses the existing owner, verifies actual predecessor bytes and rebases once; it remains
GEOMETRY_ONLY. Import/export/selection never reinterpret it as an indexed fit or adopt it implicitly.


## Desktop physical editing and sensitivity (schema 14)

Schema 14 accepts earlier project schemas and adds at most 16 KiB of sensitivity request settings:
route, canonical parameter, stored unit, positive finite step, baseline hash, affected images and
observable. Transient previews are not persisted or selected as fits. Sample initial-estimate
provenance is a bounded optional JSON object; earlier sessions default to empty provenance.
Provenance does not change frozen observations, fit equations or the launch objective.

Detector intrinsic column/row tilts in configured drafts use configuration degrees and the
canonical compiler's radians internally. An absent optional detector_tilt pair defaults to zero
without rewriting baseline YAML/hash. All fit starts retain the owner's radians/metres/native
pixels, bounds, canonical order and active/fixed scope. Drag previews validate a complete rigid
change; release commits one history action. Escape cancels. Camera navigation changes no starts.
A hBN undo restores its original review revision with its corresponding frozen pack.

Explicit sensitivity uses one selected coordinate and baseline/plus/minus, at most 4096 feature
coordinates per view and 2 MiB per result, on the existing CPU worker with BLAS limited to one.
Configured Q maps use the existing 13 by 13 native grid. Frozen site/ring identities and held-fixed
values are exact; domain or branch/topology changes remain invalid and provide no derivative.
Source spread is unobserved by nominal geometry and has no sensitivity control. Request/context
identity guards reject obsolete queued work and late completions. Display textures and retained
thumbnail memory participate in existing resource admission.

Saved hBN proposals require genuine matching qualified calibration/input lineage. Sample proposals
require a successful recorded geometry result with active detector-reference column/row calibration;
standalone sample estimates remain explicitly unqualified. The direct-beam intercept, ellipse
center and detector reference pixel are different quantities. Off-panel estimates are not clipped.
Shared hBN adoption lists compatible targets and checks configuration/CIF, detector/beam settings
and acquisition source/revision before prospective project admission and one transaction. It leaves
each private distance and immutable result intact, clears affected readiness/selection and records
source result/observation hashes. Sample adoption applies the actual coupled pose/calibration starts
through their owner. Manual/direct-spot tooling remains available in its existing panel.

Sensitivity is local feature motion, not confidence. Marginal standard errors, measurement covariance
and parameter propagation remain distinct. No prediction bands are shown without a genuine qualified
full covariance and supported observable mapping; hard bounds never supply uncertainty.


## Desktop source relocation and historical sample recovery

Relink preserves the saved reference-byte identity, including a configuration's dependent CIF.
Any later reference hash must still match that identity. OSC raw/compression bytes remain distinct
from the decoded OSC identity: decoded identity and raw-byte stability are checked before binding.
A same-path successful revalidation refreshes source/reference readiness and selected-image
admission without manufacturing a persistent edit or undo action. The original worker/review
context still gates publication; changed bytes and stale receipts cannot mutate bindings.

Sample saved-record admission checks the record/launch hashes, exact canonical review pack,
typed discovery and result structure, recorded qualification flags, named scopes/bounds/trims,
frozen observed points and finite residual shape/consistency. It does not claim live numerical
validation when original files are unavailable. Missing/unreadable/different source bytes alone
produce explicit transient unavailable readiness; malformed records or scientific/model validator
failures still reject Open atomically. Qualified/unqualified states and historical values remain
unchanged. No availability label grants downstream qualification.

Current and historical input identities have separate bounded transient checks. Missing-live
history stays inspectable; active selection/Fit is blocked until the matching current launch and
live readiness are available. Full geometry/provenance/prediction/residual/metric checks remain
mandatory on restored live record validation, import, export and supported active use. Explicit
Revalidate saved source identities uses the existing admitted worker and context checks, without
preparing observations or solving. Revalidation does not rewrite recorded inputs or fit outcomes.


## Desktop comparison replacement and copy confirmation

A locked comparison has one shared `(low, high, contrast mode)` in raw native counts. Independent
limits bind both acquisition UUID and decoded source identity. Replacing either slot admits that
image's own independent limits and reapplies the shared tuple. Pending/failed replacement retains
the prior visible snapshot under shared limits and disables export; stale admission changes nothing.
Changing contrast during incomplete replacement retains the authoritative shared tuple. Returning
to the previous cached image preserves its independent limits, including unlock while pending.
Save/Open preserves shared intent and current independent views; no units, normalization, mask,
support or detector-coordinate transformation changes. Compatible navigation retains its own gate.

A confirmed immutable copy plan carries the project UUID, revision, selected acquisition and exact
target roster from review. Deferred dispatch compares that captured context before any worker
launch or destination write. A mismatch discards the confirmation and requires review again. An
unchanged context uses the existing no-overwrite, byte-verified copy and atomic binding path, with
its original identity, dependent-CIF, cancellation and stale-publication checks.


## Desktop joint handoff original predecessors

Save handoff requires a retained immutable qualified result with an original captured launch.
Requested Bi2Se3/Bi2Te3 manifest and detector-base paths must equal that result's paths. All
captured manifest/configuration/CIF/OSC and hBN source/dark identities must still match. Ordered
image IDs and commanded angles must equal the chosen result's capture; current captures never
substitute for historical predecessors. Admission failure writes neither output and changes no
export references. Paired publication collision/cancel removes only unchanged files created by
that operation, preserving preexisting or changed destination bytes.

The canonical document builder accepts immutable report bytes and optional expected predecessor
hashes; configuration and manifest parsing use the bytes that supplied their hashes. The format
and GEOMETRY_ONLY status are unchanged. Later file changes cannot become bound predecessors
silently: reload rejects serialized identity mismatch. There is no filesystem-wide lock or atomic
two-file rename guarantee. Historical launch=None records retain exact inspection/export and
existing handoff verification. The latter proves serialized byte bindings and the owner geometry
contract, not reconstructed original fit ancestry.


## Prepared draft descriptions (schema 15)

Prepared inspection binds physics JSON, observations JSON, original plan JSON, numeric NPZ and
raw-acquisition bytes. The typed owners verify projection revision, frozen corrected counts and
full covariance. Profiles retain raw/background/signed corrected counts, validity and frozen
`observation SHA256:row index` IDs in original order. The full count-plus-background covariance
is retained externally in memory; displayed marginal standard errors never replace it. Arrays
are bounded readonly copies and are not persisted in the project. Missing/changed original files
leave descriptions inspectable and reload unavailable; wrong bindings reject before publication.

Every current/history description has a digest over exact input identities, full ordered plan and
parameter/model definition. Definitions record actual model names, canonical units/owners, source
revision and loaded FitParameter declarations, including physical/search bound kinds. Displayed
units equal canonical units. Added/removed coordinates, changed units/owners/domains/bounds/scales,
fixed-state changes or engine/model revisions block old-start reuse. Pure reordering aligns by
owner/name, never index. Authoritative loaded starts require explicit new-plan review; historical
fields remain exact. At most eight old descriptions and sixteen export references are admitted.

Pending edits retain stable identities and selected start index, including invalid text until
commit/discard. Commit uses FitParameter and the existing non-solving search validator, preserving
fixed values and final stage order. Physical bounds and active/fixed scopes cannot be silently
released. Unavailable parameter/method metadata remains inert. This is structural draft checking,
not full scientific domain/gauge/qualification or full engine launch admission. Run and indexed
adoption remain unavailable pending R4; no equivalent route around the parked script patch exists.


## Portable archives and exact exports (schema 16)

Schemas 1-15 remain readable with empty storage. A versioned storage object is bounded to 128 KiB
and 256 absolute original/stored path pairs with raw SHA256/size and archive identity. Expected
predecessor hashes must agree; relocated bytes still pass existing owner checks. Original
scientific bytes are never rewritten or assigned a derived document's digest. Compressed OSC
archive members retain `.osc.gz` for the authoritative reader.

Review displays selection, file identities/sizes/total, unavailable references and selected versus
historical qualification. Export binds current document/selection/epoch across the file chooser
and deferred dispatch. Sources are rehashed before copying, compared while streaming and rechecked
before publication. Missing selected dependencies block self-contained export. Future unavailable
R4-R6 products are separately labeled. Retained result input closure is distinct from current draft.

Limits: 256 files,512 MiB per file, 2 GiB expanded data, 1 MiB project, 512 KiB manifest and shared 4 MiB
requests. Existing CPU/resident-state admission and one worker bound file work. Disk admission
adds bounded documents and 16 MiB margin. Import rejects unsafe/colliding Windows names, links,
encryption, incomplete inventory, size/hash mismatch, unsupported version, contradictory
qualification and inconsistent selection. Existing numeric/JSON formats never execute checkpoints.

Archive publication uses a flushed/fsynced owned sibling temporary file and atomic no-overwrite
hard link where supported. Import verifies an owned staging directory before Windows rename to
a new destination. Unsupported filesystems can reject publication. There is no filesystem-wide
lock or multi-file crash-recovery guarantee. Failure/cancel cleans only owned new staging paths.
A completed directory may remain when later GUI context rejects its reference; it is neither
silently adopted nor removed. Open separately applies ordinary dirty/pending-state handling.

Prepared exact measured NPZ export contains raw/background/signed corrected counts, validity,
frozen row IDs, marginal errors, full covariance and provenance without normalization/prediction.
Reference undo/redo retains pending text. Existing figure exports retain paired numeric values.
Configured figures still require reviewed new filenames in an existing external output directory
through their owner; archive relocation does not rewrite the output declaration.


## Desktop named attempts and independent recovery (schema 17)

- `attempts_json` is `slate.attempts.v1`, at most 128 KiB inside the 1 MiB project. It holds
  at most 128 display names, one optional stable inspection key, at most eight configured and
  eight native result references, and an optional inherited source-project UUID.
- Keys are `(route, owner UUID/definition SHA256, exact result UUID/description identity)`.
  Simulation draft identities exclude only top-level storage paths; scientific declaration,
  hashes, units, revision and provenance remain bound. Result identities use the NPZ SHA256.
  A name/inspection change never rewrites the original record or grants qualification.
- Prospective project admission checks catalog bounds before adopting an output. Historical
  reference removal is explicit and never deletes external evidence. Archive selection filters
  route catalogs; inventory includes retained snapshot input closure independently of current
  drafts; relocation updates storage paths while preserving stable keys and scientific records.
- A duplicate has a fresh project/recovery UUID and a reviewed new destination. Its historical
  sessions, observation/result identities and parameter definitions retain original provenance
  as inherited inspection. Current reuse/selection uses the existing owner validators. Exact
  files can be explicitly shared; portable archive provides copied file storage.
- Duplicate publication never overwrites a destination. Reviewed duplicate/recovery Open binds
  raw SHA256 and parses those same bytes before replacing current state. Recovery candidates
  must have matching UUID filenames in the configured root, with at most 32 drafts/32 MiB total.
  Partial/malformed candidates cannot replace valid state. Missing/changed sources remain
  explicit; only existing exact-identity Relink can establish a restored binding.
- Recovery overwrite/retirement requires the exact owned/reviewed prior SHA256. A preceding
  queued publication updates ownership before the next dispatch. Unknown/changed bytes are
  preserved. Atomic publication checks cancellation before commit; after commit a complete
  copy may remain, but obsolete work cannot adopt it. No solver resumes, and no filesystem-wide
  transaction, concurrent-writer lock or power-loss guarantee is provided.
- Inherited sample Open performs structural original-record validation and labels live
  revalidation unavailable, as archived Open already does. Active selection/export keeps the
  existing canonical geometry, physical, covariance/rank and qualification checks.


## Compact Live Simulator dashboard

The detector presents input match/history, presentation versus quantitative state,
explicit recorded completion/progress, observable and qualification independently.
Configured workers record execution completion only at their successful terminal return;
older snapshots retain unknown completion. Completion never establishes convergence.
The numerical color legend describes the actual shader limits, mode and active display
aggregation. Display exposure changes limits only; clipping uses positive display cells.

Inspection connects existing Hold/Follow, exact Profiles/ROI and exact export owners.
Saved configured native-cell results can be reopened for comparison through the existing
hashed-result reader. Compatibility requires the same declared measure, entire instrument
geometry/frame/support and native shape; no alignment, rebinning or model/count residual
admission is implied. Shared display-sum bins, mode and limits are explicit and fixed across comparison resize/zoom.
Both panes reuse their worker-prepared sums at the same selected bin; Auto 99% uses only
the displayed snapshot as one shared scale anchor. Manual/full-range controls affect both
panes without normalization; exact native arrays and identities remain unchanged. Recorded-launch export uses
the displayed frame's immutable declaration, including historical inputs, and invents no
desktop Monte Carlo CLI equivalent. Native saved comparison without a bound geometry
identity remains unavailable.

Fitting preparation follows Inputs, Exclusions/counts, Geometry review, Model/stages and
Results. Typed stage selection edits the same pending declaration as Advanced; parameter
order, owner/unit/scope, fixed/gauge definitions, physical bounds and final-stage ordering
remain authoritative. Native execution/results remain unavailable; complete covariance is
retained by its existing owner. Review actions do not adopt new geometry into frozen inputs.

The desktop Simulator defaults to the side-by-side Workspace with one primary Run
action, adjacent Live and Stop visible during pending/running/timed or Live work.
Project, Display and apparatus View menus reuse existing action owners. Exact saved
image and Advanced have direct Alt+O/Alt+A routes. Common selected-device fields precede
derived incidence and a one-click mosaic summary; Sample/Mosaic expands exact mosaic
controls. One scrollable inspector keeps fields readable at compact/scaled sizes.
The compact scientific line retains history, preview/quantitative, declared completion
and qualification; full schemas/hashes/edge-bin/clipping details remain inspectable.
Inspection disclosure allocates a resizable pane; short comparison headings keep exact
frozen identities in Details and preserve B035 common-bin/limits/resource ownership. Quick controls write through the existing
configured/native schema editors to the same immutable draft and bounded undo history.
A slider drag commits on release; numeric text pauses coalesce after 300 ms. Exact
numeric text is authoritative. Slider span and step are user navigation settings,
not physical bounds; canonical constructors determine validity. Lorentzian probability
retains its actual [0,1] domain. Other vector coordinates, axes/pivots and optional
fields remain intact. Advanced retains complete schemas, routes and file/history/
inspection/export/transfer actions.

Incident-angle convenience applies only to a single +X active rotation with a +Y
beam, identity zero/sample rotations and an angle in [-90,90] degrees. Otherwise the
sidebar exposes declared axis angles. Worker admission derives signed mean-air-ray
glancing incidence from the canonical LAB-to-SAMPLE vector transform; positive means
toward the sample. This readout is neither a source-ensemble mean nor an internal-film
angle. Native geometry uses its declared transforms; full rotations remain in Advanced.

Live requests complete canonical validation then the existing Simulator Run. The
existing global worker and one newest pending snapshot remain authoritative; project,
epoch and generation reject obsolete publications. Stop, Live off, workspace exit,
route changes, Open and Close cancel timers/requests and invalidate late continuation.
First visits admit the existing starter and request one diffraction pattern through
the canonical update worker, with Live off. Stop, leaving or Open consumes/cancels
this startup request; repeated visits never restart it. Saved/imported/recovered
projects retain exact settings and Live off. Show saved image reopens a bound snapshot
by its exact hash; image-less drafts use Run without replacement by defaults. Validation failure does not retry unchanged
input. The last genuine image stays historical while updating; profiles/export metadata
remain bound to that displayed frame. Live performs no automatic configured figure
export and preserves the explicit export preference. Defaults remain nominal 64 source
samples, 4 workers, 8 draws/source and detector seed 1729; no convergence/time claim.

Legacy Geometry/Mosaic/Detector/Beam grouping informs presentation only. Native model
specimen coordinates provide supported lattice/site-displacement controls; configured
CIF input does not invent independent lattice/Debye sliders. Legacy pruning, old optical
equations and unsupported features remain absent. Native Simulator execution is separate
from the still disconnected fitting Run/adoption/stage-result hooks.
