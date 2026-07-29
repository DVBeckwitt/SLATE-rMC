# Contracts

Contract API version: **12**. Trace schema version: **4**. Reference pack version: **1**.

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
| `SourceConfiguration` / `sample_gaussian_source_rays` | configured pipeline / sampling | validated source parameters and immutable sampled source rows |
| `IncidentSampleBatch` | sampling | complete source rows, empirical weights, wavelength, polarization, provenance |
| `InstrumentConfiguration` / `CompiledInstrument` | geometry | canonical transforms, sample support, detector shape/pitch/reference coordinate, revisions |
| `IncidentStateBatch` | geometry | entrance-intersected air and film-phase `ki`, Fresnel amplitude, decay, footprint, source identity |
| `MaterialOptics` | materials | wavelength-aligned complex refractive index and material revision |
| `RodCatalog` | reciprocal | every physical `(h,k)` rod and exact family metadata |
| `RodQueryBatch` | ordered/stacking | rod-aligned `L` queries with stable IDs |
| `EventIntensityResult` | ordered/stacking | query-aligned amplitude, intensity, normalization, and model revision |
| `ParrattResult` / `SpecularResult` | reflectivity | separately named Parratt, kinematic, and composite specular outputs |
| `MeasuredPeakDiscovery` / `MeasuredIndexingResult` | selection | hashed image/mask/calibration provenance, native coordinates, reciprocal labels, decisions, and replicated branch tracks |
| `ConfiguredGeometryInputs` / `GeometryOnlyEwaldContext` | configured pipeline | one nominal ray, material optics, reciprocal basis, rods, and instrument; no strength or mosaic object |
| `OscGeometrySeriesConfiguration` / `OscGeometryIndexingRun` | selection | strict IDs/paths/commanded angles plus one provenance-bound frozen selection and retained fit-ready models |
| `IndexedGeometryImage` / `IndexedGeometryFitResult` | fitting | one frozen image block and one selected subset of the shared nine-coordinate correction pack, with fixed-coordinate provenance, per-image metrics, and rank diagnostics |
| `MosaicProfileSet` / `MosaicComponentProfileBank` / `MosaicProfileFitResult` | measurement/fitting boundary | finite-bin `S`, `N`, validity and angle layout; exact pure-component responses; fitted mosaic parameters, nuisance scales, and identifiability evidence |

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

- `evaluate_detector_geometry(column_px, row_px, *, include_surface_jacobian=True)` performs detector
  point → outgoing ray → exit refraction → film `kf` → sample-frame `Q` and reports
  validity and elastic residual. The ray is valid only on the active front face,
  `n_D dot kf_hat > 1e-14`; back-side and tangent approaches are rejected before optical or
  reciprocal work. Passing `False` skips derivative work and returns a zero Jacobian when only ray
  validity or `Q` is needed.
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

`compile_monte_carlo_sampler(*, execution_backend, seed)` creates the explicit mutable,
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

### Finite-bin mosaic-profile fitting

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
any predeclared bin exclusions are immutable and must agree throughout a component bank.
`MosaicProfileSet.source_revision` is the detector result's actual source realization revision; a
profile-layout hash cannot substitute for it.

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
An optional `MosaicProfileNuisanceBasis` is bound to the exact profile revision as well as the
identities and valid-bin mask. Its coefficients are unconstrained signed detrending terms, not a
claim of a nonnegative physical background. A component response with numerically absent
nuisance-projected energy is treated as zero; on a mixture face where a profile then has no model
energy, its fitted scale is exactly zero and it contributes its full normalized observed residual.
A width pair with one zero-energy component and one nonzero component in the same profile is
rejected because its apparent boundary optimum can be an unattained limit with divergent scale.

For every individual `(dataset, reflection, analytic branch, root side)` profile, the fitter
analytically profiles an independent nonnegative amplitude and minimizes relative squared shape
error. Absolute peak heights, structure-factor amplitudes, and cross-peak intensity ratios therefore
cannot determine the mosaic distribution; every admitted weak or strong profile has equal
profile-level leverage. Relative intensity across bins within one profile remains essential mosaic
information. Gaussian sigma and Lorentzian HWHM are selected from exact component responses; eta
is searched on both exact faces and a scale-centered logit coordinate. Repeated width pairs are
cached during deterministic refinement. A fit fails with a typed identifiability error when the
nuisance-projected local sensitivity is deficient or when the finite global audit finds distinct
tied solutions. The 8,193-point stationary audit is deterministic numerical evidence, not a formal
theorem excluding arbitrarily narrower sub-grid aliases.

Data-quality screening, when requested, occurs before fitting and removes an entire profile under
a frozen, provenance-bound policy. It may use selection-only sidebands to classify weak profiles
and explicit full-profile identities to classify known secondary lobes. It may not trim selected
phi bins according to measured intensity. Once admitted, every profile has the same profile-level
leverage regardless of absolute intensity.

A branchless measured `m=0` profile additionally requires positive finite signal from the complete
source-averaged simulated profile. This support check precedes weak/secondary screening and is
recorded in the selection audit; a nominal one-state landmark alone is not intensity evidence.

`MosaicProfileDefinition.excluded_phi_bin_indices` is a frozen, provenance-bound numerical-support
mask, never an intensity threshold. The Bi2Se3 proof excludes 43 independently audited inverse-
support boundary bins from both truth and every component response, while retaining every one of
the 32 profiles and at least 38 bins per profile. A general topology-split cubature that retains
those boundary bins is not yet implemented.

### Fixed-position ordered-intensity fitting

`compile_ordered_intensity_response(...)` freezes detector geometry, source, optics, mosaic,
physical rods, finite angular ROIs, and the atomic coordinates. Each regular inverse root retains
its observation index, physical rod, exact `L`, root sign, fixed mass coefficient, and crystal-frame
`Qr^2/Qz^2`. It also compiles the six coefficients of the homogeneous occupancy quadratic for the
three unique Bi2Se3 source sites. No detector raster, pixel deposition, detector reprojection, or
full structure-factor calculation occurs in an optimizer iteration.

For the finite-ROI response, each profile is the integrated contribution of the rods named by that frozen
reflection group. It is therefore a selected-group component mass, not the unseparated total counts
inside the same raw detector ROI. A real-image consumer must provide a provenance-bound component
extraction/deblending step or prove that omitted rods and background are below its declared error
budget. The current synthetic Bi2Se3 proof uses exact selected-component observations; it does not
claim raw-OSC intensity recovery.

For fixed Bi and Se2 Wyckoff coordinates, every term is evaluated as

```text
S = [oBi^2, oSe1^2, oSe2^2, oBi*oSe1, oBi*oSe2, oSe1*oSe2] dot C
I = fixed_mass * S * exp[-Ur*Qr^2 - Uz*Qz^2]
```

The coefficient compiler retains the configured finite-stack normalization and stacking-disorder
law. `predict_mass_direct_A2(...)` evaluates the authoritative CIF-derived quintuple-layer
amplitude and configured finite-stack model on the same frozen roots and exists as the independent
acceleration oracle; fitting uses
`predict_mass_A2(...)` only. Within every admitted reflection group, weak and baseline-extinct roots
are retained. The nominal integer-L marker catalog admits sites from geometry and detector topology
only; its aligned strength arrays are diagnostics that may be exactly zero. A zero-occupancy control
must therefore produce the same profile identities as the CIF baseline even though every diagnostic
strength vanishes. The tracked proof also freezes the expected per-incidence group counts.

`fit_ordered_intensity_series(...)` accepts an explicit active-name tuple and freezes its exact
complement. This first phase permits only the three occupancies and `Ur/Uz`; attempting to activate
either Wyckoff coordinate fails explicitly. Absolute-calibrated data may fit all three occupancies.
With one analytic scale per image, their common multiplier is an exact gauge, so one occupancy must
be fixed as a positive ratio reference and only occupancy ratios may be interpreted. Ratio
coordinates are nonnegative and may exceed one. The returned Bi2Se3 parameter object is merely the
admissible representative obtained by dividing all ratios by their maximum; frozen occupancy names
refer to the optimizer's ratio coordinates, not to the representative's numeric fields. Mixed
nonzero-`m` radial families are required to distinguish `Ur` from a common intensity scale. Within
one call, every admitted `m=0` and nonzero observation must use the same declared kinematic measure:
either finite-ROI mass or source-averaged selected-center angular signal density. Parratt, composite
specular, and mixed-measure observations are rejected.

Every response carries a derived digest of its immutable roots, coefficients, `Qr/Qz`, detector
weights, wavelength, profile identities, and source/material/geometry revisions. Its structure and
physical mosaic compatibility revisions are checked across the simultaneous series. Observations
are immutable dataset-ID-bound records with exactly one declared measure. ROI-mass observable
revisions hash profile identities, angle-frame layout, finite ROI boundaries, bin count, and frozen
topology exclusions but intentionally exclude quadrature order. Thus a refined truth/data evaluation
and a cheaper converged prediction may share one declared observable while retaining different
numerical response digests. Selected-center records instead hash their frozen center/group layout.
Tuple position is never an authority. Profile
rod-catalog revisions must equal the detector's
configured catalog before response compilation. Different wavelength-specific response digests and
reach-limited rod catalogs are allowed when the shared structure and mosaic remain compatible.
The reported rank, condition, and correlation matrix are local inverse-sensitivity diagnostics,
not statistical covariance or uncertainty: this phase declares no noise model. One bound-contact
flag is reported for every active coordinate and accepted deterministic recoveries require all of
them to be false.

The finite-ROI response compiler applies a fixed geometry-only 17-by-17 inverse-root signature probe
to every phi bin and freezes a conservative exclusion mask before quadrature. This finite probe is
not a general topology certificate. Each new material/ROI set must additionally pass an independent
response-order convergence check; topology-split cubature remains future work.

`compile_source_averaged_ordered_intensity_response(...)` is the distributed-source companion for
the synthetic fixed-position proof. For each incidence it evaluates one detector function

```text
D_a(c,r) = sum_s w_s D_a,s(c,r)
```

before selecting rods, applying a dataset scale, or constructing any residual. It exposes frozen
selected-center angular signal density in `A^2/rad^2`, not finite-ROI mass. The response therefore
uses `OrderedIntensityPeakCenterObservations` and measure ID
`selected_group_angular_signal_density_A2_per_rad2.v1`; ROI-mass observations cannot be mixed into
the same fit. Nonzero centers are frozen nominal integer-L markers, whereas admitted `m=0` centers
are frozen observed coordinates, so the common term is selected center rather than nominal peak.

At 13 canonical Chebyshev-Lobatto `Uz` nodes, six occupancy probes are evaluated through the full
source-averaged, all-root detector. Their homogeneous occupancy quadratic is exact. `Qr^2` is exact
and constant within each accepted symmetry group; the noncommuting source/root variation in `Qz`
is retained by the full detector evaluations before interpolation. Every compiled response must
pass 12 interlaced full-detector validation nodes at maximum occupancy-contracted signal error
`2e-3` or fail. At each profile and validation node the exact and interpolated homogeneous
occupancy quadratics are converted to symmetric three-site matrices `Q` and `Qhat`. A profile-local
generalized-eigenvalue bound covers every real occupancy direction `o` (and therefore every allowed
nonnegative direction): `|o.T@(Qhat-Q)@o| <= rho*(o.T@Q@o + 1e-12*S_p*||o||^2)`, where `S_p` is
the largest direct spectral norm across that profile's validation nodes. No scale is pooled across
profiles; the additive term explicitly governs exactly extinct or numerically rank-deficient modes.
The response compiler contract revision is
`source-averaged-selected-center-occ-quadratic-chebyshev-qz-spectral.v2`. Recovery schema v3 stores
that revision and the `1e-12` floor, and rendering rejects an older or missing contract before any
detector evaluation.
Its digest binds the source count/revision, instrument, material, mosaic, rods, selected centers,
all-root policy, interpolation certificate, backend, and device. A joint caller may require an
exact source count and realization; the tracked proof requires the same 250-state realization for
all three views. This is an accelerator for synthetic selected-component recovery, not a raw-OSC
intensity extractor.

The unfiltered synthetic mode retains every weak or baseline-extinct nonzero geometry anchor and
all six admitted branchless `m=0` anchors. When a measured-mosaic handoff is supplied, its exact
fit-eligible identity set is authoritative, so its weak/secondary-lobe exclusions propagate into
the synthetic selected-center structure-factor recovery. There is no per-source scale or residual:
selected-group rod restriction is linear, and all retained source/root contributions are summed
before the one dataset-scale projection and joint three-incidence residual.

The measured-mosaic handoff uses schema `rasim-bi2se3-real-mosaic-fit-v2` and support-gate revision
`positive-combined-detector-m0-profile-signal.v2`. Every candidate selection record carries its
combined-source modeled signal and Boolean support decision. Ordered-intensity consumers reject
the earlier schema or a missing/mismatched gate revision. They also reconstruct the support
decision from an integral `family_m` and a finite, nonnegative signal, rejecting inconsistent
records instead of trusting serialized Boolean state. The v2 gate includes raw-significant
nominally unsupported `m=0` observations as provisional candidates, and consumers require every
fit-eligible identity and dataset to be present in the compiled response.

Before compiling an admitted `m=0` anchor, the tracked runner evaluates its baseline structure
through the complete source-averaged selected-group detector and requires positive finite modeled
support. Raw significance remains the upstream observation gate; this forward check prevents a
nominal one-state landmark alone from establishing intensity support.

### Configured simulation

- `load_simulation_config(path, repository_root=...)` accepts one strict
  `rasim-simulation-v2` YAML document. Unknown, duplicate, aliased, or missing fields fail.
- `build_configured_simulation_inputs(config)` creates source rows, canonical incident states,
  material, rods, finite-2H strength, and Bragg space once.
- `build_configured_geometry_inputs(config)` creates only source rows, material, reciprocal basis,
  rods, and compiled instrument. `build_geometry_only_ewald_context(...)` adds one nominal incident
  state. Neither boundary constructs structure strength or mosaic probability.
- `sample_configured_source(source, sample_count=...)` is the one mapping from validated configured
  source parameters to the canonical source sampler, including the exact one-row nominal state.
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
exact fit-ready models from that provenance-bound run. Model, discovery, data, mask, policy,
commanded-angle, and indexing-context hashes are checked together; corrected global-rediscovery
runs deliberately expose no fit-ready images.
Missing, extra, duplicate, angle-mismatched, or image-provenance-mismatched IDs fail before fitting.
Geometry-only inputs recompute and require the exact configured one-row nominal source revision;
each fit-ready image also recompiles and compares the complete detector, sample, axis, and pivot
state before the residual hot path is admitted.

`fit_indexed_geometry_series(...)` owns one complete nine-coordinate vector shared by every image,
concatenates the existing canonical per-image residual blocks in sorted image-ID order, and uses
bounded TRF least squares. `SHARED_GEOMETRY_PARAMETER_NAMES` declares the only accepted coordinate
names and their canonical order. `fitted_parameter_names` selects any nonempty subset; names are
canonicalized before optimization and every omitted coordinate is copied bit-exactly from
`initial` into every residual evaluation and the final correction. The default selects all nine.
The complete pack is, in order:

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

The actual bound-scaled Jacobian must have rank equal to the selected coordinate count with
condition at most `1e8` before optimization. Singular values, weakest direction, and active-bound
flags have that same selected-coordinate length; the result explicitly reports canonical fitted
and fixed names. For the default qualifying Bi2Se3 pack the rank ladder is 5/9 for 5 degrees, 7/9
after adding 10 degrees, and 9/9 only after adding 15 degrees. Full rank does not imply precise
pivot recovery; the weakest direction must be reported. Beam center and lattice constants are not
members of this pack and cannot be activated accidentally.

`audit_indexed_geometry_series_roots(...)` brackets
`F(beta)=q(beta) dot (q(beta)+2 ki)` on its two monotone arcs without calling the production
integer-`L` root solver, assigns root sign from the oriented crossing and branch from the signed
axial derivative, and compares independent and production detector coordinates per full key. A
swapped beta/root assignment therefore fails even when the key set is unchanged.

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

### Portable staged-fit replay

`rasim-staged-fit-replay-v1` is a strict, material-case manifest for geometry, mosaic,
ordered-intensity, and optional render stages. Every path is case-relative and must remain inside
the repository. Every file has an exact SHA-256; OSC records additionally bind decoded native
shape, dtype, and `[row,column]` byte content. Paths nested inside geometry-series, simulation,
mosaic, ordered-intensity, and policy files must resolve to their corresponding declared roles;
hash-complete decoy roles are rejected. Bi2Se3 and Bi2Te3 cases require the declared
5/10/15-degree triplet, source seed 1729, one ideal source state for geometry, and the identical
250-state realization for mosaic and ordered intensity.

Each stage emits `rasim-staged-fit-replay-stage-v1` with case/material identity, case hash, backend,
its actual execution runtime, source identity, compact scientific state, and the immediately
preceding scientific revision.
`rasim-staged-fit-replay-certificate-v1` records the ordered revision chain and the verified
summary. Before creating output, the runner reloads and exact-compares the case, all role mappings,
all file hashes, decoded OSC identities, and nested consumed paths. It reads and hashes one
case-bound `uv.lock` byte snapshot, then
requires the complete transitive numerical dependency closure to equal that lock; a render replay
also requires the locked Pillow version and a successful `PIL.Image` import. The certificate's
`verification_runtime` records the verifier's lock hash, interpreter, platform, and installed
versions. Runtime is operational provenance excluded from scientific revisions, but resume requires
the stored stage runtime to match the current stage runtime exactly. Fresh and resumed stage results
receive the same exact envelope, recomputed-revision, artifact-contract, and upstream checks. A
verified stage must pass its partial scientific summary before its JSON is written or it can become
an upstream input. Resume rejects a changed case, backend, source, compact state, or externally
referenced artifact hash.

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

The Bi2Se3 ordered replay remains synthetic selected-component peak-center recovery. The Bi2Te3
ordered replay transfers measured-mosaic nuisance amplitudes onto baseline 250-source peak-center
signals and is model-limited `NO_ORACLE`; it is not direct count-calibrated raw-OSC structure
recovery. Optional named active-parameter bounds narrow the canonical ordered-intensity optimizer
bounds without activating frozen coordinates.

## Once-only factor ownership

| Factor | Owner |
|---|---|
| empirical source mass | source sampler |
| sample-footprint acceptance and entrance amplitude | incident transport |
| per-rod population | `MosaicBraggSpace` |
| CIF/finite-stack structure strength | ordered strength model |
| wrapped mosaic probability | `MosaicBraggSpace` |
| Ewald restriction / inverse-map determinant | Ewald or detector pushforward, by declared route |
| exit amplitude and uniform-depth attenuation | detector optical mapping |
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
