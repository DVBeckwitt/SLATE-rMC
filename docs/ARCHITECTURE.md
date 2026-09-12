# Architecture

## Design rule

The repository has one production path from a configured incident beam to a detector-native
observable. Scientific state is immutable, transformations are explicit, and the primary public
model is a callable function rather than a sampled cloud or image cache. Deterministic integration
and weighted Monte Carlo are explicit terminal estimators of that shared physics.

```text
strict YAML + CIF
  -> weighted correlated source rows and declared spectral lines
  -> canonical entrance transport and incident film-phase ki
  -> physical reciprocal rods + continuous mosaic/finite-stack strength
  -> analytic rod/Ewald roots, exit optics, illuminated-path weight, and external-path attenuation
  -> per-source, per-physical-rod continuous detector-coordinate density
  -> all-source, all-rod, all-root intensity sum on the detector function
  -> selected-center comparison, display-only center sampling, or detector-pixel box integration
  -> optional terminal alternative: weighted forward samples -> exact native-pixel owners
```

No retained Ewald-sphere mesh is physical model state. The intrinsic callable can evaluate the
exact almost-everywhere density at arbitrary internal-film outgoing directions over the complete
sphere. Independently, a detector coordinate determines an outgoing air ray; exit refraction
determines its film wavevector; the elastic relation and inverse latent map recover all
contributing rod/orientation branches on the active detector panel.

## Package ownership

### `painted_ewald`

- `types.py`: immutable rods, mosaic parameters, root status, and strength protocols.
- `rotations.py`: column-vector active rotations.
- `mosaic.py`: normalized folded-alpha/full-beta probability law and compact quadrature helpers.
- `normal_density.py`: opt-in spherical-area mosaic law reusing those component densities and
  quadrature; one shared probability for directed rods and signed local-lamella specular transfer.
- `rods.py`: detector-independent elastic-reach rod enumeration.
- `bragg.py`: continuous `(alpha, beta, u)` map and per-rod mosaic × population × strength density.
- `ewald.py`: stable analytic line/sphere roots with tangent and no-root status.
- `surface.py`: intrinsic Ewald restriction and its once-only coarea factor.

### `rasim_next`

- `core`: frames, units, immutable contracts, validity, transforms, and trace records.
- `geometry`: instrument compilation, source/sample intersection, incident transport, detector rays,
  and forward detector intersection.
- `materials` and `optics`: CIF-derived material data, shared complex-normal mode selection,
  refraction, Fresnel amplitudes, uniform-depth film attenuation, flat-film illuminated-path
  weight, and optional external detector-path Beer--Lambert attenuation.
- `ordered`, `stacking`, and `reflectivity`: structure amplitudes, finite stacks, stacking models,
  separately named scalar specular calculations, and the immutable optional Parratt handoff state.
- `reciprocal`: reciprocal basis and complete physical rod catalogs.
- `sampling`: deterministic weighted Gaussian phase-space sampling, optional per-axis
  position--divergence correlation, and legacy Gaussian or discrete-line wavelength laws.
- `pipeline/bragg_space.py`: binds CIF- or adopted-lattice ordered-parent strength, including the
  supported 2H and R-centered 3R catalogues, to rods and mosaic geometry.
- `pipeline/continuous_detector.py`: shared inverse latent pushforward to intrinsic internal-film
  solid angle or one-incident-state detector coordinates, plus native-pixel integration.
- `pipeline/beam_position.py`: conditional Gaussian source covariance, affine sample/detector
  projection and one scalar pixel-probability quadrature shared by CPU/CUDA forward terminals.
  It changes only the terminal position integral and reuses the canonical root and optical work.
- `pipeline/source_averaged_detector.py`: incoherent summation of complete incident-state detector
  fields, linear rod restriction/physics rebinding, compiled CPU/CUDA evaluation, and an optional
  wavelength-resolved `(0,0)` stitch compiled once per source wavelength and retained unchanged
  across candidate structure rebinding. Its explicit interface convention is either
  `local_lamella_follows_mosaic.v1`, which owns the complete local-reflection inverse map and rejects
  forward Monte Carlo, or `fixed_external_qz_m0_strength.v1`, which replaces only the `(0,0)`
  strength on the regular inverse map and supports compiled CPU/CUDA and forward sampling.
  Nonzero rods keep the regular path; these interface assumptions are not interchangeable.
- `pipeline/source_averaged_structure.py`: sparse selected-coordinate source averaging with fixed
  detector transfer and revision-bearing candidate structure strength for mosaic and ordered-
  intensity fitting.
- `pipeline/incidence_acquisition.py`: canonical continuous-incidence support, traversal
  provenance, and normalized deterministic quadrature.
- `pipeline/incidence_angle_average.py`: calibrated reduction of complete fixed-angle all-root
  detector engines over an already compiled quadrature, with one shared source realization and no
  angle Monte Carlo or retained per-angle image.
- `pipeline/configured_simulation.py`: strict YAML boundary, canonical model construction, and
  quantitative pixel integration or display-only native-center density sampling. Its geometry-only
  input/context builders stop before structure strength or mosaic construction.
- `selection`: position-free angle-chart discovery, detector-native peak refinement, reciprocal
  identity inference, immutable cross-incidence branch manifests, strict OSC-series ingestion, and
  frozen-key post-fit visibility audits.
- `fitting/geometry.py`: callable pose-bound detector fields, their exact integer- or
  commensurate-rational-L tagged landmarks,
    the shared public site-plus-line objective diagnostic, detector-coordinate and line-angle pose
    fitting, rank diagnostics, and post-fit root re-enumeration.
- `fitting/adaptive_scan.py`: material-neutral finite-ROI node records and panelwise
  covariance-whitened angular refinement with frozen calibration, source, and evaluator identity.
- `fitting/staged_scan.py`: generic fixed/scan scores and fixed-first delayed acceptance with a
  bounded exact-scan work count.
- `fitting/pbi2_geometry.py`: ideal 2H/4H/6H parent-period support in one declared single-trilayer
  PbI2 metric, exact detector-locus deduplication, and hash-bound signed-rod/parent provenance; no
  structure intensity, population weight, mosaic, or measured-centroid ownership.
- `fitting/indexed_series.py`: exact image-ID joins, one shared nine-coordinate
  detector/sample/axis/pivot correction pack, one optional common additive incidence-angle delta,
  and optional zero-sum image trims represented by Helmert contrasts. Any identifiable subset may
  be active while the complement remains exactly fixed; the common delta and overlapping sample-x
  gauge cannot be active together.
- `fitting/mosaic.py`: immutable finite-bin profile identities and response banks, exact
  per-profile nuisance-amplitude projection, deterministic width refinement, centered-logit eta
  search, and local/global identifiability diagnostics. Its continuous-profile entry point is a
  fitting-boundary adapter over the canonical measurement transform, not another angle mapping.
- `fitting/stacking_intensity.py`: direct signed-rod PbI2 finite-parent strengths, exact-rational
  reflection-group aggregation, keyed intrinsic-strength observations, and deterministic
  constrained nonnegative population fitting. The rational adapter collapses detector-root sides,
  retains their canonical definitions as provenance, and sums unique rods, but is not a
  detector/source/optics response.
- `fitting/ordered_intensity.py`: fixed detector/mosaic sparse ROI-mass responses plus certified
  source-averaged selected-center responses, fixed-position Bi2Se3 occupancy quadratics,
  directional `Qr/Qz` damping, analytic image-scale projection, and structural rank/correlation
  diagnostics. The source-averaged path sums every incident state into one detector function per
  incidence before comparison and certifies its `Uz` interpolation against full-detector probes.
  Dataset-ID-bound observations carry distinct mass/density measures, while numerical-response,
  structure, mosaic, source, instrument, backend, and rod-catalog revisions preserve provenance.
  The full strength model is retained as a proof oracle, not called by optimizer iterations.
- `measurement`: downstream detector-derived observables, including the continuous normalized
  `(phi, 2theta)` coordinate pullback, per-rod all-root angular signal, the full finite-pixel angle
  projector, and an exact cropped physical-pixel projector for fully panel-contained measured
  profiles. It also owns material-neutral layered reciprocal frames, immutable `Qr/L` profile
  regions, sample membership, and separate finite-bin signal/measure accumulation used equally by
  measured pixel centers and continuous detector cubature; never part of raw rendering.
- `proof`: compact analytic, reference, mutation, convergence, and benchmark evidence.

## Public runtime layers

1. `MosaicBraggSpace` owns the latent reciprocal measure for every physical rod.
2. `ContinuousEwaldCoating` restricts one rod to one analytic Ewald root for intrinsic diagnostics.
3. `DetectorEwaldMeasure` uses one inverse-preimage implementation to evaluate the full
   non-specular intrinsic direction density, the configured detector-visible intrinsic direction
   patch including regular nonzero `m=0`, or the raw pullback onto arbitrary detector coordinates,
   and integrates one incident state's pixels.
4. `SourceAveragedDetectorEwaldMeasure` sums independent incident-state intensities before any
   selected-center comparison, display sample, or requested detector integration. Its optional
   stochastic terminal samples the same latent/source physics and streams weighted roots directly
   into half-open native-pixel boxes.
   `CompiledMonteCarloDetectorSampler` is an explicit mutable execution resource, never hidden state
   on that immutable model. It retains a prefix-stable accumulator and either four-or-fewer stable
   CPU blocks or one thread-confined persistent CUDA workspace.
5. `IncidenceAngleAveragedDetector` streams a fixed probability quadrature over complete
   detector-native all-root engine views. The optimized Bi2X3 path may reuse one immutable static
   engine and exactly rebind every calibrated geometry node. The wrapper has no representative pose
   and cannot consume a geometry-bound outgoing-angle chart or fold plan.
6. `configured_simulation` assembles fixed-angle engines from one validated YAML document and owns
   the provenance-safe optimized scan rebind; the incidence wrapper composes an already calibrated
   node series without owning fitting state.
7. `ContinuousDetectorGeometryModel` binds callable reference and trial fields while reusing packed
   structure/mosaic state; its private exact-tag geometry rebuilds only canonical incident,
   exit-refraction, and detector geometry for each landmark trial.
8. `ContinuousNormalizedAngleFunction` reparameterizes one bound detector field with an explicit
   coordinate Jacobian and separate `S/N` measures; it does not rasterize or alter detector physics.
9. The mosaic-profile adapter evaluates only frozen angular quadrature nodes, integrates `S` and
   `N` before division, and caches exact component responses by width. CUDA performs the expensive
   detector evaluations; the small deterministic profile search remains on the CPU. The measured
   path projects only cropped raw-OSC pixels into the identical local-bin layout with exact polygon
   overlap. A frozen sideband gate removes whole weak profiles, and explicit policy identities
   remove whole secondary-lobe profiles before the one joint fit; neither operation masks bins by
   their central-profile intensity. Explicit nonzero profiles use the frozen OSC indexing
   selection, while branchless `00L` profiles require raw-significant observed support and a
   representable fixed-model landmark. Geometry-audited inverse-support boundary bins may be frozen
   out identically from every simulated component.
10. Measured selection discovers peaks without predicted coordinates, refines them on the native
   detector, infers discrete reciprocal identities, and freezes replicated branch tracks before
   fitting consumes them.
11. The OSC-series boundary joins declared files, motor angles, geometry-only material contexts, and
   frozen observations by image ID. The joint fitter concatenates canonical per-image residual
   blocks while applying one shared rigid correction vector, one common incidence delta, and
   canonical Helmert trim contrasts whose per-image values sum to zero.
12. Post-fit proof brackets the fixed-`L` elastic equation independently of the production root
    solver, then relabels exactly the selected native candidates under corrected geometry. A full
    corrected-geometry rediscovery is reported separately as a chart/candidate robustness
    diagnostic; it cannot delete or replace accepted observations.
13. The retained historical synthetic ordered-intensity proof uses the finite-ROI mass path and a
   certified source-averaged
    selected-center path. The latter produces one combined detector function per incidence before
    any dataset scale or residual, retains every weak nonzero anchor and admitted `m=0` anchor, and
    contracts only occupancy coefficients and its historical global Q-damping factors. Atomic positions,
    geometry, mosaic, lattice, optics, and stacking law are immutable. Multi-incidence observations
    join by dataset ID and exact observable-layout digest rather than tuple position. The active
    Unfiltered Bi2Se3 proof mode uses 250 shared-revision source rows and `88/78/72` centers,
    including six `m=0`; a measured-mosaic handoff instead propagates its exact fit-eligible
    identity set. Both prove synthetic selected-component recovery, not unresolved raw-OSC
    intensity recovery.
14. The synthetic PbI2 rational-landmark SF boundary collapses detector-root duplicates to one
    intrinsic structural query, sums every unique signed rod once, and fits one fixed-parent amount
    vector and scale per specimen. Exact parent overlaps stay inside one response row. Detector
    transport and arbitrary transition-law refinement remain downstream work.
15. Native-center rendering samples the final combined detector density once per native pixel. It
    is display-only `A^2/px^2`, not pixel-integrated mass, OSC counts, or a count-calibrated fit.
16. The interactive Monte Carlo viewer owns one latest-only scheduler and one long-lived render
    worker. Detector-only controls compile one batched projection and swap only four projection
    buffers; even after a nonzero sample correction they retain the already-bound incident
    transport. Sample/goniometer controls rebuild incident transport and re-enumerate rods only
    when validity topology changes. CUDA previews normalize the raw float64 accumulator into one
    full-native float32 staging frame, and a persistent OpenGL R32F texture applies only
    masking/log/colormap presentation. Matplotlib is an explicitly selected full-native software
    fallback, never an automatic backend substitution.

The scalar NumPy path is the readable oracle. Compiled CPU and CUDA kernels reuse immutable packed
state and must reproduce it within the frozen tolerance. Device initialization and caches are never
module-global or import-time side effects.

## Invalidation boundaries

- Source line, probability, correlation, or phase-space changes rebuild source rows and incident
  transport. Detector-path medium, scalar coefficient, or exact-wavelength table changes rebuild
  every detector evaluator while leaving sample geometry unchanged.
- Sample entrance geometry or material optics changes rebuild incident transport.
- CIF, finite-stack, mosaic, wavelength, or sample/crystal orientation changes rebuild Bragg and
  detector evaluator state.
- Detector rigid pose, distance, or either detector tilt uses the sampler's projection-only rebind
  without rebuilding incident transport or detector-independent structure amplitudes. Detector
  pixel pitch, shape, or reference-coordinate calibration changes invalidate the compiled sampler
  and rebuild detector state.
- The tagged-function fitters never rasterize a detector field. The OSC-series
  `ExactTagGeometryModel` constructs neither mosaic probability nor structure strength. The older
  `ContinuousDetectorGeometryModel` may receive already prepared intensity state, but its exact-tag
  trials evaluate none of it. Geometry-only trials reuse the reciprocal basis and rods, update the
  complete sample and detector rigid transforms, rebuild nominal incident transport, and rebuild
  exit-refraction and detector geometry.
- A measured selection manifest is invalidated by any image, mask, detector/cake calibration,
  reciprocal context, or policy hash change. A fitter never silently relabels a frozen manifest.
- A mosaic response bank is invalidated by any profile layout, angle frame, source, backend/device,
  material/rod catalogue, or fixed geometry revision change. Mosaic width and mixture changes reuse
  the frozen upstream geometry but require their exact component response.
- A measured mosaic observation is additionally invalidated by any OSC bytes, detector-valid mask,
  exact local projector, measured-selection policy, or nuisance-basis profile revision change.
- A source-averaged selected-center response is invalidated by source count/revision, selected-center
  layout, rods, mosaic or structure physics, instrument, backend, or device. Geometry tag/marker
  construction independently rebuilds exactly one ideal source-center, zero-divergence,
  mean-wavelength companion state and never inherits the intensity ensemble size.
- Display limits and colormaps never invalidate physics.

## Deliberately absent runtime structures

There is no retained mosaic-orientation batch, sampled scattering-candidate pool, scattering-event
table, outgoing-event table, hit table, general point depositor, bilinear rasterizer, discrete
sphere texture, compatibility adapter, or parallel legacy simulator. The optional Monte Carlo
terminal generates one fixed-width source/draw latent matrix and streams roots directly into the
final image; it retains no orientation/root/event Cartesian product. Historical equations remain
only in the immutable reference pack and proof comparisons.

## Detector-native matched-region fitting

`measurement.region_observations` owns material-neutral angular and reciprocal region definitions
and center-membership used only for preparation/display. `measurement.continuous_regions` owns
chart-rectangle quadrature and the data-only sparse projection of piecewise-constant native counts
onto those same rectangles. `fitting.radial_background` owns a shared
rise-times-decay detector halo with per-dataset amplitude and pedestal; `fitting.matched_regions`
owns the covariance-conditioned joint residual and one scale per dataset. None of these modules
knows a material or Figure 7.

`fitting.fixed_experiment` is the modular handoff between position, optional lattice, supplied
mosaic, and intensity stages. It rebuilds one immutable source/material/reciprocal state, rebinds
only the commanded incidence geometry for each image, and rejects incomplete predecessor state.
The current adapter covers the tracked R-3m Bi2X3 layered-quintuple family. Material recipes freeze
measured regions and site-specific constants; the common numerical path refines five shared
coordinates against all three OSCs. The v7 diagnostic chain fits two Wyckoff-z offsets in A, one
outer-chalcogen vacancy fraction in B with occupancy `1-v` and Bi antisite fixed to zero, the
radial/normal sample-Q envelope in C, and all five in joint. The tracked fast workflow uses the v8
`seeded_joint_only.v1` policy: the outer workflow hashes seed-v2 and passes its five-coordinate
vector as the explicit all-active joint start, without synthetic initializer predecessors. The
inner fit-v14 artifact records that numeric start and hash, not the seed-file identity.
Crystallographic site ADPs are a separate fixed profile
inside the atomic amplitude. The sample-Q envelope is applied once from sample-frame Q after mosaic
rotation and is never folded into site ADPs.

A/B/C are initializers only under the v7 policy. Each v7 child binds the exact predecessor bytes,
starts from that predecessor, and keeps all inactive coordinates unchanged. V8 instead binds an
explicit numeric start and has no predecessor chain; only the outer workflow binds seed-v2.
Under either policy only the joint result may proceed,
and fit stages checkpoint resumably;
profiles publish atomically and restart as a whole. Completed stages are immutable predecessors.
Prepare, background calibration, and render
publish atomically. Pixel-center membership is only preparation/display discovery state. The fit
integrates the verified piecewise-constant native count field over the same continuous
phi/two-theta or signed-side Qr/L chart rectangles as the continuous model and propagates the full
fractional-pixel count covariance.
Adjacent anchors condition measured background and model with the same projection but do not become
extra fitted signal rows. Only measured counts and the displayed area-detector image remain pixel
arrays.

The terminal profile is explicitly `FIT_CONDITIONED`: it evaluates complete Qz branches with the
fitted rod roster, records cubature evidence, remains `publication_ready=false`, and makes no
all-configured-rod claim. The detector panel still shows the actual integration regions. The
R-centered RichEpsilon parent is explicit proof/CPU/CUDA finite-stack state with an exact-zero fast
path. The retained production model keeps `epsilon=0` because screened nonzero values failed the
cross-image gates; it is not a fitted 4H/6H population.
If the optional lattice stage accepts a changed basis, every lattice-dependent material,
reciprocal, rod, optical, and detector object is rebuilt before downstream fitting; otherwise the
CIF basis is retained exactly.

## General-CIF sparse fitting core

Contract v13 introduced one material-neutral strength seam before detector fitting; contract v14
retains it unchanged:

```text
explicit CIF/model declaration
  -> reciprocal-basis-bound strength provider
  -> shared source/optics/mosaic/detector transfer
  -> mosaic branch: rebind mosaic -> candidate profile bank -> unchanged mosaic fitter
  -> structure branch: frozen sparse response x candidate strength
       -> exact region integration -> unchanged matched-region fitter
```

`materials.crystal` owns the resolved-CIF revision and affine expanded-site basis.
`pipeline.bragg_space` owns the generic conventional-cell repeat and specialized Bi2X3 providers.
`pipeline.source_averaged_structure` owns the shared sparse response and fitting detector;
`fitting.matched_regions` owns the parameterized region model and fail-closed identifiability gate.
PbI2's five fixed near-parent provider lives beside its existing stacking compiler and implements
the same detector-facing contract. No module dispatches on elements or a material name.

The response is reusable only while reciprocal basis, source, optics, mosaic, rods, pose,
calibration, sample-Q envelope, and requested coordinates are fixed. A candidate site structure
may change strength but not those authorities. This path is CPU sparse fitting authority, not a
replacement for the optimized Bi2X3 full-image renderer. Raw-image discovery, background/mask
policy, fitted-coordinate declarations, and any non-CIF stacking law remain explicit experiment
inputs rather than per-material Python files.

## Native Bi fitting extension

Contract v15 adds `fitting.bi_native` for symmetry-preserving 13-coordinate candidate construction
and `fitting.bi_joint` for explicitly bounded optimization. `fitting.native_input` reads typed
numerical experiment inputs; `fitting.native_observations` owns frozen native support, full GLS
covariance and historical guards. Their prediction path is `pipeline.conditional_detector`
through `fiber_detector` and `source_spatial`: conditional position is integrated over native
pixels, while signed atomic strength and spherical cone probability remain distinct factors.
The same physical evaluator supports native fits and rendering.

Generic finite-CIF strength delegates site tensors to the authoritative ordered amplitude.
Local m0 geometry and phase-Q, complex exit branches, detector revisions and finite Pb endpoint
arithmetic remain shared with their existing owners. Current main's beam-position viewer and
configured sampled-source routes are preserved. The explicit Bi execution resource may reuse
unchanged material/basis responses; cell and occupancy candidates rebuild transport. There is
no plugin backend, hidden global cache, new dependency or imported legacy runtime.

Contract v16 extends this path to the complete declared Pb phase roster and explicit
source/instrument candidates. `native_structure` is the common lattice/optics rebinder;
`native_joint` is the response owner; `native_search` and `native_accuracy` consume ordinary
prediction callables and immutable observations. They contain no scattering equations.
`scripts/refine_native.py` orchestrates numerical checks, all-active fits, discrete choices,
profiles, controls and conditional validation. The Bi-specific v15 API remains supported.
No additional dependency or parallel physical implementation is introduced.

Contract v17 keeps that ownership. `fiber_detector` owns the explicit numerical proposal
domains and composite axial/angular quadrature; `reflectivity.specular` owns the named overlap
measure. Native response identity includes both declarations. The runner separates optimizer
candidates from numerically checked selections and checks profile/conditional predictions
through `native_accuracy`, reusing the existing covariance and validation equations.
`fiber_detector` also owns the declared angular-support policy. Its optional fixed union
prevents individual observation Q boundaries from remeshing angular integration. Native
projection retains the same observation memberships, covariance and physical factors.
