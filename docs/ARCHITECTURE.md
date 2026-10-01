# Architecture

## Current native fitting ownership

Material bindings implement `NativeRefinementModel`; `native_workflow` assembles them with
the shared instrument/observation owners. `native_search` owns public bounded TRF and guarded
SLSQP, exact scale profiling and nuisance refits. `native_execution` owns raw recovery.
`fiber_detector` separates retained scattering from detector transport; `conditional_detector`
contracts current SF, mosaic and source masses. Gaussian probabilities have one shared
arithmetic owner in `source_spatial`. See [NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md).
Historical layered-stage orchestration and eager fitting-package reexports have retired;
geometry, scan and Monte Carlo paths retain their distinct live contracts.

## Selected fitting workflow

The purpose is inference from the same measured detector observables under one declared
instrument/source and specimen model. Select methods by physical validity and matched-observable
evidence before runtime or visual agreement. A different source, objective or specular assumption
is a different comparison, even if its figure looks better.

| Stage | Authoritative production owners | Scope and limit |
| --- | --- | --- |
| Geometry | `fitting/joint_geometry.py`, `hbn.py`, `joint_geometry_handoff.py`, shared `geometry` transforms | D043 reduced reference gauge with shared hBN/crystal calibration; D044 applies the accepted source/detector state once. Predictive geometry is qualified; the fixed mechanical pitch references are not measured parameters. |
| Mosaic | `painted_ewald/normal_density.py`, `fitting/native_search.py` | Directed spherical Gaussian/Lorentzian density with independent uniform spin through full-family native observations. `fitting/mosaic.py` retains continuous profile evaluation only. |
| Ordered SF | `ordered/amplitudes.py`, `ordered/finite_stack.py`, explicit Bi/Pb or generic CIF bindings | One complex atomic-amplitude equation and coherent finite structure. Geometry and source remain bound; occupancies, site ADPs and empirical sample-Q envelopes have distinct meanings. The alternate selected-center optimizer is retired. |
| SF with disorder | `stacking/transition.py`, `stacking/finite_intensity.py`, `stacking/parent_models.py` | One finite-stack transition recurrence whose prior enumeration and ordered-limit evidence is archived in Git. Independent parents mix as intensities. Synthetic fixed-parent recovery does not qualify measured disorder estimates. |
| Transport and fitting | `pipeline/fiber_detector.py`, `conditional_detector.py`, `source_spatial.py`; `fitting/native_observations.py`, `native_workflow.py`, `native_search.py`, `native_execution.py` | One conditional transport/spatial-probability path, frozen native memberships, declared covariance and shared-scale fit. Numerical routing is explicit; no extra detector engine or hidden physical fallback. |

Use geometry, mosaic and SF stages to establish supported initial values. Release additional
coordinates only under the declared inference contract and report weak directions; do not fix
unidentified parameters merely to make a full-rank claim. Keep ordered structure as the nested
control when testing disorder. Keep named specular models separate until evidence distinguishes
their assumptions. Independent inverse-coordinate APIs retain their distinct declared observables.
Development enumeration/proof harnesses have retired; they are not fitting workflows.

The empirical whole-pattern baseline is the genuine September 11 Gaussian/Lorentzian fit bound by
`configs/native_experiments.json`. It used complete native families, correlated source/spectral
inputs, a frozen background, historical weighting/guards and a shared scale. Its objective
was not the modern full-covariance GLS objective. It was nominal, not numerically qualified. The
later two-profile spline experiment changed those inputs and assumptions, improved central signed
profiles and left large radial excess. It is archived research, not the default replacement.

The two historical Bi2Se3 recovery CLIs and alternate fitters are retired; their
evidence remains archived. Bespoke point-source/full-field adapters, spline banks and
failed performance experiments remain external evidence. Do not import their orchestration into
production. The current `prepare_native`, `refine_native` and `render_native` commands own the
native-count workflow; see [NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md) for explicit nominal
integration and qualification. Historical numerical successes never waive current physical gates.

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
- `fitting/mosaic.py`: frozen angular-profile identities and continuous finite-bin evaluation
  through the canonical measurement transform.
- `fitting/stacking_intensity.py`: fixed-parent PbI2 strength and log-ratio parameters,
  using the shared finite-stack physics.
- `measurement`: downstream detector-derived observables, including the continuous normalized
  `(phi, 2theta)` coordinate pullback, per-rod all-root angular signal, the full finite-pixel angle
  projector, and an exact cropped physical-pixel projector for fully panel-contained measured
  profiles. It also owns material-neutral layered reciprocal frames, immutable `Qr/L` profile
  regions, sample membership, and separate finite-bin signal/measure accumulation used equally by
  measured pixel centers and continuous detector cubature; never part of raw rendering.
- `io/diagnostics.py`: atomic external result writing, shared by the runtime CLIs.

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
9. Frozen mosaic-profile evaluation integrates signal S and detector-area normalization N
   separately. Measurement projectors retain the corresponding native-pixel boundary.
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
13. Historical ordered-intensity evidence at Git revision `358362e` covered the finite-ROI mass
    path and a certified source-averaged
    selected-center path. The latter produces one combined detector function per incidence before
    any dataset scale or residual, retains every weak nonzero anchor and admitted `m=0` anchor, and
    contracts only occupancy coefficients and its historical global Q-damping factors. Atomic positions,
    geometry, mosaic, lattice, optics, and stacking law are immutable. Multi-incidence observations
    join by dataset ID and exact observable-layout digest rather than tuple position. The archived
    Unfiltered Bi2Se3 proof used 250 shared-revision source rows and `88/78/72` centers,
    including six `m=0`; a measured-mosaic handoff instead propagates its exact fit-eligible
    identity set. That evidence concerned synthetic selected-component recovery, not unresolved raw-OSC
    intensity recovery.
14. Fixed-parent PbI2 strengths remain available through `Pbi2ParentMixtureStrength`;
    detector-native fitting uses the shared native workflow.
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

The desktop shell's exclusion editor uses `interactive/mask_state.py` for source-bound immutable
native spans, compact session history and worker preparation. `project_state.py` persists the
mask in schema 6; `slate_app.py` owns acquisition/revision admission through the existing job and
atomic-write owners. `detector_panel.py` owns one exact profile reducer and a retained R8 reason
texture alongside the unchanged counts texture. Visibility is presentation state. A real edit
invalidates dependent profile caches; cursor and display changes reuse committed immutable state.
No numerical package import depends on these optional Qt modules.

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
- Simulated profiles bind layout, angle frame, source, backend, rods and geometry.
  Measured profiles additionally bind observation bytes and the projection revision.
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

## Generic-CIF and multiple-acquisition fitting

`measurement.continuous_regions` owns exact region quadrature and measured native-pixel
projection. `matched_regions` retains material-neutral observations, frozen background,
anchor conditioning and parameterized sparse response blocks. It prepares the same
`NativeFitObservations` consumed by `native_search`; each acquisition keeps an exposure
scale and full cross-acquisition covariance. One named specimen owns shared structural
coordinates. `fit_structure_regions` supplies data-only rank/condition diagnostics and
fitted-strength provenance after the shared search. No alternate optimizer remains.

```text
CIF + explicit repeat/site/displacement modes
 -> AffineCifFiniteStackParameterization or supported physical provider
 -> canonical sparse source/optics/mosaic/detector response
 -> region quadrature + matched measurement/background covariance
 -> NativeFitObservations -> fit_native_parameters
```

Sparse response reuse freezes basis, source, optics, mosaic, rods, geometry and requested
coordinates; admissible candidates change strength. A generic CIF does not define a
measurement, background, arbitrary disorder law or full-image renderer. The supported
native Bi/Pb evaluator is the separate physical binding for full detector fitting/rendering.
Both bindings share objective/search ownership. Retired layered-stage and Figure-7
orchestration remains archived evidence, not an active five-coordinate fitting workflow.
See [STAGED_FITTING.md](STAGED_FITTING.md) and [CONTRACTS.md](CONTRACTS.md).

## Native Bi fitting extension

Contract v15 adds `fitting.bi_native` for symmetry-preserving 13-coordinate candidate construction
and `fitting.bi_joint` for physical candidate binding. `fitting.native_search` owns bounded
optimization. `fitting.native_input` reads typed
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

## Independent desktop simulation binding

`interactive/simulation_state` owns immutable full configured drafts and external exact-result
references; schema 9 project view state persists them independently of acquisition metadata.
`simulation_fields` contains explicit UI descriptors, `simulation_panel` owns the controls and
inspection bindings, and `simulation_io` delegates equations to existing configured/detector
owners. One global `job_lifecycle` worker owns construction, sampler advance/reset/release,
profile reductions and file publication. Its one replaceable publication slot carries owned
immutable frames. No new physics or fitting owner is introduced.

The visualization extra uses `threadpoolctl` to bound nested BLAS threads during worker execution.
Display preparation uses worker-owned float32 copies; quantitative inspection uses immutable
float64 snapshots. CPU/GPU reservations include retained application data. Reservations and
software binding fidelity do not establish numerical convergence or responsiveness acceptance;
see the current Task06 disposition in `DESKTOP_UI_PLAN.md`.


### Desktop hBN owner boundary

`fitting.hbn.prepare_hbn_ring_observations` owns existing discovery and preliminary refinement;
`fit_hbn_ring_observations` fits an already reviewed immutable observation owner. The automatic
convenience function keeps its previous defaults and composes preparation as before. The residual,
curve, angle and qualification equations remain authoritative in this module. Admitted five-coordinate
seeds/bounds and cooperative residual/boundary cancellation are explicit. Solver termination fields
are separate from the retained qualification verdict and covariance/rank checks.

`interactive/hbn_state.py` bounds immutable JSON input snapshots, review decisions, frozen packs and
result records. `hbn_io.py` verifies exact source bytes and performs worker-owned OSC admission,
preparation, Gaussian proposals, frozen fitting, canonical result validation and external export.
`hbn_panel.py` presents these states and records bounded draft edits; `ShellWindow` routes requests
through the existing single latest-only worker. No widget implements scattering/geometry equations.
The optional narrow `fitting.spot` owner fits only the declared local elliptical Gaussian; it does
not qualify a geometric beam intercept. Selected hBN distance remains calibrant-private.


### Desktop sample-series owner boundary

`selection.osc_series.index_osc_geometry_series` remains the sole OSC discovery/indexing owner.
Optional complete-roster native counts/masks/revisions and cooperative checkpoints let the desktop
bind already admitted data. Explicit masks intersect the existing all-zero edge validity. Absent
optional arguments, canonical automatic inputs/defaults are retained. The existing confident-track
owner admits reviewed decisions; original candidate coordinates and covariance remain immutable.

`fitting.indexed_series.fit_indexed_geometry_series` consumes exact frozen `IndexedGeometryImage`
blocks and existing corrections/bounds/scopes/gauges. Its optional checkpoints add progress/stop
boundaries without changing equations/defaults or qualification. No desktop Fit calls the broader
CLI indexing/multistart/audit orchestration.

`interactive/sample_state.py` owns bounded immutable inputs, controls, review, frozen observations
and result history. `sample_io.py` binds canonical geometry/material, owner calls, record validation
and external exports on the global worker; `sample_panel.py` owns explicit draft/selection and
result-ID presentation. Schema 12 persists exact losslessly packed JSON; expanded bytes participate
in memory admission. Project Open admits saved structure and recorded qualification through canonical validators,
then checks live source readiness separately. Missing or changed historical sample files retain
inert inspectable history; matching live files still undergo the full prediction/residual validator
without solving. Explicit revalidation restores readiness after exact original bytes return. All
current/historical inputs and exported paths participate in project destination protection. The
widgets contain no physical equations, and sample-only operation needs no hBN state. Selection
never changes experiment geometry or supplies absent downstream qualification.


The hBN solver and desktop record admission share `hbn_calibration_is_qualified`,
`hbn_ring_residual_statistics` and `hbn_active_bounds` in `fitting.hbn`. These extract existing
rules without changing physical equations, thresholds, solver defaults or covariance/rank work.
Saved-record admission recomputes derivable diagnostics without optimizing. The hBN panel compares
visible review with committed decisions before launch; only explicit commit/freeze changes the
observation authority. Displayed point rows retain one result UUID and immutable record snapshot;
choice changes clear prior curves/points before explicit worker presentation. Inspection and
selection remain separate. No new controller, dependency or physical implementation is added.


### Desktop joint geometry mechanisms

`interactive/joint_state.py` owns immutable independent hBN/Bi2Se3/Bi2Te3 captures, canonical
controls, bounded pending drafts and historical result snapshots. `joint_io.py` binds the existing
frozen owners, calls `fit_joint_geometry` and publishes exact results and supported hash-bound
handoffs on the existing global worker. `joint_panel.py` provides explicit capture/commit/selection,
canonical-name comparison and record/image-bound native plots. No new scientific model or material
alias is introduced; optional PbI2 capture remains unsupported in this UI.

The joint fitting owner accepts optional explicit starts and phase/residual checkpoints while
retaining defaults, reduced gauge, equations and numerical qualification. `joint_geometry_report.py`
extracts the existing CLI report and qualification-failure rule for shared desktop use. Recorded
admission checks canonical roles, fixed/unobserved references, metrics, conditioning and verdicts
without fitting or computing a Jacobian. Handoff admission uses the same recorded-check validator.

Schema 13 retains schemas 1-12. Independent packed captures/history and pending visible controls
remain under the 1 MiB project cap and expanded-memory admission. Historical result presentation
requires recorded structure; live file hashes and frozen provenance are checked only when fitting
or verifying a handoff. A selected candidate is tied to the current committed launch; changed
source/metadata/masks revoke that selection. Joint handoffs remain separate from indexed result
adoption and grant GEOMETRY_ONLY, not downstream mosaic/intensity qualification.


## Desktop physical editing (U09/U05d/U13)

`physical_panel.py` connects constrained scene gestures, exact typed starts and fine adjustment
to the existing configured simulator, acquisition NumericDraft, hBN, indexed sample and joint
controls. It uses SessionHistory, explicit route adapters in `physical_io.py`, and the shell's
single admitted worker. No physical gesture starts an optimizer. Nonspatial/unsupported values
remain in their existing inspectors; absent and reduced-gauge coordinates remain readonly.

Configured detector edits compile the instrument through the configured owner; the angle-only
reuse function cannot admit them. Canonical indexed transport and the scene share
`corrected_goniometer_axis`. `hbn_detector_transform` embeds the existing private hBN ring plane
for visualization. The crystal detector distance and hBN private distance remain distinct.
The scene converts incident vectors from sample to LAB and retains actual configured source origin.
Geometry without a matching image draws a labeled schematic rather than requiring a texture.

Geometry-derived centers read matching genuine saved hBN or supported sample results, preserve
qualification, and record result/observation/input hashes. Explicit adoption updates initial values
and provenance with one history transaction. It never adds observations. Physical preview and
sensitivity jobs are bounded geometry calculations and do not consume joint handoff exports.


### Desktop comparison and reviewed copy dispatch

Comparison presenters retain one authoritative shared raw-count contrast tuple and acquisition/
source-bound independent tuples. Replacement admission reapplies shared limits, including pending
retained views; export requires current settled inputs. Persistence stores each current independent
view plus the shared tuple, so unlock after Open restores the correct occupants' limits.
Confirmed source-copy plans retain the setup context through deferred dispatch. The existing
pending-request/context gate rejects obsolete confirmation before starting the copy worker.


### Joint handoff predecessor ownership

New desktop handoffs use the displayed immutable result's frozen launch captures. The worker
checks all original hBN/crystal file identities, specimen/base paths and ordered image/command
angles before publishing either sidecar. The existing owner builds the same v1 document from
immutable report/configuration/manifest reads and checks captured hashes. The desktop rechecks
live predecessors before no-overwrite publication, removes only its own unchanged new files
on paired publication failure, and hashes serialized bytes for export references.

Historical reports without a launch remain inspectable and exactly exportable; new desktop
handoff creation is unavailable. Existing handoff verification retains its serialized byte-binding
guarantee and cannot reconstruct missing original fit lineage. Qualification and geometry rebasing
still belong to the existing core; no joint handoff becomes an indexed result implicitly.
