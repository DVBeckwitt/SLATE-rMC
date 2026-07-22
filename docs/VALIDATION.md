# Validation and proof

Tolerance selection and required negative controls are authoritative in [ERROR_INJECTION.md](ERROR_INJECTION.md).

## Proof hierarchy

1. Analytic identities and limiting cases.
2. Independent numerical oracles that do not call the public algorithm.
3. Direct manuscript-equation evaluation.
4. Shared immutable original-RASIM traces.
5. Tiny end-to-end detector result.

## Current contract-v10 runtime

The production runtime is the continuous detector pushforward described in
`CONTINUOUS_EWAL_COATING_STRATEGY.md`. The sampled mosaic/scattering-event
selection/deposition implementation is retired. Current proof authority is the latent Bragg density,
analytic line/sphere oracle,
detector-coordinate inverse map, finite pixel-box integral, and scalar-versus-compiled backend
agreement. Earlier sections in this file are retained historical evidence for the source-to-`ki`
and scientific subsystem cutovers; they do not reinstate retired APIs.

### Complete detector-density reduction cutover

Contract v10 makes the declared reduction order structural: every state-specific outgoing ray,
physical rod, and retained analytic inverse root is reduced into one continuous detector-coordinate
density before fixed macrobin quadrature. The coordinate result preserves the physical rod catalog
as provenance, an all-rod/source caustic flag, valid-source count, source revision, root policy,
`m=0` support gap, and execution identity without exposing a rod-valued detector array. The detailed
per-rod coordinate API remains the proof path. The older source-averaged per-rod native-pixel
integrator requires the explicit proof-only `include_per_rod_evidence=True` opt-in and therefore
cannot be selected accidentally by configured rendering. Configured-result schema v2 removes the
old per-rod macrobin array and its derived per-family pixel masses rather than silently retaining
the old work.

The external Bi2Se3 5 degree proof uses all 1,000 configured incident states, 85 physical rods, all
retained roots, and 10,000 continuous coordinates for the 50 by 50 fixed-quadrature preview. The
former per-rod quadrature order and the total-first order agree to `5.4211e-20 A2` maximum absolute
image error, `1.1877e-16` relative L1 error, zero relative mass error, and
`3.5527e-15` macrobin centroid shift. The default CPU configured command completed in `21.9575 s`,
including a `10.2099 s` detector stage, and emitted total mass `0.00262385385236163 A2` with image
SHA-256 `7eca4b611fa3284a6333870bc2ad984886abffedf621f8087fc017f3ff6c8453`. A warmed CUDA detector
preview took `9.1118 s`. The returned detector intensity-array payload shrank from `1,720,000`
bytes to `20,000` bytes; a separate warmed `tracemalloc` comparison reduced peak Python/NumPy
allocation from `16,572,495` to `9,589,715` bytes. These host figures exclude CUDA device memory.
A direct CUDA kernel that removed the internal rod-coordinate workspace was rejected after it
regressed the same
warmed preview to `25.6213 s`; the retained CUDA path reduces that private workspace on-device and
copies only the completed total coordinate field across the public boundary.

### Position-free measured selection activation

T08 consumes detector-native measured images only after the forward contracts are fixed. Global
angle-chart discovery accepts no marker catalogue or predicted positions; native refinement and
sample-frame `Q` evaluation precede integer-`L`, rod-family, Ewald-branch, and beta-root inference.
Only branch tracks replicated at distinct incidences with shared site identities enter the immutable
selection manifest. Missing peaks remain missing observations rather than inferred extinctions.

For the peak CSV specifically, the accepted mapping is one clockwise OSC conversion followed by
`legacy_raw_x` as native detector column and `legacy_raw_y` as native detector row. These CSV
field names are distinct provenance from the supplied-state legacy `x`/`y` variables. The CSV's
`observed_column_px = columns - 1 - legacy_raw_x` is retained only as corrected provenance. An
exact OSC regression gives 83,328 counts at native `(1455,1473)` for the 003 site and 46 counts at
its reflected `(1544,1473)` location. An independent replay of the legacy cake's provenance-only
half-pixel convention also makes the direct coordinates agree with the stored cake angles while the
reflected coordinates fail decisively; canonical detector coordinates remain pixel centers.

### Continuous angle-coordinate measurement activation

T17 starts from accepted `main` `1f3a106b0fe30b5fe92f8558a45bccf0c49655f7` and activates only
the pointwise detector-density pullback from the historically deferred caking row. The first new
declared stage is `measurement.continuous_angle`: the pose-bound inverse mapping supplies
`J=sin(2theta)/pixel_solid_angle_sr`, then returns `S=dJ`, `N=J`, and `I=S/N`. This use of solid
angle is a coordinate identity, not the excluded detector acceptance correction. A new
continuous-`S/N` finite-bin reducer and reciprocal remapping remain deferred; the accepted
finite-pixel polygon projector is unchanged.

The permanent proof compares a compound-tilted, rectangular detector Jacobian with an independent
central finite-difference determinant; checks periodic `phi`, exact-pole zero measure, and active
support; evaluates a real nominal-source all-root detector function through both coordinate routes;
and verifies that a nonzero corrected geometry exposes the same owned detector/sample pose used by
the detector field. A separate analytic square-panel oracle integrates a nonlinear detector
density in angular coordinates, conserves detector area and signal, and distinguishes
`integral(S)/integral(N)` from averaging pointwise `I`. The external 3,000 × 3,000 comparison is
display/benchmark evidence and is not committed. The flat legacy coordinate convention is
classified `MATCH`; the arbitrary-pose Jacobian and explicit continuous `S/N` measures are `NEW`.

Measured T17 proof gives a maximum relative Jacobian error of `3.2410e-10` against the independent
central-difference determinant. For the analytic nonlinear square-panel fixture, angular
Gauss-Legendre orders 4, 8, and 16 give signal errors `-7.0971e-4`, `-1.9421e-9`, and
`-2.8422e-14`, while normalization errors are `-9.9340e-5`, `-1.4427e-10`, and
`-3.5527e-15`. The conserved finite-bin ratio is `4.266666666666666`; incorrectly averaging the
pointwise ratio gives `4.17297308593442` and is rejected. All 73 compact permanent tests pass in
`100.382 s`, the production-scope Ruff lint/format gates and diff whitespace check pass, and the
independent final review reports `READY`.

The external comparison evaluates one nominal source state, all 85 physical rods, and every
retained inverse root into two true 3,000 × 3,000 center-sampled arrays. It labels all 84
positive-strength nominal exact-integer-`L` markers and uses one shared logarithmic intensity
scale. Build, detector-field, and angle-field times were `3.343 s`, `127.178 s`, and `82.099 s`;
peak working memory was `1,419,460,608` bytes. These center samples visualize the continuous
function; finite angular-bin observables still integrate `S` and `N` separately before division.

### Detector-pose validity maintenance

The 2026-07-21 maintenance branch starts from accepted `main`
`02407a966f60cc7a9af336b53f447fd93ebfa217` and corrects three first divergences:

- Detector intersection previously treated the panel as two-sided and used an absolute projected
  area. The first divergence was `geometry.detector_face_validity`: a forward ray approaching
  detector-local `z=0` from `z>0` was reported `VALID`. Scalar/batched geometry, angle conversion,
  the NumPy detector field, compiled CPU, and CUDA now share the signed gate
  `n_D dot kf_hat > 1e-14`; invalid rays carry zero geometry and intensity before reciprocal or
  optical work.
- The exact inverse of a valid `(h,k)=(0,-1)`, branch-2, `alpha=0.2 deg`, `beta=0` point produced a
  floating value just below zero whose remainder rounded to exactly `2 pi`. The first divergence
  was `reciprocal.inverse_latent_beta`, where the public latent evaluator rejected its own inverse.
  Only the exact upper endpoint is now canonicalized back to zero. The point retains two finite,
  non-caustic inverse contributions, and the NumPy and compiled CPU densities agree.
- Effective sample-normal fit corrections previously changed the sample rotation while freezing
  its LAB origin. The first divergence was `geometry.instrument_transforms` for an offset sample
  mount and nonzero mechanical pivot. Production now composes the canonical rigid motion about the
  pivot; an independent analytic oracle checks `t1 = p + (R1 R0^T)(t0-p)`. The same corrected pose
  feeds the continuous field, nonzero and m=0 exact tags, fitting, and the final marker audit.
  Missing or unequal configured pivots require an explicit override.

All 70 compact permanent tests pass, including real CUDA parity. From the clean committed tree, all
six registered proofs pass: core, references, geometry/optics (11 scientific checks and all 17
mutations), mosaic/Ewald (continuous-science check and all three mutations), ordered reflectivity,
and stacking transition. Ruff lint/format and diff whitespace checks pass.
One unrelated assertion also failed identically on unmodified `main`: batch and scalar evaluation
of the same exact integer-L strength differed by 10 binary64 ULPs while the relative test budget
allowed slightly fewer. Its numerical comparison is now explicitly bounded at 16 ULPs; production
physics is unchanged.

A one-state, all-rod compiled-CPU benchmark over 50,000 continuous detector coordinates gave a
median of `0.45518 s` on this branch versus `0.45185 s` on `main` (five post-warmup calls, `0.74%`
difference). Both runs had the same `98,947,160`-byte Python allocation peak, `39,050,000` output
numeric bytes, 25,444 valid coordinates, and summed density `2.2326975134605485e-5`; no measurable
performance or output regression was found. This reduced four-angle fitter remains an effective
end-pose correction about one fixed pivot, not a decomposition into named goniometer-axis errors.

### Continuous-runtime maintenance cutover

The maintenance branch starts from accepted `main` `0167f6dac79a66d79a94d358c041665bcb73ed38`.
It removes the sampled orientation/event/selection/deposition simulator, discrete Ewald painter and
raster, two superseded image scripts, one redundant seed test, and the duplicated `examples.zip`
archive. The surviving one-state CLI now inherits all omitted physics from the canonical YAML, and
the YAML backend is explicit rather than compatibility-defaulted.

Before the final clean-tree rerun, the compact suite passed 61 cases in about 30.4 seconds, down from 86
accepted baseline cases. All 70 surviving production/proof modules imported successfully; Ruff,
format checking, documentation links, task-index validation, strict YAML/TOML checks, and static
removed-symbol scans passed. The core, reference, ordered, and stacking registered proofs passed.
Geometry/optics passed all 11 scientific checks and 17 controls; its only precommit failure was the
intentional dirty-tree gate. Mosaic/Ewald passed signed probability, analytic-root, exact rod-sum,
catalog, and 3/3 mutation checks; its only precommit failure was the same dirty-tree gate.

The mosaic convergence sequence at quadrature orders 256, 512, 1024, and 2048 had absolute mass
errors `6.6703e-4`, `2.2646e-6`, `2.5942e-11`, and `1.5654e-13`. For 96 equivalent latent points,
the vectorized Ewald coating took 0.00393 seconds versus 0.20611 seconds for an independently
assembled scalar root/latent/coarea oracle (52.5x), with `tracemalloc` peaks 63,224 and 51,159 bytes
respectively. Forty-one points had regular positive support; maximum intensity error was
`4.89e-15`. The representative-rod reduction mutation differed from the physical rod sum by
`1.00685 angstrom^2`; the independent population-times-strength error was zero. Root-coordinate
error was `6.66e-16 inverse angstrom`, coarea-Jacobian error was `4.44e-16`, and elastic residual was
zero for the independent root fixture.

The retained limitations are explicit: the direct `m=0`, `Q=0` root remains excluded; only regular
detector-visible `m=0` support with a positive reciprocal gap is admitted. Fixed macrobin output is
a preview estimate, not an adaptive convergence claim. Detector response, background, finite
caking, and intensity fitting remain deferred.

## Historical BKI-15 incident-boundary proof

This subsection records the contract-v7 predecessor. Its material/sample revision hashes are
historical evidence, not current cache keys; the contract-v8 rebaseline is recorded below.

The 2026-07-18 proof freezes the corrected v2 source realization, explicit sample support, and
self-contained incident boundary as the common predecessor of the corrected parallel Task 1.1
reference and detector-unconditioned continuous-coating validation. Those branches remain
independent until their required join at parallel Task 1.2. The proof added no public API, module,
dependency, executor, cache, backend, or compatibility layer.

### Canonical realization and packet-layout invariance

A one-shot external harness used Python 3.13.13, NumPy 2.5.1, and SciPy 1.18.0 on Windows 11 with
an Intel64 Family 6 Model 183 CPU. For each size it invoked the public sampler exactly once, which
generated the complete realization and its source revision once. Runtime-only call instrumentation
observed exactly one canonical parameter-provenance hash and one complete ten-field realization
hash for each of 1, 33, and 129 rows; no hash-count hook is retained. Three alternate packet layouts
(three contiguous groups, four strided groups, and five reverse-ordered groups) contained only
parent row indices and the inherited parent revision envelope. No packet constructed a sliced
public batch, regenerated source rows, or called the canonical hash.

| Rows | Source-parameter revision | Complete source revision | Material revision |
|---:|---|---|---|
| 1 | `7e7add00fe28b9e5d7f5d02dc58974c63eebf52999de6e85043ed69d6700d8bc` | `2738c08208823af030d9dfc7fbd502a492672e2beca18f635926d854f2041da8` | `abb07bb8ef813b007aa6210084ab643493a6afcf704baad7a2267956f063be71` |
| 33 | `65bbf0320a236bffb1f76f9270fd74d48faf0e1d7afc13c591f6ab464b541ca9` | `a0000f48750bd8f460de923e694860acf99f6a85d7a3aa27b710d50b7be2568a` | `8f71c84032c751e5898c309012ece38abf146b4dfca3a86ace1911fbe2331bc1` |
| 129 | `40fc17228b05f24b65a675d67cfb063095862359dde84e34ffb37e6bd3c08a25` | `f378ee40fa5f5abd7d4a627a804b9371fb6ac499a12ddd2639bc040e78ede7c6` | `52f16485c28fdd40859a338f0ac353e29d403c180d0158a877321478ab0a83c0` |

All three envelopes used sample-geometry revision
`828f31face161ff0c246cdd9eb44d3d210f8584c773059c4ed0089989f1a235d` and incident model
`one_transmitted_channel.v1`. Reassembly by canonical state ID was exact for IDs, statuses,
polarization, source mass, every source array, and every incident array in every layout; the
maximum reassembled `ki` error was zero. Each of the five dimensions occupied every one of its
declared strata exactly once. The smallest reconstructed coordinate-to-edge margins were `0.5`,
`7.91e-5`, and `1.35e-5` for 1, 33, and 129 rows. Maximum absolute standardized source means were
`0`, `6.73e-17`, and `1.72e-16`; maximum second-moment errors decreased from the one-row degenerate
value `1` to `0.10084` and `0.03205`.

Scalar incident-mode evaluation agreed with the batch result exactly for air and film phase
wavevectors, complex film-normal wavevector, and entrance amplitude. Tangential-wavevector error
was zero and the maximum analytic film-dispersion residual was `7.11e-15 angstrom^-2`.

The stdout-only invocation was `PYTHONPATH=src python -` from the repository root; the inline
harness was discarded. It called public `sample_gaussian_source_rays`, `compile_instrument`,
`build_incident_states`, and scalar `solve_incident_mode`. For every row count the source fixture
used seed `1729`, LAB mean origin `[0,0,1] m`, LAB mean direction `[0,0,-1]`, transverse axes
`[[1,0,0],[0,1,0]]`, spatial sigmas `[1e-4,1.5e-4] m`, divergence sigmas
`[8e-4,1.2e-3] rad`, wavelength `1.54 +/- 0.01 angstrom`, and polarization
`UNITY_APPROXIMATION`. Material ID `external-bki15` and provenance
`external BKI-15 harness.v1` named rows at the exact unique realized wavelengths with
`n=0.999979+3.2e-7j`; contract v7 also carried derivative optical arrays removed by API v8. The instrument
used identity SAMPLE/GONIOMETER/LAB transforms, an internally derived LAB-to-SAMPLE inverse,
`unbounded_plane.v1`, detector shape `(11,7)`, row/column pitches `2e-4/1e-4 m`, reference
coordinate `(3,5) px`, detector LAB translation `[0,0,1] m`, and film thickness `500 angstrom`.
Each complete state table was reassembled from three contiguous, four strided, and five
reverse-ordered canonical-index groups carrying its unchanged parent revision envelope.

### Status-dependent work, tracing, and memory

The 129-row benchmark used seven timed repetitions after warmup; wall time is the median and peak
memory is Python `tracemalloc` peak for a separate equivalent call. Geometry attempted all 129
rows. Refraction evaluated only geometry-valid private rows. Tracing produced seven immutable
records per canonical row without changing any scientific field.

| Case | Tracing | Refraction rows | Status counts | Trace records | Median wall time | Peak bytes |
|---|---|---:|---|---:|---:|---:|
| all valid | off | 129 | `VALID=129` | 0 | 0.530 ms | 120,716 |
| all valid | on | 129 | `VALID=129` | 903 | 2.877 ms | 328,747 |
| mixed | off | 43 | `VALID=43, OUTSIDE_SUPPORT=43, PARALLEL=43` | 0 | 0.501 ms | 92,268 |
| mixed | on | 43 | `VALID=43, OUTSIDE_SUPPORT=43, PARALLEL=43` | 903 | 2.884 ms | 310,411 |
| all invalid | off | 0 | `OUTSIDE_SUPPORT=65, PARALLEL=64` | 0 | 0.370 ms | 82,606 |
| all invalid | on | 0 | `OUTSIDE_SUPPORT=65, PARALLEL=64` | 903 | 2.691 ms | 299,534 |

The performance fixture used source model `explicit_external_source.v1`, RNG model `no_rng.v1`,
seed `0`, material ID `external-bki15-performance`, provenance
`external BKI-15 performance fixture.v1`, and a material grid containing only `1.54 angstrom`.
The mixed and all-invalid cases respectively
carried 86 and 129 geometry-invalid `1.73 angstrom` rows without material-grid failure, proving
that refraction did not inspect them. Traced and untraced states were field-for-field identical.
Every geometry-failure row retained source identity, wavelength, polarization, and exact `1/129`
mass while its geometry/optical payload was zero; no survivor renormalization occurred. Peak values
include retained trace records but exclude native allocator, driver, and operating-system caches.

Both traced and untraced paths allocate the same four required full `(129,3)` scientific arrays:
sample intersection, SAMPLE direction, air-side wavevector, and film phase wavevector. The removed
trace-only parallel-wavevector scratch would have been one additional `(129,3)` float64 array,
exactly `3,096` bytes; tracing now uses compact geometry-valid rows and per-record values.

### Canonical incident-boundary negative controls

The discarded stdout harness ran all four controls assigned in `ERROR_INJECTION.md`; every mutation
was detected at its declared first boundary.

| Mutation | Fixture | Expected/observed first boundary | Detection |
|---|---|---|---|
| Duplicate one wavelength stratum, omitting another | `source.lhs.n33` | `sampling.source_lhs_strata` | exact occupancy found duplicated stratum `22` and missing stratum `10` |
| Hash a strided 17-of-33 worker slice | `source.packet.strided17_of_33` | `sampling.source_revision_ownership` | mutant `7dcf33d3e81343ec62f36d495802211e1388c75f7fc4f57b71efcdfb74bf9715` differed from inherited parent `a0000f48750bd8f460de923e694860acf99f6a85d7a3aa27b710d50b7be2568a` |
| Let detector calibration, `sample_from_crystal`, or thickness invalidate incident `ki` | `incident.excluded_instrument_fields` | `geometry.incident_revision_ownership` | all three accepted revision envelopes and `ki` arrays remained exactly identical, so the over-invalidation mutant failed |
| Erase accepted geometry on a later optical failure | `incident.optical_failure.n1` | `geometry.incident_status_payload` | construction rejected the mutant with `accepted geometry requires unit direction_sample` |

The four records used exact stratum occupancy, exact revision/array identity, or constructor
validation as their failure metric; no tolerance, snapshot, packet object, or allocation-count test
was added.

### Legacy classifications and first divergences

- The v1 pair-stratified sampler to v2 endpoint-safe antithetic N-stratum realization is
  `CORRECTED`. IDs, `1/N` source mass, and polarization match; the first trace-value divergence is
  `geometry.lab_ray`. For the 33-row fixture the maximum origin, direction-component, and
  wavelength differences were `4.62734e-4 m`, `3.54744e-3`, and `2.42409e-2 angstrom`.
- Replacing the default script's unsupported finite rectangle with explicit legacy-unbounded
  support is `CORRECTED`. A unique forward intersection at `(3e-4, 0, 0) m` agrees through
  `geometry.sample_intersection`; the first possible divergence is
  `geometry.footprint_acceptance`, where the outside-support case changes from `0` to `1` and from
  `OUTSIDE_SUPPORT` to `VALID`. The canonical 20-row default happens to realize no such row.
- Accepted downstream rows follow the scalar/analytic authority. No stale Monte Carlo image is an
  oracle for either correction.

### Gates and retained proof

Repository-wide compileall, Ruff lint, Ruff format, and all 42 permanent tests passed with bytecode,
Ruff cache, and pytest temp/cache outside the repository. The registered proof commands reported:
core `PASS` (5/5 checks, 7/7 controls), references `PASS` (7/7 checks, 4/4 controls),
geometry/optics scientific `PASS` (11/11 checks, 17/17 controls), mosaic/Ewald scientific `PASS`
(7/7 controls), ordered/reflectivity `PASS` (4/4 checks), and stacking/transition `PASS` (6/6
checks, 7/7 controls). Geometry/optics and mosaic/Ewald retain only their expected aggregate
`BLOCKED` status because their retired wrappers require a clean committed tree while this reviewed
checkpoint is intentionally uncommitted; the final coherent commit must rerun them cleanly.

The permanent incident-boundary additions remain within the five-item test budget: one compact
even/odd source parameterization; one material-grid contract and monochromatic seam; one relational
incident mutation and one revision-ownership case; extensions to the owning sample/transport tests;
and one default-builder invariant plus dispatcher registry check. Broad packet layouts,
convergence, timing, work counts, and memory measurements were stdout-only external evidence. No
benchmark file, diagnostic, snapshot, allocation-count test, packet monkeypatch, or generated image
is retained.

### Supervisor-audit correction

The first BKI-15 manifest rewrite did not freeze the final accepted text. Supervisor audit then
corrected the current finite-pool versus future detector-unconditioned coating distinction, future
replacement-ledger ownership, parallel freeze/task ownership, future Task 1.4 packet attribution,
and post-sampling detector projection/rejection terminology. The same audit required the source
batch to derive its non-omissible revisions in one construction path, required the compiled inverse
to be derived solely from the hashed forward transform, completed external-harness/control
provenance, and protected the retired T03 task/prompt from the active coating plan's T14. The
manifest therefore has two BKI-15 rewrites: the superseded first rewrite and one corrected final
rewrite after these changes froze. Relative to the superseded 223-path manifest, the corrected
rewrite retains all 223 paths, refreshes 19 size/hash records, and adds/removes zero paths. The
final cleanup also sorts exact path strings with Python's ordinal ordering and makes seed
verification reject count mismatches, duplicate paths, or noncanonical order. That corrected
223-path rewrite is the BKI-15-only state and does not include BKI-16/BKI-17 work. The final
222-path manifest and coherent commit additionally include completed BKI-16 and BKI-17.

## Beam-to-`ki` authority cutover

The 2026-07-19 contract-v8 cutover removes redundant optical/transform state without changing the
accepted incident observable. `MaterialOptics.material_revision` now uses schema
`material_optics_revision.v2`; `CompiledInstrument.sample_geometry_revision` uses
`sample_entrance_revision.v2`; and detector-angle instrument identity uses
`detector_angle_instrument_fingerprint.v2`. Transport copies owner revisions into the complete
`IncidentStateBatch` and reciprocal consumers never rejoin raw source rows.

### Exact parity and revision rebaseline

A discarded external harness ran the same three-row normal-incidence fixture from merge
`3af2f4f` and clean cutover commit `f106c45`. Status, intersections, SAMPLE direction, air and film
`ki`, complex film-normal mode, entrance amplitude, footprint, IDs, wavelength, source mass, and
validity produced the same typed byte digest
`046b0d9916d0521e36afba2b6dc97e6776328cf6184f777adaa039ee51cbf232`.
All rows remained `VALID`; film `ki` remained exactly
`[0,0,-4.0799047794078795] angstrom^-1`. Only the parity fixture's material revision changed from
`c67d6c50e6ddee94f95c2d198fca7555b80fe32a5a511d6523eea43a77dc5243` to
`b6966a011cc3cd9ca9c769656689e40df4993fca43d6585b53337c5201b55492`, and its finite sample
revision changed from `c780f155ac3d9328bcd31ce9b099db17c36cd119d9c69808b3d6b49e2d6fedbc`
to `ad22fbc9273cd0e0eec0f9b02dc30126c6c7c52698726ff5254a30fd00058eca`.

The historical BKI source parameter and realization revisions remain unchanged. Their current v2
material rebaselines are:

| Rows | Source-parameter revision | Complete source revision | Material revision v2 |
|---:|---|---|---|
| 1 | `7e7add00fe28b9e5d7f5d02dc58974c63eebf52999de6e85043ed69d6700d8bc` | `2738c08208823af030d9dfc7fbd502a492672e2beca18f635926d854f2041da8` | `1ab8d75f638e251f7add59064042a5581cd1e306d5892022347c905a8ef20dd3` |
| 33 | `65bbf0320a236bffb1f76f9270fd74d48faf0e1d7afc13c591f6ab464b541ca9` | `a0000f48750bd8f460de923e694860acf99f6a85d7a3aa27b710d50b7be2568a` | `67e32574e3b5b4c50cf9385bf0a2d005aa3a1ed605d97ef8b9c255e47c9f9194` |
| 129 | `40fc17228b05f24b65a675d67cfb063095862359dde84e34ffb37e6bd3c08a25` | `f378ee40fa5f5abd7d4a627a804b9371fb6ac499a12ddd2639bc040e78ede7c6` | `fc39d94a2da7489d975caf41359092f91c3bbcb875f45433d96f8bd26423cab7` |

The identity unbounded-plane fixture has sample revision
`42eb54b4ae6dd02ba252e76c3622d4a684067f50645af2aa432dba4a4a30383a`.
The compact angle-projector fixture has detector-only fingerprint
`sha256-34d44adf4f0d65bd8994ff087f3b96da8c2a73b74f93539adb22a48989feb7f4.v2`.
These digest changes are provenance/cache rebaselines, not numerical corrections.

Classification is `MATCH` for the declared source-to-`ki` observable. Accepting mutually
inconsistent stored material representations was `CORRECTED` first at `MaterialOptics`
construction and then made unrepresentable. Unbounded tangent-origin over-invalidation is
`CORRECTED` first at `sample_geometry_revision`, while intersection/status/`ki` remain `MATCH`.
Noncausal angle-cache invalidation is `CORRECTED` first at the detector-angle fingerprint/cache,
while projector arrays remain `MATCH`. Removing the two consumer-zero compiled transforms is an
API-v8 state cutover with no scientific-stage divergence.

### Ownership controls and retained proof

The transitional MAT-01 test independently perturbed each duplicate optical representation and
observed construction-time rejection; MAT-02 then removed that obsolete representation and test.
A one-shot guard replaced all material/sample revision helpers with raising sentinels after owner
construction: `build_incident_states` completed with zero guarded calls and exact inherited
revisions. The permanent rotated unbounded-plane test applies independent shifts along both
tangent basis vectors, preserving revision, status, intersection within `1e-12 m`, and film `ki`;
a `2e-12 m` normal shift changes revision and intersection. The permanent angle test changes every
detector-causal field and observes distinct fingerprint/cache keys, while sample/crystal/support/
film-only changes preserve fingerprint, cache, and every numeric projector array. Sorting private
packets by arbitrary state ID remains an explicitly unexecuted PAR-01 control because that runtime
path does not yet exist.

Retained tests each protect a distinct boundary: the CIF/material test protects analytic
`n_complex`, exact v2 revision, unique wavelengths, and nonnegative absorption; the incident owner
test protects revision inheritance and exclusions; the finite and unbounded compiled-revision tests
protect their different canonical payloads; the rigid-transform test protects axis order through
the retained final transform and absence of aliases; and the angle test protects causal cache
identity plus separate `AngleFrame` ownership. No allocation-count, monkeypatch, snapshot, or
duplicate relational test is retained.

Checkpoint K0 ran 44 focused permanent tests and all six registered proofs on clean `f106c45`.
Core, references, geometry/optics, ordered/reflectivity, and stacking/transition reported `PASS`;
mosaic/Ewald reported `READY` with its declared historical `PROOF_BASE_SHA`. Geometry/optics passed
11 checks and 17 injected controls under contract API v8. Ruff, formatting, documentation links,
diff checks, and zero-consumer/stale-symbol scans also passed. Full seed verification is deferred
only until the single final manifest refresh.

## Completed cross-repository audits

The mosaic and initial ordered audits below started from SLATE-rMC baseline
`caf7acd649a27dc66c6c0b73a2f66dcd520389f9`; the later PbI2 polytype proof started from
`7bccbd328220345b1a62a588d3b418bf82c1f0a9`. External repositories were consulted read-only;
`ra_sim` execution was isolated in a separate process, while the `2D_Mosaic_Sim` peak-shape
equation was evaluated independently in a proof-only harness. Neither production code nor
permanent tests import either repository. These conclusions apply to the named observable and
declared measure or atomic-factor model, not to display-normalized arrays or unrelated downstream
physics.

### Mosaic cap/ring shape and probability measure

The existing wrapped, probability-normalized mosaic density was evaluated without changing the
production model. The audit used `(0,0,3)` (`m=0`) for the cap and `(1,0,0)` and `(1,0,3)`
(`m!=0`) for rings, with pure Gaussian, mixed, and pure Lorentzian profiles. After recentering by
the reciprocal-metric Bragg polar angle, the largest cap/ring line-shape difference was
`1.9206858326015208e-14`. Profiles were scaled to unit peak only for this shape comparison;
conservation used the raw probability-normalized density. Integrating that density with the
declared spherical measure `G^2 sin(theta) dtheta dphi` returned both test intensities, `I0=1`
and `I0=7.25`, with maximum error `1.7763568394002505e-15`. The nonzero-`m` peak locus closed
through `2*pi`; the `m=0` locus remained a cap.

Four targeted mutations changed the cap/ring width, omitted the spherical measure, summed raw
surface density, or reversed the family-topology dispatch. Each failed at its intended density,
quadrature, or event-weight stage. Together with the seven existing T03 controls, all eleven
mutations were detected in the one-shot audit. Main retains the seven T03 controls after the four
temporary cap/ring controls are retired. This audit did not newly compare Ewald/Bragg intersection
loci or detector projection with `2D_Mosaic_Sim`; Ewald correctness remains supported by the
existing tracked-legacy `MATCH`, independent dense oracle, and elastic-residual proofs. Those
proofs cover narrow, broad, tail, tangent/no-root, bandwidth, and specular cases; the largest
regular-case elastic residual was `8.881784197001252e-16 A^-1`. The audit does not create a second
Ewald or mosaic implementation.

Against `2D_Mosaic_Sim` commit `5efb3233d60843f3fd4e0e3b5b73536f05c035e8`, the pure-Gaussian
peak-normalized shape is `MATCH`. Mixed and Lorentzian shapes are `CORRECTED` first at
`mosaic.wrapped_line_density`: that program mixes unit-peak components and uses an unwrapped
Lorentzian tail, whereas SLATE-rMC mixes probability mass in a wrapped density. The external
program therefore is not an oracle for absolute orientation probability. The accepted conclusion
is that the original SLATE-rMC mosaic model correctly shares one recentered shape between caps and
rings and conserves intensity under the project's declared measure; no production change is
required.

### Raw complex ordered structure factor

The structure-factor audit ran the clean `ra_sim` commit
`8fb1415e8e4695aa2ce8ec7f576b575264d4b328` in a separate Python process and compared raw complex
amplitudes in electron units before squaring, rounding, pruning, or normalization. The first
default-to-default divergence was the atomic-factor data source: SLATE-rMC uses Waasmaier--Kirfel
`f0` with XrayDB/Chantler anomalous terms, while that `ra_sim` revision uses ITC-1992 `f0` with
Henke anomalous terms. Phase sign, reciprocal coordinates, site expansion, occupancy, and
displacement factors agreed.

Holding the atomic-factor table equal to the legacy oracle produced `MATCH` for the PbI2 2H and
Bi2Se3 whole-cell amplitudes and for both physical PbI2 layer orientations. An independent audit
exercised 51,200 atomic factors and 70,400 whole-cell and layer amplitudes across
`7.92068--8.17898 keV`; the atomic factors matched exactly and the largest complex-amplitude error
was `7.7276e-13 e`, below the audit's declared `2e-12 e` absolute acceptance bound. The matched
legacy table was the `ra_sim` package default supplied by Dans_Diffraction 3.3.3. PbI2's absent
isotropic displacement value was supplied explicitly as `unknown_u_iso_A2=0.0`. The two
repositories' plus/minus layer labels are reversed, so layer results were aligned by physical
orientation rather than label.

This isolates the legacy numerical mismatch to the declared atomic-factor choice and proves the
ordered structure-factor equation and conventions. The accepted scientific default remains the
existing XrayDB/Chantler path, so no legacy atomic-factor compatibility mode or production change
is required; the temporary matched-table comparison path is not retained. That earlier audit
covers ordered PbI2/Bi2Se3 structure factors only; the separate proof below establishes the PbI2
stacking-disorder, 4H/6H transition, and phase-mixture claims.

### PbI2 2H/4H/6H structure-factor and stacking parity

The polytype proof kept three quantities distinct: file-native whole-cell `F_cell`, physical
one-layer `F_plus`/`F_minus` derived only from `PbI2_2H.cif`, and finite-stack `I_stack`. This is the
same 2H motif authority used by the historical stacking path. The 4H and 6H CIFs validate their own
expanded whole-cell structures and stacking topology; they do not replace the 2H layer pair in the
homogenized transition model. Legacy plus/minus strings were aligned by physical layer orientation.

The direct three-atom 2H sum and production layer amplitudes differed by at most
`2.4457182613372133e-14 e`. Sixty direct-sequence checks across 2H, both 4H hands, both 6H hands,
three period multiples, and four physical-Q events had maximum amplitude error
`1.2397319609735734e-12 e`. The native 2H/4H/6H files expanded to site/motif counts `3/1`, `6/2`,
and `9/3`; twelve direct whole-cell sums differed from `unit_cell_amplitude` by at most
`1.563597684016243e-13 e`. The strict legacy `N=50` finite-per-layer comparison differed by
`3.2862601528904634e-14 e^2/layer` after removing legacy `AREA` and aligning phase,
normalization, and initial-state conventions.

An independent full-six-state/direct oracle covered all three parent types at `epsilon` values
`{0, 0.01, 0.1, 0.5, 0.99}` plus 18 convex binary/ternary mixtures. The largest component and
mixture errors were both `1.8917489796876907e-10 e^2`; transition-mass error was
`2.220446049250313e-16`, and linearity and zero-weight errors were exactly zero. The structure-factor
and aggregate evidence SHA-256 values are respectively
`6379e6fa3ed3e3ab9da97fc21f9b4d3b07a15e9d8f7f2f34cb41925d96247f8e` and
`0da8be40686da52c179d8febf1791d332bcb143f12517713218e332ddf028b7f`.

The equations and convention-matched legacy comparisons are `MATCH`. Default atomic-factor
differences remain intentionally `CORRECTED` first at `ordered.atomic_amplitude`, and the accepted
registry convention is `CORRECTED` first at `stacking.registry_phase`. Comparing relaxed native
4H/6H cells directly with ideal 2H-derived parents is `NO_ORACLE` because they are different
structural models. All assigned mutations were detected, all 29 permanent tests and focused proof
gates passed, and no production code, API, dependency, CLI, or example changed. The disposable
proof design, tolerances, exact legacy anchors, performance, and branch disposition are recorded in
[WORKBRANCH_ARCHIVE_2026-07-16.md](WORKBRANCH_ARCHIVE_2026-07-16.md).

## Permanent suite

Keep a small permanent suite:

```text
test_core_coordinates.py
test_geometry_optics.py
test_mosaic_ewald.py
test_ordered_reflectivity.py
test_stacking_transition.py
test_integration.py
test_fitting.py
test_selection.py
```

Future fitting work extends the closest owning module or adds one cohesive fitting module only when
it protects a new public boundary. It must not recreate one permanent file per task or fit stage.

Permanent tests:

- use analytic and tiny direct cases
- require no original-RASIM or manuscript access
- commit no large images or experimental data
- write no diagnostics
- exclude benchmarks and broad convergence sweeps

## Branch proof gates

Every branch must pass:

- analytic or invariant checks
- independent oracle where required
- shared-pack legacy classification
- first-divergence record for each correction
- convergence only when the calculation has a real numerical refinement variable
- equivalent-work benchmark and peak memory
- clean import and dependency check
- assigned error-injection controls fail at the expected first stage
- clean Git tree

## Core analytic checks

### Shared and geometry

- transform inverse and composition
- orthogonal rotations with determinant `+1`
- pixel-to-ray-to-pixel round trip
- direct beam and detector-plane intersection
- global rigid-rotation covariance
- OSC forward and inverse marker mapping

### Optics

- tangential momentum conservation
- dispersion relation
- correct propagating direction
- decaying evanescent branch
- `n -> 1` limit
- scalar interface coefficient limits
- uniform-depth attenuation limit at zero decay and large thickness

### Mosaic and Ewald

- probability normalization
- azimuthal periodicity
- zero-width limit
- pole behavior
- tangent and no-root status
- Ewald residual
- continuous latent-density identities
- analytic roots versus an independent quadratic oracle
- per-rod to exact-family intensity reduction
- detector-visible coating and elastic closure
- detector-coordinate and pixel-quadrature convergence
- detector-angle inverse Jacobian, periodic seam, pole, and invalid-support behavior
- continuous angle `S=dJ`, `N=J`, and `I=S/N` with the corrected pose owned by the detector function

### Ordered and reflectivity

- crystallographic symmetry and occupancy
- systematic absences from amplitude
- general-cell reciprocal basis
- distinct rods with shared family metadata
- finite-stack limiting cases
- Parratt Fresnel and single-interface limits
- composite equality to declared branches outside the blend window

### Stacking

- direct sequence enumeration for small `N`
- full six-state versus exact reduced result
- `N=1`
- deterministic parent limits
- nonnegative real ensemble intensity
- `h=k=0`, `F+=F-` Laue limit
- finite total versus per-layer normalization

### Measured selection

- position-free global discovery and image-unit scaling invariance
- once-clockwise OSC conversion and detector/cake round trip
- reciprocal family, integer-`L`, Ewald-branch, and root-sign inference
- ambiguity, tangency, noncoincident-rod, and duplicated-data rejection
- exact manifest hashing and distinct-incidence/shared-site replication

## Cross-branch review gates

Before merging, an automated review checks:

```text
contract API version
trace schema version
legacy-pack hash
units and frames
array shapes
source, incident, rod, and query ID preservation
amplitude versus intensity declarations
probability density versus probability mass
model versions
owned-path compliance
mandatory proof records
```

A read-only scientific review then checks that:

- no factor is omitted or applied twice
- the same shared equation is not reimplemented inconsistently
- material optics and complex-wave branches agree
- rod and family identities agree
- latent `u`, `L`, `Q`, root, and wavelength semantics agree
- ordered and stacking return interchangeable query-aligned strengths
- outgoing film wavevectors and detector rays use the same geometry/optical branch
- OSC and simulation coordinates meet at exactly one boundary

## Integration sequence

Integrate through these live vertical slices:

1. source rows to canonical incident film-phase `ki`;
2. CIF and reciprocal basis to complete physical rods and query-aligned strengths;
3. rods, mosaic, and strength to the continuous latent Bragg density;
4. latent rods to analytic Ewald roots and the intrinsic coating oracle;
5. arbitrary detector coordinates to exit-refracted film `kf`, sample-frame `Q`, and every inverse
   latent branch;
6. per-rod contributions to one-state and then source-averaged detector-coordinate density;
7. continuous density to deterministic detector-pixel box integrals; and
8. stacking strength substitution under the same per-rod contract.

## Later fitting proof

Each fit stage must first recover parameters from synthetic data generated by the accepted forward model.

- Source fit: recover size, divergence, and declared correlations from multiple-distance direct-beam data.
- Detector fit: recover pose and beam center from calibrant or independently constrained observations.
- Sample geometry fit: recover sample/goniometer parameters from frozen peak associations and pass an outer re-index audit.
- Mosaic fit: recover core width, tail width, and mixture from fixed profiles with source and geometry frozen.
- Ordered intensity fit: recover structural parameters and image scales from fixed selected regions.
- Stacking fit: recover transition parameters from fixed rods and branches with upstream states frozen.

Measured-data improvement is not proof without synthetic recovery and parameter-identifiability diagnostics.

The accepted first fitting slice binds hidden-reference and trial detector fields as continuous
callables and fits only their exact tagged landmarks. No sampled target field, raster, centroid,
pixel integral, or detector quadrature enters the objective. Its permanent proof generates a hidden
four-angle pose with the full marker enumerator, fits 66 nonzero-m sites and 18 explicitly declared
minimum-tilt m=0 exact-L landmarks (`L=2..19`) from a different start, verifies all 33 paired-root
chord angles and the m=0 total-least-squares line, predicts 18 held-out nonzero sites, checks the
bound-scaled
Jacobian rank/condition, and independently re-enumerates all 84 visible nonzero roots. Pixel
integrators are monkeypatched to fail. The proof also rejects an underdetermined single-tag pack
before optimization and rejects off-panel trial geometry before
optimization. This proves only detector tilts plus the effective sample normal at one 5-degree
state; it does not prove separate raw goniometer mechanics or claim that exact-L landmarks are
scalar detector-density maxima.

This proof activates `PHY-FIT-002B`, `PHY-FIT-002C`, and `PHY-FIT-003B` from the physics ledger.

Tag provenance is one source-center, zero-divergence, mean-wavelength companion state for the
entire source-averaged field. Tests freeze the policy identity and verify that no Monte Carlo source
index appears in the unique `(m,L,tag_branch)` identities. A default 1,000-state catalog must be
bit-for-bit identical to the one-state companion catalog. The nonzero visual sides map
`root_sign=-1/+1` to `tag_branch=1/2`; minimum-tilt m=0 uses `tag_branch=0`.

A synthetic residual-decomposition oracle separately checks the fixed-span signed half-angle
terms for a paired-root chord and an increasing-L m=0 TLS line. An independent constrained-vector
oracle proves that every visible m=0 landmark maximizes alignment with the unmosaicked reciprocal
axis subject to its exact Ewald section; it also freezes `L=1` as backward and `L=20` as outside
the default panel.

### Multi-incidence indexed geometry proof

The shared-series proof independently constructs a nonzero hidden nine-coordinate pose and exact
observations at 5, 10, and 15 degrees. It rejects the one-image and two-image subsets at rank 5/9
and 7/9, then requires rank 9/9 for the three-image fit. The permanent fixture verifies normalized
parameter recovery, detector-native training and held-out errors, signed final-normal displacement,
canonical image-ID ordering, and an independent continuous-Ewald root audit. A second-CIF PbI2
fixture monkeypatches Bi2Se3 strength and `MosaicBraggSpace` construction to fail while exercising
both exact-tag prediction and discovery-to-indexing, proving that geometry work does not depend on
intensity or mosaic quadrature.

The same hidden-truth fixture also removes both detector tilts from the active pack, recovers the
remaining seven coordinates at rank 7/7, and requires the fixed tilt values to remain bit-exact. A
second reordered, noncontiguous three-coordinate request proves that caller order is canonicalized,
fixed coordinates remain bit-exact at arbitrary positions, and diagnostic vector shapes follow the
active count. Empty, duplicate, and unknown active-name packs fail explicitly.

The root audit is a direct oracle, not a second call to the production integer-`L` solver. It
brackets the fixed-`L` elastic residual on the two monotone beta arcs, derives root sign from the
crossing direction and analytic branch from the signed axial derivative, maps the root to native
coordinates, and compares it with the production prediction. A permanent mutation reverses the
production beta tuple without reversing `root_sign`; the audit must report `CHANGED`.

The measured qualification consumes the frozen Bi2Se3 5/10/15-degree 10/8/8-site selection once.
It must report baseline and per-image improvement, raw-pixel fit and held-out errors, the full
scaled singular spectrum and weakest direction, deterministic multi-start prediction separation,
optimizer work, wall time, and peak memory. The exact-root audit and measured outer audit are
separate. The acceptance-critical measured audit relabels only the unchanged selected native
candidates and preserves all 26 full keys and coherent tracks: 10/10 at 5 degrees, 8/8 at 10
degrees, and 8/8 at 15 degrees. A fresh global rediscovery is also recorded but is not an identity
oracle because its cake chart and same-key candidate ownership change with geometry. In the
qualifying run it is `CHANGED`: alternate broad same-key lobes are selected at 10 and 15 degrees
and a new just-above-threshold same-key competitor censors one 5-degree decision. Those are
operational discovery sensitivities, not relabels of the 26 frozen coordinates.

The qualifying real fit uses manifest
`sha256-1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132`.
Pooled raw-pixel RMS improves from `12.8312707` to `1.5589418` and maximum error from
`21.3000088` to `5.6507942`. Per-image RMS is `1.2531698`, `0.8406037`, and `2.2866603` pixels for
5, 10, and 15 degrees. The scaled Jacobian is rank 9 with condition `8689.155`; its singular values
are `(76.2284, 72.4703, 58.1766, 22.8074, 4.35393, 3.33615, 0.256518, 0.100480,
0.00877282)`. Three deterministic starts differ by at most `3.164e-5` pixels in predicted native
coordinates. Holding out `L={4,11}` gives RMS/max `1.3173871/2.0219100` pixels.

The fitted corrections, in contract order, are `(-0.004499434, -0.023011841, 0.007325549,
0.015659150, -0.015187627, -0.015250741, 49.5499 um, 99.999997 um, -27.7328 um)`.
Pivot-pitch is within `1e-6` of its normalized upper-bound span, so detector-coordinate prediction
passes but parameter precision does not. Widening that bound through 0.2, 0.5, and 1.0 mm changes
pooled RMS only from about `1.55875` to `1.55815` pixels while the pivot estimate remains
bound-seeking; a fourth incidence is required for a parameter-level pivot claim.

A follow-up constrained diagnostic freezes both detector corrections and instead uses the saved
old RA-SIM Bi2Se3 GUI-state tilts as the base detector calibration. The typed state, saved
2026-07-01, has SHA-256
`9be8c0eacd2f336946b0fa31afbe6bc01f76f7b487d43d0ad316fc8b29de7f5f` and legacy
`gamma/Gamma = 0.3935197233/-0.1480580265 deg`. Exact basis conversion gives current intrinsic
column/row tilts `-0.3935210372/-0.1480545344 deg`, reproducing the legacy basis within
`1.11e-16`. Detector-tilt corrections remain exactly zero. The beam center remains
`(column,row)=(1453.12,1596.422) px`; the CIF remains `a=b=4.143 A`, `c=28.636 A`, and
`alpha=beta=90 deg`, `gamma=120 deg`. Beam center and lattice are structurally absent from the fit
pack.

Because the base detector geometry participates in position-free discovery, this constrained run
freezes a different 9/8/7-site selection with manifest
`sha256-4626d21c9d715f31f7dcf4040b6ac7700d78faead77eb6fb36aa8d9a381f34f2` at 5/10/15 degrees.
Pooled RMS/max improves from `11.9607623/18.6106543` to `5.7518779/8.5759636 px`; per-image RMS is
`4.8268629/5.6487485/6.8563219 px`. The seven-coordinate Jacobian is rank 7 with condition
`6338.748` and scaled singular values `(56.3155, 21.8813, 6.29875, 3.93629, 0.363137,
0.102267, 0.00888433)`. In active-coordinate order, the corrections are
`(0.007755561, 0.013951613, -0.034487517, -0.017397246, 63.2707 um, 40.0814 um,
-99.999999 um)`. The final pivot-yaw coordinate reaches its lower bound. Three starts differ by at
most `0.0004286 px` in prediction. The `L={4,11}` cross-check gives held-out RMS/max
`5.5494945/9.2145733 px`.

The independent root audit is `SAME`, but the acceptance-critical frozen-candidate audit is
`CHANGED`: the 15-degree `L=5` pair is no longer preserved coherently. The run therefore rejects
the fixed-legacy-tilt hypothesis under the existing observation-space gates; it is not a substitute
qualification profile. Initial indexing took `113.696 s`, the fresh global diagnostic `111.946 s`,
the selected solve `2.003 s`, warm residual median `0.01649 s`, and traced fit peak memory
`144,527 bytes`. Source-state count remained one and fitting performed zero mosaic, intensity,
raster, or pixel work.

Measured timing was `91.851` seconds for initial OSC discovery/indexing and `95.200` seconds for
the separate fresh global diagnostic. Geometry-only setup took `0.05587` seconds, the selected
primary solve `7.6590` seconds, frozen-coordinate relabeling `0.3533` seconds, and the warm joint
residual median `0.007240` seconds. Traced fit peak memory was `99,630` bytes; OSC decoding and the
global searches are reported separately because they retain detector-sized arrays.

The nine-coordinate rank is a statement under fixed detector calibration and explicit gauge
ownership, not a claim that every viewer control is estimable. Detector roll, sample/crystal roll,
axis-parallel pivot motion, sample tangent translations, detector center/pitch/distance, wavelength,
and per-image corrections stay fixed. The pivot-pitch/pivot-yaw/plane-offset combination is the
weakest direction for the measured three-angle design, so a fourth incidence is recommended for an
independent validation and parameter-level pivot uncertainties must be reported honestly.

## Tolerance freeze and proof sensitivity

The original T02--T05 comparison proofs loaded `proof/stage_tolerances_v1.json` through the strict
loader and recorded canonical SHA-256
`d3739963a8decf481fc7ec87723854ef7628e8da02dbcb3e6f7e5bb41522b4b3`. The compact contract-v9
reciprocal proof instead binds its analytic floating-point limits directly to each stated identity
and does not claim that tolerance-artifact provenance. Tolerances may change only through reviewed
proof-base work, never after seeing branch error. Exact calculations do not invent convergence;
broad sweeps, mutations, and benchmarks remain one-shot evidence rather than permanent frameworks.

A branch is not proven when the fixture is insensitive to likely mistakes. Run the workstream
mutations from [ERROR_INJECTION.md](ERROR_INJECTION.md) and record the expected and observed first
failing stage. The integration proof covers factor omission/duplication, rod/source identity,
Ewald-root sign, OSC orientation, detector tilts, and row/column mutations.
