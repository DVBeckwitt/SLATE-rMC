# Validation and proof

Tolerance selection and required negative controls are authoritative in [ERROR_INJECTION.md](ERROR_INJECTION.md).

## Proof hierarchy

1. Analytic identities and limiting cases.
2. Independent numerical oracles that do not call the public algorithm.
3. Direct manuscript-equation evaluation.
4. Shared immutable original-RASIM traces.
5. Tiny end-to-end detector result.

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
```

Post-integration work adds only:

```text
test_selection.py
test_fitting.py
```

The sequential fitting tasks extend `test_fitting.py`; they do not leave one permanent module per fit stage. Do not create large per-feature suites.

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
- integrated event mass convergence

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

## Cross-branch review gates

Before merging, an automated review checks:

```text
contract API version
trace schema version
legacy-pack hash
units and frames
array shapes
source and event ID preservation
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
- event `Qz`, `L`, and wavelength semantics agree
- ordered and stacking return interchangeable event-aligned intensities
- outgoing film wavevectors can be consumed by geometry
- OSC and simulation coordinates meet at exactly one boundary

## Integration sequence

Integrate through vertical slices:

1. synthetic source to synthetic incident state
2. ordered rod catalog to mosaic with synthetic incident states
3. geometry incident states to mosaic event generation
4. event-aligned queries to ordered intensities
5. event outgoing wavevectors to exit refraction and continuous detector hits
6. hits to integration-owned detector deposition
7. substitute stacking strength for ordered strength under the same contract
8. run the tiny end-to-end detector case

## Later fitting proof

Each fit stage must first recover parameters from synthetic data generated by the accepted forward model.

- Source fit: recover size, divergence, and declared correlations from multiple-distance direct-beam data.
- Detector fit: recover pose and beam center from calibrant or independently constrained observations.
- Sample geometry fit: recover sample/goniometer parameters from frozen peak associations and pass an outer re-index audit.
- Mosaic fit: recover core width, tail width, and mixture from fixed profiles with source and geometry frozen.
- Ordered intensity fit: recover structural parameters and image scales from fixed selected regions.
- Stacking fit: recover transition parameters from fixed rods and branches with upstream states frozen.

Measured-data improvement is not proof without synthetic recovery and parameter-identifiability diagnostics.

## Tolerance freeze and proof sensitivity

Before T02--T05 compare with the shared pack, they load `proof/stage_tolerances_v1.json` through the strict loader and record canonical SHA-256 `d3739963a8decf481fc7ec87723854ef7628e8da02dbcb3e6f7e5bb41522b4b3`. Tolerances may change only through reviewed proof-base work, never after seeing branch error. Exact calculations do not invent convergence; broad sweeps, mutations, and benchmarks remain one-shot evidence rather than permanent frameworks.

A branch is not proven when the fixture is insensitive to likely mistakes. Run the workstream mutations from [ERROR_INJECTION.md](ERROR_INJECTION.md) and record the expected and observed first failing stage. The integration proof runs the factor-omission, factor-duplication, event-identity, OSC-orientation, and row-column mutations.
