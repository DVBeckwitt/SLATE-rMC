# Validation and proof

Tolerance selection and required negative controls are authoritative in [ERROR_INJECTION.md](ERROR_INJECTION.md).

## Proof hierarchy

1. Analytic identities and limiting cases.
2. Independent numerical oracles that do not call the public algorithm.
3. Direct manuscript-equation evaluation.
4. Shared immutable original-RASIM traces.
5. Tiny end-to-end detector result.

## Current contract-v12 runtime

The production runtime is the continuous detector pushforward described in
`CONTINUOUS_EWAL_COATING_STRATEGY.md`. The sampled mosaic/scattering-event
selection/deposition implementation remains retired. Contract v11 introduced the streaming weighted
Monte Carlo pixel-mass terminal in D035; it exposes no sampled-event runtime. Contract v12 adds the
deterministic, pointwise intrinsic Ewald outgoing-direction density in D036 and explicit progressive
CPU/CUDA execution, cancellation, causal geometry rebinding, provenance, and full-native
presentation in D037. These execution and presentation changes do not change the estimator's
measure or expose a sampled-event runtime. Current proof authority is the latent Bragg density,
analytic line/sphere oracle, detector-coordinate inverse map, intrinsic solid-angle inverse map,
finite pixel-box integral, refined forward-latent integration, and scalar-versus-compiled backend
agreement. Earlier sections in this file are retained historical evidence for the source-to-`ki`
and scientific subsystem cutovers; they do not reinstate retired APIs.

### Intrinsic Ewald outgoing-direction density

The contract-v12 result evaluates the complete internal-film outgoing-direction sphere directly in
sample coordinates. It returns density in `A2/sr`, includes both analytic inverse branches and all
regular inverse preimages, preserves both signs of sample-frame `Qz`, excludes the separately owned
`m=0` contract, and applies no exit, detector-visibility, or panel mask. Near-unit input directions
are normalized before `Q = k_film * kf_hat - ki` is formed. Returned arrays are immutable; positive
numerator caustics are explicitly marked and represented by positive infinity.

The compact independent proof differentiates the forward outgoing unit direction with respect to
the latent mosaic coordinates at a regular one-preimage point. Dividing the forward coating by the
numerical solid-angle Jacobian agrees with the inverse result: `0.005631826764734138 A2/sr` versus
`0.005631826767131236 A2/sr`, a relative error of `4.26e-10`. A known all-root point has four
regular preimages and the branchless result equals the sum of branch 1 and branch 2. A separate
change-of-measure test verifies

```text
detector_density = ewald_direction_density * J_Q / k_film^2
                   * optical_weight * source_phase_weight
```

at the same ray. These checks detect a missing or duplicated `k_film^2`, an air-wavevector
substitution, an extra Ewald coarea factor, a spurious `sin(alpha)`, dropped inverse sheets, and
detector-only masking of the intrinsic sphere. A coarse whole-sphere midpoint sum is deliberately
not a permanent total-intensity oracle: exact caustic curves and narrow structure-factor features
make uniform-raster convergence non-monotone. The publication raster is therefore declared exact
point sampling of the continuous function, while integrated-mass proof remains on regular patches
and the independently refined forward-root integral.

No accepted legacy result declares this solid-angle measure, so the legacy classification is
`NO_ORACLE`; the analytic change of variables, independent finite differences, and detector bridge
are its numerical authority.

The detector-visible intrinsic variant uses the same inverse density and adds only the canonical
active-panel selection. Its independent regular `m=0` proof differentiates the forward outgoing
direction with respect to both mosaic coordinates and sums the two inverse orientation preimages.
At the 10-degree Bi2Se3 seed (`alpha=2 degrees`, `beta=90 degrees`) the forward finite-difference
oracle and inverse result agree at approximately `0.0063736772 A2/sr`. The reported gap is
`0.7077572188 Ainv`, and the selected point has strictly larger `|Q|`; an off-panel coordinate is
zero. A nonzero rod evaluated at the same visible direction agrees with the unmasked intrinsic
direction evaluator, detecting accidental optical, source, coarea, or detector-Jacobian factors.
Deterministic row tiles assembled from two Windows `spawn` workers are bitwise identical to the
serial result on the compact proof grid.

The historical full-sphere Bi2Se3 10-degree publication render evaluated the intrinsic sphere at `720 by 1440`
(1,036,800 point samples), with 84 non-specular rods and 64-row tiles. The manifest-recorded exact
Ewald stage took `2514.9363 s`; the configured build, reciprocal sample, untilted detector, tilted
detector, and schematic-ray stages took `0.0867 s`, `37.5527 s`, `41.4108 s`, `36.8181 s`, and
`8.0886 s`, respectively. Both detector fields were evaluated independently at `1440 by 1440`.
From process start to atomic manifest installation the single-process run took approximately
`3463 s`; the largest observed Windows working set was `5,058,682,880` bytes. The renderer performs
no sphere rebinning: the publication surface consumes the direct pointwise raster at its declared
resolution, while the small sphere glyph in the projection schematic alone uses a display-only
coarsening.
### Weighted Monte Carlo detector-pixel estimate

The compact permanent test fixes one canonical source state, nonunit source and rod weights,
regular detector-visible `m=0`, one nonzero physical rod, three Philox draws, both nonzero roots,
three retained zero/miss root slots, exit optics, draw normalization, and exact native-pixel
ownership. It independently reconstructs the fixed-width Philox counter blocks and compares every
nonzero weighted pixel plus every complete-source replicate total against the public scalar
`map_latent`/detector-visible-`m=0` route at mixed tolerance `rtol=1e-8`, `atol=5e-19 A2`. The
existing inverse/forward detector Jacobian and optical parity test remains unchanged and passes.

An external independent integral used one default source state, the first physical `m=1` rod
`(-1,0)`, both roots, Gauss-Hermite order 96 in the underlying Gaussian tilt, and 32,768 periodic
beta midpoints. The refined scalar `map_latent` oracle was `2.78498e-4 A2`, with a conservative
`1e-8 A2` quadrature envelope from the checked Hermite-order sequence. One million contract-v11
PCG64 draws gave `2.7996971437331547e-4 A2` in `2.514 s`, 433,868 visible root deposits, a
left-half-panel fraction `0.4993680` versus oracle `0.5000136`, and normalized 50-by-50 total
variation `0.0299513`. The existing 50-by-50/order-2 preview gave
`6.071882071553833e-5 A2`, left-half fraction `0.7272608`, and total variation `0.35926`; this
classifies the first divergence as preview quadrature under-resolution rather than stochastic
normalization. The draw-based spread statistic was consistent with the oracle, but is not promoted
to a Gaussian error bound because a visible Ewald fold may make the importance-weight second moment
diverge.

The historical contract-v11 warm all-default benchmark used all 1,000 valid configured `ki` states,
85 rods, every
retained root, the 3,000-by-3,000 detector, seed `20260728`, and one process. The existing
60-pixel/order-2 preview evaluated 10,000 detector coordinates in `8.5131 s` and returned
`0.00262385385236163 A2`; it is explicitly a display preview, not a precise native-pixel oracle.
The streaming estimator measured:

| draws per `ki` | total mosaic draws | warm wall time | speed versus preview | mass (`A2`) |
|---:|---:|---:|---:|---:|
| 1 | 1,000 | `0.08687 s` | `98.0x` | `0.003523800840700506` |
| 40 | 40,000 | `0.41221 s` | `20.65x` | `0.003927799623402615` |
| 256 | 256,000 | `2.1507 s` | `3.96x` | `0.00384218539103489` |
| 1,024 | 1,024,000 | `8.3025 s` | `1.03x` | `0.003775901673536583` |

Cold configured-model construction in the final benchmark process took `3.1086 s` and is excluded
from both terminal timings. At 40 draws per state the estimator retained only the final 69-MiB
float64 image and small ledgers; `tracemalloc` peak Python/NumPy allocation was 153,006,120 bytes.
The timing comparison is intentionally honest: low draw counts are much faster and noisy, while
1,024 draws reached the preview's wall-time scale and a stable default mass near the independent-seed
`0.00375--0.00380 A2` range. No claim equates the coarse preview's work or accuracy with precise
3,000-by-3,000 deterministic integration.

### Fluid full-native execution and presentation qualification (T19)

Task [T19](../tasks/19_fluid_detector_viewer.md) replaces the execution realization with
`numpy.philox.fixed_width_source_draw.v1`. The independent scalar test reconstructs the Philox key
and counter blocks without calling the production sampler and covers regular `m=0`, both nonzero
roots, structure, Ewald coarea, entrance/exit optics, source/rod/phase/polarization factors, hard
native-pixel ownership, replicate totals, and mass conservation. Separate tests establish exact
draw-prefix and source-prefix identity, exact one-versus-four-worker CPU reduction, named
cancellation with no partial result, and rejection of a geometry rebind whose rods or topology
change.

On the available NVIDIA GeForce RTX 3060, the forward CUDA fixture matches CPU occupied pixels and
integer ledgers exactly; the 1,000-state/85-rod parity probe had maximum image difference
`1.11e-16 A2`. Progressive and one-shot output on the permanent CUDA fixture agree within
`rtol=8e-11`, `atol=3e-24 A2`. The cancellation proof first establishes a one-draw prefix, cancels
after a device launch, verifies the sampler resets to zero draws, and then reproduces the clean
settled result. A geometry-only CUDA rebind matches a freshly compiled sampler. Patched device
absence fails at backend selection; there is no CPU fallback under a CUDA identity.

The latest-only scheduler proof submits revisions A/B/C, proves B never starts, rejects A's stale
completion, and publishes only C's deduplicated 1/4/8/requested sequence. A failed settled stage is
retryable. Detector-only controls retain the incident transport object, including when a prior
sample correction is nonzero; validity-changing sample geometry rebuilds incident transport and
re-enumerates rods. CPU and CUDA projection-only rebound images match a fresh complete geometry
rebind, a full-rebind/pose-rebind sequence returns to the original image, and device identity checks
prove all four transport arrays remain resident. A non-square sentinel proves the
presentation boundary preserves complete `[row,column]` shape, corners, element count, source
float64 bytes, and exactly one texture-coordinate row flip. A live Qt/OpenGL smoke initialized the
R32F texture and uploaded a frame successfully.

For the full 1,000-state, 85-rod, 3,000-by-3,000 workload, cold configured construction took
`4.398 s`, CUDA sampler allocation `1.057 s`, and the first preview including JIT `2.434 s`. Median
warm incremental stages were `16.494 ms` (draw 1), `16.823 ms` (add draws 2--4), `28.666 ms` (add
draws 5--8), and `244.746 ms` (add draws 9--49). Materializing the already-computed 49-draw
authoritative float64 snapshot added `59.351 ms`. Before projection-only specialization, compatible
model rebuild/rebind/first-preview medians were `15.039/20.161/16.241 ms`, and complete detector
interaction preparation took roughly `69--81 ms`. After specialization, 10 interleaved warmed pitch
and translation revisions measured `28.074 ms` and `28.342 ms` median through full-native texture
preparation; every revision stayed below `31 ms`. The four transport device arrays retained exact
object identity. Device normalization and pinned full-native transfer are included in preview
stages. Persistent full-native OpenGL upload plus shader draw separately measured `11.691 ms`
median with a GPU completion fence. The `5 ms` scheduler cadence coalesces input continuously while
latest-only cancellation prevents a stale backlog. Long-refinement cancellation returned
`11.594 ms` median and `28.935 ms`
maximum after the user-event signal. The workspace owned `105.464 MiB` of device buffers plus one
reused `34.332 MiB` pinned host frame; process RSS was about `353 MiB` in the memory probe. These are
execution measurements, not scientific tolerances.

Legacy CPU forward results with an identical explicit latent block are `MATCH`. The change from the
v11 PCG64 realization to Philox is `NO_ORACLE` pixel-for-pixel because both are stochastic estimates
of the same measure; the independent latent integral and conservation laws remain the authority.
CUDA is `MATCH` within the frozen cross-device tolerance. Legacy release-only scheduling and
Matplotlib latency are `NO_ORACLE` operational behavior.

The clean reference-integrity gate streams and validates all eight declared compressed OSC inputs,
then compares the three named immutable-pack OSC identities and intermediates. Its inventory is
identity-based rather than frozen to the historical five-file count; Python example tools are
tracked by the repository file manifest, and tools or generated bytecode are not classified as
scientific input data. Cache residue remains a separate clean-worktree failure.

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

The limitations at this historical proof point were explicit: the direct `m=0`, `Q=0` root remains excluded; only regular
detector-visible `m=0` support with a positive reciprocal gap is admitted. Fixed macrobin output is
a preview estimate, not an adaptive convergence claim. Detector response, background, finite
caking, and intensity fitting were deferred. The later fixed-position ordered-intensity section
supersedes only that final intensity-fitting statement.

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
- intrinsic `A2/sr` density versus an independent finite-difference solid-angle Jacobian
- all-root branch decomposition and the detector-to-sphere change-of-measure identity
- complete-sphere Ewald closure, both sample-frame `Qz` signs, and absence of detector-only masks
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

The shared-series proof independently constructs a nonzero hidden pose with eight shared geometry
coordinates plus one common incidence-angle delta and exact observations at 5, 10, and 15 commanded
degrees. It rejects the one-image and two-image subsets at rank 5/9 and 7/9, then requires rank 9/9
for the three-image fit. Activating both the common delta and `sample_normal_x_tilt_rad` fails
explicitly because they share the nominal incidence-axis gauge. The permanent fixture verifies normalized
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
`21.3000088` to `5.6507934`. Per-image RMS is `1.2531695`, `0.8406036`, and `2.2866604` pixels for
5, 10, and 15 commanded degrees. The scaled Jacobian is rank 9 with condition `8689.9123`; its
singular values are `(76.2350, 72.4742, 58.2316, 4.34862, 3.34098, 2.27897, 0.256387,
0.100492, 0.00877282)`. Three deterministic starts differ by at most `2.658e-5` pixels in predicted
native coordinates. Holding out `L={4,11}` gives RMS/max `1.3173882/2.0219029` pixels.

The shared correction vector, in contract order, is `(-0.004499432, -0.023011841, 0,
0.015547811, -0.015190829, -0.015251339, 49.5499 um, 99.999996 um, -27.7326 um)`.
The ninth fitted Jacobian coordinate is the one common
`delta_theta_i=0.0073255030 rad=0.419720406 deg`, producing effective incidences
`5.419720406`, `10.419720406`, and `15.419720406 deg`. It is one scalar, not three image-specific
offsets.
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

The verified v2 shared-delta checkpoint took `246.109124` seconds for the two separately reported
OSC discovery/indexing passes (`122.717221` and `123.391902` seconds). Geometry-only setup took
`0.0641544` seconds, the selected primary solve `3.259482` seconds, frozen-coordinate relabeling
`0.626129` seconds, and the warm joint residual median `0.0285933` seconds. Traced fit peak memory
was `298,023` bytes; OSC decoding and the global searches are reported separately because they
retain detector-sized arrays.

The combined nine-coordinate rank is a statement under fixed detector calibration and explicit gauge
ownership, not a claim that every viewer control is estimable. Detector roll, sample/crystal roll,
axis-parallel pivot motion, sample tangent translations, detector center/pitch/distance, wavelength,
and per-image corrections stay fixed. The pivot-pitch/pivot-yaw/plane-offset combination is the
weakest direction for the measured three-angle design, so a fourth incidence is recommended for an
independent validation and parameter-level pivot uncertainties must be reported honestly.

## Historical one-state deterministic mosaic recovery

This retained synthetic proof predates the required 250-state source reduction and is not the active
real-OSC estimate or rendering source. It fixes accepted nine-coordinate geometry manifest
`1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132`, detector center
`(column,row)=(1453.12,1596.422) px`, lattice, material, and one source-center ray at
`1.540592925 A` with zero spatial spread, divergence, and bandwidth. Synthetic truth is Gaussian
sigma `2 deg`, Lorentzian HWHM `0.5 deg`, and Lorentzian probability `0.1`.

The objective contains 32 profiles in one simultaneous 5/10/15-degree fit: the frozen OSC-indexed
nonzero selection contributes 10, 8, and 8 profiles, and the raw-supported fixed-model `m=0`
selection contributes `{003,006}`, `{006,009}`, and `{006,009}`. Every profile owns one
independent analytically profiled nonnegative amplitude. Deterministic planted amplitudes span
`1.480148585e-6` to `1.136487411e5`; rescaling any observed or simulated profile leaves the
relative-shape objective unchanged. Absolute peak heights, structure-factor amplitudes, and
cross-reflection intensity ratios therefore do not weight the shared mosaic parameters.

Raw OSC evidence also contains significant `003` at 10 and 15 degrees. Although the nominal
minimum-tilt landmarks are `BACKWARD`, both raw centroids remain provisional candidates until the
complete source-averaged all-root detector is evaluated. Direct 6-by-6 truth-mosaic probes have
41/41 normalization-valid bins but exactly zero `m=0` signal across the combined source, so the
combined-source gate records them as `SOURCE_AVERAGED_FORWARD_MODEL_UNSUPPORTED` and excludes them.
Thus the six fitted `m=0` profiles are exhaustive for the current 250-state forward model, not for
every feature in the raw OSC files. Candidate `L` enumeration uses the union of the source-state
kinematic limits; the nominal minimum-tilt status is indexing provenance, not the eligibility gate.

A kernel-certified geometry audit found an oblique alpha-support seam in 43 bins across all 26
nonzero profiles; two 15-degree `m=1,L=5` bins also cross an exit fold. These are numerical-support
boundaries, not intensity thresholds. The frozen case excludes the same 43 bins from independent
truth and every component response, retaining every profile, at least 38 of 41 bins per nonzero
profile, and all 41 bins per `m=0` profile. Before exclusion, the 6-by-6 versus 8-by-8 same-mosaic
maximum relative shape discrepancy was `3.30581%`; afterward it was `0.0286901%` with RMS
`0.0051565%`. Each mask record and the aggregate topology hash bind the dataset, `m`, `L`,
analytic branch, exit side, representative rod, and excluded indices. A general topology-split
cubature that retains these bins remains unimplemented.

The immutable case bytes have SHA-256
`2c8d647738fb50721e2e08bd23eb405c770301e95477c8d37b9f02fa729ba295`; the runner captures those
bytes before setup and rejects any change before writing its manifest. With the predeclared
maximum per-profile relative-L2 gate restored to `0.02`, the masked independent-order recovery is
accepted:

```text
truth       = (2.000000000 deg, 0.500000000 deg, 0.100000000)
recovered   = (2.004344901 deg, 0.499233477 deg, 0.101409964)
signed error= (+0.004344901 deg, -0.000766523 deg, +0.001409964)
```

The joint objective is `3.826850872e-5`. All 32 profile-shape gates pass: maximum relative L2 is
`0.001368725`, median `0.001097962`, and RMS `0.001093568`; the six `m=0` profiles have
maximum `0.001311679`. Nuisance-projected sensitivity has rank 3, singular values
`(3.454657208, 0.248925672, 0.153301368)`, and condition `22.5351`. Direct mixed profiles agree
with the two pure-component decomposition to `3.43424e-16` relative error and component
normalization differs by exactly zero.

Distribution total variation is `0.0008826134` and Wasserstein-1 distance is
`0.00360309 deg`. Absolute `q50/q90/q99` errors are
`0.00116582/0.00670307/0.01888996 deg`. The final-byte proof run's setup, independent truth
profiles, response-bank fitting, and post-fit component checks took `84.214`, `9.073`, `150.204`,
and `1.381 s`; skip-image total wall time was `245.231 s` and traced Python/NumPy peak memory was
`126,121,913` bytes. CUDA profile evaluation used an NVIDIA GeForce RTX 3060. The accepted
numeric artifact is external under
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\mosaic_recovery_provenance_final`.

The separately generated 3,000 by 3,000 truth images use all 85 rods, every retained inverse root,
and include `m=0`. They do not use planted nuisance amplitudes and are neither fitted observations
nor recovered-model renders. Their forward model is unchanged by the profile-bin mask.

This proof is `NO_ORACLE` for real-OSC mosaic recovery: it is synthetic self-recovery with a
denser independent angle-bin rule on the retained smooth support. The fit consumes presegmented,
indexed continuous profiles; no detector raster enters the objective. Background, detector PSF,
counting noise, covariance/uncertainty, held-out subsets, additional truth regimes, automatic
arbitrary-material support-boundary planning, and topology-split cubature remain unproven. The
material-neutral profile and fitter contracts are reusable; the tracked end-to-end runner and its
frozen boundary audit remain Bi2Se3-specific. The measured-data slice below exercises real-OSC
profile extraction, but remains a model-limited estimate rather than an independent truth oracle.

## Model-limited three-OSC Bi2Se3 mosaic estimate

The measured-data run fixes the same nine-coordinate geometry manifest
`1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132`, immutable case SHA-256
`2c8d647738fb50721e2e08bd23eb405c770301e95477c8d37b9f02fa729ba295`, detector center
`(column,row)=(1453.12,1596.422) px`, lattice, and material. Each incidence uses the same
deterministic 250-row source realization, revision
`42e8108b3ba9c66cb7752bc7e9d2f9fdb8401987a8f9a0eb272d3f4edeaa39a7`, with spatial sigma
`2.12330450072e-5 m`, divergence sigma `3.705865455998e-4 rad`, mean wavelength
`1.540592925 A`, and wavelength sigma `0.010784150475 A`. All 250 state intensities are reduced to
one detector function per incidence before any profile comparison. A separate one-row source-center,
zero-divergence, mean-wavelength companion defines geometry landmarks only and contributes no
detector intensity. No geometry coordinate, beam-center coordinate, lattice constant, or structural
absolute-intensity parameter is optimized; one nonnegative nuisance amplitude is analytically
profiled per retained profile. The three detector-native observations are decoded from
`Bi2Se3_5m_5d.osc.gz`, `Bi2Se3_10d_5m.osc.gz`, and `Bi2Se3_15d_5m.osc.gz`, whose compressed
SHA-256 values are respectively
`cba9d38d9cae1e8e3be3c2ad58ad130e04ce1958f0d2d986a947ab3a90818e36`,
`faaa8f77785ddcd51c741d3b53c3ab7694fb8dacb961cd414d20daa5c21ccd67`, and
`6cfed1e0e7a00ecc3c3d4d2d18de7de4696c78a11de5ad842e31368d5e986fd1`.

The frozen measured-profile policy has SHA-256
`c85286056aab0858bc70fd5236504e193d05cbf0b30cad87ed69160a82b7c027`. It excludes an entire
profile when its central excess energy divided by local sideband scatter is below `5.0`; it never
trims individual bins. Both 10-degree `m=1,L=4` sides are also excluded as the user-authorized
secondary-lobe pair. Every branchless `m=0` candidate must first have positive finite signal in its
complete 250-state source-averaged simulated profile; the selection audit records that modeled
support. Of 34 indexed candidates, the joint fit retains these 15 profiles:

The current handoff uses mosaic schema `rasim-bi2se3-real-mosaic-fit-v3` and modeled-support gate
revision `positive-combined-detector-m0-profile-signal.v2`. Version 3 additionally carries the
upstream position revision, corrections, common incidence delta, and commanded/effective angles;
the ordered-intensity runner rejects older artifacts before compiling a response. The numerical
mosaic result below predates this v3 position handoff and awaits requalification after the verified
position checkpoint.

- 5 degrees: `m=0` orders `003` and `006`;
- 10 degrees: `m=0` order `006`, plus both sides of `m=1,L=5` and `m=1,L=10`;
- 15 degrees: `m=0` orders `006` and `009`, plus both sides of `m=1,L=5`, `m=1,L=10`, and
  `m=1,L=11`.

All 5-degree nonzero-`m` candidates, 10-degree `m=0,009`, both 10-degree `m=1,L=8` sides, and
both 15-degree `m=1,L=8` sides are weak under that frozen rule and are excluded. Raw-significant
`003` at 10 and 15 degrees remains provisional despite its backward nominal minimum-tilt landmark,
then is excluded because its complete 250-state combined-source profile signal is zero. It is
neither called absent nor fitted. Every retained profile owns
an independent nonnegative profiled amplitude, and each normalized profile contributes equally.
Consequently absolute intensity, structure-factor uncertainty, and cross-reflection intensity ratios
do not weight the shared mosaic result.

The recovered common effective radial distribution is

```text
Gaussian sigma            = 1.322875656 deg
Gaussian FWHM             = 3.115134111 deg
Lorentzian HWHM           = 0.489897949 deg
Lorentzian FWHM           = 0.979795897 deg
Lorentzian probability    = 0.448096163
```

This is an axisymmetric probability mixture, not a Voigt convolution. The objective is
`1.680259188`; all 15 fitted amplitudes are positive. Relative-profile residual RMS is `0.334690`
overall, `0.163634` for the five `m=0` profiles, and `0.393240` for the ten nonzero-`m` profiles.
The nuisance-projected sensitivity has rank 3 and condition `10.7987`. Replacing the primary
constant signed-detrending basis with an affine basis changes Gaussian sigma by `-0.0361416 deg`,
Lorentzian HWHM by `-0.0686663 deg`, and Lorentzian probability by `-0.0766298`.

The result is explicitly `MODEL_LIMITED_EFFECTIVE_RADIAL_MOSAIC_ESTIMATE` and legacy
classification `NO_ORACLE`. Both strong 15-degree `m=1,L=5` profiles remain poorly reproduced
(relative residuals at least `0.65`), and signed detrending makes eight reconstructed bins negative,
with a minimum of `-40.7514 counts/px`; that additive basis is not a physical background model.
Without a calibrated source/detector point-spread function, this envelope is not uniquely intrinsic
sample mosaic. The deterministic background sensitivity is not a statistical confidence interval.

The current gate-v2 CUDA run took `344.952 s` total: `56.379 s` setup, `40.524 s` OSC projection
and model layout, `178.265 s` component-bank construction and constant-background fit, and
`68.937 s` for the affine-background robustness fit. Traced Python/NumPy peak memory was
`206,667,033` bytes. Its JSON SHA-256 is
`69c782f67216ea595b72abced82a9660833d1acbee7b38ffaedb2de7d81df829`; the diagnostic NPZ SHA-256
is `e41245290268ab9c57fa891a3631ef401e6181282f8c2c672f3a2a6779857858`. The current external
artifacts are
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_mosaic_gate_v2\bi2se3_real_mosaic_fit.json`
and
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_mosaic_gate_v2\bi2se3_real_mosaic_fit.ra_diag.npz`.
The retired gate-v1 JSON SHA-256
`a5e80ca4842c068f5779d19481cf489b4871a45281c0650977376ea4d9b1f7d8` is historical numerical
evidence only; current consumers reject its 32-candidate catalog. No independently denser
source-support or higher-order response convergence audit was completed, so the estimate remains
explicitly model-limited.

## 250-state fixed-position full-catalog synthetic ordered-intensity recovery

The full-catalog T13 proof freezes the accepted nine-coordinate geometry, detector center, lattice,
52-layer 2H stacking law, both Bi2Se3 Wyckoff coordinates, and the model-limited 250-state mosaic
estimate `(sigma_G, HWHM_L, eta)=(1.3228756555 deg, 0.4898979486 deg, 0.4480961630)`. Each of the
5/10/15-degree views uses the same source revision and 250-state specification as the measured
mosaic fit. The compiled observable is one source/root-reduced detector function per incidence;
full detector evaluations occur only while compiling, generating independent truth, and certifying
the compact response. Optimizer candidates use that combined response before any image scale,
normalization, or residual. The three incidence blocks are then fitted together. No per-source fit
coordinate exists.

Only `oBi`, `oSe1`, `oSe2`, `Ur`, and `Uz` are active; atom positions, geometry, mosaic, lattice,
optics, and stacking remain fixed. Synthetic observations are selected-group peak-center angular
signal densities in `A^2/rad^2`, not finite-ROI masses or unresolved raw OSC intensities. The frozen
unfiltered catalogs retain all `88/78/72 = 238` geometry-admitted centers: `86/76/70` nonzero-`m` centers plus
`{003,006}`, `{006,009}`, and `{006,009}`. Weak nonzero centers are not pruned, and all six admitted
`m=0` centers enter the same joint fit. Each admitted `m=0` center additionally has positive finite
baseline support when evaluated through the complete 250-state selected-group detector; the nominal
one-state geometry companion is not accepted as intensity evidence by itself.

Six occupancy probes through each selected-group view of the full source/root detector at 13
Chebyshev-Lobatto `Uz` nodes compile the occupancy quadratic after source averaging; exact group
`Qr^2` damping is applied separately. Twelve
interlaced full-detector nodes certify the noncommuting source/root `Qz` response over every
occupancy direction through a profile-local generalized-eigenvalue bound with a declared `1e-12`
extinct-mode floor; no scale is shared across profiles. The maximum occupancy-signal certificate is
`2.084780066e-10`, and a fresh authoritative combined-detector truth
prediction agreed with the compact response to `8.976202640e-15` maximum relative error.
The current ordered result contract is `rasim-bi2se3-ordered-intensity-recovery-v4` with compiler
contract `source-averaged-selected-center-occ-quadratic-chebyshev-qz-spectral.v2`. Version 4 binds
and reconstructs the upstream v3 position state; the renderer rejects the older position handoff or
coefficient-norm certificate contract. The numerical ordered result below is the prior v3 proof and
awaits requalification after mosaic is resumed.

For the final native-center images, an exact fitted 10-degree 250-state/85-rod/all-root 96-by-96
block benchmark took `1.17849 s` on CPU and `0.499570 s` on the NVIDIA GeForce RTX 3060 (`2.36x`).
CPU/CUDA scale-normalized disagreement was `1.235e-13`, and validity/caustic masks were identical.
The fit response therefore remains CPU-compiled while rendering uses CUDA; both backends and devices
are recorded independently in the image manifest.

Both fits start away from the planted values. Absolute-calibration recovery was:

```text
parameter  truth        recovered    absolute error
oBi        0.940000000  0.940000000  1.110223025e-16
oSe1       0.780000000  0.780000000  0
oSe2       0.860000000  0.860000000  1.110223025e-16
Ur A^2     0.007000000  0.007000000  2.081668171e-17
Uz A^2     0.034000000  0.034000000  5.551115123e-17
```

Absolute sensitivity has rank 5, condition `9.42209`, and no active bound; the maximum relative
peak residual is `8.22e-15`. With one independent scale per image, the common occupancy multiplier
is an exact gauge. Fixing `oBi=1` recovers `oSe1/oBi=0.8297872340`,
`oSe2/oBi=0.9148936170`, `Ur=0.007 A^2`, and `Uz=0.034 A^2` to at most `1.11e-16` absolute error.
Relative sensitivity has rank 4, condition `4.12232`, and no active bound. The truth densities span
`8.851768476e-7` to `3.897497069 A^2/rad^2`, a dynamic range of `4.403071634e6`; every anchor is
retained irrespective of intensity.

The three compact responses contain `6864/6084/5616` coefficients. Compilation took `466.816 s`;
a fresh equivalent full-detector truth prediction took `14.3893 s`, versus `0.0008786 s` for the
cached response (`16377.5x`). The two joint solves took `0.0363 s` and `0.0382 s`; total proof wall
time was `598.787 s`, and traced peak memory was `78,770,131` bytes. The historical full-catalog
JSON SHA-256 is
`a44bfabf7ff943b00a51155c71d2f4c1e7e918e0b1ba10794c7f5a68a554f67b`; it is external at
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_sf_final\ordered_intensity_result.json`.

That archived ordered result predates exact measured-selection propagation and is not a current
`--mosaic-result` handoff proof. With a gate-v2 measured mosaic, the runner instead compiles the
exact `2/5/8 = 15` upstream fit-eligible identities, including five `m=0` profiles; unknown datasets,
missing identities, and stale gate-v1 artifacts fail before response compilation.

The current measured-selection handoff compiled `156/390/624` response coefficients in
`171.912 s`. Its 13-node response passed the 12 interlaced-node certificate at
`1.19096e-11` maximum relative error and agreed with a fresh direct combined-detector prediction
to `8.09542e-16`. Fresh and cached truth predictions took `3.83954 s` and `0.0007503 s`
(`5117.35x`). Absolute recovery had rank 5, condition `31.1434`, maximum peak residual
`1.9984e-15`, and maximum parameter error `1.22e-15`; relative recovery had rank 4, condition
`12.8583`, maximum peak residual `1.1102e-15`, and maximum parameter error `1.17e-16`. No
parameter contacted a bound. Total wall time was `249.477 s`, with traced peak memory
`62,674,047` bytes. The result binds upstream mosaic SHA-256
`69c782f67216ea595b72abced82a9660833d1acbee7b38ffaedb2de7d81df829`; its own SHA-256 is
`432a402bf675c0411c399fd7f0327175be36b54576a611dfc431cdaef5a048ea`, external at
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_sf_gate_v2\ordered_intensity_result.json`.

The post-fit display-only render then sampled the three complete source-averaged detector functions
at native pixel centers. It did not rasterize or integrate pixels during either fit. The 5/10/15
degree images evaluated `4,577,845/4,377,290/4,170,019` top-exit coordinates, respectively, with
all 250 valid source states, all 85 rods, all retained roots, and `m=0`. CUDA rendering on the RTX
3060 took `3839.456 s`; the CPU-compiled fit-response backend and CUDA render backend are both
recorded per incidence. The manifest binds mosaic JSON
`a5e80ca4842c068f5779d19481cf489b4871a45281c0650977376ea4d9b1f7d8` and ordered JSON
`a44bfabf7ff943b00a51155c71d2f4c1e7e918e0b1ba10794c7f5a68a554f67b`, and hashes every emitted
PNG. The combined-panel SHA-256 is
`d5d3639cd154576d0df3ea827f1bca510d02913488eca95a7017bf193baf254f`; the image manifest is
external at
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_images_final\bi2se3_250ki_images.json`.

Visual inspection confirms coherent, correctly oriented image files but also substantial model
discrepancy: the source-averaged forward images contain broad ring/tail intensity not comparably
visible in the raw OSC panels. Raw counts and simulation density use separate shared log transforms,
so this is not a count-calibrated residual. It reinforces the declared `NO_ORACLE` status for raw-OSC
structure recovery; the exact synthetic SF self-recovery must not be described as a raw-intensity fit.

An earlier finite-ROI attempt was rejected before this point-density design: source-dependent
root/fold topology changed across the 250 states, narrow finite-stack profiles made quadrature
oscillatory, and central integer-`L` `Qz` damping did not commute with the source/root sum. Its first
divergences included `0.6599` relative q12/q16 disagreement and `0.02726` central-`Q` error. Those
results are not rendered or reported as accepted evidence.

This is a synthetic fixed-position identifiability and acceleration proof. Raw OSC ordered-intensity
recovery remains `NO_ORACLE`: background, component deblending, count calibration, detector PSF,
model discrepancy, statistical uncertainty, atom motion, arbitrary per-site `Uij`, and a
material-neutral structure basis remain unproven.

### Superseded one-state finite-ROI proof

The following is retained only as historical evidence for the older selected-group ROI-mass path;
it is not the active 250-state result and must not be used for the current images or conclusions.

The first T13 slice freezes the accepted unconstrained nine-coordinate geometry (including its
detector-tilt corrections), detector center, lattice, one ideal source-center ray, the recovered
`(2 deg Gaussian, 0.5 deg Lorentzian, eta=0.1)` mosaic, the 52-layer 2H stacking law, and both
Bi2Se3 Wyckoff coordinates.
Only `oBi`, `oSe1`, `oSe2`, `Ur`, and `Uz` are active. The directional intensity factor is
`exp[-Ur*Qr^2-Uz*Qz^2]` in the crystal/layer frame; optimizer iterations use only cached occupancy-
quadratic coefficients and cached `Qr^2/Qz^2`, with no detector reprojection or full structure-
factor call.

The frozen 5/10/15-degree profile catalogs contain `88/78/72` selected-group component masses:
`86/76/70` nonzero-`m` profiles plus `{003,006}`, `{006,009}`, and `{006,009}`. Their exact identity
catalog revisions are `198eb585e118e0e719e13303c480cc61a5ef3f0628e64220d26844875bc1114f`,
`4e448c4c05b32813d6a998db3da65622ee4ef3a3e1757b349da9ea869984f679`, and
`73ce9345df429759d17445f9d7aa48ab4a07600be5575886c4f6a208fab652b3`. The geometry-
only signature probe conservatively excludes `186/142/135` of the `4,998` total phi bins, all from
nonzero-`m` profiles; every profile retains at least 12 of 21 bins and every one of the six `m=0`
profiles retains all 21. The probe is finite, not a general topology certificate.

The fit response uses `(two-theta,phi)` Gauss orders `(12,4)` and the independent full-strength
truth response uses `(16,8)` on the identical observable layout and topology mask. This is a
current-case refined-order agreement check, not a generalized convergence sequence. Maximum
relative planted-mass difference is `2.156462178e-4`; maximum relative six-column occupancy-basis
row-norm difference over `(Ur,Uz)={(0,0),(0.007,0.034),(0.1,0),(0,0.1),(0.1,0.1)}` is
`2.337281382e-4`.
The cached quadratic/Debye-Waller prediction agrees with the direct CIF-derived quintuple-layer and
finite-stack strength to `2.220446049e-15` relative error.

Both fits start from the CIF baseline `(oBi,oSe1,oSe2,Ur,Uz)=(1,1,1,0.019,0.019 A^2)`, without the
planted values. Hidden truth and the absolute-calibration recovery are:

```text
parameter  truth        recovered          absolute error
oBi        0.940000000  0.940001214771     1.214770939e-6
oSe1       0.780000000  0.779997717285     2.282714591e-6
oSe2       0.860000000  0.860001681396     1.681395706e-6
Ur A^2     0.007000000  0.00699952894625   4.710537540e-7
Uz A^2     0.034000000  0.0340003452570    3.452569843e-7
```

All 238 positive profiles enter the objective without intensity pruning. Their truth masses span
`5.444143579e-11` to `6.180650421e-4 A^2`, a factor of `1.135284243e7`; the maximum fractional
residual over that entire set is `2.032132531e-4`. Absolute sensitivity is rank 5 with condition
`9.47517`; no coordinate contacts a bound. With independent image scales, the exact common-
occupancy gauge is rejected, Bi is fixed as the positive ratio reference, and the recovered
`Se1/Bi` and `Se2/Bi` ratios are `0.829783719037` and `0.914894138913` versus truths
`0.829787234043` and `0.914893617021`. Relative sensitivity is rank 4 with condition `4.38329`,
maximum residual `1.997180831e-4`, and no active bound. Four predeclared noninteger-L points not
present as exact quadrature terms in any fit response agree to `1.300307410e-5`; this is exact-
model interpolation, not independent experimental validation.

The reusable q12x4 responses contain `850,944/773,568/682,752` sparse terms; the q16x8 proof
responses contain `2,269,184/2,062,848/1,820,672`. Fixed-response compilation took `351.889 s`
and refined proof compilation `891.714 s`. One equivalent three-angle full-structure prediction
took `55.0579 s`, while the cached quadratic/`Qr/Qz` prediction took `0.0468517 s`, a `1175.15x`
speedup. Absolute and relative joint solves took `2.6785 s` and `2.1458 s`; total proof wall time
was `1506.118 s`, including `147.549 s` for refined direct-truth generation. Traced peak memory was
`1,305,641,960` bytes and the observed OS peak working set was `1,497,780,224` bytes. Bounded
coordinate and structure-kernel blocks replaced the initial `5.205 GB` temporary build; a forced
multi-block parity audit retained all 2,208 fixture terms and matched unbounded cached masses to
`1.55e-15` relative error. Block order is numerical-response provenance and does not enter the
observable-layout revision.

The observable is the selected reflection-group contribution, not raw total counts inside an OSC
ROI. A spot audit of the weakest problematic 5-degree `m=4,L=2` profile found omitted all-rod tails
of `0.03708%`, below this case's `0.2%` response gate, but arbitrary materials require an all-rod
contamination audit or explicit component extraction. The upstream nominal marker catalog now
admits sites from geometry and detector topology only. Replacing all three CIF occupancies by zero
makes every diagnostic marker strength exactly zero while leaving every marker identity, coordinate,
rod group, and root unchanged. A structure-parameter trial therefore cannot add or remove profiles.

This result is `NO_ORACLE` for raw-OSC structure recovery, counting noise, background, detector
PSF, component deblending, model discrepancy, calibrated uncertainty, general per-site `Uij`, atom
motion, and arbitrary-material structure bases. The four interpolation points and local
correlations are not substitutes for independent data validation or statistical uncertainty.

## Portable Bi2Se3/Bi2Te3 staged-fit replay

The two tracked `rasim-staged-fit-replay-v2` cases make the accepted 5/10/15-degree results
repeatable from a clean clone. Exact checks cover every input hash and decoded OSC array, source
realization, fitted/fixed coordinate list, active-bound mask, profile and `m=0` identity, stage
revision, execution-runtime lineage, and classification. Numerical fits use the tolerances declared
in each case. The runner validates the complete transitive numerical dependency closure from one
hash-checked `uv.lock` read immediately before execution; resume rejects a stage from a different
runtime even though runtime provenance is deliberately excluded from its scientific revision.

Bi2Se3 now has a verified geometry-only replay checkpoint: eight shared corrections plus one common
incidence delta, one pivot bound, so parameter precision remains unqualified. The artifact is
certified only through geometry as requested. Its stage-case SHA-256 is
`c87e86f00372e7a3eaf84e71f1315eda9d55381d2513c04d5e494639cd29d6df` and its scientific revision
is `sha256-a5de9f498f8ee16f8a7d6966c958acab0250d0e416ac0bcfd305435b5c91bc72`.
Mosaic and ordered/SF code consume that exact position
state, but their numerical references below predate it and must be requalified when those stages are
resumed. The prior mosaic target was
`(sigma_G,HWHM_L,eta)=(1.3228756555 deg,0.4898979486 deg,0.4480961630)` with objective
`1.6802591883` over 15 profiles including five `m=0`. Its profile RMS/max residuals
`0.33469/0.75333` and eight negative reconstructed bins demonstrate model limitation. The ordered
target `(oBi,oSe1,oSe2,Ur,Uz)=(0.94,0.78,0.86,0.007,0.034)` is machine-precision synthetic
recovery, not unresolved raw-OSC intensity or integrated peak mass.

Bi2Te3 recomputes position-free geometry from the tracked gzip OSCs while retaining the historical
catalog for audit. Under the case-bound runtime, the tracked-container selection revision is
`sha256-97ba51fce133a27404d4b571e40fc21413fb13e8ccc70d725486b5f12fefff51`;
historical revision `sha256-79f5028d...`, catalog row-manifest `sha256-4f3755ac...`, and tracked
catalog-file SHA-256 `84cca62c...` are distinct provenance objects. The locked and historical
selections both contain 11/14/10 sites; their full hashes include floating-point localization and
covariance tokens. With detector tilts frozen, the seven-coordinate fit
has RMS/max `7.1750902/15.077870 px`, rank seven, and three active bounds; it is unqualified and
model-limited. The 33-profile/three-`m=0` mosaic target is
`(1.0340984212 deg,0.6600483356 deg,0.3446838236)` with objective `15.460968046`; median/max
profile residuals `0.69385/1.0` reject a claim of adequate intrinsic-mosaic recovery. The
31-profile/three-`m=0` ordered target is
`(Te1/Bi,Te2/Bi,Ur,Uz)=(1.0,0.8448714451,0.1,0.0)` with objective `24.118831163`; three parameters
contact bounds. It is a transferred-nuisance-amplitude estimate with `NO_ORACLE`, not direct
count-calibrated structure recovery.

An independent current-worktree CUDA replay recovered mosaic
`(1.0340984212 deg,0.6600483356 deg,0.3446812734)` with objective `15.460996255` and ordered
`(1.0,0.8448725522,0.1,0.0)` with objective `24.118711199`. The case tolerances
`(5e-6,5e-5,1e-5,2e-4)` for mosaic parameters/objective and ordered parameters/objective cover
these observed optimizer-repeat deltas while remaining much smaller than the stated model
residuals. Exact identities, profile membership, bounds, ranks, and source realizations are not
tolerance-relaxed.

The accepted Bi2Te3 mosaic and ordered stages are CUDA-qualified under the current lock; equivalent
CPU tolerance has not been established, so a CPU request through mosaic or later fails before
geometry executes or an output directory is created. The optional 250-state 3,000 x 3,000 render has the
separate historical CUDA qualification described below. Render verification hashes decoded
grayscale pixels, not PNG container bytes. Paths, timings, peak memory, and device labels are
excluded from scientific identity. Bi2Se3 rendering is disabled in this replay because available
images bind a retired mosaic result.

The locked fit-stage qualification replay used Python `3.13.13`, uv `0.11.7`, NumPy `2.4.6`,
SciPy `1.18.0`, Numba `0.66.0`, and an NVIDIA GeForce RTX 3060 with driver `572.47`. Commands use
`uv run --frozen`; the case hashes `uv.lock`, but neither the case nor uv binds the Python patch,
Windows build, CUDA driver, or GPU model. Therefore fit floats are certified through the declared
tolerances, not bitwise across arbitrary platforms. The six decoded Bi2Te3 render hashes are a
historical RTX-3060 CUDA oracle whose Python/package environment was not recorded; they await
requalification under the current lock and are not presumed to survive a different CUDA device.
Running `--through render` verifies them exactly or fails. That optional stage additionally
requires the tracked `visualization` extra for Pillow.

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
