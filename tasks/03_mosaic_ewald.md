# T03: mosaic and Ewald events

Branch: `feat/mosaic-ewald`

Start from `PROOF_BASE_SHA`.

## Goal

Produce deterministic source/orientation sampling and correct reciprocal-space scattering events with explicit probability mass, event Jacobian, and elastic residual.

## Owned paths

```text
src/rasim_next/sampling/
src/rasim_next/reciprocal/ewald.py
src/rasim_next/reciprocal/events.py
src/rasim_next/reciprocal/proof.py
tests/test_mosaic_ewald.py
this task's execution-plan and handoff sections
```

## Read-only inputs

- `IncidentSampleBatch`, `IncidentStateBatch`, `RodCatalog`, and `ScatteringEventBatch`
- immutable tracked reference pack and examples
- in-repository source/equation ledger

## Forbidden

- reciprocal-lattice or rod-catalog construction
- detector geometry or pixels
- CIF parsing, structure intensity, Parratt, or stacking
- interface optics
- branch and `Qr` selection
- fitting or acceleration frameworks
- shared contract or dependency edits

## Source and equation map

Tracked locations:

```text
Original paths resolve under reference/legacy_source/.
Manuscript equation labels resolve under reference/manuscript/.
```


Inventory: `PHY-SRC-*`, `PHY-MOS-*`, `PHY-REC-002` through `PHY-REC-009`.

Original:

```text
ra_sim/simulation/mosaic_profiles.py:15-63
ra_sim/simulation/diffraction.py:206-273, 565-1668,
2019-2600, 3113-3293
```

Manuscript:

```text
eq:mosaic_two_component_maintext
mosaic orientation-density equations in the supplement
modelling-methods Bragg-sphere and Ewald-sphere construction
detector bandwidth sum and wavelength-dependent geometry
```

## Mandatory tasks

### MOS-01: source samples

Implement deterministic spatial, directional, wavelength, and declared polarization-state samples with explicit joint or independent correlation semantics and integrated probability mass. A unity polarization assumption must be explicit metadata, not absence of data.

### MOS-02: orientation distribution

Implement independent Gaussian-like core width, Lorentzian-like tail width, and mixture weight. Normalize probability under the declared spherical measure without evaluating a singular point density at the pole.

### MOS-03: dense independent oracle

Implement a transparent dense support calculation using direct equations. It must not call the public localized solver.

### MOS-04: Ewald support

Construct the valid support for each incident state and rod. The overnight public reference may use the dense deterministic construction when it is converged; a localized/adaptive production solver is an extension unless completed and proved. Handle two-root, tangent, no-root, and specular-family cases. Use real phase wavevectors.

### MOS-05: deterministic quadrature

Integrate orientation probability and any required geometric Jacobian into `reciprocal_weight`. Do not use secondary random or quantile resampling in proof mode.

### MOS-06: event contract

Emit event-aligned internal `Q`, `Qz`, `L`, outgoing film phase wavevector, weight, residual, IDs, and validity. Do not include source weight, structure intensity, optics, solid angle, or deposition.

### MOS-07: convergence and benchmark

Compare the public method to the dense oracle across narrow, broad, tail, tangent, and bandwidth cases. When the public method is the dense reference, record its convergence and baseline timing. Any localized/adaptive method must agree before it replaces the reference path.

## Required proof

- source and spectrum weight normalization
- preservation of declared source correlations and polarization-state IDs
- orientation normalization and azimuthal periodicity
- zero-width limit and finite pole behavior
- exact elastic residual
- tangent and no-root status
- dense versus public integrated event mass
- no quantile-resampling bias
- legacy support and event cases with classifications
- convergence of total mass, event centroid, and selected quantiles

## Overnight completion rule

`READY` means the mandatory manuscript/reference slice, analytic proof, immutable-pack comparison, convergence, and public contracts all pass. Record unimplemented generalization and acceleration as an explicit extension backlog. Do not mark a scientifically required reference case as optional merely to finish.

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/sampling src/rasim_next/reciprocal/ewald.py src/rasim_next/reciprocal/events.py tests/test_mosaic_ewald.py
pytest -q tests/test_mosaic_ewald.py
python -m rasim_next.proof mosaic-ewald --json
python -m rasim_next.proof references --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the common incident-state, rod-catalog, event, or probability-measure contract is insufficient. Do not construct rods or detector pixels in this branch.

## Execution plan

State: COMPLETE; Gate 0 decisions approved and the T03 proof passes on 2026-07-15.

Planning rules:

- Keep one writer and execute tasks sequentially even where the dependency graph has independent
  branches. Read-only review may run separately.
- Modify existing owned files when possible. Add only the six missing modules required by this
  workstream; add no generic framework, dependency, configuration file, or planning sidecar.
- Each task must leave its targeted tests passing. Broad sweeps, legacy arrays, mutations,
  benchmarks, and diagnostics belong in the proof command, not the permanent test module.
- Apply the repository Definition of Done after every task: runtime verification, focused permanent
  coverage, no duplicated equations, no dead/debug code, lint clean, and no unrelated edits.

Dependency graph:

```text
Gate 0
  -> T1 -> T2 -> T3 -----------+
  -> T4 -> T5 -----------------+-> T9 -> T10 -> T11 -> T12 -> T13 -> T14 -> T15
  -> T6 -> T7 -> T8 -----------+
```

Frozen numerical acceptance targets, subject to Gate 0 approval before source editing:

```text
source weight sum                         absolute error <= 1e-12
direction norm                            absolute error <= 1e-12
wrapped probability normalization         absolute error <= 1e-10
elastic residual                          <= 64*eps*max(|ki|, 1) A^-1
legacy intersection coordinate            absolute error <= 1e-12 A^-1
public versus dense total event mass      relative error <= 1e-6
public versus dense event centroid        absolute error <= 1e-7 A^-1
public versus dense selected quantiles    absolute error <= 1e-6 A^-1
```

### Gate 0: Resolve proof-base and measure decisions

**Files likely touched:** None in this branch. The proof-base owner may need a separately reviewed
shared-contract/task correction.

**Exact intended behavior:** Before coding, obtain explicit decisions for: ownership of
`src/rasim_next/reciprocal/proof.py`; typed tangent/no-root status representation; identifiable
polarization metadata; legal source/mosaic trace-stage mappings; branch-local tolerance storage and
hashing; the continuous-rod coarea Jacobian and exact-tangent policy. Set `PROOF_BASE_SHA` to the
verified starting commit.

**Tests:** Re-run startup preflight exactly as written in `tasks/prompts/mosaic_ewald.md` after any
shared correction.

**Dependencies:** None.

**Acceptance criteria:**

- [x] The assigned proof CLI can be implemented without writing an unowned path.
- [x] Status, polarization, trace, tolerance, Jacobian, and tangent semantics are written and
      unambiguous; no branch-local invention conflicts with shared contracts.
- [x] Branch, HEAD, `PROOF_BASE_SHA`, clean status, seed verification, core proof, and reference
      proof all pass.

**Estimated scope:** External gate; no branch implementation.

### Task 1: Compile explicit-joint source samples

**Files likely touched:**

- `src/rasim_next/sampling/source.py` (add)
- `tests/test_mosaic_ewald.py` (add)

**Exact intended behavior:** Add immutable branch-local polarization/source metadata approved by
Gate 0 and compile explicitly correlated rows into the shared `IncidentSampleBatch`. Preserve row
order, joint correlations, polarization-state IDs, supplied integrated masses, and deterministic
sample IDs. Reject invalid shapes, non-finite values, negative weights, non-unit directions,
unknown polarization IDs, and non-unit total mass; never silently normalize or factorize.

**Tests:** Add `test_joint_source_samples_preserve_correlations_and_metadata`; verify successful
contract construction plus rejection of one invalid mass and one missing polarization identity.

**Dependencies:** Gate 0.

**Acceptance criteria:**

- [x] Repeated compilation is bitwise deterministic and preserves every supplied joint row.
- [x] Output satisfies `IncidentSampleBatch`; weights and directions meet frozen tolerances.
- [x] Unity polarization is accepted only through explicit approved metadata and provenance.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k joint_source`

**Estimated scope:** S; two files.

### Task 2: Generate deterministic component quadrature

**Files likely touched:**

- `src/rasim_next/sampling/source.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Generate deterministic weighted normal nodes using NumPy only for
spatial offsets, tangent-plane angular offsets, and wavelength. Map spatial offsets into metres,
map angular offsets into normalized lab directions without small-angle norm drift, and require
positive wavelengths. A zero-width component emits one node of mass one. Node ordering and masses
must be independent of process RNG state.

**Tests:** Add `test_source_component_quadrature_is_normalized_and_deterministic`; cover nonzero and
zero widths, exact mean recovery, direction norms, positive wavelengths, and identical repeated
output.

**Dependencies:** Task 1.

**Acceptance criteria:**

- [x] Every component mass sums within `1e-12`; zero width yields exactly one mean node.
- [x] Direction samples are unit vectors within `1e-12`; invalid wavelengths fail explicitly.
- [x] No RNG, SciPy, quantile resampling, or hidden clipping enters the calculation.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k component_quadrature`

**Estimated scope:** S; two files.

### Task 3: Compile explicitly independent source products

**Files likely touched:**

- `src/rasim_next/sampling/source.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Form the Cartesian product only through an explicitly named
independent-correlation entry point. Multiply component masses exactly once, assign stable
lexicographic sample IDs, retain polarization identity, and emit `correlation_model` metadata that
distinguishes independent products from explicit-joint rows. Use vectorized index construction and
avoid duplicate temporary products.

**Tests:** Add `test_independent_source_product_has_declared_mass_and_order`; compare against direct
enumeration and prove the joint entry point never expands its input.

**Dependencies:** Tasks 1-2.

**Acceptance criteria:**

- [x] Output count equals the declared component-size product and IDs/order match direct enumeration.
- [x] Product masses match direct multiplication and sum within `1e-12`.
- [x] Independent and joint correlation semantics remain observably distinct.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k independent_source`

**Estimated scope:** S; two files.

#### Checkpoint A: Source boundary

- [x] Tasks 1-3 targeted tests pass together.
- [x] `ruff check src/rasim_next/sampling/source.py tests/test_mosaic_ewald.py` passes.
- [x] Review confirms no hidden normalization, RNG, or undeclared Cartesian product.

### Task 4: Implement wrapped mosaic probability

**Files likely touched:**

- `src/rasim_next/sampling/mosaic.py` (add)
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Implement the manuscript wrapped Gaussian core and wrapped Lorentzian
tail with independent `sigma`, `gamma`, and mixture fraction. Validate domains and evaluate stable
periodic probability on `(-pi, pi]`. Represent any active zero-width component as discrete mass,
not infinite point density. Keep line probability authoritative; do not create a second surface
density implementation.

**Tests:** Add `test_wrapped_mosaic_probability_has_independent_normalized_components`; check pure
core, pure tail, mixture, periodicity, width meanings, invalid parameters, and zero-width limit.

**Dependencies:** Gate 0.

**Acceptance criteria:**

- [x] Independent numerical integration gives unit component and mixture mass within `1e-10`.
- [x] Values are finite, nonnegative, periodic, and symmetric for every positive-width case.
- [x] Zero-width active mass is represented explicitly without evaluating a singular density.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k wrapped_mosaic`

**Estimated scope:** S; two files.

### Task 5: Integrate spherical orientation mass

**Files likely touched:**

- `src/rasim_next/sampling/mosaic.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Produce deterministic tilt and uniform powder-azimuth cells with
integrated probability masses under the manuscript spherical measure. Construct orientation
rotations using column-vector active rotations. Integrate line mass times uniform azimuth directly;
never evaluate or cancel `1/sin(theta)` at a pole. Collapse only mathematically identical
zero-width support while preserving total mass and signed orientation convention.

**Tests:** Add `test_orientation_quadrature_conserves_spherical_mass_at_poles`; verify mass,
azimuthal periodicity, zero-width support, finite pole behavior, rotation orthogonality, and
determinant `+1`.

**Dependencies:** Task 4.

**Acceptance criteria:**

- [x] Orientation masses are finite, nonnegative, deterministic, and sum within `1e-10`.
- [x] Rotations are orthogonal with determinant `+1`; signed orientation and azimuth order are fixed.
- [x] Removing the spherical measure changes the sensitive proof fixture at its approved first stage.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k orientation_quadrature`

**Estimated scope:** S; two files.

### Task 6: Solve analytic continuous-rod Ewald roots

**Files likely touched:**

- `src/rasim_next/reciprocal/ewald.py` (add)
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Parameterize each oriented rod by signed reciprocal length `u`, solve
`|ki+q0+u*d_hat|=|ki|` in deterministic parallel/perpendicular coordinates, and reconstruct `Q` and
`kf` from orthonormal sphere coordinates. Classify two-root, tangent, no-root, invalid-input, and
direct-beam cases using Gate 0's typed policy. Return roots in stable order with `L=u/|b3|`; use only
real incident phase wavevectors and the column-basis convention. Geometry, status, residual, and
Jacobian remain invariant when the same line is re-originated along `d_hat`.

**Tests:** Add `test_ewald_roots_cover_two_tangent_none_and_specular`; use analytic fixtures for every
status and verify roots by direct substitution.

**Dependencies:** Gate 0.

**Acceptance criteria:**

- [x] Root counts and statuses are exact for all analytic fixtures; no sentinel values are used.
- [x] Every emitted root meets the frozen elastic-residual bound and `kf == ki + Q`.
- [x] `Qz` and `L` reconstruct the same `Q` under the declared reciprocal basis.
- [x] Three reviewed line-origin reproductions and an unclipped Jacobian above `1e6` pass.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k ewald_roots`

**Estimated scope:** S; two files.

### Task 7: Apply the event Jacobian exactly once

**Files likely touched:**

- `src/rasim_next/reciprocal/ewald.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Implement only the Gate 0-approved coarea derivative for each regular
root, with explicit units and measure. Keep raw orientation probability separate from the Jacobian
until event assembly. Apply the approved tangent policy without clipping an infinite derivative or
fabricating finite mass. Expose enough numeric state for proof to detect omission or duplication.

**Tests:** Add `test_ewald_event_jacobian_matches_finite_difference`; compare the analytic derivative
to a centered finite-difference oracle away from tangency and exercise the exact tangent policy.

**Dependencies:** Task 6.

**Acceptance criteria:**

- [x] Analytic and finite-difference derivatives agree within a predeclared `1e-8` relative bound.
- [x] Regular-root Jacobians are finite, positive, and carry the approved measure/units.
- [x] Tangent behavior is typed and explicit; no epsilon clipping silently changes event mass.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k event_jacobian`

**Estimated scope:** S; two files.

#### Checkpoint B: Probability and root foundations

- [x] Tasks 4-7 targeted tests pass together.
- [x] `ruff check src/rasim_next/sampling/mosaic.py src/rasim_next/reciprocal/ewald.py tests/test_mosaic_ewald.py` passes.
- [x] Review confirms one wrapped-profile equation, one public root equation, and one Jacobian owner.

### Task 8: Add the independent dense event oracle

**Files likely touched:**

- `src/rasim_next/reciprocal/ewald.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Add a transparent dense calculation that directly constructs
orientation support, quadratic coefficients, roots, residuals, and mass. It may reuse immutable
input/profile data but must not call the public root/event solver or localized support logic. Keep it
small, explicitly marked as proof authority, and stream cells instead of materializing a global
state-by-rod-by-orientation Cartesian array.

**Tests:** Add `test_dense_oracle_matches_tiny_analytic_event_mass`; compare a tiny regular case with
direct hand calculation and prove the oracle remains callable independently of the public solver.

**Dependencies:** Tasks 4-7.

**Acceptance criteria:**

- [x] Oracle reproduces hand-computed roots, residuals, and total mass for the tiny fixture.
- [x] Oracle never calls the public root or event functions; an injected public-solver failure cannot affect it.
- [x] Peak working storage excludes the full global Cartesian product.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k dense_oracle`

**Estimated scope:** S; two files.

### Task 9: Assemble stable scattering events

**Files likely touched:**

- `src/rasim_next/reciprocal/events.py` (add)
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Stream valid `IncidentStateBatch` rows, individual `RodCatalog` rods,
and orientation cells through the public Ewald solver. Emit the shared `ScatteringEventBatch` plus
approved sidecar status evidence. Preserve source/rod identity, assign deterministic event IDs,
calculate `wavelength_A`, internal `Q`, `Qz`, `L`, outgoing film phase `kf`, residual, and
`reciprocal_weight = orientation_mass * Jacobian` exactly once. Exclude source mass, footprint,
optics, intensity, population, polarization, solid angle, and deposition.

**Tests:** Add `test_event_batch_matches_dense_oracle_and_factor_boundary`; cover valid/invalid input
propagation, ID stability, two rods sharing family metadata, bandwidth-dependent `ki`, and direct
field-by-field comparison with the dense oracle.

**Dependencies:** Tasks 1-8.

**Acceptance criteria:**

- [x] Every event field has contract shape/dtype, stable alignment, correct frame/unit, and no family collapse.
- [x] Public/dense mass, centroid, and selected quantiles meet frozen targets on the tiny fixture.
- [x] Factor-ledger audit proves `reciprocal_weight` contains only orientation mass and one Jacobian.

**Verification:** `pytest -q tests/test_mosaic_ewald.py -k event_batch`

**Estimated scope:** S; two files.

#### Checkpoint C: Public vertical slice

- [x] Entire `tests/test_mosaic_ewald.py` passes.
- [x] Source -> synthetic incident states/rods -> events works without detector, optics, or intensity modules.
- [x] Permanent tests are reviewed for unique long-term contracts; duplicates are merged or deleted.

### Task 10: Prove immutable legacy classifications

**Files likely touched:**

- `src/rasim_next/reciprocal/proof.py` (add only after Gate 0 ownership approval)

**Exact intended behavior:** Implement the proof CLI runner and read the immutable pack without
modification. Reproduce legacy Ewald-circle support as `MATCH`; classify wrapped/spherical mosaic
measure, deterministic source behavior, direct event mass, and internal `Q` as approved
`CORRECTED` or `NO_ORACLE`. Record every correction's approved first divergence and independent
downstream authority. Compare raw arrays before resampling, detector work, normalization, or display.

**Tests:** No permanent legacy-array test. Exercise through the proof CLI against the tracked pack.

**Dependencies:** Tasks 1-9 and Gate 0 proof/trace decisions.

**Acceptance criteria:**

- [x] Pack hash matches `e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06`.
- [x] All legacy cases report `MATCH`, `CORRECTED`, or `NO_ORACLE`; every correction has a legal first stage.
- [x] `MATCH` coordinates meet `1e-12 A^-1`; corrected stages use analytic/dense authority afterward.

**Verification:** `python -m rasim_next.proof mosaic-ewald --json`

**Estimated scope:** S; one file.

### Task 11: Prove deterministic convergence

**Files likely touched:**

- `src/rasim_next/reciprocal/proof.py`

**Exact intended behavior:** Add fixed narrow-core, broad-core, Lorentzian-tail, near-tangent,
specular, and multi-wavelength cases. Refine only declared tilt/azimuth resolution. Report public and
dense total mass, event centroid, selected weighted quantiles, errors, work counts, and pass/fail
against frozen targets. Run identical inputs twice and reject any node or ordering drift.

**Tests:** Proof-only convergence sweep; no permanent parameterized sweep.

**Dependencies:** Task 10.

**Acceptance criteria:**

- [x] Every required case converges monotonically or supplies a justified bounded non-monotonic sequence.
- [x] Final mass, centroid, and quantile errors meet all frozen targets.
- [x] Repeated evaluations are bitwise identical in nodes, IDs, statuses, and ordering.

**Verification:** `python -m rasim_next.proof mosaic-ewald --json`

**Estimated scope:** S; one file.

### Task 12: Prove all mosaic/Ewald negative controls

**Files likely touched:**

- `src/rasim_next/reciprocal/proof.py`

**Exact intended behavior:** Add proof-local mutations for removed spherical measure, mixture
misnormalization, node resampling, reversed signed arc measure, omitted Jacobian, extra empirical
Lorentz factor, accepted bad-residual root, cancellation under line re-origining, reversed
noncommuting rotations, air-side wavevector ownership, Jacobian clipping, and dropped discrete atom.
Each mutation uses a sensitive fixture and records expected/observed first stage and metric.
Production APIs receive no mutation switch.

**Tests:** Proof command must fail internally if any mutation escapes or fails at an unexplained stage;
the unmutated case must still pass.

**Dependencies:** Task 11.

**Acceptance criteria:**

- [x] All twelve mutations are detected at their approved first stage and metric.
- [x] No production signature, environment flag, or retained branch enables a mutation.
- [x] Proof JSON contains complete mutation evidence without writing repository diagnostics.

**Verification:** `python -m rasim_next.proof mosaic-ewald --json`

**Estimated scope:** S; one file.

### Task 13: Measure reference performance

**Files likely touched:**

- `src/rasim_next/reciprocal/proof.py`

**Exact intended behavior:** Benchmark equivalent public and dense work after warmup with one
BLAS/OpenMP thread. Both timed paths include orientation construction; setup of incident states,
rods, transforms, import, and I/O remains outside timing. Record fixture dimensions, root/event work
count, repetitions, median wall time, and Python-tracked peak memory in proof JSON. Measure the
accepted dense reference unchanged; do not add acceleration from benchmark results in this task.

**Tests:** Run benchmark through the proof command. No timing or memory assertion enters pytest.

**Dependencies:** Tasks 1-12.

**Acceptance criteria:**

- [x] Public and dense measurements use equivalent declared work and one compute thread.
- [x] Proof JSON records deterministic workload, median wall time, and peak memory with units.
- [x] No unproved optimized path, benchmark threshold, dump, or permanent benchmark test is added.

**Verification:** `python -m rasim_next.proof mosaic-ewald --json`

**Estimated scope:** XS; one file.

### Task 14: Remove development residue

**Files likely touched:**

- `src/rasim_next/sampling/source.py`
- `src/rasim_next/sampling/mosaic.py`
- `src/rasim_next/reciprocal/ewald.py`
- `src/rasim_next/reciprocal/events.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:** Review production and permanent-test code without changing accepted
physics. Delete exploratory helpers, duplicated equations, duplicate/implementation-detail tests,
temporary diagnostics, commented alternatives, debug output, dead branches, unused imports, and any
unmeasured optimization. Retain only the smallest readable public path, independent oracle, and
tests protecting distinct scientific contracts.

**Tests:** Run the complete permanent T03 test module before and after cleanup; outputs must remain
unchanged within frozen tolerances.

**Dependencies:** Task 13.

**Acceptance criteria:**

- [x] Every retained production helper and permanent test has one distinct required purpose.
- [x] No TODO, FIXME, debug output, generated file, scratch harness, duplicate equation, or unused import remains.
- [x] Permanent tests and proof results remain unchanged; Ruff and `git diff --check` pass.

**Verification:** `pytest -q tests/test_mosaic_ewald.py` followed by the assigned Ruff command and
`git diff --check`.

**Estimated scope:** M; at most five files, deletion-focused.

### Task 15: Complete verified handoff

**Files likely touched:**

- `src/rasim_next/reciprocal/proof.py`
- `tasks/03_mosaic_ewald.md` handoff section

**Exact intended behavior:** Review proof code for the same residue standard, run every assigned
command from a clean controlled environment, inspect the final owned-path diff, and fill the T03
handoff with commit SHA, public APIs, proof state, classifications, first divergences, convergence,
benchmark, peak memory, limitations, retained-test rationale, and contract requests. Create one
coherent commit only after all gates pass.

**Tests:** Run compileall, Ruff, permanent pytest, T03 proof, reference proof, and diff checks exactly
as assigned. Confirm Git cleanliness after the commit.

**Dependencies:** Task 14.

**Acceptance criteria:**

- [x] All assigned commands pass and proof reports every mandatory evidence category.
- [x] Final diff contains only owned required files; handoff is complete and scientifically accurate.
- [x] One coherent commit exists and `git status --short` is empty afterward.

**Verification:** Run every command under this task's `Commands`, then `git status --short` and
`git rev-parse HEAD`.

**Estimated scope:** S; two files.

#### Final checkpoint: Definition of Done

- [x] All task-specific acceptance criteria and repository handoff gates pass.
- [x] Runtime behavior, edge cases, lint, tests, proofs, references, benchmark, and memory are verified.
- [x] Public interfaces and task handoff describe current behavior; no shared contract was changed.
- [ ] Human review approves implementation before integration.

## Handoff

Status: READY. T03 is complete; integration, detector work, optics, fitting, caking, and acceleration
were not started.

Commit SHA: The containing branch commit is reported in the external handoff. Proof-base SHA is
`812f896fde5b8365ff5c218fc606df674ad7dcad`; the branch contains one coherent commit above it.

Public APIs: deterministic explicit-joint, explicitly independent, and independent-Gaussian source
compilers; `WrappedMosaicParameters`, `MosaicOrientationBatch`, wrapped line-density evaluation,
and `manuscript_axisymmetric_v1_orientation_quadrature`; branch-local Ewald root/status records and
`solve_continuous_rod_ewald`; `EwaldStatusBatch`, `EventBuildResult`, and
`build_scattering_events`.

Interface boundary: `build_scattering_events` requires a `CRYSTAL -> SAMPLE` `RigidTransform`, uses
its rotation only, consumes film-side real phase wavevectors, and emits film-side outgoing phase
wavevectors. Entrance/exit refraction, attenuation, optics, validity reclassification, and detector
projection remain solely T02 work. Wavelength is joined from `IncidentSampleBatch` by
`incident_sample_id`; `reciprocal_weight` is orientation probability mass times one unclipped coarea
Jacobian and contains no other factor.

Proof summary: the assigned proof passes all `7/7` checks and detects all `12/12` mutations. It proves
the direct-alpha `p_alpha d_alpha d_phi/(2*pi)` rule without inverse-CDF/quantile resampling or a
`1/sin(alpha)` point density, source normalization/correlations, exact elastic residuals, all root
statuses, line-origin invariance, direct-beam suppression including the direct tangent, the unclipped
Jacobian, tracked-pack integrity, and the separate dense Bragg-sphere and continuous-rod oracles.

Legacy classifications: `mosaic.ewald_intersection` is `MATCH`; `mosaic.legacy_density` is
`CORRECTED`; `mosaic.deterministic_source` and `mosaic.continuous_rod_events` are `NO_ORACLE`.
The corrected density first diverges only at the tracked legacy label
`mosaic.probability_measure`. The `MATCH` and `NO_ORACLE` cases have no divergent stage.

Convergence: six required cases are bitwise repeatable. Same-node public/dense mass, centroid, and
selected tail quantiles pass the frozen `1e-6`, `1e-7 A^-1`, and `1e-6 A^-1` targets. Against the
separately converged continuous-orientation oracle, final mass and centroid errors are below their
frozen bounds and raw-node tail quantiles are below the separate `1e-3 A^-1` discretization bound;
flat-CDF cases are explicitly classified as bounded non-monotonic.

Benchmark: the final external handoff records five-run post-warmup median wall times and Python
`tracemalloc` peaks for equivalent public and dense work. Public timing includes orientation
quadrature plus public event assembly; dense timing includes orientation quadrature plus the dense
continuous-rod oracle. Case/source/state/rod/transform setup, import, and I/O are excluded.
NumPy-native allocator peaks are outside that stated memory scope. In the same-process comparison,
the separate 16,384-event fixture reduced Python-traced peak memory from `11,189,430` to `4,999,564`
bytes while retaining the same `1,851,392` bytes of numeric event arrays.

Permanent proof coverage: four combined tests remain. They protect source probability/correlation,
complex rejection and the generated-row cap; wrapped/direct-alpha mixed atom/continuous mass;
line-invariant Ewald status/residual/Jacobian behavior; and the noncommuting transform, film-side
wavevector, tangent/no-root, event/interface/factor boundary. Broad sweeps, tracked arrays, mutations,
and benchmarks remain in the proof runner rather than duplicate permanent tests. The unused
docstring-only sampling initializer was deleted after source and isolated-wheel import proof.

Known limitations: no localized/adaptive acceleration or generic SO(3) sampler is implemented.
Raw discrete-node tail quantiles can oscillate under deterministic refinement, so the proof reports
their bounded sequence separately from the much tighter same-node public/dense solver comparison.

Unresolved interface questions and contract requests: None. No shared contract, dispatcher,
configuration, reference, example, or other workstream file changed.
