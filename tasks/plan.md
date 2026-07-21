# Implementation Plan: Beam Sampling to Reciprocal-Space `ki` Audit Follow-ups

Status: **HISTORICAL ACCEPTED BEAM-BOUNDARY PLAN.** Contract API v9 retains this canonical
source-to-film-`ki` boundary. Current downstream authority is in `docs/ARCHITECTURE.md`,
`docs/CONTRACTS.md`, and `docs/SCOPE_AND_PHASES.md`.

## Overview

The accepted beam boundary is scientifically sound:

```text
IncidentSampleBatch
  -> sample intersection
  -> entrance refraction
  -> IncidentStateBatch.k_film_phase_sample_Ainv
  -> reciprocal space
```

This plan addresses the remaining audit findings without reopening the completed BKI remediation.
That work is preserved by commit `d5eed25` and the accepted evidence in
`docs/VALIDATION.md`. The scope here ends at the exact incident `ki` handed to reciprocal
space, except for narrow downstream plan amendments required to preserve this boundary during
parallel execution and fitting.

The work is divided into:

1. plan and ownership corrections that must precede parallel/fitting implementation;
2. one sequential shared-contract cleanup for material and entrance-geometry authority;
3. downstream packet, fitting, and nominal-fixture tasks owned by their existing phases; and
4. one optional, profile-gated allocation cleanup.

No production code is changed by this planning branch.

## Architecture decisions

- Keep `IncidentSampleBatch -> IncidentStateBatch` as the sole beam boundary. Do not add another
  beam wrapper, revision module, compatibility facade, or fitting-side source implementation.
- Generate and hash the complete source once. Build the complete incident table serially before
  worker dispatch; this approximately 0.53 ms stage is not itself a parallel target.
- Private workers receive the parent table and explicit `parent_row_index` values. They never
  construct or hash sliced public batches, regenerate source rows, sort physical rows by ID, or
  deduplicate equal numeric `ki`.
- Make `n_complex` the sole stored optical authority. `delta`, `beta`, and `mu_Ainv` are
  derived equations, not independently mutable material state.
- Immutable compiled material and sample-entrance objects own their revisions. Incident transport
  copies those revisions and does not recompute them.
- An unbounded sample plane is identified by orientation and signed normal offset. Translation of
  its coordinate origin within the plane is neither a revision input nor a fitted coordinate.
  Finite support retains its full pose and dimensions.
- A one-row nominal source still has empirical `source_weight == 1`. Only its diagnostic
  component tag is massless: it receives no assigned/deposited event mass and never enters a
  photon ledger.
- Freeze a physical source reference plane before any position-direction correlation is exposed.
  Do not add a general covariance merely because the parameterization can express one.
- Retain `IncidentStateBatch` intersection, SAMPLE direction, air `ki`, film phase `ki`,
  complex film `kz`, entrance amplitude, footprint, wavelength, polarization, source mass,
  status/valid, IDs, and revision envelope. They protect distinct scientific/provenance stages.
- Retain scalar intersection/refraction oracles and the ordered goniometer compilation equation.
  Keep `lab_from_goniometer` as a local compilation intermediate unless a fresh scan finds a real
  consumer; do not retain an otherwise dead compiled field for hypothetical future use.
- Do not touch transient arrays unless an equivalent-work profile after the authority cleanup
  shows that this stage is material to the accepted workload.

## Dependency graph

```text
PLAN-01 parallel/deterministic ownership
  -> PLAN-02 fitting specifications
  -> MAT-01 material revision ownership and consistency
  -> MAT-02 delete derived material storage
  -> GEO-01 canonical compiled sample revision
  -> GEO-02 delete confirmed-dead compiled transforms
  -> Checkpoint K0: implementation proof before documentation
  -> SYNC-01 live contract documentation
  -> SYNC-02 validation and task evidence
  -> MANIFEST-01 seeded-manifest refresh
  -> Checkpoint K: shared beam-to-ki boundary accepted
       |-> PAR-01 parent-row packet semantics
       |-> FIT-01 complete-wavelength fit context
       |      |-> SRC-01 fixed source reference plane
       |      `-> GEO-FIT-01 unbounded-plane active pack
       |-> NOM-01 canonical nominal incident fixture
       `-> PERF-01 optional allocation decision
```

MAT-01 through GEO-02 are sequential because they edit shared contracts. Read-only derivation,
consumer scans, and test review may run in parallel, but each implementation worktree has one
writer. PAR-01, FIT-01, SRC-01, GEO-FIT-01, and NOM-01 execute only in their named downstream
owner phases after Checkpoint K; they do not expand this branch's owned paths.

## Phase 0: correct plan ownership

## Task PLAN-01: Reconcile parallel and deterministic execution ownership

**Description:** Remove stale parallel directions that conflict with the proposed deterministic
pushforward and freeze the incident predecessor, packet identity, and nominal-tag semantics before
either implementation proceeds.

**Files likely touched:**

- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/deterministic_ewald_pushforward_plan.md`
- `tasks/deterministic_ewald_pushforward_todo.md`

**Exact intended behavior:**

- State that the complete source and `IncidentStateBatch` are constructed once, serially, before
  downstream coating/component work is tiled.
- Require every private parallel row to retain `parent_row_index`; reassembly uses that index,
  while `incident_state_id` remains identity payload rather than a sorting key.
- Route representative-tag ownership to deterministic DP-00C/DP-01/DP-06. Parallel Task 1.8 may
  consume the accepted tag contract but may not implement a competing nominal ray or coating
  representative.
- Replace “zero photon mass” on a one-row source with `source_weight == 1` and zero
  assigned/deposited tag mass.
- Require the shared-contract Checkpoint K before deterministic cache keys or parallel numeric
  records are frozen.
- Preserve all source/incident proof fields when the deterministic plan later deletes sampled-event
  APIs.

**Tests and verification:**

- Run `python tools/check_docs.py`, Markdown-link checks, and `git diff --check`.
- Use `rg` to prove there is no remaining worker-local source generation, ID-sorted reassembly,
  sliced-batch hashing, or zero-weight `IncidentSampleBatch` instruction.
- Record future tests for nonmonotonic IDs, reversed completion order, and unchanged photon ledgers.

**Dependencies:** None.

**Acceptance criteria:**

- One active owner exists for deterministic tags and one incident predecessor exists for all
  parallel work.
- No plan weakens source mass/revision invariants or revives sampled-event physics.
- The deterministic and parallel plans name the same Checkpoint K entry gate.

**Estimated scope:** S, three planning files.

## Task PLAN-02: Add the missing fitting preconditions

**Description:** Put complete material coverage and the two remaining geometric gauges into the
existing T09-T11 specifications before fitting code exists.

**Files likely touched:**

- `tasks/09_fit_foundation.md`
- `tasks/10_instrument_calibration.md`
- `tasks/11_sample_geometry_fit.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`

**Exact intended behavior:**

- T09 fit-context compilation requires exact material coverage for every unique wavelength in the
  complete parent `IncidentSampleBatch`, not only baseline geometry-valid rows.
- T09 consumes owner-provided source/sample/material revisions and rejects mismatched contexts
  before objective evaluation; it does not rehash scientific state.
- T10 fixes one named physical source reference plane before declaring position-direction
  correlations and exposes no longitudinal source-origin coordinate.
- T11 exposes one signed plane-normal translation for unbounded support and no in-plane
  translations. Finite-support translations activate only when edge-reaching data identify them.
- Existing common-pose, isotropic-roll, pivot-axis, zero-pose/mount, inactive-support, and
  detector-invalidation rules remain.

**Tests and verification:**

- Add explicit future `tests/test_fitting.py` obligations for a baseline-invalid wavelength that
  becomes valid, missing-wavelength preflight rejection, source-plane shift equivalence,
  full-column-rank active packs, rejected unbounded tangent translations, and finite-support
  contrast.
- Run docs/link checks and inspect the T09 -> T10 -> T11 dependency chain.

**Dependencies:** PLAN-01.

**Acceptance criteria:**

- Geometry trials cannot activate a source row whose material was omitted.
- Neither source correlations nor unbounded sample translation introduces an exact gauge.
- No fitting implementation or new fitting module is added by this specification task.

**Estimated scope:** S, four planning files.

### Checkpoint P

- [x] PLAN-01 and PLAN-02 pass docs/link/diff checks.
- [x] Human confirms deterministic, parallel, and fitting ownership before shared-contract edits.
- [x] No production file has changed.

## Phase 1: make material optics one immutable authority

## Task MAT-01: Make MaterialOptics own its revision and reject inconsistent representations

**Description:** Establish a safe intermediate contract in which the existing redundant optical
arrays cannot disagree, while moving material revision ownership out of incident transport.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`
- `tests/test_ordered_reflectivity.py`

**Exact intended behavior:**

- Add immutable, derived `material_revision` to `MaterialOptics`.
- Hash exactly the material ID, strictly increasing exact wavelength grid, `n_complex`,
  provenance, and an explicit `material_optics_revision.v2` schema tag.
- While `delta`, `beta`, and `mu_Ainv` still exist, require exact elementwise equality for
  `n.real == 1 - delta`, `n.imag == beta`, and
  `mu == 4*pi*beta/wavelength` after canonical float64/complex128 conversion. These are duplicate
  stored representations produced from the same arrays, so tolerance-matching would hide
  inconsistency. Stop for contract review if the public producer cannot satisfy exact equality.
- Repair existing synthetic proof/test fixtures so they are physically relational.
- Make `build_incident_states` copy `material.material_revision`; it performs no material hash.
- Preserve exact material lookup, refraction equations, branch policy, statuses, and numeric
  `ki`.

**Tests and verification:**

- Extend existing material tests with inconsistent-`n/delta/beta/mu` rejection.
- Prove identical material objects have identical revisions; changing ID, wavelength,
  `n_complex`, or provenance changes the revision.
- Prove repeated incident construction inherits the same material revision and leaves
  status/`ki`/entrance amplitude unchanged.
- Use temporary call instrumentation to show zero material-revision hash calls in transport; do
  not retain an implementation-count test.

**Dependencies:** Checkpoint P.

**Acceptance criteria:**

- An inconsistent material object cannot be constructed.
- Material revision is computed once by its immutable owner.
- The revision digest intentionally rebaselines under a named v2 schema; scientific arrays do not.

**Estimated scope:** M, five existing files.

## Task MAT-02: Delete redundant delta, beta, and mu storage

**Description:** Complete the material simplification after MAT-01 makes `n_complex` authoritative.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/materials/optics.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`
- `tests/test_ordered_reflectivity.py`

**Exact intended behavior:**

- Remove `delta`, `beta`, and `mu_Ainv` from stored fields and constructor inputs.
- Retain only material identity, exact wavelength grid, `n_complex`, provenance, and the owned
  revision.
- Preserve the physical absorption invariant by rejecting any
  `imag(n_complex) < 0`; deletion of the nonnegative `beta` field must not delete that check.
- Compute delta/beta/mu only as local analytic or reporting values where proof evidence requires
  them; do not add compatibility properties or a second constructor.
- Keep the MAT-01 revision payload unchanged because the deleted arrays were already derived rather
  than hashed authorities.
- Defer the public contract-version advance until GEO-02 so the complete Checkpoint K material and
  compiled-instrument cutover advances once rather than publishing two successive versions.

**Tests and verification:**

- Replace the transitional inconsistency test with one analytic `n_complex` producer proof.
- Retain one negative-imaginary-index rejection proving nonnegative absorption without stored
  `beta`.
- Require material revision, refraction status, `ki`, and entrance amplitude to match the accepted
  MAT-01 values exactly or within their frozen tolerance.
- Run a production/test consumer scan for `.delta`, `.beta`, `.mu_Ainv`, and obsolete
  material-revision helper calls.

**Dependencies:** MAT-01.

**Acceptance criteria:**

- One stored optical value owns entrance refraction.
- No redundant constructor field, compatibility shim, or duplicate material hash remains.
- Every retained material test protects a current equation or public invariant.

**Estimated scope:** M, five existing files; no new module.

## Phase 2: make sample entrance geometry physically canonical

## Task GEO-01: Canonicalize and own the compiled sample-geometry revision

**Description:** Make the revision describe physical entrance geometry, not an arbitrary unbounded
plane coordinate origin, and compute it once during instrument compilation.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/instrument.py`
- `src/rasim_next/geometry/transport.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Add immutable `sample_geometry_revision` to `CompiledInstrument`; transport only copies it.
- Define the sole `unique_forward_plane_intersection.v1` model-ID authority beside the canonical
  sample-revision helper in `core/contracts.py`; delete the transport-local literal.
- Compute the `init=False` revision inside `CompiledInstrument.__post_init__` after transform and
  support validation. No caller supplies it and no later stage recomputes it.
- Use an explicit `sample_entrance_revision.v2` schema tag.
- For `finite_rectangle.v1`, hash the complete SAMPLE-to-LAB rotation and translation plus finite
  width/length, support model, and intersection model.
- For `unbounded_plane.v1`, hash the complete rotation and signed LAB plane offset
  `dot(lab_from_sample.rotation[:, 2], lab_from_sample.translation_m)`, plus
  support/intersection model. Canonicalize at `1e-12 m` with nearest-integer ties-to-even rounding
  while `abs(offset / 1e-12) < 2**52`, preserve the exact float beyond that range, and normalize
  signed zero. Exclude both translation components tangent to the plane.
- Continue excluding detector geometry, `sample_from_crystal`, film thickness, and downstream
  response data.
- Preserve full `lab_from_sample` and `sample_from_lab` for numeric transport; revision
  canonicalization does not alter transform calculations.

**Tests and verification:**

- Two rotated unbounded instruments differing only by in-plane origin translation have identical
  revisions, statuses, intersections within frozen tolerance, and incident `ki`.
- A normal translation that changes the canonical offset changes the revision and physical
  intersection.
- A finite-support tangent translation changes the revision and can change support acceptance.
- Detector, crystal-mount, and thickness-only changes leave the revision and `ki` unchanged.
- Repeated incident builds inherit the exact compiled revision without rehash instrumentation on
  the normal path.
- A static scan finds exactly one intersection-model-ID literal and no transport-owned revision
  construction.

**Dependencies:** MAT-02, solely to serialize edits to `core/contracts.py`.

**Acceptance criteria:**

- Unbounded tangent translations cause neither cache invalidation nor a future fitted coordinate.
- Every entrance-geometry change resolved by the canonical contract invalidates the revision.
- `CompiledInstrument` truthfully owns the documented revision.

**Estimated scope:** M, four existing files.

## Task GEO-02: Delete confirmed-dead compiled transforms and narrow angle fingerprints

**Description:** Remove the derived compiled transform that has no numeric consumer while
preserving named goniometer provenance and future fitting evidence.

**Files likely touched:**

- `src/rasim_next/geometry/instrument.py`
- `src/rasim_next/measurement/angle_space.py`
- `tests/test_geometry_optics.py`
- `tests/test_integration.py`

**Exact intended behavior:**

- Re-run production, proof, test, script, and documentation consumer scans immediately before
  editing; stop for review if a real `lab_from_goniometer` or `lab_from_crystal` consumer has
  appeared.
- Remove stored `CompiledInstrument.lab_from_goniometer` and `lab_from_crystal`.
- Retain `lab_from_goniometer` only as a local `compile_instrument` intermediate used to produce
  final `lab_from_sample`; prove rotation-axis order through that retained final transform.
- Rebuild the instrument portion of the angle-space fingerprint from exactly
  `lab_from_detector`, detector shape, row/column pitch, and detector reference coordinate.
  `AngleFrame` remains separately keyed by its existing owner.
- Remove sample/goniometer/crystal transforms, support model/dimensions, and film thickness from
  the angle fingerprint because the angle projector consumes none of them.
- Advance the public contract version once for the coupled MAT/GEO cutover and independently bump
  the angle-space fingerprint schema.
- Do not remove any transform used by beam intersection, reciprocal orientation, detector
  projection, trace evidence, or fitting.

**Tests and verification:**

- Prove angle-space and detector results are unchanged.
- Prove each detector-causal field changes the fingerprint. Mutating only sample/goniometer/crystal
  transforms, support, or film thickness leaves the angle fingerprint and result unchanged.
- Require zero-consumer scans for both deleted fields after the change.

**Dependencies:** GEO-01.

**Acceptance criteria:**

- No dead derived transform remains in compiled state or invalidation keys.
- No causal geometry input or proof-facing transform is lost.
- No on-demand wrapper or compatibility alias is introduced.

**Estimated scope:** S, four existing files.

### Checkpoint K0: implementation proof before documentation

- [x] Focused material, geometry, reciprocal, and integration tests pass.
- [x] All registered scientific proofs pass with the new material/sample owners.
- [x] Source-to-`ki` statuses and numeric fields remain accepted; named revision digests alone
      intentionally rebaseline.
- [x] Consumer and stale-symbol scans confirm the intended deletion set before documentation
      freezes.

## Task SYNC-01: Synchronize live contracts and architecture

**Description:** Update the normative documents only after the coupled material/sample contract is
stable.

**Files likely touched:**

- `docs/CONTRACTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/DOVETAIL_MATRIX.md`
- `docs/TRACE_SCHEMA.md`

**Exact intended behavior:**

- Document sole `n_complex` authority, owner-computed material/sample revisions, both v2 hash
  payloads, unbounded-plane normal-offset canonicalization, and the causal compiled transform set.
- Preserve the complete `IncidentStateBatch` evidence/status schema and state that downstream
  consumers never rejoin raw samples.
- State that revision-digest changes are provenance rebaselines, not numerical corrections.

**Tests and verification:**

- Run `python tools/check_docs.py`, link checks, and stale-field/old-revision scans.
- Compare every documented field and invalidation cause with the accepted classes.

**Dependencies:** GEO-02 and Checkpoint K0.

**Acceptance criteria:**

- Normative documents describe the implemented contract exactly.
- No document advertises deleted material arrays, `lab_from_crystal`, or transport-owned hashes.

**Estimated scope:** M, five documentation files.

## Task SYNC-02: Record proof, performance, and error-injection evidence

**Description:** Record the compact evidence for the shared boundary and close the actionable
checklist without retaining temporary diagnostics.

**Files likely touched:**

- `docs/VALIDATION.md`
- `docs/ERROR_INJECTION.md`
- `docs/PERFORMANCE.md`
- `tasks/plan.md`
- `tasks/todo.md`

**Exact intended behavior:**

- Record numeric parity, new revision hashes/schema tags, classifications, wall time, peak memory,
  and exact retained-test rationale.
- Add executed controls for inconsistent material relations, transport-time rehash, unbounded
  tangent over-invalidation, and stale/noncausal angle-fingerprint inputs.
- Keep ID-sorted packet merge as an unchecked PAR-01 test obligation; do not claim mutation
  evidence for a parallel path that does not yet exist.
- Mark only completed tasks/checkpoints; do not copy historical BKI narratives into this plan.
- Keep full benchmark tables and diagnostics external.

**Tests and verification:**

- Run the focused suites, compact full suite, all registered proof commands, Ruff, formatting,
  compileall, docs/link checks, and `git diff --check`.
- Confirm `examples/` and `reference/` remain untouched.

**Dependencies:** SYNC-01 and Checkpoint K0.

**Acceptance criteria:**

- Each implemented Checkpoint K correction has analytic/invariant proof and mutation sensitivity;
  downstream PAR/FIT/NOM obligations remain explicitly unchecked.
- No temporary test, benchmark, cache, diagnostic, or generated artifact remains.

**Estimated scope:** M, five existing documents.

## Task MANIFEST-01: Refresh the seeded manifest once

**Description:** Update repository metadata only after production, tests, and documentation freeze.

**Files likely touched:**

- `FILE_MANIFEST.json`

**Exact intended behavior:**

- Refresh size/hash entries for exactly the accepted changed files, retain exact Python/ordinal path
  ordering, and do not add the manifest to its own catalog.
- Do not edit `scripts/verify_seed.py` unless its existing algorithm is proven wrong independently.

**Tests and verification:**

- Run `python scripts/verify_seed.py` and compare the manifest path set with `git diff --name-only`.

**Dependencies:** SYNC-02.

**Acceptance criteria:**

- The manifest is exact, uniquely ordered, and verifies on the committed candidate.

**Estimated scope:** XS, one metadata file.

### Checkpoint K: shared beam-to-ki boundary

- [x] MAT-01 through MANIFEST-01 are accepted in one coherent shared-contract branch.
- [x] Focused geometry/material/integration tests and all registered scientific proofs pass.
- [x] Existing source-to-`ki` numeric outputs remain accepted; only named revision/API digests
      intentionally rebaseline.
- [x] Static scans find no deleted material fields, dead transform, transport hash, raw-source
      rejoin, worker RNG, sliced public batch, or duplicate incident equation.
- [x] Human approves the public contract and revision changes before deterministic/parallel/fitting
      consumers freeze their records.

## Phase 3: downstream owner tasks

## Task PAR-01: Implement canonical parent-row packet semantics

**Description:** Make parallel layout an execution detail of one immutable parent incident table.

**Files likely touched:**

- `src/rasim_next/pipeline/simulate.py`
- `tests/test_integration.py`
- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/deterministic_ewald_pushforward_plan.md`

**Exact intended behavior:**

- Private state partitions carry a one-dimensional integer `parent_row_index` view into the
  complete parent `IncidentStateBatch`.
- Validate indices as in-range and nonduplicate for state partitions; validate coverage of the
  controller's explicitly scheduled parent-row set at merge. Invalid or unsupported parents may
  intentionally schedule no downstream record. Density/component records may legitimately repeat
  a parent index.
- Kernels preserve input-record alignment. The controller scatters results into canonical parent
  row slots, then applies the declared state/family/component ordering.
- Verify `incident_state_id` at boundaries but never use it as a row-order key.
- Inherit parent source/sample/material/incident revisions unchanged. Never construct/hash a public
  slice or regenerate source state inside a worker.
- Keep complete source-to-state compilation serial; parallelize only measured downstream bulk work.

**Tests and verification:**

- Use nonmonotonic/nonconsecutive IDs with contiguous, strided, reverse, odd-sized, one-worker,
  two-worker, and reversed-completion layouts.
- Require exact parent order, IDs, status, wavelength, source mass, polarization, revisions, and
  accepted `ki`; duplicate/out-of-range indices fail deterministically.
- Compare scalar and packed paths without an executor-specific golden snapshot.

**Dependencies:** Checkpoint K and accepted deterministic DP-02 through DP-04 kernels. PAR-01 is the
multi-row identity slice of DP-05's sole staged record seam and precedes DP-14 reconciliation.
Parallel Task 1.3 consumes that accepted record contract and adds no second packet type.

**Acceptance criteria:**

- Packet layout, worker count, and completion order cannot change identity, mass, revision, or
  result.
- No new public packet contract, backend abstraction, or source parallelism exists.

**Estimated scope:** M, two implementation/test files and two owning plans.

## Task FIT-01: Compile fit material over the complete source realization

**Description:** Prevent geometry-dependent material coverage when repeated trials can activate a
previously rejected source row.

**Files likely touched:**

- `src/rasim_next/fitting/context.py`
- `src/rasim_next/fitting/invalidation.py`
- `tests/test_fitting.py`
- `tasks/09_fit_foundation.md`

**Exact intended behavior:**

- Fit-context construction requires exact coverage of
  `np.unique(parent_samples.wavelength_A)` in the applicable `MaterialOptics`.
- Missing wavelengths fail before the first objective evaluation.
- Trial transport still refracts only that trial's geometry-valid rows.
- Source-wavelength changes rebuild complete material; geometry-only trials reuse it; detector-only
  trials reuse both material and incident states.
- Consume owner revisions from the immutable parent objects and reject mismatched contexts without
  rehashing them.

**Tests and verification:**

- A two-wavelength source starts with the second row geometry-invalid, then makes it valid in a
  trial; the unchanged full material revision supplies correct `ki`.
- A context missing the second wavelength fails at preflight.
- Detector-only invalidation does not rebuild material or incident state.

**Dependencies:** Checkpoint K and the base T09 contracts.

**Acceptance criteria:**

- Geometry topology cannot expose a missing material row at runtime.
- Fit code owns no material equation, interpolation, or source reconstruction.

**Estimated scope:** M, three future T09 files and its task record.

## Task SRC-01: Fix the source reference plane before correlations

**Description:** Give source position and position-direction correlation a physical, identifiable
reference before T10 implements them.

**Files likely touched:**

- `src/rasim_next/fitting/source.py`
- `src/rasim_next/sampling/source.py`
- `tests/test_fitting.py`
- `tests/test_mosaic_ewald.py`
- `tasks/10_instrument_calibration.md`

**Exact intended behavior:**

- Freeze one named plane origin, normal, and transverse basis in the source-owned LAB frame, using
  the accepted nominal LAB beam axis. A fitted/trial beam direction is expressed at this fixed
  plane and never rotates or translates the reference plane.
- Parameterize mean position only by two transverse coordinates on that plane; expose no active
  longitudinal origin coordinate.
- Define any position-direction covariance at that plane and propagate rays physically to other
  distances.
- Keep fixed latent stratified coordinates separate from the deterministic physical mapping.
  `IncidentSampleBatch`, exact `1/N` mass, IDs, antithetic semantics, and worker-independent RNG
  remain unchanged.
- Keep `independent_gaussian_antithetic_lhs.v2` byte-for-byte frozen. Extend the canonical
  sampling owner only after a separate shared-contract approval, using a new versioned model ID
  such as `correlated_gaussian_antithetic_lhs.v1`.
- Include the fixed reference-plane origin/normal/basis and every declared covariance/correlation
  parameter in canonical source-parameter provenance and its revision. Do not add an unconstrained
  5x5 covariance without identifiability evidence.

**Tests and verification:**

- Prove equivalent propagated beams from the declared plane, positive-semidefinite accepted
  correlation parameters, full-rank active packs, held-out detector-distance prediction, and
  deterministic fixed-latent evaluation.
- Reject a longitudinal reference-plane coordinate and underdetermined correlations.

**Dependencies:** FIT-01 and the T10 shared-path expansion review.

**Acceptance criteria:**

- Source correlation has one physical plane and one equation owner.
- Fitting never resamples privately or introduces a longitudinal gauge.

**Estimated scope:** M, four future implementation/test files and the T10 task record.

## Task GEO-FIT-01: Expose only causal unbounded-plane translations

**Description:** Encode GEO-01's physical plane equivalence in T11 active-parameter construction.

**Files likely touched:**

- `src/rasim_next/fitting/contracts.py`
- `src/rasim_next/fitting/geometry.py`
- `src/rasim_next/fitting/context.py`
- `tests/test_fitting.py`
- `tasks/11_sample_geometry_fit.md`

**Exact intended behavior:**

- Unbounded support exposes exactly one signed plane-normal offset and no tangent translations.
- Finite support may activate tangent translations only when corresponding edge-sensitive
  observations identify them.
- Reject redundant coordinates before objective evaluation; do not silently freeze or ignore a
  supplied null coordinate.
- Preserve existing common-pose, pivot-axis, roll, mount/zero, and support-extent rules.

**Tests and verification:**

- Require a full-column-rank unbounded active pack and deterministic rejection of either tangent
  coordinate.
- Demonstrate finite-support activation only with edge-reaching data and held-out recovery.

**Dependencies:** GEO-01, FIT-01, and accepted T10 calibration.

**Acceptance criteria:**

- T11 cannot construct the exact unbounded-plane null identified by the audit.
- Finite-support freedom is retained only when physically observable.

**Estimated scope:** M, four future fitting files and the T11 task record.

## Task NOM-01: Implement the DP-00C canonical nominal incident fixture

**Description:** Let the deterministic pushforward own the diagnostic center ray and component-tag
semantics without weakening the source contract or duplicating source literals.

**Files likely touched:**

- `src/rasim_next/proof/bi2se3_pushforward_fixture.py`
- `tests/test_integration.py`
- `tasks/deterministic_ewald_pushforward_plan.md`
- `tasks/deterministic_ewald_pushforward_todo.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`

**Exact intended behavior:**

- The sole DP-00C fixture declares the source parameter mapping consumed by proofs, tests, and the
  optional image CLI; no script or parallel task duplicates its literals.
- The fixture builds explicit source/material/instrument arguments and passes them into production.
  Production `rasim_next.pipeline` and lower layers never import `rasim_next.proof`.
- Construct `sample_count=1` with canonical seed `1729`, zero spatial/divergence/wavelength
  spreads, declared exact mean origin/direction/wavelength, and `source_weight == 1`.
- Numeric center coordinates are seed-independent under zero spreads, but the source revision still
  records the fixed seed; do not require different seeds to have equal revisions.
- Build material optics containing the exact mean-wavelength row and derive the incident state
  through `build_incident_states`; never inject `ki` by hand.
- The component-tag result contains no photon-mass/deposition field and never changes detector
  pixels, assigned mass, rejected mass, or clipped mass.

**Tests and verification:**

- Prove exact declared means, `source_weight == 1`, fixed revision repeatability, wavelength
  coverage, and accepted analytic incident `ki`.
- Prove the ordinary multi-row source realization and every detector/mass ledger remain unchanged.
- Static scans find no second nominal source builder or “zero source weight” instruction.
- Static import scans find no production-to-`rasim_next.proof` dependency.

**Dependencies:** Checkpoint K and deterministic Checkpoint 0/DP-00A tolerance authority. NOM-01 is
the incident-fixture slice of DP-00C; it precedes DP-01 and DP-06. No second post-DP-00C fixture
task is permitted.

**Acceptance criteria:**

- One canonical diagnostic incident state exists and one massless tag contract consumes it.
- No source exception, duplicate beam definition, or hand-authored wavevector exists.

**Estimated scope:** M, one planned fixture, one existing test, and three owner plans.

## Task PERF-01: Decide the incident-allocation cleanup from measurement

**Description:** Treat the approximately `80N` transient-array opportunity as a measured decision,
not an automatic refactor.

**Files likely touched if the gate passes:**

- `src/rasim_next/geometry/sample.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- First benchmark equivalent all-valid, mixed-valid, all-invalid, and repeated-geometry builds with
  external caches and diagnostics.
- If this seam is not material to the accepted workload, record `NO_CHANGE` and edit no production
  file.
- If material, reuse contract-ready zero-masked intersection/direction arrays and delete redundant
  `point_sample_output`, `distance_output`, `intersection_lab_m`, and `direction_output`
  allocations.
- Retain the immutable public contract copy, scalar oracle, status-dependent payload rules, traces,
  row order, and numerical tolerance.

**Tests and verification:**

- Existing scalar/batch, rotated-frame, invalid-payload, traced/untraced, and revision tests protect
  behavior; add no allocation-count or private-implementation test.
- Record equivalent-work wall time and peak memory externally before/after.

**Dependencies:** Checkpoint K. This task does not block deterministic pushforward, parallel Task
1.3, or fitting.

**Acceptance criteria:**

- Either a documented no-change decision exists, or measured memory/work improve with identical
  accepted scientific outputs and no public API change.

**Estimated scope:** XS decision; at most M if the four-file cleanup is justified.

## Downstream checkpoints

### Checkpoint D: parallel and nominal boundary

- [ ] PAR-01 scalar/packed/worker layouts agree in canonical parent order.
- [ ] NOM-01 tags are deterministic and massless while their source state retains mass one.
- [ ] No worker RNG, sliced public batch, slice hash, ID-sort merge, or duplicate tag physics exists.

### Checkpoint F: fitting boundary

- [ ] FIT-01 accepts every parent wavelength before geometry trials.
- [ ] SRC-01 fixes the source reference plane and proves identifiable correlation parameters.
- [ ] GEO-FIT-01 rejects unbounded tangent translations and proves active-pack rank.
- [ ] Detector-only changes remain downstream of incident revisions.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Revision cleanup silently preserves obsolete digests | High | Use explicit v2 payload tags, record a provenance-only rebaseline, and never force old hashes to match. |
| Derived optical arrays disagree before deletion | Critical | MAT-01 rejects inconsistent objects before MAT-02 removes the redundant inputs. |
| Unbounded tangent origin enters cache/fitting state | High | Hash and fit only full orientation plus the resolved canonical signed normal offset. |
| Worker merge sorts arbitrary state IDs | Critical | Carry parent indices, scatter by parent slot, and test nonmonotonic IDs/reversed completion. |
| Fit material is compiled from baseline survivors | Critical | Preflight exact coverage of all parent wavelengths. |
| Source correlation depends on an arbitrary longitudinal plane | High | Freeze a named physical reference plane and reject longitudinal origin parameters. |
| Nominal tag weakens source mass semantics | Critical | Keep source weight one; make only the tag/deposition ledger massless. |
| Cleanup removes proof-facing state | High | Preserve the complete incident evidence schema and scalar oracles; delete only confirmed consumers-zero fields. |
| Incident micro-optimization distracts from downstream bottlenecks | Medium | PERF-01 defaults to no change unless equivalent-work profiling justifies it. |

## Definition of done

Every implemented task must satisfy its own acceptance criteria and the standing project gate:

- one writer and one coherent commit per implementation worktree;
- no task touches more than five files unless an inseparable public-contract migration is explicitly
  re-reviewed;
- runtime behavior is proven, not only typechecked;
- optimized and proof paths agree within frozen tolerances;
- every retained test protects a distinct scientific invariant or public boundary;
- all temporary tests, benchmarks, diagnostics, caches, generated output, dead code, compatibility
  shims, and unused dependencies are removed;
- formatting, lint, compileall, docs/links, compact permanent tests, scientific proofs, manifest,
  and assigned error injections pass;
- `git status --short` contains only the task's intended paths before its coherent commit and is
  clean afterward;
- a human approves this plan before implementation and each shared public-contract cutover before
  downstream work resumes.
