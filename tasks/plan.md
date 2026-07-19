# Implementation Plan: Beam Sampling to Reciprocal-Space `ki` Remediation

Status: COMPLETE

This plan addresses the complete audit of source-ray sampling through the exact
`ki_sample_Ainv` consumed by reciprocal-space construction. It is a post-integration remediation
gate, not a restart of retired T02/T03 work. It must be accepted before the parallel plan freezes
its scalar reference or treats incident rows as an immutable Stage-A input.

The numerical scope ends at `IncidentStateBatch.k_film_phase_sample_Ainv`. Outgoing `kf`, detector
projection, deposition, and fitting implementation are excluded, except where their plans must be
amended to consume the corrected incident boundary.

## Architecture decisions

- Implement genuine randomized `N`-stratum Latin-hypercube sampling while retaining exact
  antithetic pairs, an odd central ray, independent dimensions, and empirical mass `1/N`. Bump the
  sampler revision; fixed-seed row values are intentionally rebaselined before the scalar forward
  reference is accepted.
- Pin the source RNG to an explicitly named bit generator and retain the seed and complete source
  parameter encoding in provenance. NumPy's changeable default generator is not part of the
  scientific contract.
- Define `MaterialOptics.wavelength_A` as a strictly increasing, unique exact grid. The public
  material producer coalesces repeated requested wavelengths; refraction never interpolates or
  tolerance-matches wavelengths.
- Make `IncidentStateBatch` the sole cross-domain numeric authority handed to reciprocal space.
  It owns aligned wavelength, polarization ID, first-failure status, source mass, `ki`, entrance
  optics, and source/sample/material revision identities. Do not add another public beam wrapper.
- Retain `valid` as a vectorized convenience but require exact agreement with `status == VALID`.
  `IncidentTransportResult` becomes only the authoritative states plus optional traces.
- Use separate deterministic content revisions for the complete source realization, sample
  entrance geometry, and material optics. Hash canonical typed bytes, not Python object identity or
  `hash()`. Worker packets inherit the already-computed parent revisions and canonical row IDs;
  they never hash a slice as though it were a different physical realization.
- Generate the complete canonical source batch once before worker dispatch. No source RNG key may
  include worker count, tile size, backend, completion order, or another execution-layout value.
- Classify every sample-parallel ray, including a coplanar ray, as `PARALLEL`; an infinite set of
  plane intersections is not a valid unique sample hit.
- Represent sample support explicitly as either `finite_rectangle.v1` with finite positive width
  and length or `unbounded_plane.v1` with no width/length. Zero is never a sentinel. The immutable
  legacy `forward_case.toml` remains provenance; its zero dimensions are mapped at the runtime
  boundary to the explicit unbounded-plane model rather than replaced by unauthoritative literals.
- Reconcile the active plans to the parallel plan's detector-unconditioned coating measure:
  detector projection occurs after sampling, detector misses retain rejected mass, and detector
  changes do not invalidate incident states or coating masses/CDFs.
- Retain the scalar ray/mode functions as proof oracles. Retain `direction_sample`, complex
  `kz_film_Ainv`, the sample intersection, footprint factor, and fitting-facing compiled transforms.
  Retain `k_air_sample_Ainv` until a separately justified trace-contract redesign; only its
  trace-only scratch allocation is removed here.
- Preserve accepted upstream evidence by status: a geometry failure has no intersection or optical
  payload, while an optical failure retains its accepted intersection, SAMPLE direction, air-side
  wavevector, source mass, and provenance but exposes no usable film-side `ki` or amplitude.

No new production module, backend abstraction, scheduler interface, compatibility facade, or
generic configuration framework is planned. The only new repository files for this planning turn
are this skill-mandated plan and `tasks/todo.md`.

## Dependency graph

```text
P0 current parallel/coating edits reviewed and committed, or ownership handed to BKI writer
 -> BKI-00 plan authority and entry gate
      |-> BKI-01 source sampler/RNG/provenance
      |-> BKI-02 canonical material grid -> BKI-03 refraction seam
      |-> BKI-04A explicit sample support
      `-> BKI-13 delete stale proof registration

 {BKI-01, BKI-04A} -> BKI-04B tracked default inputs
 {BKI-01, BKI-03, BKI-04A} -> BKI-05 incident contract

 BKI-05
      |
      v
          BKI-06 reciprocal migration -> BKI-07 transport consolidation
            -> BKI-08 relational/status invariants -> BKI-09 compiled inverse
            -> BKI-10 geometry-valid-only refraction -> {BKI-11 trace scratch, BKI-12 coplanar policy}

 {BKI-01 ... BKI-13} -> BKI-14 live docs -> BKI-15 proof/validation/manifest gate
                                                        |-> corrected parallel Task 1.1 --\
                                                        `-> coating validation ----------+-> parallel Task 1.2

Post-gate follow-ups (do not block corrected Task 1.1):
    BKI-15 -> BKI-16 future fitting gauges (must precede fitting Phase 2)
    BKI-15 -> BKI-17 delete dead unit scaffold
```

Write tasks remain sequential under one-writer ownership. Read-only derivation review, test review,
and benchmark-log analysis may run in parallel. Before BKI-00, the current owners must either land
the existing parallel/coating plan edits or explicitly hand both dirty files to the BKI writer.
Before implementation, any active parallel worktree that overlaps `core/contracts.py`,
`geometry/transport.py`, `pipeline/simulate.py`, reciprocal entry seams, tests, either active plan,
or `FILE_MANIFEST.json` must pause, land, and rebase. BKI-05 is a separately reviewed
shared-contract slice. `FILE_MANIFEST.json` is normally regenerated once in BKI-15 after the
validation summary and final pre-gate file set are known, rather than being rewritten at every
contract task. The first BKI-15 rewrite was invalidated by supervisor-audit corrections; the
corrected frozen set therefore requires one transparent second and final rewrite.

## Phase 0: authority and correctness

## Task BKI-00: Make the two active plans agree on the incident predecessor and measure

**Description:** Establish this remediation as an explicit predecessor of the continuous-coating
and parallel scalar-reference work, and remove the detector-conditioned versus
detector-unconditioned contradiction.

**Files likely touched:**

- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `tasks/continuous_ewald_coating_replacement_plan.md`

**Exact intended behavior:**

- The parallel plan may edit `geometry/transport.py` only through BKI remediation and freezes it
  again after BKI-15.
- Stage A consumes the accepted self-contained incident batch produced after BKI-08.
- Source sampling happens once before Stage-A workers and is never regenerated per worker/tile.
- Coating mass is detector-unconditioned; detector misses are post-sampling rejected mass and do
  not alter source/incident revisions or coating CDFs.
- Preserve the current uncommitted plan work; edit only the conflicting entry-gate, Stage-A,
  invalidation, and measure statements.

**Tests and verification:**

- Use `rg` to confirm neither plan says detector validity conditions coating mass/CDFs.
- Check that both plans name BKI completion before their scalar/parallel freeze.
- Run `git diff --check` and validate every relative Markdown link.

**Dependencies:** P0: the current uncommitted edits in both active plans are reviewed and either
committed or explicitly handed off to the same BKI writer. No implementation starts while another
writer owns an overlapping path.

**Acceptance criteria:**

- Both plans state one probability measure and one invalidation graph.
- Neither plan introduces a second beam contract or worker-local source sampling.
- Existing unrelated working-tree edits remain intact.
- Both plans branch from BKI-15 into the corrected parallel Task 1.1 reference and accepted
  continuous-coating validation, then join those prerequisites at parallel Task 1.2.

**Estimated scope:** S, two documentation files.

## Task BKI-01: Correct and version the Gaussian source stratification

**Description:** Replace the current pair-stratified implementation with the declared genuine
`N`-stratum randomized LHS while preserving antithetic and equal-mass semantics.

**Files likely touched:**

- `src/rasim_next/sampling/source.py`
- `src/rasim_next/core/contracts.py`
- `tests/test_mosaic_ewald.py`
- `docs/RESULT_MEASURE.md`

**Exact intended behavior:**

- For `N=2p`, independently permute lower strata `0..p-1` in each of five dimensions, draw
  `(stratum + U)/N` with endpoint-safe `U` strictly inside `(0,1)`, and place its exact complement
  in stratum `N-1-stratum` on the paired row.
- For `N=2p+1`, use the same `p` lower/upper pairs and place the final row exactly at `0.5`, the
  middle stratum.
- Preserve row IDs, adjacent pair layout, `1/N` source weights, the spherical exponential-map
  divergence transform, polarization IDs, and whole-batch fixed-seed determinism.
- Change the model identifier to `independent_gaussian_antithetic_lhs.v2`. Do not preserve v1
  fixed-seed numeric rows under the new identifier.
- Construct the generator explicitly as `Generator(PCG64(seed))` and record
  `numpy_pcg64.v1`, the nonnegative seed, the sampling-model ID, and a deterministic revision of
  every source input (means, axes, sigmas, wavelength parameters, count, and polarization) on the
  source batch. Retain the complete unit/frame-labelled parameter provenance as canonical text with
  exact float encodings, not only its digest, so trace/manifest output can reproduce the request;
  worker count and packet layout are absent from it.
- Put the canonical typed SHA-256 encoding in one small core helper in `core/contracts.py`; BKI-05
  reuses it for all incident revision domains. Do not create a second hashing implementation or a
  revision module for one function.

**Tests and verification:**

- Replace the existing 4097-row statistical test with one small even/odd parameterization that
  reconstructs all five unit coordinates and requires every `N` stratum exactly once.
- In that same compact test, cover exact antithetic pairing, odd center, equal mass, unit
  directions, polarization and provenance preservation, same-seed equality, and changed-seed
  inequality. Do not retain both the old statistical sweep and the exact-strata proof.
- Run `pytest -q tests/test_mosaic_ewald.py -k source` and the mosaic proof command.

**Dependencies:** BKI-00.

**Acceptance criteria:**

- Metadata and implementation describe the same algorithm.
- Every dimension occupies all `N` strata exactly once.
- No generating Gaussian PDF is applied as a weight.
- The intentional source-realization rebaseline is recorded before any scalar image is frozen.

**Estimated scope:** M, four files; one shared-contract metadata change, no new module.

## Task BKI-02: Canonicalize the material wavelength authority

**Description:** Make one unique exact wavelength grid the material-optics contract so repeated
source wavelengths do not repeat optical work or create ambiguous lookup rows.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/materials/optics.py`
- `tests/test_ordered_reflectivity.py`
- `docs/CONTRACTS.md`

**Exact intended behavior:**

- `material_optics()` canonicalizes requested wavelengths with sorted exact `np.unique` before
  evaluating atomic/optical data and returns one row per unique wavelength.
- `MaterialOptics` requires a nonempty, strictly increasing, finite positive wavelength grid with
  all optical arrays aligned to it.
- Duplicate or unsorted manually constructed material contracts fail at construction rather than
  being interpreted later.
- Optical constants remain exact table values; no interpolation or tolerance-based matching is
  introduced.

**Tests and verification:**

- Pass repeated monochromatic and repeated polychromatic requests to `material_optics()` and prove
  one sorted row per unique wavelength with unchanged optical values.
- Prove a manual duplicate or unsorted `MaterialOptics` is rejected.
- Run the material-optics portion of `tests/test_ordered_reflectivity.py`, core-contract tests, and
  Ruff. The consolidated seed-manifest update and verification are owned by BKI-15.

**Dependencies:** BKI-00.

**Acceptance criteria:**

- Repeated requests perform material work once per exact wavelength.
- The contract, producer, documentation, and manifest agree.
- Existing distinct-wavelength optical results are unchanged within frozen tolerances.

**Estimated scope:** M, four files; shared-contract review required.

## Task BKI-03: Make refraction consume the canonical grid and prove monochromatic batches

**Description:** Simplify exact material lookup and close the confirmed `wavelength_sigma_A=0`,
`N>1` runtime failure.

**Files likely touched:**

- `src/rasim_next/optics/refraction.py`
- `tests/test_geometry_optics.py`
- `src/rasim_next/geometry/proof.py`

**Exact intended behavior:**

- `_material_indices` uses one `searchsorted` against the already unique sorted material grid.
- Every requested incident wavelength must match exactly; absent wavelengths still fail with an
  informative error.
- A multi-ray monochromatic source maps all rows to the same material row and produces valid
  incident modes.
- Entrance/exit equations, branch selection, dtypes, and exact-match policy remain unchanged.

**Tests and verification:**

- Add one permanent seam regression for a public monochromatic `N>1` source through
  `build_incident_states`.
- Retain an exact absent-wavelength rejection test.
- Run `pytest -q tests/test_geometry_optics.py -k 'refraction or incident'` and
  `python -m rasim_next.proof geometry-optics --json`.

**Dependencies:** BKI-02.

**Acceptance criteria:**

- The confirmed monochromatic failure is eliminated.
- Lookup no longer computes `unique`/counts on every incident-mode build.
- No interpolation, nearest-neighbor selection, or silent fallback exists.

**Estimated scope:** S, two production/proof files and one permanent test file.

## Task BKI-04A: Add an explicit sample-support contract

**Description:** Replace implicit zero/placeholder dimensions with one explicit finite-or-unbounded
support model at the geometry boundary.

**Files likely touched:**

- `src/rasim_next/geometry/instrument.py`
- `src/rasim_next/geometry/sample.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/measurement/angle_space.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Add one versioned `sample_support_model_id` to instrument configuration/compiled state. Accept
  only `finite_rectangle.v1`, with finite positive width/length, or `unbounded_plane.v1`, with no
  width/length. Reject zero dimensions and inconsistent mode/dimension combinations.
- An unbounded plane accepts every unique forward plane intersection with footprint acceptance
  one; a finite rectangle retains the existing closed-edge test and dimensions.
- Propagate the support model through compiled instrument signatures and the sample intersection;
  no downstream consumer may inspect missing dimensions without first checking the explicit mode.
- BKI-05 includes the support model and finite dimensions, when present, in the sample-geometry
  revision; detector fields, `sample_from_crystal`, and film thickness remain excluded.

**Tests and verification:**

- Extend the existing scalar/batch sample-intersection tests with explicit finite and unbounded
  cases, invalid mode/dimension combinations, and footprint mass.
- Exercise the compiled instrument signature for both modes and run the geometry proof.

**Dependencies:** BKI-00.

**Acceptance criteria:**

- Sample support has one typed/versioned meaning; zero is never overloaded.
- Finite-support behavior remains available and unchanged for cases with authoritative dimensions.
- Unbounded support does not carry a fake width/length or alter detector/film ownership.

**Estimated scope:** M, five production/proof files and one existing test file. This is one coupled
support-mode migration because splitting configuration from its consumers creates an invalid
intermediate contract; no new module is added.

## Task BKI-04B: Correct and centralize the tracked default-case inputs

**Description:** Apply the explicit support contract to the canonical script and eliminate split
beam/sample literals without treating immutable provenance as runtime configuration.

**Files likely touched:**

- `scripts/generate_bi2se3_detector_image.py`
- `tests/test_integration.py`

**Exact intended behavior:**

- Replace the script's unauthoritative `0.2 mm x 0.5 mm` literals with explicit
  `unbounded_plane.v1`, matching the tracked legacy meaning of zero as disabled finite clipping.
- Consolidate source and instrument literals behind one pure `build_default_case_inputs()` seam so
  exact source parameters, RNG/seed provenance, transforms, support model, detector values, and
  film thickness have one runtime authority. `main()` only consumes the compiled result.
- Do not parse, copy, or edit the immutable TOML. If footprint classification changes, record the
  script's old finite rectangle as `CORRECTED` at that named first divergence.

**Tests and verification:**

- Add one default-builder invariant covering exact source inputs, seed/RNG ID, transform frames,
  unbounded support, and absence of zero-sentinel or replacement dimensions.
- Run the tiny integration test. Keep full-image comparison and first-divergence artifacts outside
  the repository.

**Dependencies:** BKI-01 and BKI-04A.

**Acceptance criteria:**

- The tracked runtime case has one source/instrument authority and no unsupported finite footprint.
- `examples/` and `reference/` are byte-for-byte untouched.
- No new configuration framework or tracked generated output is added.

**Estimated scope:** S, two existing files.

## Checkpoint A: source and wavelength authority

- [x] BKI-00 through BKI-04B pass focused tests, core/mosaic/geometry proof gates, lint, and diff
      checks.
- [x] A human approves the intentional v1-to-v2 source-realization change.
- [x] The scalar detector reference has not yet been frozen or, if already present, is explicitly
      invalidated and scheduled for regeneration after BKI-15.

## Phase 1: one self-contained incident boundary

## Task BKI-05: Extend `IncidentStateBatch` into the sole Stage-A authority

**Description:** Perform one coupled shared-schema migration so reciprocal construction no longer
needs a companion source batch for physical metadata.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/proof/core.py`
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_core_coordinates.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:**

- Add required aligned `wavelength_A`, `polarization_state_id`, and exact `status` fields.
- Add batch-level `source_sampling_model_id`, `source_rng_model_id`, `source_seed`,
  `source_parameter_provenance`, `source_parameter_revision`, `source_revision`,
  `sample_geometry_revision`, `material_revision`, and `incident_model_id` values.
- Reuse the one canonical SHA-256 helper introduced in BKI-01: length-prefixed field
  names/text/model/frame/unit IDs;
  fixed little-endian dtype tokens; rank and shape; and contiguous C-order numeric bytes. Reject
  noncanonical/nonfinite content before hashing. Never use Python object identity or `hash()`.
- Source revision covers the complete canonical realization: RNG/model/seed and parameter
  revision, sample IDs, ray arrays, wavelengths, weights, and polarization. Sample-geometry
  revision covers only `lab_from_sample`, explicit support model/dimensions, and the
  intersection-model ID. Material revision covers the exact optical grid and constants.
- Compile and hash the complete canonical batch once. Parallel work units carry canonical row
  indices/views plus the parent's revision envelope; they do not construct or rehash sliced public
  batches. Reassembly sorts by canonical state ID and verifies the inherited revisions.
- When diagnostics are explicitly enabled, include source model/RNG/seed/parameter revision and the
  source/sample/material/incident revision envelope in the existing trace/manifest path; keep
  diagnostics disabled by default and numeric rows in the accepted single `.ra_diag.npz` format.
- Keep `valid` during migration and require it to agree with status. Keep duplicate
  `IncidentTransportResult` metadata temporarily until BKI-07, avoiding a compatibility facade.
- Bump the shared contract API and update every direct constructor in the same reviewed slice; no
  required field receives a default or sentinel.

**Tests and verification:**

- Construct synthetic valid/invalid batches and check shapes, immutability, row order, exact text
  alignment, and nonempty version/revision fields.
- Build incident states twice from identical inputs and require identical revisions.
- Change source, sample pose/footprint, and material independently and require only the owning
  revision to change; change detector geometry, crystal mount, or film thickness and require the
  incident revision tuple and `ki` to remain unchanged.
- Split private row-index packets, reassemble them, and require inherited parent revisions and
  canonical state identity; explicitly prove that no packet is rehashed.
- Run core, geometry, and mosaic focused tests. Defer the intentionally consolidated seed-manifest
  rewrite and `verify_seed.py` to BKI-15.

**Dependencies:** BKI-01, BKI-03, and BKI-04A.

**Acceptance criteria:**

- One incident-state object contains every row-aligned value reciprocal Stage A needs.
- Revision ownership matches the parallel invalidation table.
- Contract construction cannot omit or silently fabricate provenance.
- Runtime tests and proof commands pass immediately after the coupled schema migration; the known
  seed-manifest delta is synchronized once in BKI-15 before handoff.

**Estimated scope:** M/L only because all constructor sites must move atomically; numerical logic
remains narrow. This is the separately reviewed shared-contract commit.

## Task BKI-06: Remove the raw-source join at reciprocal entry

**Description:** Migrate reciprocal event construction and pipeline orchestration to consume only
the self-contained incident-state authority.

**Files likely touched:**

- `src/rasim_next/reciprocal/events.py`
- `src/rasim_next/pipeline/intersections.py`
- `src/rasim_next/pipeline/simulate.py`
- `src/rasim_next/reciprocal/proof.py`
- `tests/test_mosaic_ewald.py`

**Exact intended behavior:**

- Remove `incident_samples` from `build_scattering_events` and the pipeline intersection/chunk
  helpers after incident compilation.
- Read wavelength and aligned metadata only from `IncidentStateBatch`.
- Delete the ID-to-wavelength dictionary join, its missing-ID bridge, and duplicate companion-batch
  error path.
- Preserve valid-state filtering, state-major order, source IDs, exact `ki`, event geometry, and
  reciprocal weights. Compact only private valid execution rows; retain the canonical incident
  ledger unchanged.

**Tests and verification:**

- Update existing Ewald/event proofs to call the state-only boundary.
- Compare pre/post event IDs, roots, `Q`, `L`, `kf`, residuals, weights, and row order for the same
  corrected source realization.
- Add a compact ordering check using nonconsecutive state IDs; no companion sample batch should be
  accepted or required.
- Run `tests/test_mosaic_ewald.py` and the mosaic proof.

**Dependencies:** BKI-05.

**Acceptance criteria:**

- Reciprocal construction has one incident input authority.
- The obsolete dictionary join and raw-source parameters are deleted.
- No reciprocal equation or probability factor changes.

**Estimated scope:** M, five files.

## Task BKI-07: Delete duplicate wavelength and status storage from transport results

**Description:** Finish consolidation after all reciprocal consumers use the state batch.

**Files likely touched:**

- `src/rasim_next/geometry/transport.py`
- `src/rasim_next/geometry/proof.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- `IncidentTransportResult` contains only `states` and optional `traces`.
- Outgoing transport reads wavelength and first-failure status from `incident.states`.
- Delete duplicate wavelength/status copying and comparisons while retaining event-to-state exact
  wavelength validation at the sole boundary.
- Do not add proxy properties or a deprecated compatibility layer.

**Tests and verification:**

- Update geometry proofs and tests to assert status/wavelength on `incident.states`.
- Deliberately mismatch an event wavelength and require the existing exact consistency failure.
- Run focused geometry tests and the geometry proof command.

**Dependencies:** BKI-06.

**Acceptance criteria:**

- There is exactly one wavelength array and one incident first-failure status authority.
- Existing outgoing calculations and traces are numerically unchanged.
- No obsolete result field or compatibility shim remains.

**Estimated scope:** S, three files.

## Task BKI-08: Enforce relational `ki` and status-dependent payload invariants

**Description:** Make it impossible to construct a row-aligned incident object whose fields are
individually well-shaped but physically inconsistent.

**Files likely touched:**

- `src/rasim_next/core/contracts.py`
- `src/rasim_next/geometry/transport.py`
- `tests/test_core_coordinates.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Require `incident_state_id` to be unique. Treat `incident_sample_id` as a required foreign key
  into the complete source realization, not an alias. The current
  `one_transmitted_channel.v1` model requires one state per source sample; a future version may
  repeat the foreign key only after it declares channel identity and once-only channel weights.
- On valid rows require positive wavelength, unit `direction_sample`,
  `0 <= footprint_acceptance <= 1`, `k_air=(2*pi/lambda)*direction`, conserved tangential
  components, and `k_film_phase_sample_Ainv[:,2] == real(kz_film_Ainv)` within frozen tolerances.
- Require `valid` to equal `status == VALID` exactly.
- Apply status-dependent payload rules. Geometry failures preserve source identity, wavelength,
  mass, polarization, and provenance but zero intersection/footprint and all air/film optical
  payload. Optical failures preserve the accepted intersection, SAMPLE direction, air-side `ki`,
  footprint, mass, and provenance, but zero unusable film-phase `ki`, `kz`, and entrance amplitude.
  Valid rows retain the complete payload. Never renormalize survivors.
- Keep validation vectorized and deterministic.

**Tests and verification:**

- Add one compact parameterized test that mutates each relation and proves construction fails for
  the intended reason.
- Check geometry-invalid and optical-invalid builder rows against their distinct stage-validity
  policies, including retention of the first accepted stage for divergence evidence.
- Check a normal valid batch passes without changing accepted `ki`.
- Run core and geometry tests and Ruff. Defer the consolidated seed-manifest gate to BKI-15.

**Dependencies:** BKI-07.

**Acceptance criteria:**

- A mismatched wavelength/direction/`ki`/status object cannot cross into reciprocal space.
- Invalid rows retain mass, first-failure identity, and accepted upstream evidence but cannot leak
  numeric state from the failed or later stages.
- Validation overhead is measured in BKI-15 and remains a one-time compile cost.

**Estimated scope:** M, five files; shared-contract review required.

## Checkpoint B: incident authority

- [x] BKI-05 through BKI-08 pass core, geometry, mosaic, and integration seam tests.
- [x] Static search finds no reciprocal or pipeline consumer joining raw samples back to incident
      states.
- [x] Detector-only changes leave incident `ki` and its revision tuple unchanged.
- [x] A human approves the shared contract before performance cleanup proceeds.

## Phase 2: remove upstream work and settle edge behavior

## Task BKI-09: Compile and reuse the SAMPLE-from-LAB transform once

**Description:** Remove the repeated transform inversion and duplicate LAB-to-SAMPLE direction
calculation without introducing a new geometry wrapper.

**Files likely touched:**

- `src/rasim_next/geometry/instrument.py`
- `src/rasim_next/geometry/sample.py`
- `src/rasim_next/geometry/transport.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Add immutable `sample_from_lab` to `CompiledInstrument`, constructed once as the exact inverse of
  `lab_from_sample` and validated for frame/inverse consistency.
- The batched sample intersection consumes this compiled inverse and returns the already-computed
  SAMPLE-frame direction.
- `build_incident_states` reuses that direction; it does not call `inverse()` or transform the same
  direction again.
- The scalar intersection oracle may compute one inverse per scalar call and remains independent.

**Tests and verification:**

- Compare rotated/translated scalar and batch intersections and incident `ki`, including valid,
  backward, parallel, and outside-support rows.
- Prove compiled forward/inverse composition is identity within the existing transform tolerance.
- Run geometry tests and proof; inspect the hot path to confirm one compiled inverse.

**Dependencies:** BKI-08.

**Acceptance criteria:**

- Batch incident construction performs no repeated SAMPLE/LAB inverse or direction transform.
- IDs/status are exact and numeric outputs remain within frozen tolerance.
- No detector field enters the entrance-geometry revision.

**Estimated scope:** M, four files.

## Task BKI-10: Refract only geometry-valid incident rows

**Description:** Stop material lookup and mode solving for rays that never reach the sample while
preserving the full canonical ledger.

**Files likely touched:**

- `src/rasim_next/geometry/transport.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Intersect all canonical rows first, compact only the private `geometry_valid` row indices, solve
  material/refraction for that compact batch, and scatter results/status back to original rows.
- Skip refraction entirely when no row is geometry-valid.
- A geometry-invalid row does not require a matching material wavelength because it never enters
  the material, but its source wavelength and `1/N` mass remain in the state ledger.
- Preserve stable row order and the status-dependent payload policy from BKI-08.

**Tests and verification:**

- Use a mixed batch whose valid wavelength exists in material and whose missed-ray wavelength does
  not; require successful compilation with exact geometry failure on the missed row.
- Cover the all-invalid batch and compare source mass/status/order.
- Run focused geometry tests and proof.

**Dependencies:** BKI-09.

**Acceptance criteria:**

- Material and normal-mode work scales with geometry-valid rows, not source row count.
- No invalid row is silently dropped or renormalized.
- Valid-row `ki` is unchanged.

**Estimated scope:** S, two files.

## Task BKI-11: Remove trace-disabled `k_parallel` scratch allocation

**Description:** Eliminate a full-batch array that exists only to emit an optional trace.

**Files likely touched:**

- `src/rasim_next/geometry/transport.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- With `trace_case_id=None`, allocate no full-batch `k_parallel_output` array.
- With tracing enabled, emit the same canonical `optics.ki_parallel_sample` values and zero-invalid
  semantics, using the already available compact mode data.
- Incident states are identical with tracing on or off.

**Tests and verification:**

- Compare every incident-state field for tracing on/off and compare enabled trace values/schema to
  the current accepted trace.
- Use the BKI-15 peak-memory measurement to confirm the normal path loses one `(N,3)` float64
  allocation.
- Run focused geometry tests and trace-schema proof.

**Dependencies:** BKI-10.

**Acceptance criteria:**

- The non-tracing path has no trace-only full-batch allocation.
- Trace IDs, units, frames, values, and tolerances remain unchanged.

**Estimated scope:** XS/S, two files.

## Task BKI-12: Reject nonunique coplanar sample intersections

**Description:** Replace the implicit zero-distance acceptance of a parallel coplanar ray with one
declared status policy.

**Files likely touched:**

- `src/rasim_next/geometry/sample.py`
- `tests/test_geometry_optics.py`

**Exact intended behavior:**

- Any ray with `abs(direction_sample_z) <= parallel_tolerance` receives `PARALLEL`, whether or not
  its origin lies within the sample plane tolerance.
- No point, footprint mass, or `ki` is emitted for that row.
- Backward and footprint-edge policies and existing tolerances remain unchanged.

**Tests and verification:**

- Test parallel off-plane, parallel coplanar inside the rectangle, parallel coplanar outside it,
  and a just-nonparallel forward ray on each side of the threshold.
- Require exact status and zero payload, then run geometry proof and convergence checks.

**Dependencies:** BKI-10.

**Acceptance criteria:**

- Every valid sample hit is a unique forward plane intersection.
- The edge policy is synchronized into the live contracts in BKI-14 and protected by the existing
  compact sample-status regression; retired T02 handoff text remains historical evidence.

**Estimated scope:** S, two files.

## Checkpoint C: incident work count and edge semantics

- [x] BKI-09 through BKI-12 preserve accepted valid-row `ki` and exact row/status identity.
- [x] One inverse transform, valid-row-only material work, and trace-disabled allocation removal are
      visible in a representative work-count/peak-memory comparison.
- [x] No executor, cache, or backend abstraction has been introduced.

## Phase 3: live contract synchronization

## Task BKI-13: Remove the stale registered integration-proof command

**Description:** Make the proof dispatcher advertise only commands backed by an importable compact
proof instead of retaining a dead registration.

**Files likely touched:**

- `src/rasim_next/proof/__main__.py`
- `tests/test_core_coordinates.py`

**Exact intended behavior:**

- Remove the current `integration` registry entry because `rasim_next.pipeline.proof` does not
  exist. Do not add an empty module, compatibility stub, or command that merely runs pytest.
- Keep `tests/test_integration.py` as the current integration seam. If an accepted T07 compact
  pipeline proof has landed before execution, retain the entry only after the dispatcher test
  proves its module and callable exist.
- Parallel Task 1.1 remains responsible for defining any later scalar/tiny end-to-end proof; this
  remediation does not invent that downstream oracle.

**Tests and verification:**

- Extend the existing dispatcher test to import every registered module/callable and reject an
  unknown command deterministically.
- Run the dispatcher test and `pytest -q tests/test_integration.py`.

**Dependencies:** BKI-00 and the P0 ownership handoff.

**Acceptance criteria:**

- Every advertised proof command is executable.
- No fake integration proof or new production module is added.
- The final verification matrix makes no claim about a command that does not exist.

**Estimated scope:** XS/S, two existing files; delete-first.

## Task BKI-14: Synchronize the live boundary documentation

**Description:** Bring every authoritative live boundary document into agreement before the final
proof summary and single pre-gate manifest rewrite.

**Files likely touched:**

- `docs/ARCHITECTURE.md`
- `docs/CONTRACTS.md`
- `docs/DOVETAIL_MATRIX.md`
- `docs/RESULT_MEASURE.md`
- `docs/TRACE_SCHEMA.md`
- `docs/EXAMPLES.md`
- `docs/PHYSICS_LEDGER.md`
- `docs/ERROR_INJECTION.md`

**Exact intended behavior:**

- Describe beam origin/direction/transverse geometry under source sampling; pin the RNG/model/seed
  provenance and exact antithetic N-stratum measure.
- Use the implemented `sample_from_crystal` name/direction and document explicit finite versus
  unbounded sample support, coplanar rejection, and the default case's immutable legacy provenance.
- Replace aspirational/nonexistent compiled-source names with `IncidentSampleBatch` and the
  self-contained `IncidentStateBatch`; delete descriptions of the removed raw-source join.
- Document the unique material grid; state/sample identity cardinality; status-dependent field
  validity; canonical revision encoding; full-batch revision ownership; worker row-index views;
  and the exact detector/crystal/thickness invalidation exclusions.
- Classify the sampler and default-support corrections in the physics ledger and name one-shot
  mutations for stratum duplication, packet rehashing, detector over-invalidation, and erased
  first-divergence evidence.
- Preserve retired T02/T03 handoffs as historical evidence. Do not bless or modify anything under
  `examples/` or `reference/`.

**Tests and verification:**

- Run `python tools/check_docs.py`, validate every relative Markdown link, and use `rg` to find stale
  source-join, pair-stratified-LHS, zero-sentinel, or detector-conditioned claims.
- Require `git status --short -- examples reference` and `git diff HEAD -- examples reference` to
  be empty, including `examples/MANIFEST.toml` and `reference/reference_manifest.toml`.
- Record the exact expected seed-manifest delta for BKI-15; do not rewrite it early.

**Dependencies:** BKI-01 through BKI-13, including BKI-04A and BKI-04B.

**Acceptance criteria:**

- Live code, contracts, trace schema, measure, examples documentation, ledger, and error-injection
  ownership agree.
- Retired task records and immutable evidence are untouched.
- The only expected pre-proof documentation mismatch is the explicitly deferred seed manifest.

**Estimated scope:** M, eight small documentation edits; no production or evidence file.

## Checkpoint D: pre-parallel boundary gate

- [x] BKI-00 through BKI-14 are complete under one-writer ownership.
- [x] Current active plans name the corrected incident predecessor and detector-unconditioned
      coating measure; retired handoffs remain unchanged.
- [x] The immutable example/reference pack is unchanged at Git level, not merely hash-consistent.
- [x] Static scans find no source-to-state wavelength join, worker-local source generation, stale
      proof command, or zero sample-dimension sentinel.

## Permanent-test budget

Retain only the smallest tests that own distinct long-term failures:

1. one small even/odd exact-strata source parameterization, replacing the 4097-row statistical test;
2. one material-grid contract test and one public monochromatic multi-ray transport seam;
3. one compact incident relational-mutation test and one revision-ownership/invalidation test;
4. the existing sample-status/transport tests extended for transform reuse, private valid-row
   refraction, trace parity, coplanar policy, and explicit support; and
5. one default-builder invariant and one proof-dispatch registry check.

Do not retain sampled-array snapshots, full images, allocation-count assertions, benchmark sweeps,
or one new test per optimization task. Replace weaker tests in their owning task; do not defer an
unnamed blanket test deletion to cleanup.

## Phase 4: incident-boundary proof and handoff

## Task BKI-15: Prove scientific correctness, packet-layout invariance, and reduced work

**Description:** Run the complete incident-boundary handoff gate without retaining broad sweeps,
temporary tests, or repository-local diagnostics.

**Files likely touched:**

- `docs/VALIDATION.md`
- `FILE_MANIFEST.json`
- only if a permanent proof gap is discovered, the existing owning test/proof file

All large benchmark, convergence, and first-divergence artifacts remain outside the repository.

**Exact intended behavior:**

- Accept the corrected v2 source realization, explicit sample support, and self-contained incident
  boundary as prerequisites to both the parallel Task 1.1 reference and continuous-coating
  validation. Those branches join only at parallel Task 1.2.
- Compare one-row scalar modes against batch incident states and analytic refraction identities.
- Generate and hash each complete canonical realization once for 1, 33, and 129 rows. Execute
  alternate private row-index packet layouts, reassemble by canonical state ID, and require exact
  inherited revisions, IDs, statuses, source mass, and source arrays plus accepted-tolerance `ki`.
  Never rehash a slice and compare that new hash to the parent.
- Benchmark all-valid, mixed-valid, and all-invalid batches with tracing off/on. Record equivalent
  work, wall time, peak memory, platform, revisions, and numeric error externally; do not snapshot
  allocation counts in a permanent test.
- Record two named first divergences when applicable: v1 source realization to v2 exact strata,
  and the default script's unauthoritative finite rectangle to explicit legacy unbounded support.
  Downstream accepted rows still follow scalar/analytic authority; do not require equality to a
  stale Monte Carlo image.
- Record compact durable results, convergence bounds, legacy classifications, error-injection
  detections, work counts, wall time, and peak memory in `docs/VALIDATION.md`; then perform the
  reviewed pre-gate `FILE_MANIFEST.json` rewrite. If supervisor audit invalidates that frozen text,
  record the correction and perform exactly one replacement rewrite after the corrected set freezes.

**Tests and verification:**

- `python -m compileall -q src`
- `python -m ruff check src tests scripts`
- `python -m ruff format --check src tests scripts`
- `pytest -q` with its temporary directory outside the repository
- `python tools/check_docs.py`
- `python -m rasim_next.proof core --json`
- `python -m rasim_next.proof geometry-optics --json`
- `python -m rasim_next.proof mosaic-ewald --json`
- `python -m rasim_next.proof references --json`
- `python scripts/verify_seed.py`
- `git diff HEAD --check`
- Require empty `git status --short -- examples reference` and
  `git diff HEAD -- examples reference`; then inspect the complete `git status --short`, remove all
  temporary residue, and map every retained test to the permanent-test budget.

**Dependencies:** BKI-00 through BKI-14.

**Acceptance criteria:**

- All analytic, contract, proof, full-suite, integration-test, lint, format, documentation,
  manifest, immutable-evidence, and diff gates pass.
- Monochromatic multi-ray transport works; exact absent wavelengths still fail.
- Every valid row satisfies the declared `ki` relations. Each invalid row retains precisely the
  upstream stages accepted before its first failure, exposes no failed/later-stage payload, and
  keeps its exact source mass without survivor renormalization.
- Execution packet layout cannot alter the canonical realization, inherited revisions, incident
  ledger, or numeric result.
- Measured work/allocation reductions are recorded without adding an executor, cache, or backend.
- The residue-free corrected boundary is ready for the parallel Task 1.1 scalar reference and the
  continuous-coating validation branch.

**Estimated scope:** S implementation, proof-heavy; no production addition expected.

## Phase 5: nonblocking follow-ups

These audit items do not block the corrected parallel Task 1.1 simulation reference. BKI-16 must
land before the parallel plan begins fitting work; BKI-17 is independent deletion-only cleanup.

## Task BKI-16: Remove exact geometry-fit gauges from future task specifications

**Description:** Prevent later source/sample fitting from exposing redundant parameters identified
at the incident boundary.

**Files likely touched:**

- `tasks/09_fit_foundation.md`
- `tasks/10_instrument_calibration.md`
- `tasks/11_sample_geometry_fit.md`
- `tasks/parallel_simulation_geometry_fitting_plan.md`
- `FILE_MANIFEST.json`

**Exact intended behavior:**

- Source characterization owns the fixed LAB beam frame; sample fitting may not vary a compensating
  common beam/sample rigid transform.
- Parameterize beam direction/transverse axes by one minimal beam-frame rotation. Fix beam roll
  when both spatial and divergence widths are isotropic.
- Parameterize each fitted rotation axis with two tangent coordinates and each pivot with only its
  two components perpendicular to that axis; never fit the invisible axis-parallel pivot component.
- Anchor zero pose versus sample-mount transform rather than fitting both unconstrained.
- Keep finite support dimensions inactive unless observations reach a footprint edge; when active,
  use a positive transform. Never fit an unbounded support dimension or
  `sample_from_crystal` translation at the incident stage.
- Record detector geometry as a downstream revision that does not invalidate incident `ki`.

**Tests and verification:**

- This task is specification-only. Add explicit future `tests/test_fitting.py` obligations for
  full-rank active-parameter Jacobians, rejected redundant packs, isotropic-roll deactivation,
  perpendicular-pivot reconstruction, support-mode activation, and revision invalidation.
- Run `python tools/check_docs.py`, Markdown-link checks, `git diff HEAD --check`, and
  `python scripts/verify_seed.py` after updating hashes for the three seeded task files.

**Dependencies:** BKI-05 and BKI-15. It must precede parallel fitting Phase 2.

**Acceptance criteria:**

- Future fitting tasks cannot construct common-pose, isotropic-roll, pivot-axis, zero-pose/mount,
  or inactive-support null directions.
- Parameter ownership and invalidation agree with the corrected incident boundary.
- No fitting code is added prematurely.

**Estimated scope:** S/M, four planning files and one manifest synchronization.

## Task BKI-17: Delete the unused core unit scaffold

**Description:** Remove the only statically confirmed dead production file without bundling an
unnamed test or broader cleanup sweep.

**Files likely touched:**

- delete `src/rasim_next/core/units.py`
- `docs/ARCHITECTURE.md`
- `FILE_MANIFEST.json`

**Exact intended behavior:**

- Re-run import/export/consumer scans, delete `core/units.py` only if it still has no production,
  test, proof, or documentation consumer, and remove its architecture/manifest entries.
- Do not delete scalar oracles, trace fields, fitting transforms, raw evidence, or tests under this
  task. Raw-source bridges are already owned and deleted by BKI-06/BKI-07.

**Tests and verification:**

- Run before/after `rg` scans for `core.units` and `Unit`, `python -m compileall -q src`, the full
  suite, `python tools/check_docs.py`, and `python scripts/verify_seed.py` after the manifest update.

**Dependencies:** BKI-15. This deletion does not block parallel Task 1.1.

**Acceptance criteria:**

- The unused file and all references to it are gone, with no public or proof consumer removed.
- The repository, documentation, and manifest pass after the deletion.

**Estimated scope:** XS/S, one deletion and two metadata edits.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Correct N-stratum LHS changes fixed-seed images | High | Land before scalar-reference acceptance, bump sampler revision, and rebaseline once with analytic source proofs. |
| Explicit unbounded support changes default footprint acceptance | High | Treat the old hardcoded rectangle as `CORRECTED`, name the footprint first divergence, and validate finite/unbounded modes independently before freezing the reference. |
| Shared incident schema migration crosses historical T02/T03 ownership | High | Use one separately reviewed contract slice, update every constructor atomically, and keep pipeline as the only concrete cross-domain orchestrator. |
| Concurrent work overwrites the current dirty plan amendments | High | Preserve/rebase existing edits and keep one writer on both plan files. |
| Strict relational checks add compile cost | Medium | Keep validation vectorized, measure once in BKI-15, and do not move checks into per-node/per-draw hot loops. |
| Unique material ordering changes callers that assumed source-row alignment | Medium | Make exact mapping explicit, update all consumers, and reject rather than silently reorder a manually invalid contract. |
| Source generation is incorrectly divided among workers | Critical | Generate the complete canonical batch once and prove packet-layout invariance after compilation. |
| A worker slice is hashed as a new realization | Critical | Hash the full canonical batch once; packets inherit the parent envelope plus canonical row indices. |
| Detector or crystal changes over-invalidate incident states | High | Hash only entrance-relevant sample geometry and test negative invalidation cases. |
| Blanket invalid-row zeroing erases first-divergence evidence | High | Validate payload availability by first-failure stage and retain every accepted upstream field. |
| Example cleanup mutates immutable evidence | Critical | Leave `examples/` and `reference/` untouched; clarify provenance and centralize only the runtime script inputs. |
| Cleanup deletes proof-facing state | Medium | Require an import/consumer/trace scan and retain scalar oracles and trace-required fields. |

## Definition of done

Every BKI task must satisfy its own acceptance criteria plus the standing project bar:

- runtime behavior is verified, not only typechecked;
- new behavior has a regression that fails without the change;
- existing compact tests and proof commands pass;
- code remains small, explicit, immutable, frame/unit aware, and free of duplicate equations;
- lint, formatting, manifest, documentation, and integration checks pass;
- no temporary tests, diagnostics, benchmarks, generated outputs, dead code, compatibility shims,
  or unrelated refactors remain;
- a human reviews the plan before implementation and the final diff before merge.
