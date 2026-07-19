# Beam Sampling to `ki` Remediation Checklist

This checklist mirrors [the detailed implementation plan](plan.md). BKI IDs are local remediation
work items, not replacements for retired T00--T15 handoffs.

## Ownership precondition

- [x] **P0** — Review and either commit the current parallel/coating plan edits or explicitly hand
      both dirty files to the same BKI writer; pause, land, and rebase every overlapping writer.

## Phase 0: authority and physical inputs

- [x] **BKI-00** — Reconcile active plan entry gates, detector-unconditioned measure, Stage-A
      ownership, and the BKI-15 branch/join. Dependencies: P0.
- [x] **BKI-01** — Implement true endpoint-safe antithetic N-stratum LHS; pin PCG64; retain
      separate model/seed fields and source-parameter provenance; replace the 4097-row statistical
      test. Dependencies:
      BKI-00.
- [x] **BKI-02** — Make the material wavelength grid sorted, unique, exact, and producer-owned.
      Dependencies: BKI-00.
- [x] **BKI-03** — Simplify exact refraction lookup and prove monochromatic `N>1` transport.
      Dependencies: BKI-02.
- [x] **BKI-04A** — Add explicit `finite_rectangle.v1` / `unbounded_plane.v1` sample support; remove
      zero/placeholder dimensions. Dependencies: BKI-00.
- [x] **BKI-04B** — Centralize tracked default inputs and replace unsupported finite dimensions
      with explicit legacy unbounded support. Dependencies: BKI-01, BKI-04A.

### Checkpoint A

- [x] Focused source/material/refraction/support tests and core/mosaic/geometry proofs pass.
- [x] Human approves the v2 source and default-footprint first divergences before reference freeze.
- [x] Immutable `examples/` and `reference/` remain untouched.

Checkpoint A evidence (2026-07-18): the five owning test files pass `34 passed`; focused Ruff
check and format check pass; `tools/check_docs.py` and `git diff --check` pass. The core proof is
`PASS` with 7/7 mutations detected. Mosaic/Ewald and geometry/optics report scientific `PASS`
(7/7 and 17/17 mutations respectively); their aggregate status is `BLOCKED` only because the
checkpoint is intentionally uncommitted and therefore fails each retired proof's clean-worktree
policy. Git status and diff restricted to `examples/ reference/` are empty.

Intentional classifications awaiting human approval: source v1→v2 is `CORRECTED`; stable IDs,
uniform empirical mass, and polarization agree, with the first trace-value divergence at
`geometry.lab_ray`. Replacing the unsupported default finite rectangle with legacy unbounded
support is `CORRECTED`; unique forward plane intersections agree through
`geometry.sample_intersection`, with the first possible divergence at
`geometry.footprint_acceptance` (outside-support acceptance `0→1`). The fixed 20-row default
fixture happens to place every row inside the old rectangle, so it has no realized footprint-row
divergence; the owning outside-support fixture proves the named boundary.

## Phase 1: one incident authority

- [x] **BKI-05** — Extend `IncidentStateBatch` with wavelength, polarization, exact status,
      canonical revisions, and explicit state/sample identity semantics. Compile/hash once;
      packets inherit parent revisions. Dependencies: BKI-01, BKI-03, BKI-04A.
- [x] **BKI-06** — Remove raw source samples and the wavelength dictionary join from reciprocal
      entry/pipeline chunks. Dependencies: BKI-05.
- [x] **BKI-07** — Delete duplicate wavelength/status storage from `IncidentTransportResult`.
      Dependencies: BKI-06.
- [x] **BKI-08** — Enforce IDs, status/valid, dispersion/tangential relations, and status-dependent
      geometry/optics payload validity. Dependencies: BKI-07.

### Checkpoint B

- [x] Core, geometry, mosaic, and integration seam tests pass.
- [x] Static search finds no raw-source-to-incident rejoin.
- [x] Detector/crystal/thickness-only changes leave incident `ki` and revisions unchanged.
- [x] Human approves the shared contract and canonical hash encoding.

Checkpoint B evidence (2026-07-18): the four owning seam files pass `29 passed`; focused Ruff
lint, formatter checks on the formatter-clean owning files, `tools/check_docs.py`, and
`git diff --check` pass. The two pipeline files changed only by removal of the obsolete raw-source
arguments retain their pre-existing full-file formatter drift for the final repository-wide gate.
Core proof is `PASS` with
7/7 mutations; mosaic/Ewald and geometry/optics scientific gates pass with 7/7 and 17/17
mutations, respectively, while their aggregate status remains `BLOCKED` only on the expected
intermediate dirty-tree policy. The 512-row geometry benchmark retains exact status agreement,
zero point/`kz` error, and maximum amplitude error `2.220446049250329e-16`.

Canonical source-parameter provenance contains exactly `frames`, `units`, and `values` for the
declared means, axes, sigmas, wavelength parameters, count, and polarization; sampling model, RNG
model, and seed remain separate fields. For the deterministic revision envelope
`(source_parameter, source, sample_geometry, material, incident_model)`, the reviewed fixture is
`(8fe644bb084fc87febdc5d76857abb1d0a33edf2bba823a47a061376efdc3c20,
9c25af9b33750683e3de7f6fc50d3a6082697e4710fac3cfc25315517f44f6a7,
c780f155ac3d9328bcd31ce9b099db17c36cd119d9c69808b3d6b49e2d6fedbc,
083012c12f15a2fd4c15f7122a5d2bcc8d3f2feca74c0865750e1012cb37ca06,
one_transmitted_channel.v1)`. A seed-only change alters tuple position `{1}`; sample pose or
support alters `{2}`; material optics alters `{3}`; detector calibration, crystal mount, and film
thickness alter `{}` and leave `ki` exactly unchanged.

Alternate-layout/no-slice-rehash proof remains external until a real private packet path exists;
no test-local tuple construction is retained as a permanent test. Geometry-failure rows retain
source ID, wavelength, mass, polarization, and
revisions while zeroing intersection/direction/air/film payload; optical-failure rows retain the
accepted intersection, SAMPLE direction, air `ki`, footprint, mass, and provenance while zeroing
film payload. Relational mutations of IDs, wavelength, direction, air `ki`, tangential `ki`, film
normal, footprint, status/valid, provenance digest, and both failure payload policies are rejected.
The guarded untraced path returns `traces == ()`; external work-count evidence records zero default
JSON serialization calls versus one traced call. Opt-in traces retain the T02 scientific provenance
plus source/sample/material/incident identities. Git status and diff restricted to
`examples/ reference/` are empty.

## Phase 2: remove work and settle edges

- [x] **BKI-09** — Compile `sample_from_lab` once and reuse the transformed direction.
      Dependencies: BKI-08.
- [x] **BKI-10** — Refract only private geometry-valid rows and scatter to canonical order.
      Dependencies: BKI-09.
- [x] **BKI-11** — Remove trace-disabled full-batch `k_parallel` scratch allocation.
      Dependencies: BKI-10.
- [x] **BKI-12** — Classify every parallel/coplanar sample ray as `PARALLEL` while retaining
      accepted-stage evidence rules. Dependencies: BKI-10.

### Checkpoint C

- [x] Valid-row `ki`, IDs, statuses, and source mass match scalar/analytic authority.
- [x] Work counts show one compiled inverse, valid-row-only refraction, and no trace-only normal-path
      allocation; no allocation-count snapshot is retained as a permanent test.
- [x] No backend, executor, cache, compatibility facade, or new production module exists.

Checkpoint C evidence (2026-07-18): the four focused seam files pass `29 passed`; focused Ruff
lint/format and `git diff --check` pass. Core proof is `PASS` with 7/7 mutations. Geometry/optics
passes every scientific check and all three convergence checks; its 512-row vector/scalar benchmark
has exact status agreement, zero point and `kz` error, and maximum amplitude error
`2.220446049250329e-16`. Mosaic/Ewald science is `PASS`; both retired aggregate proof statuses are
`BLOCKED` only by the expected dirty-tree check at this intermediate checkpoint.

External work-count instrumentation records exactly one `RigidTransform.inverse()` call while
compiling the instrument and zero additional calls in `build_incident_states`. A two-row mixed
batch invokes incident refraction once with only its one geometry-valid row; its missed row has
wavelength `1.73 A`, absent from the singleton `1.54 A` material grid, and retains canonical status
`(VALID, OUTSIDE_SUPPORT)`, IDs, wavelength, and `1/2` mass. The two-row all-invalid batch invokes
refraction zero times and retains `(OUTSIDE_SUPPORT, PARALLEL)` in original order. For 129 rows,
both untraced and traced paths make the same four full `(N,3)` scientific output allocations; the
traced path derives canonical parallel values row-wise from compact mode data instead of adding a
full-batch scratch. They return zero versus 903 trace records and both omit the former `3,096`-byte
parallel scratch. Tracemalloc peaks were 126,452 bytes untraced and 331,889 bytes traced; the latter
includes the required immutable trace records rather than a trace-only scientific batch.

The retained owning tests extend existing geometry/transport/revision cases rather than adding a
new test file: they uniquely protect compiled inverse consistency and rotated scalar/batch parity,
valid-row-only exact material lookup including the all-invalid skip, status/payload/order/mass
preservation, traced/untraced public-state identity and trace schema/value parity, and off-plane,
coplanar-inside, coplanar-outside, and both just-nonparallel threshold sides. No allocation-count or
private implementation test is retained. The tracked legacy coplanar acceptance is `CORRECTED` at
`geometry.sample_intersection`: declared ray/plane inputs agree, then the nonunique hit changes from
accepted to `PARALLEL` with zero point, footprint, and optical payload. Git status and diff restricted
to `examples/ reference/` are empty; no production file/module or dependency was added.

## Phase 3: live contract synchronization

- [x] **BKI-13** — Delete the stale registered integration-proof command unless a real accepted T07
      proof exists, and verify every advertised dispatcher target. Dependencies: BKI-00, P0.
- [x] **BKI-14** — Synchronize live architecture/contracts/measure/trace/example/ledger/error-
      injection docs and record the expected manifest delta. Dependencies: BKI-01 through BKI-13,
      including BKI-04A/B.

### Checkpoint D

- [x] `python tools/check_docs.py` passes; the seed-manifest rewrite remains explicitly deferred to
      BKI-15.
- [x] `git status --short -- examples reference` and `git diff HEAD -- examples reference` are
      empty, including both evidence manifests.
- [x] Active plans agree; retired T02/T03 handoffs remain historical and unchanged.

Checkpoint D evidence (2026-07-18): the proof-dispatch registry test passes `1 passed`, every
advertised module/callable imports, and two identical unknown-command calls return canonical JSON
with exit code 2. The nonexistent `rasim_next.pipeline.proof` registration is deleted without a
stub or pytest wrapper; the retained integration seam passes `7 passed`. Removing the final
proof-only source/wavelength dictionary join leaves the state-only dense reciprocal oracle at
`5 passed`, with mosaic/Ewald science `PASS` and only the expected dirty-tree aggregate blocker.
Focused Ruff lint/format, `git diff --check`, `tools/check_docs.py`, and every relative Markdown
link pass.

The initial Checkpoint D stale-claim scan was incomplete because it covered only the eight named
boundary documents. The corrected repository-wide scan found and removed live residues in
`docs/PERFORMANCE.md`, `docs/FITTING_ROADMAP.md`, and
`docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`: live documentation now names
`IncidentSampleBatch`/`IncidentStateBatch`, assigns detector geometry only to hit/response
invalidation, and makes coating masses/CDFs detector-unconditioned with detector projection a
post-sampling rejection. A scan over every `docs/**/*.md` file now reports zero matches for the
retired compiled-source names or detector-conditioned/incident-state-invalidating claims. The
unexcluded repository-wide Markdown scan reports only three plan/ledger meta-requirement matches:
this correction record and the two explicit contradiction/removal requirements in `tasks/plan.md`.
The broader stale-claim scan also reports zero live matches for `crystal_from_sample`, contract API
v5, source-to-state wavelength joins, worker-local source generation, zero sample-dimension
sentinels, or stale proof commands. Its raw expected matches are explicit no-regeneration/no-rehash
requirements in live docs, the legacy-zero explanation in `docs/EXAMPLES.md`, the historical v1
comparator in `PHY-SRC-007`, the retired T03 record, and plan/ledger requirements. Repository-wide
`rasim_next.pipeline.proof` matches are confined to the plan's deletion requirement and this ledger
record; no live documentation or dispatcher registration advertises it. The active plans explicitly
agree on
`BKI-15 -> {corrected Task 1.1, continuous-coating validation} -> Task 1.2`.

The deferred BKI-15 manifest rewrite is exact relative to the current 160-entry manifest: retain
all 160 paths, refresh size/hash metadata for 35 existing entries, add 63 paths, remove zero paths,
exclude `FILE_MANIFEST.json` itself as before, and set `file_count` to 223 after the final
`docs/VALIDATION.md` and checklist text are frozen. The 35 refreshes are:
`AGENTS.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/DECISIONS.md`,
`docs/DOVETAIL_MATRIX.md`, `docs/ERROR_INJECTION.md`, `docs/EXAMPLES.md`,
`docs/FITTING_ROADMAP.md`, `docs/PERFORMANCE.md`, `docs/PHYSICS_LEDGER.md`, `docs/RESULT_MEASURE.md`,
`docs/SCOPE_AND_PHASES.md`, `docs/TRACE_SCHEMA.md`, `docs/VALIDATION.md`, `pyproject.toml`,
`scripts/verify_seed.py`, `src/rasim_next/core/contracts.py`, `src/rasim_next/core/transforms.py`,
`src/rasim_next/io/orientation.py`, `src/rasim_next/proof/__main__.py`,
`src/rasim_next/proof/core.py`, `src/rasim_next/proof/reference.py`,
`src/rasim_next/proof/traces.py`, `tasks/02_geometry_optics.md`,
`tasks/03_mosaic_ewald.md`, `tasks/04_ordered_reflectivity.md`,
`tasks/05_stacking_transition.md`, `tasks/06_parallel_review.md`, `tasks/07_integration.md`,
`tasks/OVERNIGHT_RUNBOOK.md`, `tasks/index.yaml`, `tasks/prompts/integration.md`,
`tasks/prompts/mosaic_ewald.md`, `tests/test_core_coordinates.py`, and `uv.lock`. The three additions
to the refresh count are formatter-only changes required by the repository-wide locked Ruff
0.15.21 gate.

The 63 additions are: `docs/CONTINUOUS_EWAL_COATING_STRATEGY.md`,
`docs/HBN_RING_FITTER_MINIMUM_SPEC.md`, `docs/WORKBRANCH_ARCHIVE_2026-07-16.md`,
`scripts/generate_bi2se3_detector_image.py`, `src/rasim_next/core/scattering.py`,
`src/rasim_next/core/traces.py`, all eight current `src/rasim_next/geometry/` implementation files,
`src/rasim_next/io/osc.py`, `src/rasim_next/materials/{__init__,crystal,optics}.py`,
`src/rasim_next/measurement/{__init__,angle_space}.py`,
`src/rasim_next/optics/{__init__,attenuation,refraction}.py`, all six current
`src/rasim_next/ordered/` files, `src/rasim_next/pipeline/{intersections,selection,simulate}.py`,
`src/rasim_next/proof/{stage_tolerances_v1.json,tolerances.py}`,
`src/rasim_next/reciprocal/{events,ewald,lattice,proof,rods}.py`,
`src/rasim_next/reflectivity/{__init__,parratt,specular}.py`,
`src/rasim_next/render/deposition.py`, `src/rasim_next/sampling/{mosaic,source}.py`, all six current
`src/rasim_next/stacking/` files, the five active plan/ledger additions
`tasks/{continuous_ewald_coating_replacement_plan,hbn_ring_fitter_plan,hbn_ring_fitter_todo,
mosaic_distribution_fitting_plan,parallel_simulation_geometry_fitting_plan}.md`, `tasks/plan.md`,
`tasks/todo.md`, and `tests/{test_geometry_optics,test_integration,test_mosaic_ewald,
test_ordered_reflectivity,test_stacking_transition}.py`. `scripts/verify_seed.py` currently fails
only on this deliberately deferred size/hash/catalog mismatch; evidence-pack checks remain clean.

The permanent-test audit matches the five-item budget: one compact even/odd exact-strata
parameterization; one material-grid contract plus one public monochromatic seam; one relational
incident mutation case plus one revision/invalidation case; the existing sample/transport cases
extended for support, compiled inverse, private valid-row refraction, trace parity, and coplanar
status; and one default-builder invariant plus the single dispatcher registry check. There are no
4097-row sweeps, snapshots, allocation-count tests, packet monkeypatches, full images, or new test
files. Git status/diff for `examples/`, `reference/`, their two evidence manifests, and retired
T02/T03 task records are empty; `FILE_MANIFEST.json` is unchanged.

## Phase 4: proof and handoff

- [x] **BKI-15** — Run analytic/proof/full-suite/docs/manifest/lint/format gates and external
      equivalent-work benchmarks for 1, 33, and 129 canonical rows. Dependencies: BKI-00 through
      BKI-14.
- [x] Record the compact proof/convergence/classification summary in `docs/VALIDATION.md`, perform
      the corrected second and final pre-gate manifest rewrite after supervisor-audit changes
      freeze, and run `python scripts/verify_seed.py`. The first rewrite is transparently
      superseded, not claimed as the only BKI-15 rewrite.
- [x] Compare alternate private packet layouts using inherited parent revisions; never rehash a
      slice as a new realization.
- [x] Record source-v1 and default-footprint first divergences, wall time, peak memory, work counts,
      hardware, revisions, and numeric errors outside the repository.
- [x] Retain only the plan's compact permanent-test budget; remove temporary tests, benchmark files,
      diagnostics, caches, and generated outputs.
- [x] Run `git diff HEAD --check`, inspect the complete intended diff, and obtain human approval.
- [x] The user-authorized supervisor review approves the BKI-15 branch point for corrected parallel
      Task 1.1 and continuous-coating validation; the two branches join only at parallel Task 1.2.

BKI-15 evidence (2026-07-18): the stdout-only external harness generated each complete 1-, 33-,
and 129-row source once, then exercised three contiguous/strided/reverse row-index layouts carrying
only canonical indices and the unchanged five-part parent revision envelope. Reassembly was exact
for source/state IDs, all statuses, all source and incident arrays, polarization, and `1/N` mass;
packet hash calls were zero and maximum reassembled `ki` error was zero. Every dimension occupied
all `N` strata exactly once. Maximum standardized second-moment error decreased from `1` at the
degenerate one-row case to `0.10084` and `0.03205`; scalar/batch fields agreed exactly and the
largest analytic film-dispersion residual was `7.11e-15 angstrom^-2`.

For 129 rows, all-valid/mixed/all-invalid refraction work was respectively 129/43/0 rows with
tracing both off and on. The mixed and invalid cases retained 86/129 geometry-invalid `1.73
angstrom` rows absent from the `1.54 angstrom` material grid. Tracing added 903 immutable records
without changing a scientific field. The post-correction rerun's untraced median wall times were
0.530/0.501/0.370 ms with `tracemalloc` peaks 120,716/92,268/82,606 bytes; traced values were
2.877/2.884/2.691 ms and 328,747/310,411/299,534 bytes. Geometry-failure payloads were zero after their preserved source
fields, and no mass was renormalized.

The source v1-to-v2 case is `CORRECTED` first at `geometry.lab_ray`; the finite-to-legacy-unbounded
default support is `CORRECTED` first possibly at `geometry.footprint_acceptance` after a matching
unique `geometry.sample_intersection`. Core, references, ordered/reflectivity, and
stacking/transition proofs report `PASS`; geometry/optics and mosaic/Ewald scientific gates pass
with only their expected dirty-tree aggregate blocker until the final coherent commit. Compileall,
repository-wide Ruff lint/format, and all 42 permanent tests pass with every cache/temp directory
outside the repository. The compact durable tables, revisions, classifications, limitations, and
proof counts are frozen in `docs/VALIDATION.md`. No benchmark, diagnostic, snapshot, allocation
test, packet monkeypatch, new module, or dependency remains.

Supervisor-audit correction (2026-07-18, approved by user-authorized supervisor review): retain
the five reviewed live-boundary
documentation corrections; derive both public source revisions through the sole batch construction
path so parameter provenance and the complete realization are each hashed exactly once; derive
`sample_from_lab` internally from hashed `lab_from_sample`; add reproducible external-harness and
four-control evidence; and preserve retired T03 task/prompt files in the active coating plan. The
first manifest rewrite is invalidated by these corrections. The corrected rewrite retains all 223
paths, refreshes 19 size/hash entries, and adds/removes zero paths. Exact Python/ordinal path order,
unique paths, and `file_count` agreement are now enforced by `scripts/verify_seed.py`. Focused tests, all 42 permanent
tests, compileall, Ruff lint/format, docs/links, all six registered proof commands, protected-path
checks, the successful 1/33/129 layout/work/memory rerun, exact hash-once instrumentation, and all
four assigned controls pass scientifically. Geometry/optics and mosaic/Ewald retain only their
expected dirty-tree wrapper blocker until the final coherent commit. The approved BKI-15 branch
point is ready; BKI-16 and BKI-17 proceed sequentially as nonblocking follow-ups.

## Phase 5: nonblocking follow-ups

- [x] **BKI-16** — Amend future fitting specifications to remove common-pose, isotropic-roll,
      pivot-axis, zero-pose/mount, and inactive-support gauges; update the seed manifest.
      Dependencies: BKI-05 and BKI-15; must precede parallel fitting Phase 2, not Task 1.1.
- [x] **BKI-17** — After a fresh consumer scan, delete only unused `core/units.py` and synchronize
      architecture/manifest metadata. Dependencies: BKI-15; does not block Task 1.1.

BKI-16 evidence (2026-07-18): the four live fitting specifications now fix the LAB beam frame,
forbid common beam/sample pose, deactivate isotropic beam roll, retain exactly two tangent-axis and
two perpendicular-pivot coordinates, give one owner to zero-pose/sample-mount, and expose finite
sample support only after edge-reaching observations. Unbounded support and incident-stage
`sample_from_crystal` translation stay inactive, while detector calibration remains a downstream
revision. Future full-rank/redundant-pack, roll, pivot, support, and invalidation tests are prose
obligations only. Documentation/link and focused diff checks pass; no fitting code or test was
added. The manifest refresh remains deferred until BKI-17 and the final ledger text freeze.

BKI-17 evidence (2026-07-18): fresh text, AST import, export, test, script, and documentation scans
found no consumer of `rasim_next.core.units.Unit`; only the scaffold itself, its architecture entry,
and the BKI deletion instructions named it. The sole production deletion is
`src/rasim_next/core/units.py`, and `docs/ARCHITECTURE.md` removes only that package-tree entry.
Compileall and all 42 permanent tests pass with external cache/temp paths. The exact-sorted seed
manifest now contains 222 unique paths and retains all 12 reference cases.
