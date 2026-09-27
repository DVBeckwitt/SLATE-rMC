> Historical record: task instructions and test/proof commands below describe earlier
> revisions. Current `AGENTS.md` and `docs/VALIDATION.md` supersede those workflows;
> this file does not authorize launching or recreating them.

# Full physical-intensity refit plan (2026-09-06)

Base: `04489309c606e9cd87505dcc4d8482b26060ce50`.
One writer, isolated branch `codex/full-physical-intensity-refit`.
The user authorizes corrections and new six-sample fits/figures after the intensity audit.

## Current scope and acceptance

1. Correct the compiled Parratt equal-zero interface limit and replace silent model substitution
   with explicit numerical failure. Verify against the existing independent scalar oracle and CUDA.
2. Reconcile one normalized mosaic probability measure for every rod and stitched `m=0`, reusing
   the existing plane-normal implementation where valid. Prove normalization, detector transfer,
   finite-area convergence, and one common parameter state before fitting.
3. Reuse the six existing sample loaders, geometry/source states, raw observations, physical
   structure/phase providers, nonnegative backgrounds, and region quadrature. Evaluate every
   declared rod in every region; remove independent empirical Q damping and all simulation-side
   sideband subtraction. Fit one common mosaic and physical structure state per sample, with one
   count calibration per exposure. Fit and render must consume identical physics.
4. Rerun Bi2Te3, Bi2Se3, GD1, SiD1, Clean1, and B4 using prior physical parameters as initial guesses.
   Compare old and new results on the identical raw-data support; do not hide failed bins or
   remove inconvenient peaks. Retain low-angle reflectivity. Keep old results immutable.
5. Reproduce log-view `m=1+/-,3+/-,4+/-` profiles with full-width `m=0` below and transparent
   detector-region overlays. Validate `total = scale * full simulation + background`, saved model
   identity, support, and quadrature convergence. Nonpositive observations remain explicit rather
   than being floored for a log plot.
6. Independent review, compact regression suite, registered proofs, formatting/lint, inventory,
   timing/memory evidence, and one coherent commit. All generated output remains external.

The physically valid detector domain and predeclared integration regions remain geometry, not
intensity-dependent masks. Historical transformed-observable proofs remain historical, never
evidence for a raw-count fit. New numerical or scientific failures block promotion, not visibility.
The two untracked B4 sources and source-less replay cache in main are preserved.

## Previous completed cleanup

The cleanup plan below was completed at `04489309c606e9cd87505dcc4d8482b26060ce50`.

## Repository-wide simplification plan (2026-09-06)

Base: `229f1bcef4a54166313c090da8bd31175ef0ca16`.
One writer, isolated branch `codex/repo-wide-simplification`.

## Scope and acceptance

Audit all tracked production packages, fitting/figure scripts, interactive tools, tests, dependencies,
and active documentation. Remove confirmed dead consumers and exact duplicate implementations;
preserve public scientific contracts, independent oracles, reference/example bytes, and live fit
replay dependencies. This is behavior-preserving cleanup, not a sample refit or new physical model.

1. **Inventory and consumer audit:** inspect each subsystem and distinguish obsolete adapters from
   active proof, fit, and presentation boundaries. Removal requires a known consumer migration or
   zero live consumers.
2. **Small verified changes:** remove dead indirection; share identical CPU/CUDA arithmetic,
   covariance operations, and validators only through existing module relationships. Run focused
   tests after each group; compare outputs and exceptions against the accepted base.
3. **Handoff gate:** independent diff review, full suite, formatting/lint, docs and seed verification,
   all registered proofs, and equivalent-work wall-time/peak-memory evidence. Update the inventory,
   commit one coherent result, then fast-forward main only if its tracked state remains unchanged.

No new dependencies, generic framework, broad cache deletion, or reference regeneration is needed.
The untracked B4 mosaic/SF sources and compiled replay dependency remain in the main checkout.
The checklist is [todo.md](todo.md); current architecture and contracts are in
[ARCHITECTURE](../docs/ARCHITECTURE.md) and [CONTRACTS](../docs/CONTRACTS.md).

## Historical beam-boundary provenance

The original long-form plan and checklist are recoverable from Git at
`229f1bcef4a54166313c090da8bd31175ef0ca16:tasks/plan.md` and
`229f1bcef4a54166313c090da8bd31175ef0ca16:tasks/todo.md`.
Historical file/line citations refer to those archived versions, not this compact summary.
The earlier BKI remediation is `d5eed2524a636c7c190b8f9be300d1d73728884e`.

### Checkpoint K: shared beam-to-ki boundary

Accepted at `6267301421e6dba1b0d121fa26585c24eae05f3d`: one canonical
`IncidentSampleBatch -> IncidentStateBatch` boundary, sole complex-index optical authority,
owner-computed material/sample revisions, physically canonical unbounded-plane offset, and removal
of noncausal stored transforms. Numeric source-to-film-`ki` outputs remained accepted; named
API/revision digests changed deliberately. Evidence remains in [VALIDATION](../docs/VALIDATION.md).
PLAN-01/02, MAT-01/02, GEO-01/02, SYNC-01/02, and MANIFEST-01 were completed at this checkpoint.

The six downstream entries below were unchecked in the archived checklist. Their historical
implementation mechanisms may have been superseded. They require reconciliation against current
contracts before any new work; they are neither new authorization nor evidence of a current defect.

| Historical entry | Obligation and acceptance evidence to reconcile | Historical dependency/owner |
|---|---|---|
| PAR-01 | Preserve parent-row identity through partitions and completion order; reject duplicate/out-of-range state indices. No worker source generation, public-slice hashing, or ID-sorted merge. Verify nonmonotonic IDs and scalar/packed parity. | Checkpoint K and accepted deterministic staged seam; deterministic/parallel plans |
| FIT-01 | Preflight material coverage for every parent-source wavelength, including rows made valid only by a geometry trial. Reject missing wavelengths; detector-only changes reuse incident/material state. | Checkpoint K; T09 |
| SRC-01 | Freeze one physical source reference plane before fitting correlations; no longitudinal-origin gauge. Verify deterministic fixed-latent mapping, covariance admissibility, rank, and held-out distance prediction. | FIT-01; T10 shared-path review |
| GEO-FIT-01 | Unbounded support exposes only a signed normal offset; finite tangent translations require edge-sensitive identifying data. Reject redundant coordinates and verify active-pack rank. | GEO-01, FIT-01, accepted T10; T11 |
| NOM-01 | Nominal geometry uses exact declared means and canonical incident transport. Its source retains unit mass; tags cannot alter detector mass. Reconcile the retired DP-00C fixture mechanism with the current nominal-geometry API. | Checkpoint K and historical DP-00A/DP-00C |
| PERF-01 | Default to `NO_CHANGE` unless equivalent all-valid/mixed/all-invalid/repeated-geometry profiling justifies an allocation reduction with unchanged scientific state and scalar-oracle agreement. | Checkpoint K; optional, nonblocking |

Do not recreate retired T02--T05 worktrees or restore sampled-event/raster architecture from these
historical plans. [SCOPE_AND_PHASES](../docs/SCOPE_AND_PHASES.md) is the current runtime authority.
