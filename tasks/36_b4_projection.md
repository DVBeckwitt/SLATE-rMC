# Bounded B4 detector-projection acceleration

## Authorized plan (2026-09-16)

The user authorizes reducing the measured detector-projection bottleneck, proving
preserved predictions, resuming numerical qualification, and refitting only after
qualification succeeds. Main is the sole writer. The isolated worktree starts from
main `2b80a2c` and carries forward the existing frozen prerequisite `108a454` without
changing either original checkout. Relevant ledger: PHY-SRC-001 and PHY-FIT-025.

Baseline evidence: full B4 support has 1,352 stored rows, 1,000 valid rows,
810 training rows and 190 held-out rows. One GH3 source and the complete m1 group
takes 137.95 seconds for four saved candidates; Gaussian projection takes 121.35
seconds. Its stricter counterpart exceeds the 180-second pilot budget. The old
fit is numerically unstable; no candidate or physical parameter is accepted.

1. Measure existing Gaussian rectangle arithmetic on bounded real-kernel batches.
   Retain only a material equivalent-work speedup with unchanged probability
   measure, tail policy, covariance, physical parameters and acceptance gates.
2. Change only the common spatial-probability owner and its nearest regression.
   Reuse existing independent Gaussian/oblique/weighted-overlap oracles. Keep
   failed experiments external and record why each was rejected.
3. Compare complete original and optimized pilot vectors, including the actual
   iodine-z and mosaic stencils. Run compact software and registered proof gates.
4. Resume full-target numerical qualification with bounded jobs and a measured
   total-work estimate. Preserve all rods, the training marginal covariance and
   conditional held-out covariance. A partial prediction never qualifies a fit.
5. Only after qualification, run a bounded refit and requalify its endpoint and
   actual optimizer stencils. Report numerical and physical acceptance separately.

Owned paths: `src/rasim_next/pipeline/source_spatial.py`,
`tests/test_source_spatial.py`, this task, `tasks/index.yaml`, and the generated
tracked-file inventory. Other shared contracts and immutable references remain
read-only. No dependency, optimizer, raster, physical approximation, threshold
relaxation, or simulated-image mirroring is authorized by this narrow change.

Diagnostics remain external single `.ra_diag.npz` files with embedded manifests.
Individual forward diagnostics are capped at 180 seconds; setup/short arithmetic
probes have tighter limits. Do not launch another expensive fit while numerical
qualification is unresolved. End with one coherent performance commit, clean
worktree, measured timing/memory, retained-test rationale, and explicit remaining
scientific limits.

## Projection result

The existing compiled rectangle index now also identifies shared physical corners
and separate X/Y marginal endpoints. Completed angular corner integrals and
Gaussian tails are reused only within the same Gaussian kernel. Per-kernel stamps
prevent reuse across changed means/covariances. Empty-X rejection still precedes
cache access, signed correlations are unchanged, and cancellation falls back to
the existing independent conditional-CDF integration. There is no public API,
dependency, physical-model, source-count, mesh, tail-radius or tolerance change.

One thousand identical real B4 kernels, six warmed alternating comparisons:
median throughput improves 1.2054x. Maximum probability difference is 3.33e-16;
maximum per-kernel L1 difference is 4.44e-16. Full original observation support,
source0 and complete m1, four saved candidates: 137.95s before versus 119.46s
after (projection 121.35s versus 102.92s). These full-pilot timings used different
concurrent workloads and are descriptive, not the controlled speedup authority.
Actual optimized worker peak RSS is 497,213,440 bytes. The older pilot's launcher
RSS is not a valid comparison. Sparse nonzero counts can differ at cancellation
roundoff; the accepted observable is probability/prediction, not sparse storage.

Maximum full-pilot prediction difference: 2.647e-23 raw, 2.480e-13 counts using
the unchanged saved scale 9,370,283,389.323324. The 810-row training-whitened RMS
difference is 5.667e-17 sigma; maximum stencil RMS difference is 7.044e-17 sigma.
This is software `MATCH` against the frozen predecessor, not full-source
numerical qualification. No physical correction or legacy reclassification is
introduced. First floating-point differences arise from regrouping the same
weighted corner sums, not from new equations.

Rejected work: direct conditional-CDF-only integration is 4.58x slower; caching
only the previous rectangle's four integrated corners gains 1% and was reverted;
candidate pruning has no useful envelope rejects in the measured workload.
Globally indexed corners alone gain 14.6%; reusing endpoint tails adds the
remaining benefit. No rejected implementation or scratch file remains.

The existing oblique Gaussian test now projects individual pixels as overlapping
observations alongside its whole-panel and row observations. This uniquely checks
shared-corner reuse against an independent conditional-CDF oracle for both signs
of correlation, including near-singular cases. Existing multi-kernel, weighted
overlap, Monte Carlo, and empty-ROI checks remain; no test function was added.
Two in-memory faults (reusing Gaussian stamps and swapping corner identities)
are detected at native region probability. Independent read-only review reports
no actionable findings.

Evidence: external `b4_projection_acceleration_20260916.ra_diag.npz`, containing
numeric comparisons, source hashes, the full pilot driver, benchmark ledger,
error injections, and software proof records. No diagnostics are in this checkout.

Software gate: all 371 permanent tests pass in 600.61s (14 existing CUDA
under-utilization warnings). Maintained-code Ruff lint and formatting pass for
166 files; `git diff --check` passes. No type checker is configured. Six registered
proof commands pass before commit; geometry-optics and mosaic-ewald pass their
scientific checks but require a clean checkout and are rerun after commit.

## Bounded numerical continuation

A single failure screen uses training rows 39, 40, 265, 580, and 1165, outside the
previously passed 157-row representative check. It retains all 91 signed rods,
original full-support transport bounds and frozen 827/1654-panel meshes. Both
rules must be completed over all 18 baseline/32 strict sources before any signed
difference is judged. Four candidates reuse identical geometry. Individual jobs
have a 175s worker watchdog and 180s parent timeout; dispatch stops on failure.
One complete-source/all-rod pilot took 33.26s baseline and 99.99s strict.

For the five-row marginal covariance, a prediction-difference norm above
0.1*sqrt(810) proves failure of the full training prediction gate; a corresponding
stencil-difference norm above 0.05*sqrt(810) proves contrast failure. Below either
threshold cannot prove full-target acceptance. A subset-profiled objective is not
a bound on the full objective. No refit or accepted parameter update is permitted
on the strength of this screen alone.

All 50 source jobs completed without timeout. Longest job: 124.90s; summed worker
wall time: 4,373.06s; resumed eight-worker batch: 574.88s, after the 99.99s pilot.
Maximum individual worker RSS: 759,775,232 bytes. These are worker wall times,
not measured process CPU times, and the memory value is not aggregate job memory.

The screen does not reject the new rule. Its maximum full-training prediction
RMS lower bound is 0.000673342 sigma; the contrast lower bound is 0.0000190585
sigma. Those are lower bounds, not full-target RMS errors or upper error bounds.
At the center the largest individual marginal disagreement on the five rows is
0.0151734 sigma. Complete numeric source partitions, sums, covariance, exact
driver, inputs and assessment are retained in the external
`b4_full_rule_screen_20260916.ra_diag.npz`. There is no new fitted result.

## Handoff and remaining gate

The narrow software acceleration is ready; scientific fit acceptance remains
blocked by incomplete full-target qualification, not by a failure of this screen.
Main and the frozen predecessor checkouts are unchanged. No public API changed;
all legacy `MATCH`/`CORRECTED`/`NO_ORACLE` classifications remain as before.
No integration into main is performed by this branch.

Do not repeat smaller screens or launch a fit merely because these five rows
agree. Remaining work is one complete fixed-support comparison over all 1,000
valid rows (810 training, 190 held out), the endpoint and actual bound-aware
stencils for all 25 active specimen parameters. Preserve covariance and evaluate
the full training-profiled objective contrasts and conditional held-out check.
The 21 intensity-only moves can share geometry; lattice a/c and Pb/I occupancies
require distinct geometry/optics. Each source/group or exact node partition must
retain the same original support, mesh and weights, then be summed before
qualification. Refit only after that gate, and requalify the resulting endpoint.

The full-support baseline m1 pilot alone takes 119.46s for one of 18 sources;
scaling that measured source cost gives roughly 36 worker-minutes for this one
group and four shared-geometry candidates. This is a workload extrapolation,
not a timing bound. Other groups, the denser strict rule and geometry-changing
stencils add work. The 20% measured throughput gain does not turn full numerical
qualification into a short diagnostic; the bounded five-row screen stops here.
