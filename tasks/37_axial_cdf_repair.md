# Bounded axial inverse-CDF repair

## Authorized plan (2026-09-17)

The user authorizes fixing the demonstrated stalled inverse-CDF solve, proving
coordinate/importance-density consistency, and resuming bounded qualification.
This worktree starts from clean main `2b80a2c` and carries forward the verified
prerequisite `e18eb745`. Main is the only writer; review agents are read-only.
Relevant ledger: PHY-FIT-025. Shared contracts and physical measures stay fixed.

1. Extend the existing public joint-quadrature invariant with the Clean1 narrow
   31-peak proposal. Demonstrate unordered nodes/incorrect mass before the fix.
2. Preserve the existing Newton fast path and stopping tolerances. On stalled
   solves, guarantee bounded bracket contraction; return only a tested coordinate
   with its matching PDF. Reject exhausted or nonfinite solves explicitly.
3. Compare the failure witness with an independent bracketed CDF oracle, detect
   restored exhaustion and stale-PDF mutations, and measure ordinary/fallback cost.
4. Run focused sampler/native-transfer tests, compact software and registered
   proof gates. Repeat the failed Clean1 full-support pilot with a 120-second
   child cap; partial observations or contributions never qualify a fit.
5. Resume numerical qualification only within measured bounded work. No refit
   starts until complete original observations and admitted stencils pass the
   unchanged numerical and applicable conditional-validation gates.

Owned paths: `src/rasim_next/pipeline/fiber_detector.py`, the existing
`tests/test_integration.py` invariant, this task, `tasks/index.yaml`, and the
tracked-file inventory. No new API, dependency, physical approximation, source
pruning, angular solver change, or tolerance relaxation. Retained diagnostics
are one external `.ra_diag.npz` with numeric arrays and an embedded manifest.

Classification: CORRECTED against the earlier numerical implementation at the
axial inverse-CDF solution and returned importance density, after identical
proposal inputs/CDF. The independent oracle solves that same normalized CDF.
This does not change legacy crystallographic/optical classifications.

## Repair and numerical evidence

The inverse now retains the first 70 Newton evaluations and allows at most 80
additional evaluations with forced bisection. Nonfinite/nonpositive density and
unresolved exhaustion raise. A successful return always pairs the tested
coordinate with the density evaluated there. There is no public API change.

The public regression failed before the fix with unordered axial coordinates.
At cap `.02`, the repaired 2,912-node axial rule gives relative constant-mass
and first-moment errors `7.98e-12` and `8.51e-12`, below the unchanged `1e-7`
test tolerance. Cap `.0025` gives 22,064 axial / 353,024 joint nodes and errors
`4.25e-14` or less. This denser public sampler takes 2.665 seconds including JIT;
its measured process peak working set is 270,503,936 bytes.

An independent arctangent-difference CDF and SciPy Brent oracle cover 53
quantiles, including endpoints and the adjacent pair that previously diverged
by `.06216` inverse angstroms. Maximum coordinate difference is `1.85e-14`,
CDF residual `6.22e-15`, and relative PDF difference `6.67e-16`. Restored silent
exhaustion fails coordinate ordering; returning the old density with corrected
coordinates fails the physical-mass check by `7.05e-4`. Both mutations are
detected. Only the existing public quadrature invariant was extended.

Five warmed alternating timing pairs on 8,193 ordinary quantiles give medians
1.053 ms old / 1.049 ms repaired; no speedup is claimed. On 256 deliberately
stalled quantiles, 3.571 ms old / 5.509 ms repaired compares an incorrect old
return with a correctly converged result, not equivalent scientific work.

The previously failing Clean1 full-support pilot now completes both requested
batches in 64.507 seconds: all 1,334 stored observation rows, one unchanged
source row of 36, and the complete six-rod first radial family. A cached
strength contraction agrees exactly with the ordinary response on its parity
batch. Final process exit is 1 because the separate post-result RSS reporter
had an incomplete Windows counter structure; numerical arrays were emitted
and retained before that error. Actual pilot peak RSS is unavailable. The
separate sampler-memory measurement above does not stand in for detector RSS.

Evidence is the external `axial_cdf_repair_20260917.ra_diag.npz`, with exact
inputs/source hashes, worker and oracle sources, timings, numeric arrays and
the explicit post-result reporting limitation. A first software-suite attempt
was stopped after temporary-directory permission errors; the retry uses a
verified ordinary Windows temporary directory. Task-owned failed-run scratch
directories were removed. Main and both frozen predecessor worktrees remain
unchanged.

## Software handoff

The compact suite passes: 371 tests in 564.17 seconds, with 14 Numba GPU
under-utilization warnings and no failures. Ruff lint and formatting checks
pass for `src`, `tests`, and `scripts`; no type checker is configured. The
registered core, references, ordered-intensity-fit, ordered-reflectivity,
PbI2-polytype-Bragg, and stacking-transition proofs pass. The immutable
reference pack remains unchanged (12 cases, 73 arrays; 8 MATCH / 4 CORRECTED).
The geometry-optics and mosaic-ewald commands require a clean committed tree;
their post-commit results and final inventory check belong to the same external
diagnostic, bound to the final commit. No proof threshold was relaxed.

Independent read-only reviews approve the narrow diff, contract compatibility,
and saved numerical evidence. The artifact audit verifies all 123 recorded code
hashes, 20 input hashes, parent and worker-source hashes, and independently
reproduces the oracle results. The only retained test change extends the public
joint-quadrature invariant to detect stalled inversion and mismatched importance
density; no exploratory test or script remains. Integration needs only this
branch's coherent commit on top of the verified projection prerequisite.

## Remaining scientific qualification

The bounded integration work has resumed, but it is not a complete numerical
comparison: remaining source rows, rod families, full active optimizer
stencils and conditional held-out checks are still required. The five
prepared full-observation plans remain unqualified. No fit or parameter
update is authorized by these partial vectors, and no refit was started.
