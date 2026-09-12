# Exact native rectangle coalescing

Continue the authorized all-parameter fitting work by reducing measured detector-response
construction cost. Base main is `e6b36b5a5a89c54aa0852434f1ea59d2ad3cc4dc`; this isolated
worktree fast-forwards the reviewed numerical prerequisite `24ec7b2`. Main's two inherited
untracked B4 scripts remain untouched. One writer; subagents review only.

## Plan and ownership

1. Coalesce adjacent columns only when their observation owner, exact pixel weight and
   vertical bounds agree. Deduplicate the resulting rectangles. Keep the existing per-column
   candidate search and evaluate each wider rectangle once per continuous Gaussian kernel.
2. Preserve every native pixel membership, weight, source factor and scalar Gaussian
   probability authority. No raster, rounding, normalization, physical cutoff or public API
   is added. Relevant ledger rows are PHY-SRC-001 and PHY-FIT-025. Their physical treatments
   remain unchanged; the conditional source model still has no legacy oracle.
3. Extend the compact public spatial invariant only for weighted overlapping/gapped regions
   and a strongly correlated kernel reaching a rectangle's far end. Retain the existing
   independent near-singular conditional-CDF oracle.
4. Compare the complete native observable at the actual occupancy stencil against the
   unchanged prerequisite. Record real-kernel benchmark and observable parity externally.
5. Finish the already launched standard checks and independent review, then commit this
   cohesive change. Follow the user's instruction to prioritize fitting over additional
   proof sweeps: start response-reusing optimization and check promising candidates.

Owned files: `pipeline/source_spatial.py`, its existing spatial test, this task/index,
the corresponding contract/validation notes and file manifest. Other scientific equations,
contracts and immutable reference inputs are read-only.

## Baseline evidence

External `qualified_bi2se3_wide_rectangle_benchmark.ra_diag.npz` measures 4096 actual
source-0 m0 kernels across four Q windows and all 1250 native observations. Seven alternating
warm repeats give median 0.2850083 seconds before and 0.2134083 seconds for the prototype
(1.3355 times faster). Maximum probability disagreement is 4.44e-16; comparison with doubled
scalar quadrature order gives 5.55e-16. This is a microbenchmark, not a whole-fit speedup or
certified error bound. Wider rectangles must pass probability, tail and full-observable checks.

## Result and next fitting step

The production partition reduces the same 241396 unique vertical rectangles to 196472
wider rectangles. Candidate searches still use native columns; a rectangle is marked visited
only after its row-overlap check and probability evaluation. This preserves contributions
from oblique beams that reach the same rectangle in a later column. Different overlapping
owners can increase column-reference storage; it is bounded by the original owner-specific
vertical runs. Public APIs and physical weights are unchanged.

`qualified_bi2se3_projection_coalescing_parity.ra_diag.npz` recomputes every configured rod
family at both the baseline and actual occupancy stencil. The maximum whitened prediction
RMS difference is 1.02e-14, raw absolute difference is 4.34e-19, and the profiled objective
contrast is unchanged. Both normalized occupancy gradients are -11.62590591734216. The
two full predictions take 192.958 seconds, with peak working set 1041346560 bytes. These
timings are not a controlled whole-fit speed comparison. The first-order source still
limits this comparison to projector parity, not physical-fit qualification.

Two existing permanent tests are extended: the independent conditional-CDF test now
checks whole-panel and narrow-row regions for strongly correlated beams; the weighted
region test includes a gap, overlapping regions and a full-panel region. Both pass before
and after the implementation. No new test function, dependency or numerical equation is
introduced. Independent read-only reviews find no correctness defect.

The complete permanent suite passes 510 tests in 457.18 seconds (459.230 seconds
subprocess wall time), with the 14 expected CUDA underutilization warnings. Formatting
and lint pass; all six registered proofs that permit an uncommitted worktree pass.
`qualified_projection_coalescing_proofs.ra_diag.npz` retains those command results.
The two clean-worktree proof wrappers and final tracked-file inventory complete the
handoff gate. No additional exploratory test or benchmark sweep is retained in the package.

The next execution uses a nondegenerate two-order divergence/wavelength source and positive
bandwidth. Sixteen Bi coordinates that reuse the response are optimized first. The remaining
coordinates are fixed only during initialization; all-parameter release, integer choices,
multistarts, candidate refinement and physical validation remain required. Exploratory
initialization is never reported as an accepted physical fit.
