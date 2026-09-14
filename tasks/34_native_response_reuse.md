# Native response reuse and workflow consolidation

Implement the authorized performance and branch audits from 2026-09-14. The scientific
objective remains detector-native joint geometry, source, mosaic and structure fitting
for Bi2Se3, Bi2Te3 and the four PbI2 acquisitions. All admitted physical parameters
remain available; numerical qualification and parameter identifiability remain separate
from optimizer completion. This task does not claim accepted new fits or error bars.

Base: `349524d24f960198d75df8def104adbca204944a`, fast-forwarded onto main before
creating `codex/native-response-reuse`. Only the root agent writes. Preserve all refs,
dirty files and ignored satellite payload before retiring old worktrees. Retain main
and its inaccessible cloud placeholders. Historical physics experiments are archived,
not merged over the accepted equations.

## Implementation order

1. Reuse Gaussian rectangle corner arithmetic in the shared native integration owner.
   Compare complete outputs and equivalent warmed work against the base revision.
2. Separate reusable scattering from source-position and detector transport. Declare
   proposal validity and exact dependencies; retain events before detector pruning.
   Separate spectral mixture masses from geometric event coefficients. Never reuse
   a response outside its declared validity or claim an arbitrary permanent response.
3. Add public bounded TRF fitting with shared batched differences and durable raw
   prediction checkpoints. Preserve SLSQP for explicit inequalities. Resume completed
   predictions, not private optimizer internals. Keep candidate/converged/qualified
   states distinct, including interrupted discrete and profile searches.
4. Consolidate preparation, fitting and rendering for all six frozen acquisitions.
   Preserve observation support, signed dark covariance, accepted background arrays,
   calibration ownership and absolute intensity scale. Resolve relocated inputs using
   explicit paths and hashes. Full images integrate the continuous model over pixels.
5. Retire duplicate Bi fitting and historical workflow orchestration after migrating
   their surviving contracts. Keep immutable references and reproducible historical
   source in the verified recovery bundle. Update current commands and manifests.
6. Run focused scientific boundary tests, equivalent Bi/Pb performance comparisons,
   configured handoff checks and compact permanent suite. Record limitations and
   measured improvements. Commit one cohesive result, integrate main, retire verified
   superseded worktrees/branches, then stop.

## Acceptance

PHY-FIT-023 through PHY-FIT-025 and the source/spatial contracts remain authoritative.
No changed physical equations, observation weights or relaxed tolerances. Reuse must
agree with direct evaluation on the same proposal. Event count, cache ownership,
wall time and retained memory must be explicit. Retain only tests protecting distinct
scientific invariants or preparation/search/render integration boundaries. No generated
diagnostics or temporary harnesses belong in production or the repository root.

## Implementation and evidence

The common material boundary is `NativeRefinementModel`: parameter names/units,
physical binding and exact inactivity. `BiJointModel` and `PbJointModel` own their
symmetry/stacking rules; no Bi/Pb dispatch remains in the response evaluator.
`native_workflow` is the explicit built-in assembler and accepts another supplied
model. `native_prediction_group` accepts that same model seam. The public JSON CLI
currently admits built-in recipes, not arbitrary automatic CIF refinement.

`FiberScatteringCache` retains bounded pre-projection state (256 MiB/4096 entries).
`NativeFiberResponse` v2 separates current source masses. Transport perturbations
reuse upstream state only when actual quadrature and optical dependencies match;
the optional frozen Ewald envelope explicitly validates Q/angular coverage. New
detector probabilities remain necessary. Gaussian pixel corner reuse preserves the
stable independent fallback; fitting rectangles retain direct corner arithmetic.

The common runner binds public TRF, physical scaling, bound-aware batched differences,
persistent isolated processes and parent-owned exact raw recovery. Each worker group
owns its own bounded evaluator. SLSQP remains for historical inequalities. All
admitted parameters are released in the final stage. Scope-correct lower unfinished
candidates and discrete/profile ownership survive recovery. The raw store publishes
whole completed records; the renderer publishes only complete additive batches.
Neither resumes private optimizer internals or promotes unqualified candidates.

All six frozen experiments passed raw-pixel reprojection, count/dark covariance,
unchanged net/background and control-support verification. Portable copies reside in
`prepared_native_inputs_20260914` under the external evidence root. Preparation is
adoption of accepted calibration, not a new background estimator. Full native rendering
uses one whole-panel proposal, actual physical mosaic/roughness/scale, and independent
fit/image numerical status. Synthetic targets retain their target-kind label.

The duplicate Bi optimizer/evaluator, historical layered/replay orchestration, unused
workflow configs and their superseded tests are retired. Remaining geometry, scan,
Monte Carlo, direct atom/transition and covariance boundaries remain live. All imports
now name their owning modules instead of eager fitting-package reexports.

### Validation

The full permanent suite passes: **365 tests in 486.25 seconds**, including CPU/CUDA
parity. The 14 warnings are expected small CUDA-grid utilization warnings. Ruff lint
and formatting pass; no separate type checker is configured. The subsequent atomic
raw-record publication change also passes its focused interrupted-execution test.
Standard clean-worktree proof wrappers run after the coherent commit; their external
diagnostic records final status, so dirty-worktree precommit status is not hidden.

New/extended permanent tests protect distinct boundaries: pre-pruning/scattering reuse
and exact spectral mass, insufficient envelope rejection, frozen-proposal native
pixel/batch equivalence, all-parameter TRF bounds and calibrated search, lower unfinished
evidence, discrete/profile ownership, interrupted raw recovery, portable frozen-input
adoption and complete-image resume. No exploratory sweep, full-image snapshot or benchmark
harness remains in the repository. Immutable reference and example bytes are unchanged.

### Equivalent work and limitations

The complete center-plus-one-sided stencil on the actual frozen observations agrees
with baseline `349524d` at the same coarse numerical rule:

| Sample | Continuous coordinates | Old/new warm batch seconds | Relative norm discrepancy |
|---|---:|---:|---:|
| Bi2Te3, N=15 | 39 | 16.396 / 15.850 | 1.32e-16 |
| B4, N=72 | 43 | 22.778 / 22.252 | 1.18e-16 |

These are single sequential runs with axial power 3, angular power 1, two source lines
at Gauss-Hermite order 1, all original rods/observations and one worker. They demonstrate
equivalent implementation and about 2–3% speed improvement for this bounded workload;
they do not predict full-resolution fit speed or qualify derivatives. Warm-up is excluded
from the tabulated wall time but included in recorded cumulative compiler counters.
Full-process peak working sets are approximately 503/505 MiB for Bi and 444/446 MiB for
Pb, including imports and JIT. All four remaining acquisitions also pass complete center
prediction parity: relative norm discrepancies 1.31e-16 through 2.32e-16.

An additional fixed-envelope Bi stencil agrees with fresh direct reconstruction at
relative norm 2.55e-17; only the extra-thickness row differs bitwise (maximum 1.09e-19),
from the already accepted attenuation-rescaling arithmetic. The proof uses the existing
5e-12 response tolerance, not a new physical acceptance threshold. It records 224 upstream
cache hits/176 builds including warm-up, with 93,360 retained peak cache bytes. Warm
contraction took 10.478 s, current transport 0.164 s and region probability 0.735 s over
2,136 projected events at this deliberately small rule. Those proportions must not be
extrapolated to the production integration rule. Fresh reconstruction of every candidate
took 40.133 s versus 16.969 s with all response caches; that comparison includes preexisting
SF/intensity reuse and is not the incremental speedup of this change.

The same fixed-envelope B4 stencil passes at relative norm 2.79e-17, with only the
extra-thickness row differing bitwise. It records 224 cache hits/160 builds including
warm-up, 87,120 peak retained bytes and 2,185 projected events. Warm transport/probability/
contraction times are 0.163/0.641/20.305 s. Total warm cached/direct-reconstruction times
are 25.427/45.267 s; the same nonincremental-cache and coarse-rule limitations apply.
The common two-process CLI also executes a complete 39-coordinate TRF stencil and
restarts through `--resume`; returned raw rows remain exact and the result stays
unqualified with no selected fit. Worker groups are atomic returns: in-flight groups
may repeat work after interruption; already returned rows do not need recomputation.

Five alternating Gaussian microbenchmark pairs reproduce all six full 4096-kernel arrays
bitwise. Median new/old time: real Bi native 0.94967, synthetic narrow native 0.97903;
8/128-pixel fitting rectangles retain 1.6–5.3% overhead. Whole-worker peak working set
238.76→242.61 MiB includes JIT and fixtures. This local pixel gain is not a whole-fit
speedup. Final LF source SHA256 is
`9da08c4ff64ab196768a601ef5e6d1df6edda2599e627a7c5368246e7f433395`;
the measured CRLF source differs only by line-ending normalization.

No analytic probability Jacobian was introduced. It remains a possible measured next
optimization, not a claim of this task. Domain coverage and optimized/direct agreement
do not establish quadrature convergence. PHY-FIT-025 thresholds and all existing
`MATCH`/`CORRECTED`/`NO_ORACLE` classifications remain unchanged; this extension introduces
no new physical first divergence. Current Se fixed/free fits remain unqualified and no
new parameter-confidence interval is claimed.

### Preservation and integration

The verified external `consolidation_recovery_20260914/all_refs.bundle` preserves 74
original refs with complete history. The satellite manifest and byte-verified ZIPs
preserve all 25 old worktrees, dirty overlays, untracked/ignored payload and virtual
environments. All 25 HEAD/branch/status snapshots were rechecked unchanged; an additional
20-tree byte recheck, including every dirty tree, found no mismatch. Main's two B4
scripts were separately hash-verified. Main's environment and inaccessible cloud
placeholders are retained, without claiming a backup of inaccessible bytes.

The public recovery command successfully restored the native baseline and dirty Se
worktree to new external roots with recorded source-path mappings. Historical divergent
physics is preserved rather than merged over the accepted native implementation.
Commit the cohesive result, fast-forward main, and retire only verified superseded
worktrees/local branches. Keep remote, stash, recovery and Codex evidence refs.
