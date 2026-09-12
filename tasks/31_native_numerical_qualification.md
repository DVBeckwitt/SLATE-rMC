# Numerical qualification and physical fitting

User authorization: plan and implement the remaining scientific fitting workflow, 2026-09-12.
Base main: `e6b36b5a5a89c54aa0852434f1ea59d2ad3cc4dc`; reviewed prerequisite:
`9f8a477664fbbe60c54104d0dffb64261b829922`. Branch: `codex/native-numerical-qualification`.
The new worktree starts from main and fast-forwards the prerequisite. Main's only untracked
files are the two inherited B4 scripts; this continuation preserves them and does not merge main.
The main agent is the only writer. Review agents are read-only.

## Ordered work

1. Isolate Bi overlap-normalization error and source/axial proposal coupling. Implement the
   smallest justified numerical corrections. Preserve legacy observables or explicitly name
   and document a changed empirical prescription. Prove normalization, support and weights.
2. Qualify one Bi and B4 at baseline and representative perturbations. Fixed-scale whitened
   prediction RMS must be at most 0.1, parameter-contrast RMS at most 0.05, absolute profiled
   objective-contrast error at most 0.5, with applicable guards. Refine axial, angular, source,
   cone, spatial and handoff rules independently, then use an independent seed/combined rule.
3. Demonstrate synthetic recovery of identifiable combinations using independently qualified
   truth. Report unidentifiable directions explicitly; do not manufacture recovery by fixing them.
4. Run staged initialization followed by all-active multistart fits, independent N alternatives,
   nuisance profiles and conditional validation for all six specimens. Qualify every reported
   candidate and profile; a baseline pass alone cannot qualify a fit.
5. Evaluate control contamination and residuals. Add physics only if a qualified discrepancy
   establishes its need. Report estimates, profiles, bounds, correlations, unresolved freedoms,
   and targeted calibration or measurement needs. Integrate only through the clean-main gate.

## Scope and verification

Owned numerical implementation: `pipeline/fiber_detector.py`, `reflectivity/specular.py`,
their existing native configuration/binding/trace call sites, compact scientific tests, and
the corresponding contract, ledger, validation and task documentation. The shared equations,
coordinates, observable, observation supports and reference pack remain authoritative.
Relevant rows: PHY-FIT-020/022/023/024/025 and existing specular/conditional transfer rows.
No arbitrary priors, altered error thresholds, physical cutoff, raster or extra blur is admitted.
No new dependency is planned. Reuse existing runner, data bundles and declarations.

Each numerical slice requires an independent invariant/oracle and a measured convergence
artifact before expensive downstream fits. Keep artifacts external as one `.ra_diag.npz`
with numeric arrays and an embedded manifest. Record the first divergence, code/input hashes,
wall time and peak memory. Retain only unique scientific regressions. Before handoff run
formatting, lint, the compact suite, eight registered proofs and file-inventory verification;
obtain read-only review, then create one coherent commit with clean worktree status.

Implementation, numerical, predictive and identification status are separate. Failure of a
numerical gate prevents accepted parameter estimates, not further numerical investigation.

## Implemented numerical and reporting changes

Contract API v17 adds the explicitly named `continuous_q_median` overlap prescription,
conservative frozen Q proposal domains, and composite Gauss quadrature with optional physical
axial-panel width control. The manuscript's sampled overlap median and Sobol rule remain the
defaults. No physical scattering equation or observation is replaced. The continuous median
is `NO_ORACLE` against legacy, with first divergence at overlap normalization after identical
Parratt and structure strengths. Its independent authority is the uniform-coordinate
sublevel-set measure and analytic constant, affine and finite-fringe identities.

Native result schema v2 distinguishes an optimizer candidate from a numerically checked
selection. The default initial gate exits 2 before optimization when qualification fails.
Every fitted N, nuisance-refitted profile center and conditional-validation prediction has
its own numerical checks. Cross-N/profile objective offsets are checked together. An admissible
profile point below the corresponding N's joint minimum blocks selection pending joint refitting,
including guard-feasible improvements found by an unguarded profile. These controls do not
turn empirical agreement at declared probes into a certified error bound.

All artifacts below are external under
`C:/Users/Kenpo/.codex/visualizations/2026/09/11/01a09194-718e-7e83-ab46-0fc959c9d928`.
Each diagnostic contains numeric arrays and one embedded manifest. Plans bind exact inputs;
source hashes describe startup filesystem snapshots, not loaded-bytecode attestations.

- `qualified_bi2se3_continuous_overlap.ra_diag.npz` holds distinct baseline/occupancy
  candidates on fixed outgoing/source rules. Refining 32769 to 65537 overlap nodes gives
  maximum prediction RMS `3.54625e-5`, contrast RMS `7.24498e-6`, and objective-contrast
  error `-0.0851168`; all three gates pass. The study takes 40.0557 s.
- `qualified_bi2te3_handoff.ra_diag.npz` passes the same isolated overlap-grid check with
  maximum prediction RMS `5.34377e-7`, in 61.3759 s. These qualify the handoff calculation
  at these probes, not complete detector integration or parameter estimates.
- `qualified_<sample>_source_envelope.ra_diag.npz` records source-rule and occupancy probes
  underlying frozen domains: Se regular `(0.35,4.95)` and local `(0,4.95)`; B4 regular
  `(0.25,4.9)`, all in inverse angstroms. Future candidates must still pass enclosure checks.
- `qualified_bi2se3_integration.ra_diag.npz` records a failed Sobol p9 to p10 axial check:
  maximum whitened RMS `14.5794`. The remaining expensive sweeps were stopped explicitly;
  the B4 sibling contains only its completed baseline checkpoint.
- `qualified_<sample>_gauss.ra_diag.npz` records a rejected global-Gauss prototype. Combined
  refinement disagreements are `39.3455` Se and `40.2837` B4. That prototype is removed;
  its retained plans name an obsolete experimental option and are provenance only.
- `qualified_<sample>_composite.ra_diag.npz` records the initial composite p7/g3 trials.
  Axial/angular/combined disagreements are `35.1906/14.2980/38.0580` Se and
  `15.1264/20.2061/24.1655` B4. Complete runtimes are 670.0715 and 878.8149 s. All fail.
- `qualification_gate_smoke.ra_diag.npz` verifies the default failed-initial gate and
  subprocess exit code 2. `qualification_profile_smoke.ra_diag.npz` exercises all-active
  and nuisance-profile execution with an explicit unqualified override: 184 evaluations,
  177.4609 s, `selected=null`. This is execution evidence, not physical recovery.
- `qualified_sampler_parity_benchmark.ra_diag.npz` compares 65536 equivalent Sobol nodes
  with the previous own implementation. Coordinates and weights are bitwise identical.
  Five-run median sampler times are 0.174539 s before and 0.200680 s after; peak working
  set is 258654208 bytes. Concurrent numerical trials were active. This narrow benchmark
  shows a sampler overhead, not a whole-fit speedup or a memory bound for fine grids.

Read-only source-0 probes explain a major failure mechanism. Se's largest-error native bins
contain only one to four axial nodes; one physical inverse-CDF panel spans 0 to 0.220803
inverse angstroms, leaving a neighboring node gap of 0.113827. B4 also has important bins
with only three to seven nodes, while broader regions retain angular/intensity errors.
Exact pixel-run expansion would introduce 241396 Se or 198653 B4 runs, with only about
1.4--1.5 times potential savings from removing empty events in the sampled slices. The
chosen next experiment therefore refines physical axial panel widths before adding geometry
clipping. Its `.02` versus `.01` inverse-angstrom plans use two first-order spectral source
nodes to isolate axial behavior; they cannot establish source convergence.

Completed finer trials are retained as `qualified_<sample>_physical_axial_followup.ra_diag.npz`
and `qualified_<sample>_physical_angular_followup.ra_diag.npz`. Axial cap `.01` to `.005`
gives maximum prediction RMS `0.0193593` Se and `0.0141682` B4; objective contrasts are
`-3.91267` (fail) and `0.412069` (pass). Angular order 16 to 32 at cap `.01` gives RMS
`8.75063` Se and `11.1784` B4, with objective contrasts `-174.400` and `-1630.60` (fail).
Axial times are 657.052/1196.462 seconds; angular times are 644.851/1040.562 seconds.
Maximum peak working set is 2,081,529,856 bytes. These files reuse the preceding refined
raw vectors as references and bind those artifacts by hash. Source hashes in these
follow-ups describe end-of-run filesystem snapshots, not loaded bytecode.

## Reassessment: current failure points and next actions

The user paused broad sweeps to reassess the fitting goal. All such sweeps are stopped.
The observation-grouping prototype, upstream membership mask and evaluation-local stitch
cache are removed: grouping multiplied axial work and did not solve numerical convergence.
Their external plans and timing artifacts remain historical evidence, not supported controls.
The completed uncapped Se axial comparison reverses the occupancy objective contrast from
`+2435.014` to `-1741.035`. Angular 32-to-64 comparisons still fail: prediction RMS is
`1.11333` Se and `1.64797` B4, objective-contrast errors `13.0531` and `-9.27726`.
The B4 uncapped axial and Se 128-angular follow-ups were paused without complete comparisons;
their checkpoints explicitly retain incomplete status and no selected candidate.

1. **Artificial angular remeshing creates axial discontinuities.** At an actual Se
   source-0/local-m0 Q boundary (`2.0614578928089955` inverse angstroms), a change of
   `2e-9` changes coarse angular acceptance from `1.29277e-6` to `0.01869534`.
   Individual observation bounds activate different arc unions, moving all quadrature nodes.
   The implemented `angular_support="fixed_union"` uses the complete conservative source
   angular union throughout its Q interval. The existing integrator retains all proposal
   masses and native projection retains contribution ownership. This removes the artificial
   jump; a public analytic Gaussian/continuity regression passes. The compatibility default
   remains `"q_conditioned_union"`. Geometry changes can still change union endpoints.
2. **Angular resolution remains inadequate.** Independent uniform physical-angle integration
   at that same geometry agrees between 32769 and 65537 points to `2.83e-15` relative.
   Fixed-union g12 agrees to `7.76e-14`; g10 differs by `1.44e-4`. Actual mosaic/source/
   polarization/envelope weighting with unit signed-sheet strength preserves this result.
   Eight-point fixed physical-angle panels at width `0.00625 rad` reach `4.30e-10` relative
   error using 3872 nodes, versus 8192 for g12. Intermediate refinements are nonmonotone.
   Five additional low/high-Q and nonzero-rod Se/B4 slices now compare against independent
   uniform-angle oracles converged within `1e-9` relative. A B4 mosaic-width stress case
   rules out a generic panel replacement: local-CDF GL8 panels differ by `4.29e-4`, direct
   physical GL8 panels by `1.17e-6`, while the existing global g10 differs by `8.59e-9`
   using fewer nodes. No angular-panel API is added. Exact arrays and input/source hashes
   are retained in `qualified_angular_slice_reassessment.ra_diag.npz`.
3. **Qualification probes must cover optimizer decisions.** The occupancy probe `0.01`
   does not test the actual SLSQP step `2e-5` or sensitivity step `1e-6`, much less other
   coordinates. Check actual parameter stencils, response rebuild counts and complete
   iteration cost before all-active optimization. Initialization may use explicitly
   unqualified warm starts only when independent refinements preserve improvement direction
   and satisfy a predeclared relative decision-error budget. Final `0.1/0.05/0.5` gates remain.
4. **Identification and missing physics are downstream questions.** Source integration is
   still unqualified. Occupancy/scale freedoms, inactive mixtures and absent independent
   calibration cannot be repaired by releasing more coordinates or absorbing errors into
   blur, strain or DWBA. After numerical proof, perform identifiable synthetic recovery,
   qualified all-active fits, N comparisons, profiles and conditional validation.

`qualified_optimizer_stencil_preflight.ra_diag.npz` now retains exact baseline and
one-sided SLSQP candidate vectors: 40 for Se (39 coordinates) and 44 for B4 (43).
The signed normalized steps agree exactly with installed SciPy's bound-handling oracle.
This is candidate-construction evidence only: no predictions or gradients were evaluated.
The interface guide documents replay using the existing runner; no API or gate was added.

The next work is deliberately bounded: refine one additive rod family and update the
complete native vector as `coarse_total - coarse_family + refined_family`, preserving every
other contribution. Begin with m0 on uncapped p7 to isolate angular error. Combining angular
g10 with the `.01` axial cap would require at least 16.2 million m0 events; uncapped p7
uses at most 524288 under the same two-source/two-arc assumptions. These are work estimates,
not performance or accuracy claims. A family-only correction cannot qualify the full model.
Then check optimizer-sized contrasts and one representative Bi/Pb pilot. Do not resume
all-six sweeps first.

## Completed bounded attribution check

`qualified_bi2se3_m0_angular_attribution.ra_diag.npz` retains two complete coarse native
predictions and the corresponding m0 contributions at g4/g8/g10. The perturbation is the
actual SLSQP occupancy step `-2e-5`, not the earlier `-0.01` probe. The experiment uses the
fixed two-line source, fixed angular union and uncapped axial p7; every non-m0 contribution
remains in the complete corrected vectors. Candidate optics, mosaic, thickness, roughness,
source weights and original observation memberships are preserved.

For corrected g8 versus g10, maximum prediction RMS is `0.00821780135`, parameter-contrast
RMS `2.44181813e-7`, and objective-contrast error `6.67758286e-6`: all three declared gates
pass for this angular-family comparison. The normalized occupancy gradients are
`443.13709847` and `443.13702293`, a relative discrepancy of `1.70462552e-7`, with the same
improvement direction. This measures m0's numerical contribution to an actual optimizer
step; it does not qualify the still-coarse other families, axial rule, source integration,
all coordinates, or any physical fit.

The first 90-second run checkpoints seven of eight evaluations. A separate bounded
continuation verifies source hashes and candidate equality, reuses all completed arrays,
and computes only the missing candidate-1 m0 g10 result. Recorded accumulated analysis time
is 128.677 seconds, excluding stopped incomplete work and parent-process overhead. m0 g10
retains 228853 events and 3602456 sparse probabilities per candidate; its measured compile
and evaluation times are 31.989 and 42.614 seconds. These are diagnostic costs, not a full-fit
benchmark; no whole-pilot peak-memory measurement is claimed.

Next numerical experiment: the first nonzero radial family under the same exact additive
comparison. Then independently refine axial/source integration and the remaining optimizer
directions. There is now a demonstrated way to attribute error without refining every
family globally. No all-six fitting sweep resumes from this isolated pass.

## Compact proof and retention

The complete suite passes 508 tests in 503.79 s with 14 expected CUDA underutilization
warnings. The final public-result overlap-measure metadata adds one parameterized case;
all 30 reflectivity tests subsequently pass. One earlier invocation exited inside Numba
compilation with a Windows access violation; the isolated CPU/CUDA case, eight changed
specular cases, and full file all passed unchanged on recheck. The external proof record
retains this limitation rather than treating the interrupted invocation as a pass.

The final current-source suite passes **510 tests** in 861.32 seconds (864.720 seconds
subprocess wall time), with 14 expected CUDA underutilization warnings. Formatting and lint
pass over `src`, `tests` and `scripts`. The added public fixed-support test protects interior-Q
continuity and absolute angular/axial normalization against an analytic Gaussian. Explicit
Python error injection detects both restored Q-conditioned remeshing and doubled quadrature
mass; the ordinary JIT path passes in the full suite. The earlier full 509-test snapshot
predates removal of the observation-grouping prototype and is not the final proof.

All eight subsystem scientific proofs pass; geometry/optics and mosaic/Ewald wrappers
additionally require a clean committed worktree. Their final results are retained externally. The 378-file/12-reference-case inventory passes.
External `qualified_native_final_proofs.ra_diag.npz` and
`qualified_reassessment_subsystem_proofs.ra_diag.npz` retain command outputs and timings.
The original repository-wide formatting invocation incorrectly included immutable legacy
evidence; those files remain unchanged and outside the authoritative scope. No type checker
or new dependency is introduced.

Retained tests protect distinct long-term behavior: uniform-coordinate overlap normalization
with isolated zeros and tiny medians; exact compiled low/high branches and explicit result
measure; nonseparable joint quadrature with arc masses and narrow support between proposal
peaks; fixed-support continuity and absolute mass; frozen-domain source reuse and insufficient-support rejection; and profile/held-out
numerical error that an initial/training screen alone would miss. Existing integration tests
are extended rather than retaining experimental scripts, images or numerical sweeps.

Public additions are `continuous_overlap_scale`, explicit `overlap_measure` on stack and
result types, the optional Q-domain/quadrature/panel/angular-support controls on `FiberIntegrationRule`,
`compare_conditional_predictions`, and the documented native runner/result-v2 behavior.
Trace v4 and the immutable reference pack remain unchanged. Main's inherited B4 scripts
remain untouched; integration requires the normal clean-main gate.

Current default-path benchmark: `qualified_reassessment_sampler_parity.ra_diag.npz`
compares 65536 Sobol nodes against the previous own committed sampler. All coordinates and
weights are bitwise identical. Five alternating-order warmed repetitions give median times
0.134746 seconds before and 0.135670 seconds after, with a 256319488-byte peak working set.
The compact suite ran concurrently. This is sampler overhead, not whole-fit performance.
