# Native Bi atomic/cell joint refinement

Base: `e6b36b5a5a89c54aa0852434f1ea59d2ad3cc4dc`.
Branch: `codex/bi13-native-joint`. Main agent is the sole writer.

## Goal and authorization

The user explicitly requested implementation of the audited fitting plan on
2026-09-11. This first milestone exposes a,c; Bi and outer-chalcogen z; all three
orbit occupancies; and six physical orbit radial/normal ADPs, together with the
existing mosaic, finite coherent size, surface fractions and film parameters.
Initialize in blocks, then exercise a joint fit. Do not claim parameter
identification or numerical qualification from optimizer completion alone.

The reviewed plan is the external `fitting-audit/tasks/plan.md` in the current
task's visualization directory. The nominal September 11 all-sample archive is
immutable baseline evidence. Main contains two inherited untracked B4 scripts;
they are outside this worktree and remain untouched.

## Read-only design findings

The current native evaluator exists in the unmerged modern-fit-defaults worktree.
Port its dependency closure additively; preserve current main's beam-position
viewer, existing simulation configuration and older declared model routes.
Extract shared equations/revisions instead of duplicating them. Retain the
generic expanded-CIF termination ensemble: conventional-cell repeats and the
integer lifts of its site rows are physical state at continuous L. Do not switch
to canonical quintuple-layer surfaces while adding fitted coordinates.

The archived Bi2Te3 observation bundle supplies one fixed native projection,
1322 rows (1268 valid), full count/background covariance, historical objective
and guard operators. New lattice candidates change predictions on that fixed
support. The historical weighted objective remains a separate comparison;
primary inference uses each supported observation once with its full covariance.
Background is independently fixed and its uncertainty is propagated once.

## Owned scope and sequence

1. Add the conditional source, spherical cone average, fiber/native spatial
   integration and native input dependencies already used by the nominal fits.
   Preserve exact inactive behavior and extract common detector identities.
2. Add explicit optional site tensors to the existing generic CIF strength;
   use the authoritative atomic-amplitude equation and immutable revisions.
3. Add an explicit symmetry-preserving Bi candidate owner. Preserve orbit row
   signs and termination lifts; rebuild direct/reciprocal bases, occupied-density
   optics and incident/exit transfer when their dependencies change. Enforce the
   declared coherent-height/film constraint for these fit candidates.
4. Add native observation loading and a reusable fit entry point with one scale,
   fixed covariance, explicit bounds/scales, active-parameter reporting and
   separate historical guards. Full cell/occupancy candidates require physical
   response rebuilding; no stale bank reuse is allowed.
5. Replay the baseline, measure cost and sensitivity, run bounded staged/joint
   experiments and independent numerical checks. Expensive production-order
   work is reported separately from lower-order exploratory optimization.

Primary ownership: `sampling/source.py`, `painted_ewald/normal_density.py`,
`core/wave_modes.py`, `optics/refraction.py`, `ordered/amplitudes.py`,
`ordered/motifs.py`, `stacking/finite_intensity.py`, native/fiber pipeline modules,
the narrow shared helpers in existing detector modules, the four small native fitting/input
modules and CLI, compact proof tests, and documentation.
No new dependency, dynamic registry, general backend framework or reference edit.

The new physical candidate contract is explicitly additive to the historical
fixed-cell/isotropic affine-CIF and five-coordinate sample-Q-envelope fits. Those
older contracts are not silently reinterpreted. Trace-stage numbering is unchanged.

## Scientific gates

- Analytic tensor attenuation and direct complex atom/repeat enumeration at
  signed noninteger L; isotropic reduction and termination-lift preservation.
- Complete candidate dependencies: A^T B=2*pi I, symmetry/occupancy budgets,
  composition-derived optical density, incident revisions and fixed observations.
- Conditional source moments, spatial probability conservation, independent
  crystal-azimuth cone average and native fit/render parity. Preserve the
  existing beam-position suite. Pb endpoint support, if ported, retains its
  direct enumeration proof.
- Full rebuild versus permitted strength/mosaic/thickness reuse. Record Bi
  stitch normalization/interval selection; inspect branch-switch sensitivity.
- Full covariance GLS, nonnegative scale profiling and historical F/G replay.
  Common occupancy remains active with density-derived optics; report weak or
  prior-dominated directions rather than treating a bound as information.
- Report independent quadrature changes in the covariance-whitened observable,
  timing, peak memory and numerical status. Coarse search is not certification.

Relevant ledger rows: PHY-ORD-003/004/007/008/009/012/013/017,
PHY-REF-003/005/007/010/012, PHY-STK-006/007/011/013,
PHY-FIT-009/014A/015/016/019A/022 and PHY-ORD-018. Existing tolerances apply to unchanged paths;
generic amplitude/direct enumeration uses rtol=3e-13, atol=2e-18 A2.
Mutation checks target wrapped termination lifts, exchanged tensor axes, stale
material/basis, moved observations and duplicated background covariance.
Only unique long-term scientific/interface proofs remain permanent.

## Handoff requirements

Run formatting/lint, compact suite, assigned proofs, equivalent-work benchmark
and seed verification with the regenerated tracked-file manifest. Keep numerical
artifacts outside the repository. Commit one coherent result after independent
review, recording APIs, classifications/first divergences, convergence, timing,
peak memory, retained proofs and unresolved inference. Do not merge into main
or label a rejected/unfinished fit accepted.

## Implemented contract and independent review

Contract API 15 adds all 13 physical coordinates to generic expanded-CIF strength, plus the
eight continuous morphology coordinates and integer repeat refits. It retains each surface's
integer lifts and per-orbit ADPs. The measured native observation and source remain fixed.
An explicit two-response execution resource invalidates on changed material or reciprocal basis;
zero-ADP and original isotropic strengths reproduce the existing path.

Read-only scientific review independently perturbed every coordinate at signed noninteger L:
all 13 changed strength; only a/c changed basis; exactly a,c and all occupancies changed optics.
Review caught and resolved Bi coherent-height validation, unequal orbit reference ADPs,
guard-constrained scale profiling, malformed observation/parameter inputs, discrete N refitting,
and empirical stitch interval evidence. Final optimizer/covariance review found no blocking defect.

The fixed 85-rod roster covers a=4.34214--4.42986 A and c=30.19203--30.80197 A with each
occupancy in [0.8,1]. The absorption-aware global elastic Q bound is 8.156850154955768 A^-1;
the first omitted radius is 8.188969241620356 A^-1. Exhaustive indices [-8,8] plus the
reciprocal-metric eigenvalue bound prove completeness beyond that square. The production
validator repeats this proof for every CLI cell box and rejects an incomplete roster.

## Measured experiments and retained evidence

Artifacts are external under the task visualization directory
`2026/09/11/01a09194-718e-7e83-ab46-0fc959c9d928`; every diagnostic is one NPZ with an embedded
JSON manifest. Numeric JSON plans contain bounds and complete repeat/source/integration choices.
No experimental output replaces the nominal September 11 fits.

`bi13_baseline_replay.ra_diag.npz` uses the fixed archival continuous response with the new
canonical Bi13 strength evaluator. It reproduces all 1322 incumbent count predictions to
1.5134e-8 count maximum absolute error (2.2176e-12 relative): **MATCH**. At archived scale,
GLS chi-square is 1,258,551.618681115 and historical loss 5903.609165255841; all guards pass.
The feasible scale interval is [150892252.52327472,150921827.9615696]. Unconstrained GLS selects
152541206.97174653 and fails two m0 count guards. Constrained profiling recovers the old scale
and passes. This exposes a real constraint, not evidence for relaxing it.

`bi13_fresh_baseline_comparison.ra_diag.npz` compares a freshly rebuilt p12/g5, 32-source
baseline with that archive. The first complete prediction took 2122.419 s, including
1295.866 s response compilation; observed Windows peak working set was 4,367,552,512 bytes.
At the archived scale it passes every guard, has GLS chi-square 1,258,159.634760466 and
historical loss 5898.491180405545. Its prediction differs from the archive by 0.00321102
relative L2, 16.3495 covariance-whitened L2, or 0.459141 RMS per supported row. Free scale
profiling gives chi-square 1,257,657.768909310 but fails guards; guard-constrained profiling
gives 1,258,135.469643420 and passes. These are numerical comparisons, not accepted gains.

The first demonstrated divergence is `quadrature.proposal_construction`, before event
transport. The archived event generator used axial half-width 1/32 L (its proposal bank
had N=16), phase-Q Bragg centers transformed to external Q for m0, and angular proposal
sigma/gamma 0.3329091562/0.0300000000 degrees. The new declared proposal uses 1/30 L,
uniform centers in the external-Q coordinate for m0, and 0.3290720092/0.1995209388 degrees.
Both divide by the complete axial and conditional-angular proposal densities and therefore
target the same integral. The native projection and finite support are identical. Relevant
read-only archived evidence is `continuous_fiber_quadrature_20260911.py:239,268,277,313`
and `all_material_mixed_thickness_refit_20260909.json:42` in the September 6 visualization
directory; the new equations are in `pipeline/fiber_detector.py`.

Neither independent proposal is established as a converged oracle, so this comparison has
**NO_ORACLE** and `not_qualified` status; it is not an exact replay or a corrected legacy
physics claim. The complete first prediction remains in
`bi13_baseline_production_order.ra_diag.npz`. That process started before prediction
memoization was added and was stopped during its redundant final reevaluation after the
complete baseline checkpoint. No optimization or sensitivity stage was configured. The
comparison artifact records this explicitly and binds the retained checkpoint by SHA256.

`bi13_exploratory.ra_diag.npz` contains atomic initialization followed by the all-active
21-coordinate joint search on the full 1322 native rows/1268 supported covariance rows,
all 32 source states, p7/g2 integration and 8 projection workers. It made 56 response
compilations in a 597.612 s complete experiment. The coarse GLS objective decreased from
1,579,074.491830201 to 1,468,743.057363992. The optimizer hit the explicit iteration budget;
the candidate failed guards. Its local SVD is diagnostic only. The companion repeat-refit
experiment independently optimizes all continuous coordinates at N=14 and N=16.

`bi13_repeat_refits.ra_diag.npz` completed both all-active refits, each with a two-iteration
budget and historical constraints enabled. N=14 returned chi-square 1,490,374.621256;
N=16 returned 1,498,062.280394. Both reached their iteration limit and failed guards, so
neither replaces the N=15 candidate; this does not establish a converged preference for N.
This experiment took 1240.548 s with 103 response
compilations. It records the empirical specular stitch at the baseline and all 21 sensitivity
perturbations: no interval-selection branch changed in these local probes; every surface/line
used the declared fallback interval [3,6] Qc. This local observation does not establish global
smoothness or physical adequacy of the empirical composite.

`bi13_numerical_checks.ra_diag.npz` records independent axial and angular checks on the selected
coarse candidate. At one fixed physical scale, p8/g2 changes the whitened native prediction by
446.661 in L2 norm (12.5435 RMS per supported row); p7/g3 changes it by 280.087 (7.8656 RMS).
The independently seeded p9/g3 check differs by 560.762 (15.7478 RMS). These fail qualification.
The first coarse baseline also differs from the archived-resolution result by 13.8273 whitened
RMS per row. No parameter interval or new best physical structure is inferred from this run.

The new response engine is the intended fitting capability. The measured trials establish that
all requested coordinates are wired through it, and establish that this coarse integration is
inadequate for accepting their estimates. The next measured fit must qualify the detector
observable at a feasible computational cost before trusting small atomic-parameter changes.
Full occupancy/source/ADP identifiability and missing PSF/strain/extra-channel physics remain
separate scientific questions.

## Performance and permanent handoff proof

`bi13_projection_benchmark.ra_diag.npz` compares identical warmed source-0 p9/g3 work on all
1322 observation rows: five batches, 20,055 events, 155,488 sparse probability entries.
Serial projection took 10.878821 s; four bounded workers took 3.039432 s (**3.579x**), with
bitwise-identical CSR data/indices/pointers. Other validation jobs were concurrent; this is a
representative work-equivalent benchmark, not an end-to-end fitter speedup. Windows peak process
working set was 509,493,248 bytes, including the loaded observations/runtime.

The optimizer logs also exposed duplicate physical evaluations between SLSQP's objective and
guard finite differences. The final execution resource retains at most 64 immutable prediction
vectors keyed by the full candidate, in addition to its two response objects. Independent review
confirmed complete keys and immutable ownership. The reported exploratory timings precede this
small-vector memoization and are not claimed as final-version speed benchmarks.

The full suite passed 498 tests without skips in 619.98 s (14 expected CUDA small-grid warnings).
After the final bounded-cache change, all 13 native fitting/spatial tests passed in 18.29 s;
the same 13 passed after the final range-validation refinement.
All registered proof commands pass scientifically; their historical clean-tree wrapper gates
are rerun on the final commit. Formatting, lint and whitespace checks pass. No type checker is
configured. The only environment repair was installing the existing locked Matplotlib optional
dependencies required by pre-existing figure tests; no project dependency or lockfile changed.

Retained new tests cover four distinct fitting/atomic contracts and nine conditional spatial
contracts; existing integration, cone-density and stacking tests gained one distinct physical
invariant each. Large numerical runs and benchmark arrays remain external proof artifacts.
No temporary script, notebook, fixture, diagnostic writer, model registry, new dependency,
sample-Q stand-in for atomic ADPs or obsolete adapter was added to production modules.

Minimum integration request: review and fast-forward the coherent branch from its recorded
base after the ordinary clean-main gate. The main checkout and its two inherited B4 scripts
are untouched. Existing nominal fits remain authoritative. This core milestone is ready with
explicit measured-fit limitations; neither parameter identification nor a new accepted fit is
claimed.

## Entry points and rerunning the experiments

`BiNativeStructureModel.bind` owns the physical atomic/cell update; `BiNativeFitEvaluator`
evaluates its detector-native prediction; `fit_bi_joint` controls active coordinates and guards;
`bi_joint_sensitivity` supplies the range-scaled local diagnostic. `NativeFitObservations`
loads and verifies the frozen data/projection/covariance bundle. The JSON CLI is
`scripts/fit_bi_native.py --physics <physics.json> --observations <observations.json>
--plan <plan.json> --output <external.ra_diag.npz>` with `src` on `PYTHONPATH`.

The numerical physics and observation inputs live in the September 6 visualization directory's
`native_fit_inputs_20260911/bi2te3_{physics,observations}.json`. Plans and diagnostic outputs
named above live in this task's September 11 visualization directory. Each diagnostic records
both input revisions and its full numeric plan; the observation manifest binds its NPZ by SHA256.
The initial exploratory plan intentionally permits infeasible candidates; the separate integer
refit plan enables historical constraints. Neither plan supplies numerical qualification or
changes the authoritative nominal result.
