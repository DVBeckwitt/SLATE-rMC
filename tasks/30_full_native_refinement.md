# Complete physical native fitting workflow

User authorization: implement the entire audited fitting roadmap, 2026-09-11.
Main base: `e6b36b5a5a89c54aa0852434f1ea59d2ad3cc4dc`; the new worktree fast-forwards
the independently reviewed Bi dependency `c4d231cd2ee0b56c50c321025ec21e79feb7a9be`.
Branch: `codex/full-native-refinement`. Main and inherited B4 files remain untouched.
The main agent is the sole writer; independent reviewers are read-only.

## Intended result

One physical detector-native evaluation path supports all admitted Bi and Pb specimen
coordinates, calibrated instrument/source variation, staged initialization and final joint
optimization, independent discrete repeat alternatives, numerical qualification, parameter
profiles and prospective validation. Deliver runnable all-six experiment declarations and
measured results with separate implementation, numerical, predictive and identification status.
An unqualified calculation or unresolved parameter must remain explicit. Missing calibration
cannot be replaced with invented priors; a hypothetical missing physical effect cannot be
added as an arbitrary blur or background adjustment.

## Implementation sequence and gates

1. Profile the complete native evaluation; eliminate demonstrated repeated work with exact,
   dependency-bound reuse. Compare optimized predictions to the existing proof path and
   benchmark equivalent work. Qualify axial, angular, source, spatial and specular-handoff
   accuracy at baseline and representative perturbations in the full covariance measure.
2. Complete the parameter/measurement inventory for all six bundles. Preserve masks, native
   memberships, covariance and historical guards. Add explicit source/instrument corrections
   with frames, units, provenance and complete rebuilding. Distinguish coordinate gauges,
   physical boundaries and search limits. Requalify newly active directions.
3. Add the appropriate Pb physical candidate and joint fit through the same detector authority.
   Preserve stacking-law versus intensity mixtures, endpoints, repeat units and declared
   FINITE_TOTAL/FINITE_PER_LAYER normalization. Retain all independent Bi atomic freedoms.
4. Provide common bounded fitting, multistart and independently refitted integer alternatives;
   profile linear scale exactly. Distinguish best evaluated from converged feasible solutions.
   Exercise real-forward synthetic recovery and measured fits. Inspect weak directions before
   expensive searches, then refit nuisance coordinates for parameter profiles.
5. Diagnose residuals after numerical effects. Admit an additional physical response only with
   evidence, a declared operator/measure and an independent invariant or converged oracle.
   Record supported and unsupported physical hypotheses explicitly.
6. Run the workflow on both Bi and four Pb specimens. Share parameters only with explicit
   material/acquisition ownership. Freeze prospective validation before additional tuning,
   retain correlated train/validation covariance, and do not call reused historical guards an
   untouched test. Report bounds, profiles, prior dependence and remaining measurement needs.

## Owned paths and authorities

Owned: `fitting/` native candidate/objective/workflow modules; narrow extensions to authoritative
`pipeline/conditional_detector.py`, `pipeline/fiber_detector.py`, `pipeline/source_spatial.py`,
`pipeline/bragg_space.py`, `ordered/`, `sampling/source.py` and `painted_ewald/normal_density.py`
when required by a demonstrated dependency; fitting CLI and compact scientific tests;
contract/architecture/ledger/validation/task documentation and file inventory.
No original-RASIM execution, reference edits, model raster, silent normalization, changed
observation association, diagnostic directories or new general framework.
The measured repeated handoff work additionally owns `reflectivity/specular.py`: carry
existing phase/zero strength through typed results instead of reevaluating the same inputs.

Relevant existing ledger: PHY-FIT-000--004, PHY-FIT-014/014A/015/016/019A/020/021/022;
PHY-ORD-003/004/007/008/009/012/013/017/018; PHY-REF-003/005/007/010/012;
PHY-STK-006/007/011/013 and the existing conditional-source and native-measure contracts.
New proof rows: PHY-ORD-019 and PHY-FIT-023/024/025.

## Verification and handoff

Retain only unique scientific/interface tests: exact numerical reduction, candidate dependency
invalidation, physical symmetry/tensors/endpoints, observation/covariance ownership,
constrained fitting/profiling and all-six integration. Large profiles, fits, convergence and
timings are external `.ra_diag.npz` artifacts with embedded manifests. Run configured lint,
formatting, compact suite, registered proofs and seed inventory; provide a coherent commit,
clean status, APIs, classifications, first divergences, timings/memory, convergence evidence,
retained-test rationale and any genuine external-data limitations. Completion of code is
reported separately from acceptance of physical parameter estimates.

## Implemented result and proof

The common native workflow exposes 39 coordinates for each Bi and GD1/SiD1 specimen,
45 for Clean1 and 43 for B4, plus independent integer N alternatives. This includes
all admitted atomic/cell, mosaic, termination, Pb phase/parent/fault, optical and
source/instrument coordinates. The final stage releases the complete roster.
Source/cell/occupancy/pose changes rebuild their dependencies; exact mixture-boundary
inactivity, search versus physical bounds, calibration versus assumption terms,
unfinished minima and raw nuisance-refitted profiles are reported explicitly.

Public entry points are `BiNativeStructureModel`, `PbNativeStructureModel`,
`PbJointModel`, `NativeInstrumentModel`, `NativeJointEvaluator`, `FitParameter`,
`GaussianCalibration`, `fit_native_parameters`, `refit_native_choices`,
`profile_native_parameter`, `training_observations`, `conditional_validation`,
`compare_native_predictions`, `native_sensitivity`, and `scripts/refine_native.py`.
The detailed interface and unsupported physical responses are in
`docs/NATIVE_REFINEMENT.md`. Contract API is 16; trace schema remains 4.

The final scientific-source suite passed **503 tests**, with 14 expected CUDA
underutilization warnings, in **475.38 s**. Focused native/statistical tests also pass
after the final search/reporting changes. Ruff and formatting pass. No type checker
is configured. All eight registered proof commands pass, including the geometry/optics
and mosaic/Ewald clean-worktree wrappers. The 377-file/12-reference-case inventory
passes. Reference and example bytes remain immutable. Main's inherited B4 scripts
are preserved.

Retained tests protect distinct invariants: independent signed Pb atom/path and finite
window enumeration; all material/source/pose dependencies against fresh native responses;
projection-owner rejection and exact projection reuse; cone-component reuse under width,
mixture, thickness and order changes; real-forward identifiable mosaic/scale recovery;
analytic nuisance/scale profiles; calibration metadata and duplicate evidence rejection;
training/guard separation and conditional covariance/group partition; unfinished versus
infeasible minima; and numerical feasibility/control-variance/sensitivity-boundary errors.
No temporary sampler, scratch harness, timing assertion or measured image snapshot is retained.

## External evidence

All current artifacts are under
`C:/Users/Kenpo/.codex/visualizations/2026/09/11/01a09194-718e-7e83-ab46-0fc959c9d928`.
The six hash-verified physics/observation/seed bundles remain under the September 6
visualization directory `01a0790a-e521-7951-8cf8-97659242b079/native_fit_inputs_20260911`.

- `full_native_measurement_inventory.ra_diag.npz` binds the six complete parameter
  declarations, masks, native support, normalization, prospective grouped splits and
  independent calibration status. Every archived prediction reproduces its old score
  and passes its historical guards (`MATCH`). The same counts have separately reported
  full-covariance GLS scores; the measures are never added together.
- `full_native_<sample>_plan.json` declares the complete staged/multistart/discrete/profile
  workflow. `full_native_<sample>_screen.ra_diag.npz` contains all-six p7/g2 numerical
  screens with original 32-row sources. All fail axial/angular/source qualification;
  cone and spatial refinements agree at roundoff. `docs/VALIDATION.md` gives the values.
- `full_native_bi2te3_stitch_check.ra_diag.npz` and the corresponding Bi2Se3 artifact
  refine the Bi handoff grid from 513 to 1025. Maximum prediction RMS is 0.041189 and
  0.387748, but parameter-contrast RMS is 0.065841 and 0.382708, and objective-contrast
  errors are 661.25 and -4615.36. Both fail the declared contrast gates.
- `full_native_<sample>_exercise.json` / `.ra_diag.npz` exercise the complete measured
  workflow with deliberately coarse p3/g1 quadrature and one-iteration budgets. These
  are execution evidence, never physical fit acceptance. Per-run completion, optimizer,
  numerical, control and profile statuses are in the embedded manifests. Concurrent
  runs record startup filesystem source hashes; these are not loaded-bytecode
  attestations. The final runner also records startup worktree status.
  Their single occupancy grid point is a profile execution check, not a measured
  curve. Early best-evaluated records omit boundary flags; empty lists there do
  not establish absence of boundaries. The final search records boundaries for
  every evaluated point, including fixed profile coordinates.
- `full_native_<sample>_sensitivity.ra_diag.npz` retains local data-only scaled
  derivatives and exact boundary inactivity at the declared training baseline. The
  coarse Jacobians are explicitly unqualified and provide no rank or uncertainty claim.
- `full_native_target_check.ra_diag.npz` verifies the initial numerical check uses
  the exact synthetic-training observation revision and its scale, with historical
  guards excluded. Early concurrent exercise runs used full-observation initial
  checks; their recorded observation revisions expose this limitation. Fitted
  checks use their actual targets. `full_native_final_proofs.ra_diag.npz` records
  all eight registered proof results and the final inventory on the clean commit.
- `full_native_final_benchmark.ra_diag.npz` uses the same p8/g3 253,269 retained events
  and ten strength grids as the original profile. Warm baseline medians are 1.86547 s
  direct and 0.26988 s with cone reuse (6.91x for that evaluation, not the whole fitter).
  Width replacements are checked separately. Cache/direct relative differences are at
  most 4.17e-16; the original pre-optimization prediction agrees to 4.38e-16. Reused
  spatial projection is bitwise identical. Projector construction costs 0.3497 s;
  first compile is 24.3077 s including JIT, and the warm compile using a retained
  projector is 18.0815 s. Peak working set is 607,453,184 bytes with two responses
  present. Concurrent fits ran during timing; this is not a peak-memory bound for
  broader spectra or finer rules.
- `full_native_region_mixture_pilot.ra_diag.npz` records a rejected numerical experiment.
  Four and eight nodes per conservative region took 165.84 and 315.05 s for two
  predictions; disagreement was 63.56 whitened RMS. The sixteen-node attempt was
  stopped without retaining a result. Exhaustive endpoint partitioning was also
  rejected before implementation after counting 94.8 million Te / 74.5 million B4
  cells. Neither experimental sampler enters production.

## Scientific and integration state

This implements the declared fitting capability. It does **not** finish numerical
acceptance or identify all physical parameters in these acquisitions. Fresh coarse
predictions and the nominal archived quadrature have `NO_ORACLE` for convergence;
the first demonstrated difference is proposal construction, not a corrected physical
equation. Independent source/axial/angular and empirical handoff qualification remain
required. Consequently no new measured parameters, preferred N or confidence intervals
replace the nominal archive. Background-control predictions remain diagnostic because
their numerical accuracy and admissible contamination budget are unresolved.

No detector PSF, finite-footprint, strain, size ensemble or extra off-specular response
is admitted from these numerically unresolved residuals. Independent calibration
covariances are absent; no artificial prior substitutes for them. Weak/gauge directions
remain explicit rather than fixed silently.

Minimum integration: preserve the original main work and review/fast-forward this
coherent branch after the clean-main gate. Implementation, numerical qualification,
predictive validation and identification remain distinct statuses in the handoff.
