# Deterministic Ewald pushforward checklist

Status: **CORRECTED PLANNING ONLY — implementation blocked pending human re-review.**

Authoritative specification:
`tasks/deterministic_ewald_pushforward_plan.md`.

## Workbranch and proof authority

- [ ] Confirm this approved plan/checklist are committed on the intended base and record both Git
      object hashes before BKI-15 executes.
- [ ] Before BKI-15 executes, replace both obsolete coating-validation and parallel Task 1.1
      successor edges with this deterministic workbranch; no overlapping successor may start.
- [ ] Import the corrected planning commit into the BKI-approved base and recompute BKI-15's
      predeclared manifest file count, path set, hashes, and delta to include this plan/checklist.
- [ ] Confirm BKI-00 through BKI-15 completed successfully on the approved main.
- [ ] Require BKI-16/BKI-17 and every parallel writer to land before the base or remain explicitly
      paused; preserve any existing retired validation ref until its unique work has an
      owner-approved port/archive decision.
- [ ] DP-00: resolve the current dirty main checkout without overwriting user work; require
      warning-free status with every untracked path visible and no permission errors.
- [ ] Record the approved post-BKI `PUSHFORWARD_BASE_SHA` and reference hashes.
- [ ] Re-audit the exact post-BKI API/import/fixture tree; any material mismatch returns to human
      plan review before branch creation.
- [ ] Create a separate worktree on `codex/deterministic-ewald-pushforward`; do not reuse the old
      coating-validation branch.
- [ ] DP-00A: add and hash scalar/global-L1 coating, sphere/detector mass, ledger,
      discriminant/transversality, topology/chart/indicator allocation, and `L_peak` stage
      tolerances; a reported bound must itself fall below its stage tolerance. Without a full
      array oracle, use only the independently certified corresponding scalar mass as the
      candidate-independent global-L1 scale or fail the gate.
- [ ] Update `docs/VALIDATION.md` atomically in DP-00A so the prior tolerance hash is explicitly
      historical and the new hash is current before coating implementation begins.
- [ ] DP-00B: verify BKI-13 already removed the dead `integration` registration without editing it,
      and migrate mosaic/stacking proofs from retired baselines to `PUSHFORWARD_BASE_SHA`.
- [ ] Checkpoint 0: obtain human approval for the immutable tolerance artifact and prove all six
      real proof commands are green with external caches.
- [ ] DP-00C: add the sole pure 5-degree fixture authority, migrate the BKI permanent builder
      invariant away from the old image script, relabel that old builder as legacy-only, and prove
      the new authority imports without Matplotlib.

## Internal coating

- [ ] DP-01: consume the sole fixture authority and freeze the centered source, default wavelength,
      +5-degree sample pose, VESTA CIF/hash, material-optics values/provenance, pure 2-degree
      Gaussian mosaic, detector transform, explicit unbounded support, and 121/120 rod counts.
- [ ] Derive the internal `ki` through the source/intersection/entrance-refraction boundary; never
      inject it by hand.
- [ ] Add the pure continuous reciprocal rod/root kernel with exactly one Ewald coarea factor and
      classify 120 rod identities plus all 240 ordered `(rod_id,root_label)` slots.
- [ ] Prove root labels, residuals, mosaic normalization, beta periodicity, and no-root/tangent
      status.
- [ ] DP-02: use analytic/interval branch-and-bound to certify every `D>0/D=0/D<0` component,
      fold, seam, pole, and status region; unresolved topology fails rather than passing by sample
      coverage.
- [ ] Use `D=s^2` only in a certified nonsingular `(D,eta)` chart and apply
      `2*abs(s)/abs(det d(D,eta)/d(alpha,beta))` exactly once; stop a vanishing gradient or chart
      denominator as `DEGENERATE_TANGENCY`/typed unresolved failure.
- [ ] Accumulate the frozen 512x1024 `(mu,phi)` equal-solid-angle raster by exact seam-safe bin-box
      clipping or validated indicator integration, with a separate certified allocation bound.
- [ ] Prove both poles, the periodic seam, and coordinate-fixed front/back panel handedness.
- [ ] Prove sphere mass, per-rod/root/family totals, moments, topology/chart/allocation bound
      coverage, a missed-node bin crossing/support island mutation, and refinement convergence.
- [ ] Freeze a canonical detector-independent sphere partition/result; detector refinements cannot
      feed back into it.
- [ ] Checkpoint A: review the internal coating numeric summary.

## Exit and detector mapping

- [ ] DP-03: prove surface-tangential conservation and the accepted propagation-direction/
      non-growing exit rule at `z_sample=0`; do not invent a blanket `kf_z > 0` rule.
- [ ] Prove `kf_air` in SAMPLE, its one-time SAMPLE-to-LAB transform, and the actual surface origin.
- [ ] Prove detector ray/pixel round trips and rigid-rotation covariance.
- [ ] DP-04: conservatively push mapped simple cells into exact detector pixel boxes.
- [ ] Freeze pixel `(r,c)` ownership as `[c-.5,c+.5) x [r-.5,r+.5)` with explicit final outer-edge
      closure; certify every fold/status/support edge, reject self-intersections, and retain every
      preimage.
- [ ] Split exit-critical, detector parallel/backward, support, and other status boundaries, or use
      validated interval indicator bounds; never hide unresolved/numeric-failure mass in a residual.
- [ ] Prove separately: sphere closure, exhaustive pre-optics/post-optics `ValidityCode` ledgers,
      certified numerical bounds below frozen tolerances, row/column/edge ownership, detector-change
      invariance of the sphere result, and no solid-angle/plane-Jacobian multiplier.

## End-to-end path and peak-mosaic component tags

- [ ] DP-05: compose the temporary deterministic path beside the old runtime; production consumes
      the fixture's explicit one-row `IncidentSampleBatch` and never samples a source itself.
- [ ] Move accepted orchestration from private proof helpers into the sole pipeline owner; reject
      nonhexagonal catalogs, nonzero Lorentzian mixtures, and active zero-width atoms before work.
- [ ] Add the final image-script path first as a Matplotlib-free numeric-only CLI; its JSON reports
      exact config/array hashes, `perf_counter` wall time, and Windows process peak RSS from
      `GetProcessMemoryInfo`, with no repository-local output.
- [ ] Classify all 120 nonzero-`m` rods and all 240 ordered root slots; integrate every supported
      regular component.
- [ ] Compare factor ledgers through the declared first divergence and run the independent latent
      oracle.
- [ ] DP-06: add the immutable zero-mass peak-mosaic component-tag cache.
- [ ] For every nonempty `(incident_state_id,family_id,intersection_branch_id)` support closure,
      emit exactly one internal row holding `(L_peak,m,intersection_branch_id)`.
- [ ] Globally certify the pure-Gaussian maximum on every closed support, including all interior,
      seam, and `D=0` boundary candidates; compare log density and use the total
      `(-density,alpha,abs(beta),beta,rod_id)` order, never local/adaptive discovery order.
- [ ] Keep every same-`m` rod/orientation in physical mass and ignore all off-peak mosaic `L` values
      in metadata; never round `L` or search for integer peaks.
- [ ] Retain invalid exit/detector rows with explicit status and absent optional projected fields;
      detector pose must not change the internal representative.
- [ ] Freeze canonical component IDs/row order and enumerate every internal/projection cache-key
      field; mutate each dependency and prove exact invalidation behavior.
- [ ] For branch 0, remove the direct `u=0` root, retain only the non-direct metadata root, evaluate
      no physical mass, allow its closed-support coalescence only as `TANGENT_BOUNDARY`, and mark
      `M0_INTENSITY_EXCLUDED`.
- [ ] Prove tags cannot change either numeric image.
- [ ] Before Checkpoint B, prove the complete 120-rod/240-root-slot case meets `<=15 min` and
      `<=1.5 GiB` using the exact twice-run `--numeric-only --json` command; failure stops before
      deletion.
- [ ] Checkpoint B: human review authorizes old-runtime deletion only after correctness and resource
      gates pass.

## Delete the old framework

- [ ] DP-06A: synchronize architecture/contracts/dovetail/result/trace authorities before cutover;
      mark the old runtime as temporary comparison code only.
- [ ] DP-06B: synchronize scope/physics/validation/error-injection/decision authorities, including
      tangent-chart, topology-certificate, certified-bound, and `m=0` rules before cutover.
- [ ] DP-07A: make deterministic `simulate_ordered` the sole path; delete
      `pipeline/intersections.py`, `pipeline/selection.py`, and the old image script after moving
      only the needed rod bound. DP-00C must already have migrated the script’s permanent caller.
- [ ] Remove `selection_seed`, `draw_count`, candidate pools, selected-event compaction, and all
      outgoing-event RNG selection.
- [ ] DP-07B: migrate reciprocal proof/tests and then delete `reciprocal/events.py`.
- [ ] DP-08: delete `ScatteringEventBatch` and event transport, repair `geometry/__init__.py`,
      `proof/core.py`, and core contract tests atomically in exactly five files.
- [ ] DP-09: delete fixed mosaic-orientation construction, `render/deposition.py`, and obsolete
      mosaic/Monte Carlo/bilinear implementation-detail tests.

## Two required images

- [ ] DP-10: add a locked optional `image` dependency group and extend the DP-05 numeric-only
      `generate_bi2se3_pushforward_images.py`; the script consumes DP-00C’s sole fixture builder
      and forces the headless `Agg` backend before importing `pyplot`.
- [ ] Generate external `bi2se3_5deg_ewald_coating.png` from the deterministic sphere bins.
- [ ] Generate external `bi2se3_5deg_detector_pushforward.png` from correctly refracted outgoing
      rays in LAB.
- [ ] Overlay zero-mass peak-mosaic `(L_peak,m,intersection_branch_id)` tags; detector PNG overlays
      only valid projections and marks branch-0 intensity excluded.
- [ ] Print configuration, hashes, numeric-array hashes, convergence, and separate mass ledgers as
      JSON; write no numeric sidecar by default.
- [ ] Render sphere bins as one flat value per exact numeric bin on a fixed `2400 x 1200` canvas;
      deterministically tessellate curved orthographic edges to `<=0.1` output-pixel chord error,
      with no value resampling/smoothing, deterministic seam/pole ownership, explicit zero
      handling, and one shared positive absolute log scale. Use the matching zero/log rule for the
      native detector image.
- [ ] Confirm exactly two PNGs are external and no image/numeric output exists in the repository.
- [ ] Keep Matplotlib out of core/dev and permanent-test imports; run the exact frozen
      `--group image` command with external `MPLCONFIGDIR`/output and record its locked version.
- [ ] Checkpoint C: full suite, six proofs, replacement script, and legacy-symbol scans are clean.

## Live documentation and task retirement

- [ ] DP-14: first reconcile the pure coating strategy, parallel/mosaic-fitting plans, runbook, and
      this checklist; remove every inbound link to the files DP-13 will delete and run docs checks.
- [ ] DP-13: only after DP-14 passes, retire T03/T07/coating task instructions and update the task
      index/review; rerun link and index checks immediately after deletion.
- [ ] DP-13A: retire remaining integration/mosaic prompts, task README, and residual BKI checklist
      language if still live after its required pre-BKI successor correction; run inbound-link,
      docs, index, and live-instruction checks immediately around every conditional deletion.
- [ ] DP-13B: update the stale bootstrap dataflow, import every package initializer, and run exact
      production plus reviewed historical-text scans including event transport symbols.
- [ ] Confirm no active instruction can recreate candidates, CDF sampling, `T/N`, sampled outgoing
      events, or bilinear points; retain upstream source characterization.

## Final gate

- [ ] DP-15: compile with external bytecode cache, lint, format-check, run the compact suite, and
      run all six valid proof commands.
- [ ] Detect every required physics/frame/coarea/Jacobian/identity/tag mutation at its owning stage.
- [ ] Meet scalar/global-L1 mass, certified-bound coverage, image-convergence, coordinate, runtime,
      and peak-memory thresholds with zero accepted unresolved-topology, degenerate-tangency, or
      numeric-failure mass.
- [ ] Audit every retained test against a unique long-term invariant.
- [ ] Remove temporary comparisons, grids, harnesses, diagnostics, profiles, and generated files.
- [ ] After final validation and file-set changes, regenerate/review `FILE_MANIFEST.json` and pass
      `verify_seed.py`.
- [ ] Run working and cached diff checks before the coherent commit; afterward run the exact
      quoted PowerShell command `git diff --check "${env:PUSHFORWARD_BASE_SHA}..HEAD"` and require
      warning-free empty status.
- [ ] Show both external PNGs to the user.
- [ ] Squash temporary task commits into one coherent commit and confirm a clean worktree.
