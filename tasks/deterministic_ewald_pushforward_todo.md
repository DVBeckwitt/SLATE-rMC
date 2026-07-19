# Deterministic Ewald pushforward checklist

Status: **PLANNING ONLY — implementation blocked pending approval.**

Authoritative specification:
`tasks/deterministic_ewald_pushforward_plan.md`.

## Workbranch and proof authority

- [ ] Before BKI-15 executes, replace its obsolete continuous-coating-validation successor edge
      with this deterministic workbranch; do not create the retired branch.
- [ ] Confirm BKI-00 through BKI-15 completed successfully on the approved main.
- [ ] DP-00: resolve the current dirty main checkout without overwriting user work.
- [ ] Record the approved post-BKI `PUSHFORWARD_BASE_SHA` and reference hashes.
- [ ] Create a separate worktree on `codex/deterministic-ewald-pushforward`; do not reuse the old
      coating-validation branch.
- [ ] DP-00A: add and hash scalar/global-L1 coating, sphere-mass, detector-mass, ledger,
      discriminant/transversality, and `L_peak` stage tolerances before adding scientific code.
- [ ] DP-00B: remove the dead `integration` proof registration and migrate both mosaic and stacking
      proofs from retired baselines to the declared `PUSHFORWARD_BASE_SHA` contract.
- [ ] Checkpoint 0: obtain human approval for the immutable tolerance artifact and prove all six
      real proof commands are green with external caches.

## Internal coating

- [ ] DP-01: freeze the centered source, default wavelength, +5-degree sample pose, VESTA CIF/hash,
      material-optics values/provenance, 2-degree Gaussian mosaic, detector transform, finite
      sample support, and 121/120 rod counts.
- [ ] Derive the internal `ki` through the source/intersection/entrance-refraction boundary; never
      inject it by hand.
- [ ] Add the pure continuous reciprocal rod/root kernel with exactly one Ewald coarea factor.
- [ ] Prove root labels, residuals, mosaic normalization, beta periodicity, and no-root/tangent
      status.
- [ ] DP-02: locate `D=0`; use `D=s^2` only after the frozen transversality proof; stop a vanishing
      gradient as `DEGENERATE_TANGENCY`; split folds/seams and add bounded adaptive latent cells.
- [ ] Accumulate the frozen 512x1024 `(mu,phi)` equal-solid-angle raster in `angstrom^2/sr`.
- [ ] Prove both poles, the periodic seam, and coordinate-fixed front/back panel handedness.
- [ ] Prove sphere mass, per-rod/root/family totals, moments, estimator coverage, and refinement
      convergence.
- [ ] Checkpoint A: review the internal coating numeric summary.

## Exit and detector mapping

- [ ] DP-03: prove surface-tangential conservation and the accepted propagation-direction/
      non-growing exit rule at `z_sample=0`; do not invent a blanket `kf_z > 0` rule.
- [ ] Prove `kf_air` in SAMPLE, its one-time SAMPLE-to-LAB transform, and the actual surface origin.
- [ ] Prove detector ray/pixel round trips and rigid-rotation covariance.
- [ ] DP-04: conservatively push mapped simple cells into exact detector pixel boxes.
- [ ] Split zero-determinant folds, reject self-intersecting cells, bound curvature/density/pixel
      allocation error, and keep every preimage.
- [ ] Split exit-critical, detector parallel/backward, support, and other status boundaries, or use
      independently bounded indicator quadrature; never hide numeric-failure mass in a residual.
- [ ] Prove separately: sphere closure, exhaustive pre-optics/post-optics `ValidityCode` ledgers,
      independent numerical-error coverage, row/column orientation, and no solid-angle/Jacobian
      multiplier.

## End-to-end path and peak-mosaic component tags

- [ ] DP-05: compose the temporary deterministic path beside the old runtime; production consumes
      the fixture's explicit one-row `IncidentSampleBatch` and never samples a source itself.
- [ ] Move accepted orchestration from private proof helpers into the sole pipeline owner; reject
      nonhexagonal catalogs and active zero-width mosaic atoms before work.
- [ ] Classify all 120 nonzero-`m` rod/root attempts and integrate every supported regular
      component.
- [ ] Compare factor ledgers through the declared first divergence and run the independent latent
      oracle.
- [ ] DP-06: add the immutable zero-mass peak-mosaic component-tag cache.
- [ ] For every nonempty `(incident_state_id,family_id,intersection_branch_id)` support closure,
      emit exactly one internal row holding `(L_peak,m,intersection_branch_id)`.
- [ ] Maximize only `p_alpha/(2*pi)` on the closed support; include `D=0` tangent-boundary winners
      and use the analytic periodic-beta/rod tie order, never adaptive-node order.
- [ ] Keep every same-`m` rod/orientation in physical mass and ignore all off-peak mosaic `L` values
      in metadata; never round `L` or search for integer peaks.
- [ ] Retain invalid exit/detector rows with explicit status and absent optional projected fields;
      detector pose must not change the internal representative.
- [ ] For branch 0, remove the direct `u=0` root, retain only the non-direct metadata root, evaluate
      no physical mass, allow its closed-support coalescence only as `TANGENT_BOUNDARY`, and mark
      `M0_INTENSITY_EXCLUDED`.
- [ ] Prove tags cannot change either numeric image.
- [ ] Checkpoint B: human review authorizes old-runtime deletion.

## Delete the old framework

- [ ] DP-07A: make deterministic `simulate_ordered` the sole path; delete
      `pipeline/intersections.py`, `pipeline/selection.py`, and the old image script after moving
      only the needed rod bound.
- [ ] Remove `selection_seed`, `draw_count`, candidate pools, selected-event compaction, and all
      outgoing-event RNG selection.
- [ ] DP-07B: migrate reciprocal proof/tests and then delete `reciprocal/events.py`.
- [ ] DP-08: delete `ScatteringEventBatch` and repair `proof/core.py` plus core contract tests.
- [ ] DP-09: delete fixed mosaic-orientation construction, `render/deposition.py`, and obsolete
      mosaic/Monte Carlo/bilinear implementation-detail tests.

## Two required images

- [ ] DP-10: add `generate_bi2se3_pushforward_images.py` with one pure frozen-fixture builder.
- [ ] Generate external `bi2se3_5deg_ewald_coating.png` from the deterministic sphere bins.
- [ ] Generate external `bi2se3_5deg_detector_pushforward.png` from correctly refracted outgoing
      rays in LAB.
- [ ] Overlay zero-mass peak-mosaic `(L_peak,m,intersection_branch_id)` tags; detector PNG overlays
      only valid projections and marks branch-0 intensity excluded.
- [ ] Print configuration, hashes, numeric-array hashes, convergence, and separate mass ledgers as
      JSON; write no numeric sidecar by default.
- [ ] Confirm exactly two PNGs are external and no image/numeric output exists in the repository.
- [ ] Keep Matplotlib out of permanent tests/core dependencies; exercise rendering manually in the
      handoff environment with external `MPLCONFIGDIR`.
- [ ] Checkpoint C: full suite, six proofs, replacement script, and legacy-symbol scans are clean.

## Live documentation and task retirement

- [ ] DP-11: update architecture, contracts, dovetail, result measure, and trace schema.
- [ ] DP-12: update scope, physics ledger, validation, error injections, and decision ledger.
- [ ] DP-13: retire T03/T07/coating task instructions and update the task index/review.
- [ ] DP-13A: retire remaining integration/mosaic prompts, task README, and residual BKI checklist
      language if still live after its required pre-BKI successor correction.
- [ ] DP-14: reconcile the pure coating strategy, parallel plan, mosaic fitting plan, and runbook.
- [ ] Confirm no active instruction can recreate candidates, CDF sampling, `T/N`, sampled outgoing
      events, or bilinear points; retain upstream source characterization.

## Final gate

- [ ] DP-15: compile with external bytecode cache, lint, format-check, run the compact suite, and
      run all six valid proof commands.
- [ ] Detect every required physics/frame/coarea/Jacobian/identity/tag mutation at its owning stage.
- [ ] Meet scalar/global-L1 mass, estimator-coverage, image-convergence, coordinate, runtime, and
      peak-memory thresholds with zero accepted degenerate-tangency/numeric-failure mass.
- [ ] Audit every retained test against a unique long-term invariant.
- [ ] Remove temporary comparisons, grids, harnesses, diagnostics, profiles, and generated files.
- [ ] Show both external PNGs to the user.
- [ ] Squash temporary task commits into one coherent commit and confirm a clean worktree.
