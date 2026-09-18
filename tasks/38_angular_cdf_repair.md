# Bounded angular inverse-CDF repair

## Authorized plan (2026-09-18)

The user authorizes the targeted repair of demonstrated angular inverse-CDF
exhaustion. Fits and scheduled check-ins remain stopped. The isolated branch
starts from clean main `2b80a2c3` and carries the audited execution prerequisite
`68b29da7`; main and the failed-run checkout remain unchanged.

1. Reproduce the saved GD1 failure with a compact public sampler regression
   against analytic arc mass and sine moment, before changing production code.
2. Preserve the first 70 Newton evaluations, proposal measure, support and
   stopping tolerances. Add bounded bisection fallback and reject unresolved or
   nonfinite inversions; successful coordinates and densities must be paired.
3. Check the saved actual angular nodes against an independent wrapped-Cauchy
   CDF/PDF and bracketed inverse; detect restored exhaustion/stale-density errors.
4. Test detector impact on the actual failing axial row, one original source,
   all six signed family rods and all observations, with a direct physical-angle
   integration reference. Two 120-second blocks maximum; no full fit campaign.
5. Measure ordinary/fallback costs, run compact software and registered proofs,
   obtain independent review, and commit one coherent repair. Retain only the
   unique public regression; external evidence is one `.ra_diag.npz`.

Owned paths: `src/rasim_next/pipeline/fiber_detector.py`,
`tests/test_integration.py`, this task, `tasks/index.yaml` and the tracked-file
inventory. Relevant ledger: PHY-FIT-025. No API, dependency, physical-factor,
normalization, support, source or observation change. Main agent alone writes.

Classification: CORRECTED at angular inverse-CDF solution/returned importance
density after identical proposal inputs and CDF. This does not certify complete
detector integration, optimizer convergence, or any previously failed fit.

## Repair and focused proof

The loop retains its first 70 Newton evaluations and permits at most 80 more
evaluations with forced bisection. Every acceptance checks a finite CDF error
and finite positive PDF first. Exhaustion raises instead of returning an
untested coordinate with the preceding iterate's density. No API changes.
An arc spans at most `2*pi`; 49 bisections suffice for the existing `2e-14`
bracket-width tolerance, so the fallback has explicit margin.

The new public 256-node regression fails before the repair: integrated arc mass
is `0.006107777167942617` versus analytic `0.005979459368185669` (2.146% excess).
After repair, relative mass error is `9.55e-15` and the sine moment differs by
`1.50e-18`. Restored exhausted weights, including stale densities paired with
corrected angles, fail the mass invariant. This is the sole new permanent test;
it protects a distinct multimodal angular-inversion failure via public outputs.
The four focused sampler cases pass in 5.95 seconds.

All 1,394,432 saved actual GD1 source-8/family-1 angular nodes were checked.
Exactly 1,031 previously erroneous coordinates change; the other 1,393,401 are
bitwise unchanged. Independent periodic SciPy wrapped-Cauchy CDF and analytic
tanh PDF give maximum errors `3.55e-15` and `1.98e-15` relative, respectively.
All arc fractions remain bitwise unchanged. Seventeen independent Brent roots
agree within `2.78e-15` rad. No complete detector prediction was recomputed.

## Bounded detector impact and limitations

The actual failing axial row 3794 (`3.1781816914396286` inverse angstroms) was
evaluated with the original source 8, all six signed family-1 rods, N72, original
candidate, full angular union and all 1,226 observation outputs. The independent
angular integration route uses physical GL8 azimuth panels, bypassing the CDF
transform but sharing the implemented physics and spatial deposition.

Define local discrepancy as `max(abs(value-reference))/max(abs(reference))`.
It is not an elementwise relative error or covariance-whitened fit gate.
Direct panel widths `.005` and `.0025` rad differ by `5.30e-8`. Against the finer
reference, old p7 differs by `0.12869034`, repaired p7 by `0.000940543`, and
repaired p8 by `0.000239990`. The repair therefore removes the demonstrated
dominant local error, while finite angular resolution remains visible. These
are one axial-row contributions, not complete source/family predictions.

Two supervised proof children finish and are reaped in 31.79 and 26.20 seconds
total wall time, below their separate 115-second caps. Mapping/detector peak
working sets are 445,423,616 / 597,106,688 bytes. Seven alternating warmed timing
pairs on 4,096 ordinary nodes give old/repaired medians 11.119 / 11.332 ms with
bitwise-identical outputs. Fallback-case medians are 18.928 / 20.896 ms, but the
old result is wrong and is not equivalent scientific work. No speedup claimed.

External evidence: `angular_cdf_repair_20260918.ra_diag.npz`, alongside the
unchanged `gd1_bounded_root_cause_20260918.ra_diag.npz`. The repair artifact
retains source/input hashes, proof source, numerical arrays, timings, memory,
shutdown and review evidence. Independent read-only audit reproduces every
reported array metric and verifies all 123 source hashes and input hashes.

The numerical fix does not establish pointwise physics correctness, explain the
fraction of full GD1/SID1 disagreements, certify all integration controls, or
accept an optimizer endpoint. Fits and check-ins remain stopped. Future work
must separately qualify complete predictions under the repaired code; old
failed artifacts and controller plans are not silently relabeled or resumed.

## Software handoff

All 372 compact-suite cases pass across two runs: the initial run passes 349 in
414.95 seconds, with 23 cases blocked solely by Windows sandbox temporary-folder
or named-pipe permissions; only those 23 are retried with normal process access
and pass in 263.37 seconds. The first run emits 14 GPU-underutilization warnings.
Ruff and formatting pass for all 166 Python files; no type checker is configured.
Core, immutable references, ordered-intensity-fit, ordered-reflectivity,
PbI2-polytype-Bragg and stacking-transition registered proofs pass. The two
clean-commit-required geometry-optics/mosaic-ewald proofs and final inventory
verification are recorded post-commit in the same external evidence artifact.

Independent code review finds no blocking issues. Five empty test-created
directories are removed after the failed sandbox run; no scientific evidence
is removed. The temporary external proof harness is embedded in its NPZ and
removed as a loose file. Integration requires this one repair commit on the
audited `68b29da7` prerequisite, not a restart of any old queue or executor.
