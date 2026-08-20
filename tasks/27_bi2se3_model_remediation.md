# T27 — Bi2Se3 model remediation and bounded re-evaluation

Status: CORRECTED_5_TO_25_SCAN_CONVERGENCE_REJECTED_NO_PARAMETER_UPDATE
Branch: `codex/bi2se3-staged-scan-fit`

## Objective

Correct the smallest independently supported omissions in the Bi2Se3 detector model, then
re-evaluate the three fixed-incidence OSCs and the physical 5--25 degree integrated scan without
turning uncalibrated instrument effects into fit parameters.

## Accepted implementation scope

1. Preserve the existing complex-kz film transmission and uniform-depth attenuation exactly.
2. Add the incidence illuminated-path factor as a separate, once-only measurement weight; finite
   sample clipping remains the independently sampled footprint acceptance.
3. Add an explicit weighted discrete source-line model while preserving the legacy Gaussian null
   path. Use it to test Cu K-alpha1-only and the published K-alpha1/K-alpha2 doublet; do not fit the
   line ratio from Bi2Se3.
4. Add optional wavelength-resolved external detector-path attenuation. Standard dry air is a
   sensitivity case until the sample-to-detector medium is confirmed.
5. Extend the existing exact RichEpsilon transition law to the native 3R parent in the optimized
   CPU and CUDA detector kernels. Preserve the exact epsilon-zero fast path and prove the nonzero
   path against direct finite-stack enumeration.
6. Correct the Figure-7 outer-site coordinate from Bi substitution to chalcogen vacancy. Keep Bi
   substitution as a discrete competing hypothesis, not a simultaneous free coordinate.
7. Re-evaluate the stale-dark assumption and nested ROI aperture growth. Do not clip signed
   contrasts.

## Evidence-gated omissions

- Keep the detector PSF as a delta response. The direct-beam distance series measures the combined
  beam/source/divergence/plate response and cannot identify a detector-only kernel without double
  counting.
- Keep uniform mosaic azimuth and the current axisymmetric tilt law. The apparent sixfold signal is
  dominated by one nonrepeatable band and reverses in the integrated scan.
- Keep unpolarized Thomson scattering, unity flat field, and linear detector response until an
  independent calibration exists.
- Do not add detector solid angle or a textbook Lorentz factor; the detector pushforward already
  owns that Jacobian.

## Compute-bounded fit order

1. Run analytic/invariant and short-stack proofs.
2. Freeze corrected source/measurement hypotheses and regenerate only invalidated immutable state.
3. Screen pure/vacancy/Bi-substitution and a bounded one-dimensional 3R-epsilon grid with one
   profiled scale per OSC.
4. Jointly refine only a surviving interior model, with a hard cap on completed detector calls.
5. Re-evaluate the physical 5--25 degree scan with uniform normalized motor dwell, deterministic
   panelwise angular quadrature, one scan scale, and the corrected incidence weights. The fixed
   three images remain the inner fit and exact scan evaluation remains an outer gate.

## Acceptance gates

- Direct-enumeration, scalar, optimized CPU, and CUDA paths agree within the frozen proof tolerance.
- Source masses sum to one and the doublet line masses match the declared probabilities within the
  finite stratification bound.
- Air attenuation is exactly unity at zero coefficient and is applied once.
- The retained fit is full-rank, condition number at most `1e5`, numerically converged, and has no
  artificial active bound. A nested epsilon result at exactly zero selects the fault-free model.
- No one reflection group supplies more than half of the score improvement; all three fixed images
  are noninferior under the frozen working covariance.
- The final candidate passes the existing per-family cubature gate and a higher-order replay.
- Scan claims remain model-limited until angle/flux telemetry and absolute source/draw convergence
  are available.

## Diagnostics and residue

Each proof run may retain at most one external `.ra_diag.npz` with numeric arrays and one embedded
JSON manifest. No generated data, raster stack, scratch runner, or exploratory test belongs in the
repository.

## Result

The corrected weighted source, incident illuminated-path factor, wavelength-resolved external-path
operator, vacancy coordinate, scale-zero dark contract, and nonzero-epsilon 3R recurrence are
implemented and covered by compact permanent proofs. Production keeps the external medium at
unity because the sample-to-detector gas is unverified; dry air remains a named sensitivity.

The bounded fixed-image screen did not produce an admissible candidate. Under one scale per OSC,
the exact fixed-optics vacancy profile decreases from chi-square `10973.4668` at `v=0` to
`10204.2988` at the 3% diagnostic cap, but the 10-degree image worsens, `m=3` worsens, and most of
the joint gain is coupled to `m=0`. Off-specular-trained scales predict the held-out `m=0` rows at
WRMS `96.35`. Bi substitution is worse, and the apparent `U_radial` gain nearly vanishes when
family scales are allowed. The 3% endpoint is therefore a censored model-compensation direction,
not a fitted value.

The superseded pre-acquisition-correction 5--20-degree post-hoc scan retained the full angle-node
response so exposure laws could
be replayed without detector reruns. Candidate C2 improves the full uniform-exposure chi-square by
`85.860`, but only `0.857` remains after removing the suspect mirror pair and `0.070` after removing
its detector-row band; uniform exposure wins every grouped held-out comparison. Absolute scan
prediction fails the angular-refinement and independent-seed gates (`0.928` and `1.710` working
sigma). It has no angle/flux telemetry, source/draw refinement, or deterministic region oracle and
was not fed back into the fit.

No screened state passed the no-bound, all-image, reflection-diversity, and numerical-qualification
gates. The nonlinear joint optimizer was therefore intentionally not run and the canonical fit was
not replaced. This fixed-image phase ended with the intermediate status
`MODEL_REMEDIATION_COMPLETE_NO_ACCEPTED_PARAMETER_UPDATE`; the task-level final status is the
corrected 5--25-degree result recorded below.

The consolidated external diagnostic is
`bi2se3_fit_remediation.ra_diag.npz`, SHA-256
`2429fefad8edc7d9412405c74db4de395b0189348185124a509a6fbe6fed051b`; its embedded manifest and
all numeric array hashes were independently verified.

## Corrected 5--25-degree staged-scan result

The acquisition contract now uses physical support 5--25 degrees with uniform normalized dwell.
The legacy `ScanImageStep=0.1 degree`, traversal/cycle counts, measured endpoint excess, and
forward/reverse speed difference are provenance only and do not enter the numerical nodes,
weights, cache identity, or prediction. Direction reversal canonicalizes to the same observable;
the fixed-angle engine remains the sole owner of common incidence calibration and illuminated-path
weighting.

The newly regenerated fixed-three fit reached
`(-0.000127041382, 0.005346965016, 0.030000000000, 0.020000000000,
5.99e-18)` with full data chi-square `13928.158584`. Vacancy, radial-envelope, and normal-envelope
coordinates are on compensating bounds. One corrected exact scan baseline was then evaluated with
448 commanded-angle node evaluations (64 initial plus 384 refinement evaluations), yielding a
retained 256-node fine rule. Its profiled scan scale is
`9.58607775467e8 count/A2`, scan chi-square is `1179.662263`, and the provisional four-dataset
joint chi-square is `15107.820848`.

The absolute angular convergence metric is `11.426760` working sigma and the independent spatial
finite-region cubature metric is `3.092562`, both failing the strict `<0.25` gate. The 20--25-degree
physical panel contributes signed predicted contrast count `152127.073561` across the 18 retained
scan rows. No scan-informed candidate was constructed or evaluated: exact scan calls `1`, outer
iterations attempted/accepted `0/0`, accepted update `false`. The exact stop reason is
`baseline_absolute_scan_convergence_failed`.

The compact external diagnostic is `bi2se3_staged_scan_fit.ra_diag.npz`, SHA-256
`f89be0fae0ca78ceeba35c50689e2ebdbf6016883ee3cf263eb887bee6f214b6`. No detector figure was
generated because the corrected baseline failed convergence and therefore is not a qualified
visualization model.

That diagnostic over-evaluated its secondary spatial comparison at every angle. The retained fine
region prediction and angular failure are unchanged. The retained generic oracle accepts a
caller-owned finite-region node evaluator, freezes its source/evaluation revisions, and recursively
refines only failing or event-bearing leaves. It neither compiles nor caches a detector-region
operator and does not own secondary spatial cubature. No material-specific scan runner, detector
image, additional parameter, or scan candidate is retained.

Ledger coverage: `PHY-MEA-014`, `PHY-FIT-020`, `PHY-FIT-021`.
