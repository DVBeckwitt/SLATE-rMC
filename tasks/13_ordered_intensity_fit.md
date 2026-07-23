# T13: ordered intensity fit

Status: `COMPLETE_FIXED_POSITION_SYNTHETIC_SLICE`.

This is the completed contract-v10 fixed-position slice. References to historical event geometry,
hits, or detector responses do not authorize restoring deleted APIs.

Branch: `codex/structure-factor-intensity-fit`

## Goal

Fit relative ordered Bragg intensities from immutable detector-native ROI selections while reusing
the continuous Bragg and detector mappings.

## Owned paths

```text
src/rasim_next/fitting/ordered_intensity.py
tests/test_fitting.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/gui/ordered_structure_fit.py:53-99,322-540
    ra_sim/fitting/rod_profiles.py

manuscript
    sections/refinement_workflow.tex:39-43,57
    2D_Supplemental/SI_failure_modes.tex:691-703
```

## Required work

- freeze source, detector, sample, mosaic, selection, ROI, mask, and background-policy revisions
- record every individual rod contributing to each ROI
- compare measured and simulated detector mass under one declared noise model
- support exact nonnegative per-image scale where applicable
- keep structural parameters global and nuisance scales/backgrounds separate
- reuse analytic root geometry and continuous detector-coordinate response
- fit raw amplitudes/relative intensities without maximum normalization, pruning, or independent peak amplitudes
- expose held-out reflection validation and parameter identifiability

## Proof

- synthetic structural recovery
- ROI mass conservation
- exact scale solution
- held-out exact-model interpolation checks (not independent experimental validation)
- parameter rank/correlation evidence
- original-RASIM raw-intensity comparison before normalization and rounding

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/fitting/ordered_intensity.py tests/test_fitting.py
pytest -q tests/test_fitting.py
python -m rasim_next.proof ordered-intensity-fit --json
git diff --check
```

## Execution plan

State: COMPLETE

This branch implements the first contract-v10 ordered-intensity slice for the tracked Bi2Se3
three-incidence case. It does not restore the retired event/hit runtime.

1. Extend the existing CIF-derived quintuple-layer strength with one occupancy per unique source
   label and one shared transverse-isotropic displacement tensor `U_radial/U_normal`. Keep the two
   symmetry-allowed 6c fractional-z coordinates available to the explicit freeze mask, but freeze
   both at their CIF values for this first user-requested recovery. The directional Debye-Waller
   amplitude replaces the current shared isotropic factor and reproduces it exactly when both
   components are `0.019 A^2`.
2. Keep source, nine-coordinate geometry, lattice, material optics, layer count, stacking law, and
   recovered mosaic immutable. Prepare detector-native finite-angle profile masses once and cache
   their fixed response relative to the exact ordered strength; no detector raster is evaluated in
   an optimizer iteration.
3. Fit every geometry-visible positive nonzero integer-L marker in the 5, 10, and 15 degree
   simulations, including weak `m=1,3,4` groups, plus the six admitted `00L` profiles. Use one
   analytically profiled scale per image and equal relative peak leverage. Never fit per-peak
   amplitudes.
4. Record both Wyckoff coordinates explicitly as frozen state and reject attempts to activate them
   in this first phase. Permit any subset of the three occupancies and two displacement factors.
   With relative image scales, reject the exact common-occupancy gauge unless one occupancy is
   fixed; report occupancy ratios in that case. Also prove the absolute-calibration mode in which
   all three occupancies are recoverable.
5. Generate hidden truth with the authoritative full structure-factor and 52-layer stacking model
   on an independently refined frozen detector response, then fit it with the cached occupancy quadratic and
   directional damping. Recover from a deterministic non-truth start, verify the acceleration
   kernel against the direct oracle, report rank, singular values, correlations, held-out
   exact-model interpolation points, wall time, and peak memory, then retain only compact invariant and integration
   tests. Bind both numerical responses to one quadrature-independent observable-layout revision
   and require planted-mass plus six-column occupancy-basis convergence before acceptance.

## Handoff

Status: COMPLETE — fixed-position deterministic synthetic slice

Commit SHA: recorded in the final branch handoff (a commit cannot contain its own SHA)

Accepted ordered-model revision: `bi2se3_fixed_position_occ_directional_u.v1`

Proof summary: one simultaneous 5/10/15-degree fit retains 238 profiles (`88/78/72`), including
all six admitted `m=0` profiles and all frozen weak nonzero identities. The q12x4 cached response
agrees with an independently refined q16x8 full-strength response to `2.1565e-4` maximum relative
planted-mass error and `2.3373e-4` maximum six-column basis row-norm error. Cached and direct
strength agree to `2.22e-15`. Absolute recovery is rank 5 with condition `9.475`; relative ratio
recovery is rank 4 with condition `4.383`; neither contacts a bound. The exact common-occupancy
scale gauge is rejected.

Benchmark: q12x4 fit-response compilation `351.889 s`; q16x8 refined-proof compilation
`891.714 s`; cached versus direct equivalent three-angle prediction `0.0468517/55.0579 s`
(`1175.15x`); absolute/relative fits `2.6785/2.1458 s`; total proof `1506.118 s`. Traced/observed
peak memory `1,305,641,960/1,497,780,224` bytes. Response term counts are
`850,944/773,568/682,752`; refined-proof counts are `2,269,184/2,062,848/1,820,672`.

Held-out exact-model interpolation errors: maximum `1.3003e-5` at four predeclared noninteger-L
points that do not occur as exact quadrature terms; this is not independent experimental
validation.

Identifiability: absolute calibrated masses identify the three occupancies and `Ur/Uz`. Free image
scales remove the common occupancy multiplier, so Bi is fixed as a positive ratio reference and
only `Se1/Bi`, `Se2/Bi`, `Ur`, and `Uz` are identified. Correlations are local inverse-sensitivity
diagnostics, not statistical covariance.

Legacy classification: `CORRECTED` for raw, unrounded CIF-derived strength with explicit occupancy
and directional damping; `NO_ORACLE` for raw-OSC component extraction, background/noise/PSF, and
real-data parameter recovery.

Permanent tests retained: direct directional structure sum; fixed-position quadratic versus full
strength; PSD behavior at a numerical extinction; sparse `m=0` plus nonzero response/topology/direct
oracle; geometry-only marker identity under zero versus CIF strength; freeze/gauge/revision/rank/
bound contracts; and asymmetric scalar/CPU/CUDA parity. Each protects a distinct public or
scientific invariant.

Limitations: Bi2Se3-specific five-parameter basis and finite topology probe; selected-group
component masses rather than raw total ROI counts; no atom motion, per-site `Uij`, background,
noise model, uncertainty, or real-OSC deblending. New materials require their own structure basis,
identity catalog, support audit, and response-order proof.

Minimum integration request: consume immutable `(dataset_id, observable_revision, mass)` records
from a separately proven component-extraction boundary; do not join by tuple order or numerical
response digest.
