# T12: response-folded mosaic fit

Status: `READY_DETERMINISTIC_FULL_SET_SLICE`.

Branch: `codex/mosaic-response-fit`

Dependencies: T17 continuous angle measurement and T18 accepted fixed multi-OSC geometry.

## Accepted slice

Recover Gaussian sigma, Lorentzian HWHM, and mixture probability from immutable finite-bin
`(phi,2theta)` response profiles while geometry, source, material, rods, wavelength, detector center,
and lattice stay fixed. The objective integrates `S` and `N` before division and profiles one exact
nonnegative nuisance amplitude per individual profile. Absolute peak heights and cross-reflection
intensity ratios therefore do not weight the shared mosaic parameters.

The tracked proof uses one source-center, zero-divergence, mean-wavelength state at 5, 10, and 15
degrees. It jointly includes the frozen 10/8/8 indexed nonzero profiles and six raw-supported
collapsed `|00L|` profiles representable by the fixed top-exit model. The 43 independently audited
inverse-support boundary bins are excluded identically from truth and every component; every
profile remains in the objective. It fits no detector raster; the three 3,000 x 3,000 images are
external configured-truth forward visualizations. Raw-significant `003` at 10 and 15 degrees is
reported as outside the current forward channel.

## Owned paths

```text
src/rasim_next/fitting/mosaic.py
src/rasim_next/measurement/continuous_angle.py
scripts/recover_bi2se3_mosaic.py
examples/bi2se3/experiment/mosaic_fit_truth.toml
tests/test_fitting.py
tests/test_integration.py
this task and the live contract/validation documentation
```

## Public result

- material-neutral finite-profile identities and component banks;
- explicit nonzero branch and collapsed `00L` cardinality;
- deterministic width refinement with repeated-pair caching;
- exact eta faces plus a finite 8,193-point centered-logit stationary audit;
- nuisance-projected rank/condition and typed local/global identifiability failures;
- CUDA forward-profile and image evaluation with CPU-side profile search.

## Proof command

```powershell
uv run python scripts/recover_bi2se3_mosaic.py `
  --output-directory C:\path\outside\the\repository\mosaic-recovery
```

## Remaining proposed work

`tasks/mosaic_distribution_fitting_plan.md` remains `PROPOSED` for topology-split cubature that
retains the excluded boundary bins, automatic arbitrary-material boundary planning, half/quarter subsets,
leave-group-out prediction, additional truth regimes, stochastic noise/background/covariance,
uncertainty intervals, CPU/GPU crossover studies, and real-OSC profile extraction/recovery. The
masked independent-order calculation is an angle-bin convergence check on smooth retained bins,
not the broader plan's independent source/orientation Tier-B qualification.

## Handoff

The numerical recovery, distribution distances, convergence, timing, memory, external artifact
paths, and retained limitations are recorded in `docs/VALIDATION.md`; the final handoff response
records the resulting commit SHA and retained-test rationale.
