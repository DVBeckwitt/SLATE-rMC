# PbI2 optional polytype-landmark mosaic fit

Status: READY_SYNTHETIC_PROFILE_RESPONSE_NO_PBI2_DETECTOR_FORWARD
Branch: `codex/pbi2-rational-mosaic`

## Objective

Connect the frozen exact rational-layer landmarks from T22 to the accepted finite-bin mosaic
profile fitter and prove that qualified 4H half-order and 6H third-order peaks can constrain one
shared mosaic distribution. Optional peaks that are missing or off-panel contribute no profile.
Exact parent/rod overlaps remain one physical profile rather than duplicated residuals.

This task is an internal synthetic response-bank proof. It does not implement a detector-native
PbI2 stacking-strength adapter or fit parent populations during the mosaic stage.

## Scientific contract

- `MosaicReflectionGroupKey` may carry either its retained integer-L metadata or one exact reduced
  `CommensurateLayerOrder` plus the reciprocal-basis revision. The two identities cannot be mixed.
- `build_layer_l_mosaic_profile_definitions(...)` converts an already frozen, active
  `LayerLMarkerObservations` pack into one profile definition per physical landmark. It preserves
  all contributing signed rods and rejects a changed reciprocal basis or invalid angular center.
- `None` means that no optional pack was admitted and produces no profiles. A nominally absent
  pure-parent site is never used as an intensity mask: finite stacks and disorder may still place
  intensity there.
- Every admitted profile retains its own analytically profiled nonnegative amplitude. A fixed
  population vector may set the synthetic amplitudes, including a sum at exact overlaps, but
  absolute and cross-peak intensity ratios do not enter the mosaic objective.
- Only the transverse within-profile shape constrains the shared Gaussian sigma, Lorentzian HWHM,
  and mixture probability. Parent populations and stacking-disorder parameters remain a later
  stacking-intensity problem.

## Permanent proof

The retained 5/10/15-degree fixture reuses the exact T22 detector-locus roster. Its integer-order
baseline contains 22 profiles (`8/8/6`); half- and third-order landmarks expand this to 62
(`24/20/18`). The test checks exact rational identity, denominator coverage, missing-site omission,
one-profile overlap handling, reciprocal-basis rejection, population-weighted scale invariance,
shared-parameter recovery, full projected rank, and an independent nuisance-projected Fisher
calculation.

Synthetic truth is Gaussian sigma `2 deg`, Lorentzian HWHM `0.5 deg`, and Lorentzian probability
`0.2`; aggregate phase fractions are `[2H, 4H, 6H] = [0.65, 0.20, 0.15]`. Both fits recover the
truth and every planted profile scale to roundoff at rank 3. The weakest sensitivity singular value
rises from `0.113453905211` to `0.191405863494`
(`1.68708043x`), and the weakest projected Fisher eigenvalue rises from `0.0128717886076` to
`0.0366362045800` (`2.84624039x`). The three eigenvalues of the information increment are
`0.0235148383159`, `0.632742343524`, and `2.38065717512`, so every information direction gains.
The focused test takes about `7.01 s`; a post-import `tracemalloc` run peaks at `34.5577 MiB` of
Python allocations (excluding process RSS and native-library allocations).

## Limitation

The fitted profiles are analytic synthetic responses whose population-weighted amplitudes span ten
orders of magnitude. Exact keys determine identity, upstream qualification determines availability,
and detector coordinates set only the angular centers. The repository still lacks a PbI2
source-averaged detector response that folds the transition model, finite-stack shape, optics, and
mosaic into these bins. Consequently measured PbI2 mosaic recovery,
parent-specific mosaics, background/PSF effects, the eventual three separately calibrated
specimens, relaxed native 4H/6H metrics, and simultaneous mosaic/population fitting remain
`NO_ORACLE`.
