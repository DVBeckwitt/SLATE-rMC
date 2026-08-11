# PbI2 exact-landmark fixed-parent SF population fit

Status: READY_SYNTHETIC_INTRINSIC_LANDMARK_POPULATION_NO_PBI2_DETECTOR_FORWARD
Branch: `codex/pbi2-rational-sf`

## Objective

Connect the qualified exact rational-layer landmarks from T22/T23 to the existing five-parent
PbI2 stacking-strength response and prove internally that separate 2H, 2H+6H, and
2H+4H+6H specimens can recover their declared parent populations. A physical peak is one
residual even when several parents overlap there, and a missing optional peak contributes no row.

This is a synthetic pointwise intrinsic-strength boundary. It does not claim a detector-native or
measured PbI2 intensity fit.

## Scientific contract

- `compile_pbi2_layer_l_stacking_response(...)` accepts an all-parent ideal landmark catalogue
  plus an already qualified `LayerLMarkerObservations` subset. It collapses detector root-side
  duplicates into one structural `(m, exact L, reciprocal basis)` row.
- Every unique contributing signed rod is evaluated once with the existing finite-parent compiler
  and summed as intensity. Parent support metadata is never used as a multiplicity or an intensity
  mask; all five fixed parent columns are evaluated at every admitted row.
- `LayerLStackingObservations` binds intrinsic strengths and variances to the canonical exact
  reflection groups and response sampling revision. The fitter joins by identity, not tuple
  position.
- One nonnegative amount vector and one scale tie every row in one specimen. Exact parent overlap
  is `prediction[row] = response[row, :] @ amount`, never a parent-expanded residual list.
- The three specimens are fitted independently. Their allowed component rosters are respectively
  `2H`, `2H+6H+/-`, and all five components. Their synthetic responses and scales are separate;
  detector calibration is not represented at this intrinsic-strength boundary.
- The allowed aggregate phase contrast must have rank `K-1` for `K` allowed phases. A pure 2H
  specimen is a scale-only fit, while a full three-phase fit remains rejected from integer peaks
  alone.

## Permanent proof

The compact fixture uses the complete six-rod `m=1` group, exact integer orders 1--4, odd
half-orders 1/2--7/2, third-orders 2/3--11/3, wavelength `1.540592925 A`, 52 layers, the tracked
2H motif, fixed `epsilon=0.001`, plus-only initialization, and finite-per-layer normalization.
The two detector root sides collapse to 12 structural rows.

The grouped integer row equals an explicit sum of the six retained pointwise signed-rod responses
to roundoff. Finite-stack/disorder leakage remains nonzero in nominally unsupported columns.
Restricted-parent catalogues, incomplete or branch-inconsistent source provenance, changed crystal
bases, and changed sampling revisions fail closed. An extreme response-scale alias is rejected
after scale-normalized nuisance projection.

Separate train/held-out fits give:

| specimen | train / held-out rows | allowed rank | phase rank | planted domain fractions | scale |
|---|---:|---:|---:|---|---:|
| 2H | 3 / 1 | 1 | 0 | `[1,0,0,0,0]` | 2.75 |
| 2H+6H | 6 / 2 | 3 | 1 | `[.70,0,0,.18,.12]` | 7.0 |
| 2H+4H+6H | 9 / 3 | 5 | 2 | `[.55,.17,.08,.12,.08]` | 13.5 |

All amounts, phase totals, scales, and held-out strengths recover to the frozen numerical
tolerance. The full fit's allowed-response condition is `64.2505`; its phase-contrast condition is
`1.59552`. Integer-only rows have full-model domain rank 3 but phase-contrast rank 1, so the
three-phase fit raises `StackingPopulationIdentifiabilityError` as required.

The three focused tests take `7.85839 s`. Grouping all 12 rows/72 signed-rod events in one compiler
call takes `0.0624491 s` (best of three), versus `0.683017 s` for 12 equivalent pointwise calls;
the `10.9372x` faster grouped result agrees within `4.33681e-19 A2`. A focused `tracemalloc` run
that includes test/module loading peaks at `77,400,472` bytes (`73.8148 MiB`) and takes `52.5931 s`;
native-library allocations and process RSS are outside that measurement.

## Limitation

This estimator fits incoherent populations of five fixed near-parent responses. It does not refine
continuous disorder `epsilon`, arbitrary transition probabilities, layer count, initial population,
the 2H-derived motif, or the registry convention. The response excludes source averaging,
geometry/mosaic/optical transfer, polarization, detector Jacobians, finite detector-bin
integration, exposure/count calibration, background, PSF, masks, and covariance derived from
measured OSC data. Mosaic nuisance amplitudes are not SF observations. Relaxed native 4H/6H
metrics, three measured specimens with separate detector calibrations, and a detector-native PbI2
population fit remain `NO_ORACLE`.
