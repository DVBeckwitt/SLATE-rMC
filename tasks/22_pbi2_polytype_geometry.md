# PbI2 optional polytype-landmark geometry fit

Status: READY_SYNTHETIC_DECLARED_SINGLE_TRILAYER_METRIC_NO_MEASURED_PBI2
Branch: `codex/pbi2-rational-geometry`

## Objective

Extend the accepted exact-tag geometry fitter with an additive rational-layer-coordinate path so
prequalified 4H half-order and 6H third-order landmarks can constrain the same detector/sample
geometry as 2H integer landmarks. Missing optional peaks contribute no residual. Exact overlaps
between parent periods contribute one physical geometry observation, with every contributing
signed rod retained as provenance.

This task is a synthetic validation of ideal parent periods in the model-supplied one-trilayer
PbI2 metric; the permanent proof uses the tracked 2H CIF. It does not activate measured PbI2
OSC discovery, mosaic fitting, detector-native stacking fitting, or relaxed-native 4H/6H lattice
metrics. The existing integer-L selection, serialization, and Bi2X3 staged workflows remain
unchanged.

## Scientific contract

- Layer coordinate identity is an exact reduced rational `(numerator, denominator)`; floating
  values never define identity.
- The reciprocal basis remains the one-layer 2H basis. A virtual six-layer supercell is proof-only
  and is not a production representation.
- A physical marker key contains family, rational layer order, analytic branch, root sign, and the
  reciprocal-basis revision. Signed rods are stored separately in a canonical provenance tuple.
- The analytic fixed-L Ewald equation is implemented once. The retained integer solver delegates
  to the same equation with denominator one and must remain numerically unchanged.
- The position objective evaluates no structure strength, parent fraction, stacking recurrence,
  mosaic distribution, detector raster, or pixel integral.
- Optional half-/third-order observations must already be frozen and centroid-qualified by an
  upstream process. Empty optional input returns the 2H baseline unchanged. Nearby but distinct
  rational sites are never resolution-merged.
- The common-incidence-delta/sample-normal-x gauge from T18 remains unchanged; additional peaks do
  not remove it.

## Implementation boundary

1. Add an exact commensurate layer-order value type.
2. Add a rational fixed-L root solver while retaining the integer API as a denominator-one wrapper.
3. Add rod-free rational marker keys, explicit marker definitions, observations, predictions, and
   an independent fixed-L root audit.
4. Let the existing multi-incidence residual and optimizer consume either the retained integer
   observations or one rational observation pack containing integer and fractional landmarks.
5. Add an ideal-PbI2 catalogue that uses the single 2H trilayer metric, exact parent-period support,
   detector-locus deduplication, and hash-bound provenance.
6. Add a frozen-observation merge that returns the baseline object unchanged when no optional
   landmarks are supplied and rejects conflicting duplicate physical observations.

## Permanent proof

- Canonical rational reduction/order and invalid-input rejection.
- General fixed-L roots against an independently bracketed elastic equation, including regular,
  tangent, and no-root cases.
- Bit-exact denominator-one solver and detector-prediction parity with the retained integer path.
- Exact PbI2 signed-sector support and one-residual overlap deduplication.
- Three-incidence hidden-pose recovery for 2H-only and augmented 2H/4H/6H landmark packs, with the
  same planted geometry, full rank, independent root audit, and a strictly stronger weakest
  information direction when the optional peaks are present.
- Spies that reject any intensity, mosaic, stacking, raster, or pixel work in the rational geometry
  path.

## Acceptance and limitation

Noiseless synthetic fits must recover the planted geometry within the existing T18 normalized
parameter tolerance, remain off all bounds, and pass the direct fixed-L root audit. The augmented
fit must add information without changing the parameterization or the recovered truth. Measured
PbI2 use remains `NO_ORACLE`: the supplied relaxed 4H/6H cells have a different per-layer repeat,
and no measured PbI2 OSC/calibration data or centroid-invariance certificate is tracked.

## Result

The retained 5/10/15-degree proof compares the unchanged legacy integer-L 2H baseline with one
complete rational pack containing 4H+ halves and 6H- thirds. Active site counts are `8/8/6` and
`24/20/18`. Both recover all nine planted shared geometry coordinates at rank 9 with no active
bounds. The weakest scaled singular value increases from `0.193651473287` to `0.296279736884`
(`1.52996376x`), while condition improves from `24920.4269954` to `16842.9140450`. Maximum
normalized parameter errors are `1.68284e-11` and `6.41023e-11`; maximum detector-site residuals
are `2.14504e-12` and `4.86069e-12 px`.

Independent bracketed root audits are `SAME`; a swapped rational beta/root assignment is
`CHANGED`. Denominator-one roots and predictions are bit-exact with the retained integer API.
Exact overlaps union their signed rods into one residual, conflicting duplicates and mixed-basis
packs fail, and the no-option merge returns the original legacy observation object. Strength,
stacking, mosaic, macrobin, and pixel-center paths are all guarded against execution. The focused
diagnostic took `71.67 s`; full-proof traced Python allocation peaked at `3.74982 MiB` after
imports (excluding process RSS and native-library allocations).
