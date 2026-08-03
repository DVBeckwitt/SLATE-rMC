# Parratt specular stitch

Status: READY_MODEL_LIMITED_FIT_CONDITIONED_NO_OBSERVED_HANDOFF_OVERLAP
Branch: `codex/parratt-specular-stitch`
Ledger rows: `PHY-REF-001`--`PHY-REF-007`, `PHY-REF-010`

## Objective

Add the smallest material-independent, detector-native implementation of the named empirical
Parratt-to-kinematic `m=0` handoff used by the manuscript and corrected RA-SIM calculation.  Apply
it to the Bi2Se3 three-OSC qualification without changing any nonzero-`m` rod, fitting a new optical
nuisance parameter, rasterizing the model, or duplicating the pure Parratt and finite-stack
equations.

## Fixed plan before editing

1. Retain `parratt_reflectivity` and `manuscript_specular_composite` as the numerical authorities.
2. Add a unit-preserving kinematic-scale result whose high branch is exactly the continuous
   internal-phase finite-stack strength and whose low branch is scaled pure Parratt reflectivity.
3. Bind an explicit air/film/substrate stack to the source-averaged detector.  Film optics remain
   wavelength-resolved CIF/XrayDB values; substrate index, thickness, and both interface
   roughnesses are fixed inputs and part of the model identity.
4. Replace only the `(h,k)=(0,0)` strength in each source-state evaluator.  Preserve all existing
   source, mosaic, detector-measure, and fit integration contracts; the model stays continuous.
5. Prove high-branch identity, strict `m=0` scope, CPU/CUDA equivalence, and immutable stack
   provenance with the smallest distinct permanent tests.
6. Reuse accepted position and mosaic checkpoints, warm-start the staged structure fit, generate
   new continuous profiles and Figure 7 externally, and compare the low-angle region and fitted
   parameters against T20.

## Interpretation

This is the named empirical manuscript/legacy handoff, not a DWBA calculation or a new additive
surface-reflection channel.  It assumes the stitched `m=0` line follows the locally tilted
lamellae represented by the mosaic distribution.  Pure Parratt, pure kinematic, and composite
outputs remain separately accessible.

Detector events supply external normal momentum transfer directly; the implementation does not
infer it from internal phase `L`.  The high branch nevertheless evaluates the finite stack at that
internal phase coordinate, as the refracted detector model already did before this task.

The legacy SiO2 substrate and zero roughness values are compatibility inputs, not established ORNL
sample metadata.  They must be declared in the Bi2Se3 recipe and reported as a limitation.

## Completion gate

- No change to `m!=0` model values and exact recovery of the old kinematic `m=0` branch above the
  frozen handoff.
- No detector pixelization, hidden scale, fitted optical constant, or fallback that widens the
  handoff merely to improve the data.
- External fit/profile/figure artifacts bind the optical stack and implementation hashes.
- The before/after report states whether the actually observed low-angle region lies inside the
  handoff and whether the fit improved.

## Bi2Se3 qualification result (2026-08-03)

The strict external rerun is retained under `bi2se3_parratt_stitch_v17`.  Every checked source
wavelength selected the legacy fallback handoff `3 <= Qz/Qc <= 6`.  At the representative
`lambda=1.54059065 A`, `Qc=0.0506540596 A^-1`, so the composite covers
`Qz=0.151962--0.303924 A^-1` (`2theta=2.13496--4.27067 deg`).  Across the complete configured
wavelength support the upper endpoint is at most `0.304025 A^-1`.  The retained `m=0` display and
data objective begin at `2theta=6.55 deg`, `Qz=0.459319 A^-1`, entirely above the handoff.  The
implementation therefore recovers the exact refracted finite-stack branch over every displayed
point; widening the fallback after inspecting the data is explicitly rejected.

The final joint vector is
`(-0.00023470731574029476, 0.005835113709169384, 7.237450881995888e-15,
0.01999999999999276 A2, 0.0017923284519703931 A2)` for Bi fractional `delta z`, outer-chalcogen
fractional `delta z`, outer-site Bi antisite fraction, sample-envelope `U_r`, and sample-envelope
`U_z`.  Weighted RMS changed from `20.84361406300731` to `20.843613947573456`; the low-angle
`6.55--7.15 deg` summed model/data ratio changed from `1.6046810284471114` to
`1.6046810219763552`.  Both changes are numerical-noise sized.  The `003` model/data ratio changed
from `0.993124360488` to `0.993124359110`, also only numerical noise.  The result is therefore not
a fit improvement.  It remains `MODEL_LIMITED_FIT`, `FIT_CONDITIONED`, and
`publication_ready=false`; the antisite fraction is on its lower bound and `U_r` is on its upper
bound.

The direct fixed-parameter proofs retain exact `m!=0` identity and exact high-branch recovery.
After the deliberately repeated joint optimization, the largest relative change in any plotted
model value is `5.94e-7`, caused by the tiny parameter-vector change rather than direct Parratt
weighting.  The model is unsmoothed and unpixelized; only the measured detector panel remains a
native pixel image.

External artifact SHA-256 values are prepared diagnostic
`056ee4940f03a47a986c23ded7c47502a0c3010f7d89e128efc854ea9f081d4d`, background
`a42469e2af9497350da6abd01d720fff8b824822bb5aa6f12b99ee5881336d8d`, joint fit
`081ba343b85a5b68fcfe4157e857370fd9352ca90242b98f88fbf77aed5f02ef`, continuous profiles
`4e56da248c354560d8085e09d33fd8c39b90364bc316cba2ca31b547bcd290ef`, and rendered PNG
`153803ff7a809d7abe2c95850b7933b3581440676e19f6cf31b196e0baa6b7aa`.
