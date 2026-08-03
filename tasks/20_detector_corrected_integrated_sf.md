# Detector-corrected integrated structure-factor fit

Status: READY_MODEL_LIMITED_FIT_CONDITIONED
Branch: `codex/dark-thomson-integrated-sf`

## Objective

Make the ordered-intensity stage a small, resumable, material-independent pipeline whose fitted
observable is trusted integrated peak mass.  Qualify it on the existing three-image Bi2Se3 series
with the supplied acquisition dark field.  Retain the accepted geometry, mosaic, continuous model,
mixed coordinate charts, and empirical radial-background policy.

The qualification model is the exact fault-free R-centred 3R limit of the generalized stacking
transition model.  It is not a separate perfect-crystal intensity shortcut.  No 2H, 4H, 6H, or
nonzero fault probability is fitted in this task.

## Ordered modular stages

1. Load and orient raw OSC data exactly once at the I/O boundary.
2. Build a hash-bound detector-native dark artifact and subtract it linearly before observation
   projection.  Preserve signed corrected counts; do not clip them.
3. Reuse the supplied accepted geometry checkpoint.
4. Reuse the supplied accepted mosaic checkpoint.
5. Evaluate the fault-free 3R parent through the generalized stacking transition recurrence.
6. Apply the event-wise unpolarized Thomson factor exactly once in detector intensity.
7. Project data and the continuous model onto identical trusted peak regions, condition them with
   the existing frozen background operator, and contract each complete region to one peak mass
   with its full covariance.
8. Fit ordered coordinates jointly with one shared scale per OSC and the declared displacement
   gauge; render full continuous Qz branches only after the authoritative joint fit.

Each stage consumes an immutable artifact or checkpoint from the preceding stage.  A later stage
may be run alone only when every prerequisite artifact is supplied and identity hashes agree.

## Scientific contracts

### Fault-free 3R

Use `RichEpsilonModel(Parent.THREE_R, 0).transition_law()`,
`InitialPopulation.plus_only()`, and zero fault probability.  The transition model is
authoritative.  Its algebraically collapsed one-hot CPU/CUDA evaluator is retained as an optimized
path and must agree with the finite recurrence and direct deterministic enumeration.

### Thomson polarization

For incident and exit unit vectors in external air, apply

`P_Th = (1 + (ki_hat_air dot kf_hat_air)^2) / 2`.

The structure strength already owns the Thomson-length scale, so this dimensionless intensity
factor is multiplied once and is never squared.  A configured scalar polarization weight remains
only an explicit calibration multiplier.

### Dark correction

For an explicit nonnegative dark exposure scale `s_dark`, the corrected detector field is

`D = raw - s_dark * dark`.

Raw and dark are projected by the same continuous rectangle operator before subtraction.  For
independent Poisson acquisitions the propagated covariance is

`Sigma_D = W diag(max(raw, 1) + s_dark^2 max(dark, 1)) W.T`.

The artifact records canonical orientation, dimensions, scale, source SHA-256, decoded-array hash,
and correction-model identity.  Qualification uses `s_dark = 1` unless the file headers establish
a different exposure ratio.

### Integrated peak mass

Every declared trusted peak owns one complete continuous integration rectangle in its accepted
chart: `phi` versus `2theta` for `m=0`, and signed-side Q space for `m!=0`.  The measured
piecewise-constant pixel field and continuous model are integrated over the same rectangle.  The
existing frozen background conditioning is applied without redesign.  If prepared rows partition
a peak, a fixed aggregation matrix `G` gives

`y_peak = G y`, `mu_peak = G mu`, and `Sigma_peak = G Sigma G.T`.

The result is invariant to how a peak rectangle is partitioned.  Pixel centres are never used as
the model measure, and the model is never rasterized.

### Displacement gauge and occupancy

Crystallographic site ADPs remain fixed inside structure amplitudes.  The separate global sample-Q
envelope `exp[-(Ur Qr^2 + Uz Qz^2)]` may be fitted only with its explicit zero-centred regularizer
and declared bounds.  A fit plan that varies site ADPs at the same time is rejected.  The existing
full-site-conserving antisite occupancy parameterization is unchanged.

## Owned paths

- `tasks/20_detector_corrected_integrated_sf.md`, `tasks/index.yaml`
- the narrow scattering, continuous-detector, OSC-correction, matched-observation, and fitting
  modules required by the contracts above
- Bi2Se3 experiment recipes/configuration and the compact tests/documentation protecting unique
  invariants

## Forbidden changes

- background-model redesign, occupancy-rule changes, Parratt stitching, detector solid angle, or
  polarization analysis of the exit beam
- legacy execution/copying, model rasterization, centre-selected count masses, family-specific
  scales, hidden normalization, clipped dark subtraction, or new 2H/4H/6H fit branches
- generated diagnostics under the repository root

## Permanent proof

- stacking recurrence agrees with deterministic 3R enumeration and CPU/CUDA agree
- Thomson factor is one forward/backward, one half at ninety degrees, rotation/sign invariant, and
  CPU/CUDA equivalent
- dark orientation and hashes are exact; raw-minus-dark mass and covariance agree with direct
  enumeration and retain negative values
- integrated mass and covariance agree with direct rectangle integration and are invariant to row
  partitioning
- the gauge rejects simultaneous site-ADP variation, leaves occupancy semantics unchanged, and
  records the model identity in resumable artifacts
- optimized and proof paths agree within frozen tolerance; qualification records convergence,
  wall time, peak memory, and first divergence from the historical centre-selected observable

## Completion gate

The external qualification artifacts include the accepted geometry, mosaic, joint structure
parameters, fitted per-OSC scales, hashes, alignment checks, and a full-Qz Figure 7 comparison.
Only the joint integrated-mass result is authoritative.  Publication readiness remains false unless
the data-only Jacobian is full rank, parameters are admissible, and the declared numerical gates
pass.

## Bi2Se3 qualification result

The detector-corrected integrated-mass fit completed as `MODEL_LIMITED_FIT`.  It used the supplied
dark image, event-wise Thomson polarization, the accepted position and mosaic checkpoints, and the
exact fault-free 3R transition recurrence.  No model image was rasterized and no measured profile
was smoothed.

- Position: common incident-angle shift `+0.4315725 deg`; zero-sum OSC trims
  `[-0.0520939, +0.0023599, +0.0497340] deg`; effective angles
  `[5.3794786, 10.4339324, 15.4813065] deg`.
- Mosaic: Gaussian sigma `1.5193932 deg`, Lorentzian HWHM `0.1409394 deg`, Lorentzian probability
  `0.3776327`.
- Joint structure: Bi fractional `delta_z=-0.0002347073`, outer-Se fractional
  `delta_z=+0.0058351138`, outer-Bi antisite fraction `6.99e-27`, sample envelope
  `Ur=0.0200000 A^2`, `Uz=0.001792329 A^2`.
- The crystallographic site ADPs stayed fixed and distinct from the sample-Q envelope.  Lattice
  constants stayed fixed at `a=4.143 A`, `c=28.636 A`.
- Weighted RMS decreased from `23.09225` at the same-observable Stage-B initializer to `20.84361`
  after the joint fit (`9.74%`); data half-chi-squared decreased from `7732.154` to `6299.616`
  (`18.53%`).

This is an improvement for the new dark-corrected integrated observable, but not a publication-ready
fit.  `Ur` reached its upper bound, the antisite fraction reached its lower bound, and refined versus
coarse projected covariance has not converged.  The result therefore remains fit-conditioned and
model-limited even though the data-only five-parameter sensitivity is full rank.
