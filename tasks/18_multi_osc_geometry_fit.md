# Multi-OSC detector-native geometry fit

Status: READY
Branch: `codex/multi-osc-geometry-fit`

## Objective

Fit one shared, scientifically identifiable geometry correction to frozen integer-L peak
associations from an arbitrary nonempty series of OSC images.  The qualifying data set is the
Bi2Se3 5 degree, 10 degree, and user-authoritative 15 degree series.  Every objective evaluation
uses one source-center incident state with zero sampled divergence and the mean wavelength.  It
predicts exact detector-native `(column_px, row_px)` tag coordinates without evaluating structure
intensity, mosaic probability, a raster, or pixel integration.

The first public identity remains the layered-hexagonal
`(family_m, integer_L, tag_branch)` contract backed by analytic branch, root sign, and one physical
`(h,k)` rod.  General crystal-system family identifiers are outside this change; the reusable
series and material boundaries must not otherwise assume Bi2Se3.

This task activates `PHY-FIT-001`, `PHY-FIT-002A`, `PHY-FIT-002B`, `PHY-FIT-003`,
`PHY-FIT-003A`, `PHY-FIT-003C`, and `PHY-FIT-004`.  One fit group has one material and mount;
varying materials are
handled by running the same manifest/API contract for separate groups rather than sharing a
physically meaningless correction vector across unrelated specimens.

## Inputs and immutable joins

One series manifest owns:

- one simulation-geometry configuration path;
- the configured goniometer axis index that each image commands;
- a nonempty ordered image list containing a unique `image_id`, OSC path, and complete commanded
  axis-angle tuple in degrees.

Paths are resolved relative to the manifest.  Image records are joined to indexed observations by
`image_id`, never tuple position or filename inference.  Incidence is never read from an OSC header
or inherited from the historical 12 degree provenance row.  The Bi2Se3 qualification manifest
declares the third file as 15 degrees.

The fitter consumes one immutable data record per image:

- `image_id` and commanded-angle provenance;
- a geometry-only exact-tag model;
- frozen `IntegerLMarkerObservations`, including native coordinates, covariance, wavelength, and
  complete marker keys;
- optional indexing/image hashes for reporting.

All records use the same source, material, crystal orientation, detector calibration, goniometer
zero pose, and fit bounds.  Only declared commanded motor angles vary by image.

## Active parameter pack and gauge ownership

The shared parameter vector has this exact order:

1. detector local-column tilt, radians;
2. detector current-local-row tilt, radians;
3. effective sample-normal local-x tilt, radians;
4. effective sample-normal current-local-y tilt, radians;
5. goniometer-axis pitch, radians;
6. goniometer-axis yaw, radians;
7. signed unbounded-sample-plane normal offset, metres;
8. goniometer-pivot pitch-tangent offset, metres;
9. goniometer-pivot yaw-tangent offset, metres.

The first six half-spans are respectively 10, 10, 5, 5, 5, and 5 degrees.  Each translation
half-span is `1e-4 m`.  Goniometer pitch/yaw reorient exactly one configured commanded axis while
preserving every commanded angle.  The corrected axis is

`(cos(pitch) cos(yaw), -cos(pitch) sin(yaw), sin(pitch))`.

At the corrected axis pitch/yaw, define transported orthonormal tangents
`e_pitch=(-cos(yaw) sin(pitch), sin(yaw) sin(pitch), cos(pitch))` and
`e_yaw=(-sin(yaw), -cos(yaw), 0)`.  The corrected pivot is the nominal pivot plus the two fitted
tangent offsets.  Motion along the axis is excluded because it is a gauge.

The transform order is corrected axis and pivot, commanded motion about them, intrinsic
sample-local x/current-y end-pose rotations about that pivot, the signed plane-offset correction,
then detector intrinsic x/current-y rotations.  If the final sample normal is `n = R[:, 2]`, the
final translation is `t' = t + delta_d * n`; therefore the canonical signed plane offset changes by
exactly `delta_d`.  No tangent sample translation is exposed.

The detector reference coordinate, detector distance and pitches, source center and ray,
wavelength, pivot motion along its axis, crystal in-plane roll, detector roll, crystal mount,
motor angles, and all per-image corrections remain fixed.  Detector roll is an exact gauge with
sample-y and axis-pitch corrections; crystal roll is an axial-powder gauge.  Detector distance is
not fitted with the plane offset because detector calibration owns it.  A parameter pack that is
not full column rank under the actual image/key set is rejected before optimization; no
regularization, silent reduction, or parameter freezing is allowed.

Expected bound-scaled rank for the qualifying key set is 5/9 for 5 degrees alone, 7/9 for 5 plus
10 degrees, and 9/9 only for all three images.  The accepted full-pack condition ceiling is `1e8`.
The two pivot offsets and plane offset form the weakest measured direction; full rank is not a
precision claim, and a fourth incidence is recommended for independent validation.

## Objective and optimizer

For each image, predict its frozen exact tags and evaluate the existing detector-native residual:

- covariance-whitened `(column,row)` site displacement in key order;
- paired-root chord half-angle residual in sorted `(m,L)` order.

Concatenate image residuals in deterministic sorted-image-ID order and optimize one shared bounded vector
with trust-region reflective least squares.  Frozen identities may not be reassigned inside the
objective.  A missing, tangent, branch-changed, integer-L-mismatched, refractively invalid, or
off-panel tag is a typed topology failure.

The result reports the shared correction, rank, condition, scaled singular values, active bounds,
evaluation counts, pooled metrics, and per-image raw site/chord metrics. Predictions and independent
frozen-root survival audits remain public so held-out sites and fitted root validity can be checked
without refitting. The oracle directly brackets the fixed-`L` elastic equation and compares detector
coordinates; it does not call the production integer-`L` solver.

## Reusable command boundary

One generic command accepts a series manifest and can:

1. read every OSC with the one accepted clockwise boundary conversion;
2. run the existing blind discovery, reciprocal indexing, and cross-image selection once;
3. build geometry-only nominal models for every commanded angle;
4. fit the shared nine-coordinate pack against `observations_for(image_id)`;
5. report indexed-manifest hashes, fit metrics, held-out predictions, and post-fit audits as JSON.

The existing Bi2Se3 indexing command remains a retained independent qualification path; this task
does not claim to route that older command through the new series orchestrator. Real full images
and large discovery sweeps remain external proof work, not permanent tests.

## Permanent proof budget

Retain only compact tests that protect distinct long-term failures:

- exact source-center/mean-wavelength behavior already covered by the existing one-state test;
- a geometry-only context built from a second CIF, with no structure-strength, mosaic-density,
  raster, or pixel-integration dependency;
- a prescribed nonzero nine-parameter hidden truth recovered jointly from 5, 10, and 15 degree
  synthetic observations, including the 5/9 -> 7/9 -> 9/9 rank ladder;
- exact signed-plane-offset ownership and structural exclusion of arbitrary sample tangent
  translations and axis-parallel pivot motion;
- deterministic image-ID joining, wavelength/key preservation, per-image metrics, and rejection of
  duplicate IDs or a rank-deficient subset;
- held-out exact-tag prediction, independent post-fit root-coordinate audit, and swapped-beta
  mutation detection;
- wrong OSC rotation, transpose, row/column swap, and half-pixel controls remain protected by the
  existing coordinate tests.

Synthetic acceptance requires maximum parameter error at most `1e-5 rad` for angles and `1e-6 m`
for translations, normalized error at most `2.5e-5` of each half-span,
training RMS/max at most `1e-3/5e-3 px`, held-out max at most `1e-2 px`, line-angle error at most
`1e-7 rad`, no active bounds, and condition at most 15000.

The real 10/8/8-site proof must record baseline and fitted metrics, full rank, condition, bounds,
wall time, peak memory, direct-root/frozen-candidate audits, and the global-rediscovery diagnostic.
Its manifest declares `bi2se3-osc-5-10-15.v1`; formal acceptance requires the frozen initial
indexed-manifest hash, the `L={4,11}` cross-check, and benchmark evidence. Generic manifests omit a
qualification profile and report numerical run completion without inheriting Bi2Se3 thresholds.
Provisional observation-space gates
are pooled RMS at most 3 px, each image RMS at most 4 px, maximum at most 8 px, and no identity
change.  An active bound must be reported and prevents a parameter-precision claim, but does not by
itself invalidate an otherwise stable detector-coordinate prediction.  Common `L={5,8,10}` sites
train a cross-check while `L=4` and `L=11` are
held out; held-out RMS/max gates are 5/10 px.  If the zero-correction baseline is already below the
pooled gate, the fit may not worsen it; otherwise it must reduce pooled RMS by at least 25 percent.
The acceptance-critical outer audit recomputes angles and reciprocal labels for exactly the selected
position-free native coordinates while preserving their support matrices, scores, image/mask
hashes, and policy. It requires every full key and coherent frozen tracks and performs no OSC I/O,
cake search, or coordinate refinement. A separate fresh corrected-geometry global search is
reported as an operational diagnostic; its alternate same-key lobes and newly visible candidates
never drop data or trigger a refit.

## Qualification evidence

The real series freezes manifest
`sha256-1de21e03a801fa38390ef5280133666474bfd969377024ef6dd4fb34e40f3132` with
10/8/8 sites at 5/10/15 degrees. Pooled RMS/max improves from `12.8312707/21.3000088` to
`1.5589418/5.6507942 px`; per-image RMS is `1.2531698/0.8406037/2.2866603 px`. Rank is 9/9,
condition `8689.155`, and the scaled singular spectrum is `(76.2284, 72.4703, 58.1766,
22.8074, 4.35393, 3.33615, 0.256518, 0.100480, 0.00877282)`. Three deterministic starts agree
within `3.164e-5 px` in detector prediction. The `L={4,11}` cross-check gives held-out RMS/max
`1.3173871/2.0219100 px`.

The fitted correction vector is `(-0.004499434, -0.023011841, 0.007325549, 0.015659150,
-0.015187627, -0.015250741, 49.5499 um, 99.999997 um, -27.7328 um)` in contract order. The
pivot-pitch coordinate is bound-seeking, so detector-coordinate acceptance passes but parameter
precision is unqualified and a fourth incidence is recommended. Both the direct root-coordinate
oracle and frozen-candidate relabel audit are `SAME`; all 26 selected candidates retain their full
keys and coherent tracks. Fresh global rediscovery is `CHANGED` because of alternate same-key broad
lobes and a threshold competitor, and is explicitly diagnostic rather than acceptance-critical.

Initial discovery/indexing took `91.851 s`, the fresh global diagnostic `95.200 s`, geometry-only
setup `0.05587 s`, the selected solve `7.6590 s`, and frozen-coordinate relabeling `0.3533 s`. Warm
joint residual median was `0.007240 s`; traced fit peak memory was `99,630 bytes`. Exact-tag fitting
performed zero mosaic, structure-intensity, raster, or pixel work.

Legacy classification is `CORRECTED`. The retained older Bi2Se3 selection manifest was
`sha256-1db880c48002bb5e3cff653044a268bb89b9df89c38c11ee8b2c05ce1bdcc584`; the generic boundary
retains the same 10/8/8 full keys but first diverges at the corrected incident-intersection
`AngleFrame`, provenance revision, detector-native residual, and shared rigid-transform ownership.

## Handoff

The final branch handoff records the commit, public APIs, permanent tests and their distinct
invariants, real-fit and cross-validation metrics, rank/singular/weak-direction evidence,
multi-start stability, outer audits, benchmark/memory, legacy classification and first divergence,
limitations, and the minimum downstream integration request.

## Classification and exclusions

The detector-native implementation is a `CORRECTED` successor to legacy caked-space geometry
fitting.  It may agree through frozen association and the named local correction signs, then
diverges at native-coordinate residuals, canonical rigid transforms, multi-angle rank ownership,
and the exact nominal-ray forward path.

This change does not fit intensity, mosaic width, wavelength, source divergence, crystal lattice,
ordered structure, stacking disorder, detector distance, beam center, detector pitch, gauge
rotations/translations, arbitrary per-image corrections, or general-crystal family labels.  It
does not write diagnostics under the repository root and does not import or execute the tracked
legacy snapshot.
