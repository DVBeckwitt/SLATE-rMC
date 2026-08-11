# Optional-polytype staged fitting integration

Status: READY_INTEGRATED_BI2X3_MATCH_PBI2_POSITIONAL_PREFLIGHT_ONLY
Branch: `codex/polytype-staged-integration`

## Objective

Dovetail the exact rational-layer geometry, mosaic, and fixed-parent PbI2 population work from
T22--T24 with the accepted staged fitting path. The optional path must add qualified 4H/6H
landmarks when they are present, disappear cleanly when they are absent, and leave the existing
Bi2Se3/Bi2Te3 benchmark results unchanged. Then apply the defensible stages to the PbI2 state and
raw OSC data saved by RA-SIM, render external figures, and report both results and unclosed
scientific boundaries.

## Acceptance contract

- Existing integer-L selection, geometry, mosaic, ordered-SF, configuration, artifact, and replay
  contracts remain unchanged. A missing optional landmark pack dispatches directly to the legacy
  implementation rather than passing through an empty rational catalogue.
- Bi2Se3 and Bi2Te3 retain their declared deterministic 3R, epsilon-zero structure model. The
  PbI2 2H/4H/6H parent names and rational support rules never participate in their material path.
- Optional measured landmarks originate only from catalogue-free peak discovery followed by a
  frozen-geometry exact-rational qualification pass. Missing candidates add no residual; exact
  overlaps remain one physical observation; ambiguous candidates fail closed. The returned
  admission record retains the complete hashed discovery, complete parent catalogue, exact source
  row join, and frozen observation pack.
- Geometry may use admitted integer, half-order, and third-order centroids because its objective is
  intensity-free. Mosaic may use them only through detector-native profile shapes with independent
  nuisance amplitude per physical peak. Population/SF fitting uses one detector-folded response
  row per physical observation, so overlapping parent intensities sum inside one residual.
- The three PbI2 specimens own separate detector calibrations, mosaic states, population vectors,
  and nuisance scales. A common beam revision may be shared only when the raw acquisition records
  justify it.
- RA-SIM GUI values, manuscript disorder labels, synthetic truth, and supplied PONI files are
  quarantined during the blind analysis and may be compared only after the fitted artifacts are
  frozen.
- All generated diagnostics, plots, and reports are written outside the repository.

## Required validation

1. Run the current Bi2Se3 and Bi2Te3 staged benchmarks and compare geometry, mosaic, ordered-SF,
   profile, scale, and stacking-parent outputs to the tracked accepted artifacts. Record the first
   divergent stage for any nonzero difference.
2. Prove the optional-absent dispatch preserves the legacy observation objects and does not invoke
   the PbI2 rational or population compiler.
3. Prove any retained detector-native PbI2 response against the existing intrinsic strength at the
   structure-response boundary, including half/third landmarks, overlap, source/mosaic folding,
   and a held-out region.
4. Run catalogue-free discovery on the exact external PbI2 OSC inputs. Freeze hashes, selection
   policies, calibration grouping, ranks, conditioning, residuals, held-outs, and uncertainty or
   state why a stage is not identifiable.
5. Render an external detector/landmark figure, mosaic-profile figure, and population/stacking
   diagnostic only for stages whose declared measures have been validated.

## Stop conditions

Stop with a named `NO_ORACLE` or `BLOCKED` boundary rather than manufacturing a result if the raw
series cannot identify the requested geometry, if rational peaks cannot be qualified independently,
or if detector/source/background transfer is missing for mosaic or population fitting. A CIF alone
is never evidence for detector geometry, mosaic, or measured stacking fractions.

The accepted measured slice is an auditable fitter-ready admission API and external preflight, not
a PbI2 staged runner. Existing Bi2X3 runners remain byte-compatible integer-path callers; measured
PbI2 execution stops before any absent detector-native boundary.

## Accepted result

- The PbI2-only admission pass consumes frozen catalogue-free discoveries, rejects incomplete or
  ambiguous half-/third-order root groups, and retains the exact discovery/catalogue row join.
  `None` leaves the original integer observation object untouched.
- Pre-extension and current integer root solutions agree exactly for 31,110 real Bi2Se3/Bi2Te3
  root problems; all selected Bi2Se3 detector coordinates are bit-exact.
- Fresh current workflows completed for both Bi2Se3 and Bi2Te3. Geometry is scientifically
  unchanged. Joint parameter changes are at most `8.16e-8` for Bi2Se3 and `6.11e-9` for Bi2Te3;
  profile relative-L2 changes are at most `2.83e-7` and `2.04e-8`. Both retain exact deterministic
  3R stacking with `epsilon=0`.
- Catalogue-free external PbI2 inspection finds repeatable third-order positional evidence in the
  Y2 and biggerB series and repeatable half-/third-order evidence in Clean2. The RA-SIM-linked
  biggerB image alone has geometry rank 5/9 and only two admitted structural intensity groups, so
  complete geometry and fixed-parent populations are not identifiable.
- Measured PbI2 mosaic, ordered structure factor, parent populations, and arbitrary transition-law
  parameters stop at `NO_ORACLE`/`BLOCKED`: accepted geometry and a PbI2
  source/optics/mosaic/detector-bin response are absent. No numerical result is manufactured past
  that boundary.
- The two retained rational-admission tests complete in `6.91271 s` under `tracemalloc`; traced
  Python peak memory is `76,594,633` bytes (`73.0463 MiB`) after importing pytest but before loading
  the test module. The production pass evaluates one frozen discovery against one active catalogue
  without image-sized materialization.
