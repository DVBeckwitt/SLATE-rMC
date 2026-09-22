# T41: automatic hBN and joint Bi geometry

Status: BLOCKED_GONIOMETER_PIVOT_UNIDENTIFIABLE
Branch: `codex/joint-hbn-bi-geometry`
Base: `4a2c2940a7de7f05004b77b173dc22a76a879b9f`

## Goal

Automatically extract hBN powder-ring observations and fit them jointly with every declared
Bi2Se3 and Bi2Te3 incidence image.  Report the complete fixed, session-global, and specimen-local
geometry with rank, conditioning, covariance, correlations, held-out prediction, and measured
detector overlays.

## Ownership

- hBN constrains exactly the detector local-column tilt, detector current-local-row tilt, and the
  native continuous beam-center column and row.
- hBN ring scale or calibrant-plane distance is a private nuisance and never changes the shared
  detector distance.
- Bi images constrain the shared goniometer-axis direction, its two axis-perpendicular pivot
  coordinates, global beam-to-center displacement `zB`, and one common commanded-angle zero when
  identifiable.
- Bi2Se3 and Bi2Te3 each own their sample-normal end-pose corrections and signed sample-plane
  displacement `zS`.
- Detector pitch, detector shape, detector distance, source declaration, wavelength declaration,
  OSC mapping, crystal mount, and detector roll stay fixed.

The joint parameterization must be minimal.  If `zB` is a derived component of the fitted beam
line relative to the goniometer pivot, report it with propagated covariance instead of adding a
duplicate optimizer coordinate.

## Plan before implementation

1. Add a GUI-free hBN observation module that dark-subtracts canonical detector-native OSCs,
   predicts the fixed hBN rings from a bounded nuisance scale, traces their centerlines
   automatically, and returns accepted full-resolution points with uncertainties and ring IDs.
2. Add the physical powder-ring residual with only four shared derivatives plus its private scale.
   Retain ring/sector grouping so dense angular tracing cannot overwhelm the Bi observations.
3. Add one cross-material geometry parameterization that maps session-global and specimen-local
   values into the existing per-image exact-tag predictor.  Reuse the existing detector-native Bi
   residual and fixed associations; do not duplicate projection physics.
4. Add the smallest external-artifact runner for hBN, Bi2Se3, and Bi2Te3 manifests.  Persist no
   generated images or diagnostics in the repository.
5. Prove synthetic recovery and derivative ownership, including zero hBN sensitivity to Bi-only
   coordinates and distinct recovery of the two local `zS` values.  Reject rank-deficient packs.
6. Run the measured joint fit, ring/peak holdouts, deterministic multistarts, and covariance audit.
   Render detector-native measured/predicted overlays externally and classify every reported
   parameter as confident, correlated/model-limited, or unidentifiable.

## Stop conditions

- Stop `BLOCKED` rather than report a parameter value when the actual joint Jacobian is rank
  deficient, the selected topology changes, the automatic ring assignments fail their gates, or
  a parameter remains bound-seeking without an independent predictive qualification.
- Structure intensity, mosaic, background, detector integration, and all-SF fitting are outside
  this geometry task.

## Verification

```powershell
python -m compileall -q src scripts
ruff check src/rasim_next/fitting tests scripts
pytest -q tests/test_fitting.py tests/test_fit_osc_geometry_cli.py
python -m rasim_next.proof geometry-optics --json
python scripts/verify_seed.py
git diff --check
```

## Handoff

Status: The click-free hBN detector calibration and combined six-image observable fit pass their
residual gates.  The complete physical parameter pack is not confidence-qualified because the
goniometer pivot pitch coordinate seeks the expanded +1.0 mm bound and dominates the weakest
Jacobian direction.  `zB` therefore remains unqualified.

Commit SHA: branch HEAD

Measured geometry artifact:
`C:\Users\Kenpo\.codex\visualizations\2026\09\18\01a0b67f-592e-76b3-b9b4-5d6d86215329\joint_hbn_geometry\joint_geometry_report_final.json`

Fitted fixed/global/local parameters: the artifact reports every value and one-standard-error
precision.  Detector center `(1452.7382, 1596.7348) px` and intrinsic tilts
`(-0.38865, -1.34264) deg` qualify.  The pooled Bi site RMS is `1.2433 px`; hBN ring RMS is
`0.8250 px`.  Goniometer axis and pivot values, and derived `zB=-1.0313 mm`, do not qualify.

Confidence and limitations: data Jacobian rank is 15/15 and its scaled condition is 12141.6, but
the active pivot bound fails the declared stop condition.  The weakest scaled direction has
0.9845 loading on the pivot-pitch coordinate.  The two local `zS` estimates are statistically
consistent with zero at about 0.046 mm one-standard-error precision.  A mechanical pivot datum or
a wider-angle incidence observation that changes the pivot lever arm is required to qualify the
complete decomposition.
