# Bi2Se3 example

This example supports OSC decoding, detector-native geometry, refraction/attenuation, mosaic and
Ewald events, ordered rods, reflectivity, rod-family selection, and the later geometry, mosaic,
and ordered-intensity fit sequence.

The saved legacy state used `background_backend_rotation_k = 3`, equivalent to one clockwise
array rotation. It also named detector-native row as `x` and detector-native column as `y`.
The sanitized files never use that naming. Continuous coordinates are `(column_px, row_px)` and
arrays are indexed `[row, column]`.

The measured OSC files are gzip-compressed to keep the repository modest. A proof may stream them
through `gzip.open`; it must not create uncompressed copies under the repository root.

The tracked mosaic-recovery case is
`experiment/mosaic_fit_truth.toml`. Run
`uv run --frozen python scripts/recover_bi2se3_mosaic.py --output-directory <external-directory>` to recover
the prescribed `(2 deg Gaussian sigma, 0.5 deg Lorentzian HWHM, eta=0.1)` distribution jointly from
the 5, 10, and 15 degree continuous profiles. The fit jointly uses the frozen 10/8/8 indexed
nonzero profiles and six raw-supported branchless `00L` profiles representable by the fixed
top-exit model. Every profile has its own nuisance amplitude, so peak heights and cross-reflection
structure-factor ratios do not weight the answer. A frozen geometry audit excludes 43 inverse-
support boundary bins identically from truth and every component, while retaining all profiles.
Its 3,000 x 3,000 images are external configured-truth forward visualizations without the planted
nuisance amplitudes, not fitted observations or recovered-model renders. Raw-significant `003` at
10 and 15 degrees is reported but not fitted because this forward model predicts zero signal there.

For the measured images, supply the completed verified position stage atomically; raw geometry
corrections or a loose revision are not accepted:

```powershell
uv run --frozen python scripts/recover_bi2se3_mosaic.py `
  --observation-mode osc `
  --position-artifact C:\external\bi2se3-replay\geometry.json `
  --source-sample-count 250 --skip-images `
  --output-directory C:\external\bi2se3-mosaic
```

The case-bound
`experiment/mosaic_fit_measured_policy.toml` removes entire weak profiles using frozen local
sidebands and removes both 10-degree `m=1,L=4` secondary-lobe profiles; it never trims individual
central-profile bins by intensity. The three OSC files are still fitted simultaneously. The fitted
`m=0` profiles are `003/006` at 5 degrees, `006` at 10 degrees, and `006/009` at 15 degrees. This
real-data result is reported as a model-limited effective radial envelope, not a unique intrinsic
mosaic distribution.

`experiment/staged_fit_replay.toml` binds the full geometry, measured-mosaic, and synthetic
selected-component ordered-intensity sequence. Run it with an external output directory so the
tracked dependency lock is part of the replay:

```powershell
uv run --frozen python scripts/replay_staged_fit.py `
  examples/bi2se3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2se3-replay --backend cuda --through geometry

uv run --frozen python scripts/replay_staged_fit.py `
  examples/bi2se3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2se3-replay --backend cuda --through mosaic --resume
```

The second command validates and reuses `geometry.json`, fits only mosaic, and stops without
ordered-intensity or detector rendering. A later stage requires this verified predecessor and must
be requested explicitly.

The ordered stage is an identifiability proof at fixed geometry and mosaic; it is not raw-OSC
structure-factor recovery.
