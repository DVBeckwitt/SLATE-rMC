# Examples

## Canonical Bi2Se3 simulation

[`configs/bi2se3_simulation.yaml`](../configs/bi2se3_simulation.yaml) is the sole default authority.
It currently declares:

- the tracked Bi2Se3 CIF;
- 1,000 source phase-space states with wavelength variation;
- a 5 degree sample incidence and no other sample rotations;
- a 3,000 × 3,000 detector, 100 micrometre pitch, 75 millimetre distance, beam center, and two
  independently configurable intrinsic detector tilts;
- a 1 degree Gaussian mosaic with no active Lorentzian component;
- a 52-quintuple-layer finite-2H model with shared disorder epsilon 0.001;
- every elastically reachable physical rod, every retained root, and regular detector-visible
  `m=0` intensity with its positive direct-root gap certificate;
- explicit CPU or CUDA detector execution; and
- independent switches for reciprocal-space, Ewald-coating, and detector figures.

Run it with:

```powershell
uv run python scripts/run_configured_simulation.py configs/bi2se3_simulation.yaml
```

Use an external output override when desired:

```powershell
uv run python scripts/run_configured_simulation.py `
  configs/bi2se3_simulation.yaml `
  --output-dir C:\path\outside\the\repository
```

Set `numerics.detector_execution_backend` to `cuda` for GPU evaluation. CUDA availability is checked
before simulation; there is no silent CPU fallback. Detector tilts are set in
`instrument.detector_tilt.about_column_axis_deg` and `about_row_axis_deg`. Both are folded into the
one canonical detector pose before ray construction.

The three optional outputs are views of callable models:

- `reciprocal-space.png`: a user-controlled display sampling of continuous latent Bragg space;
- `ewald-surface.png`: a display sampling of the detector-visible intrinsic analytic coating;
- `detector.png`: fixed-quadrature detector macrobin integrals of the source-averaged coordinate
  field.

Sampling used to render the two 3D figures is not retained as physics. The detector figure labels
its macrobin estimate and shows the direct beam and invalid regions. It also marks the exact
nominal-source, peak-mosaic (`alpha=0`) integer-L centers that survive exit refraction and intersect
the active panel. Each tag declares `m`, integer `L`, and Ewald-root branch; the external diagnostic
retains the contributing physical `(h,k)` rods and their exact beta coordinates. A center that is
back-facing or off-panel is deliberately absent even if a nonzero-mosaic tail becomes visible.

## Fit tagged landmarks between continuous detector functions

Bind the base simulation and every trial as continuous detector functions. The fit uses the exact
tag identities and their detector-coordinate line geometry; it does not construct an image,
centroid, pixel integral, or detector quadrature:

```python
from rasim_next.fitting import (
    ContinuousDetectorGeometryModel,
    GeometryCorrectionBounds,
    GeometryCorrections,
    IntegerLMarkerObservations,
    audit_integer_l_marker_selection,
    fit_tagged_detector_function_geometry,
)

catalog = evaluate_nominal_integer_l_markers(build_nominal_ewald_context(inputs))
keys = IntegerLMarkerObservations.from_markers(
    catalog,
    selection=selected_rows,
).keys
model = ContinuousDetectorGeometryModel(inputs)
reference = model.bind(reference_geometry_corrections)
fit = fit_tagged_detector_function_geometry(
    model,
    reference,
    nonzero_keys=keys,
    m0_integer_L=tuple(range(2, 20)),
    sigma_px=0.25,
    initial=GeometryCorrections.zero(),
    bounds=GeometryCorrectionBounds.rasim_reduced_pose(),
)
fitted = model.bind(fit.corrections)
audit = audit_integer_l_marker_selection(
    fitted,
    IntegerLMarkerObservations.from_markers(catalog).keys,
)
```

The nonzero lines join user-facing `tag_branch=1/2`, which retain analytic `root_sign=-1/+1` at
the same `(m,L,Ewald branch)`; the sides are not separate Ewald branches. The m=0 line uses
`tag_branch=0` and explicitly named minimum-mosaic-tilt
exact-L landmarks because `(m=0,L)` alone is a curve. These landmarks are not advertised as
intensity maxima. The fit adjusts both detector tilts and the two identifiable components of the
effective sample normal about the common configured goniometer pivot. Configurations without a
unique common pivot must pass `sample_correction_pivot_lab_m` when constructing the model. The fit
intentionally does not report separate sample-mount and goniometer-axis errors; that decomposition
requires tags from multiple commanded goniometer angles.

## Bi2Te3 quintuple-layer configuration

[`configs/bi2te3_simulation.yaml`](../configs/bi2te3_simulation.yaml) exercises the same finite
quintuple-layer implementation with the tracked COD 9011962 Bi2Te3 structure. The generic
`r3m_quintuple_finite_2h.v1` model resolves the Bi, central-chalcogen, and outer-chalcogen CIF
orbits from their expanded multiplicities, so Te form factors and site labels come from the
Bi2Te3 CIF rather than a Bi2Se3-specific constant. The historical Python parameter type retains
its `se1` and `se2` field names for compatibility; those fields mean central and outer chalcogen
for this material-generic path.

The configuration is a reusable forward-model input. Its detector tilts and beam center are fixed
calibration values; its 250 source states are reduced incoherently into one detector function
before any mosaic profile, ordered-intensity comparison, or image display. The exact compressed
Bi2Te3 OSC inputs and frozen catalog used by the accepted model-limited fit are tracked.

## Portable Bi2Se3 and Bi2Te3 staged-fit replay

The two `staged_fit_replay.toml` cases bind every input by repository-relative path, file SHA-256,
and decoded detector-native OSC-array SHA-256. Validate a clean clone without running a fit:

```powershell
uv run --frozen python scripts/replay_staged_fit.py examples/bi2se3/experiment/staged_fit_replay.toml --inputs-only
uv run --frozen python scripts/replay_staged_fit.py examples/bi2te3/experiment/staged_fit_replay.toml --inputs-only
```

Replay every fit stage into a new directory outside the repository:

```powershell
uv run --frozen python scripts/replay_staged_fit.py examples/bi2se3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2se3-replay --backend cuda
uv run --frozen python scripts/replay_staged_fit.py examples/bi2te3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2te3-replay --backend cuda
```

`--through geometry`, `--through mosaic`, and `--resume` retain strict stage order and scientific
revision chaining. Resume also requires the exact per-stage execution runtime recorded in the
stage artifact. Geometry uses the ideal source center, zero divergence, and mean wavelength.
Mosaic and ordered intensity each use the case's identical 250-state realization and compare the
single fully reduced detector function per incidence; neither fit constructs 3,000 x 3,000 images.
Bi2Te3 can additionally render the six native images with `--through render`. Bi2Se3 render is
disabled because the available images bind a retired mosaic gate.

The frozen replay targets are:

- Bi2Se3 geometry: nine active coordinates, RMS/max `1.55894/5.65079 px`; measured mosaic:
  `(sigma_G, HWHM_L, eta)=(1.3228757 deg, 0.48989795 deg, 0.44809616)` from 15 profiles including
  five `m=0`; ordered stage: exact synthetic selected-component recovery
  `(0.94, 0.78, 0.86, 0.007, 0.034)`.
- Bi2Te3 geometry: detector tilts fixed, seven active coordinates, RMS/max
  `7.17509/15.07787 px`; measured mosaic `(1.0340984 deg, 0.66004834 deg, 0.34468382)` from 33
  profiles including three `m=0`; relative SF parameters
  `(Te1/Bi, Te2/Bi, Ur, Uz)=(1.0, 0.84487145, 0.1, 0.0)` from 31 profiles.

Exact verification covers input bytes, decoded OSC values, identity lists (including `m=0`),
source revision, masks, classifications, stage lineage, and decoded display pixels. Fit floats use
the declared absolute tolerances; PNG container bytes, paths, timings, memory, and device strings
are not scientific identity. The Bi2Te3 mosaic and ordered stages are current-lock CUDA-qualified;
the render has the separate historical qualification below. Both measured mosaic fits are
model-limited effective envelopes, and the Bi2Te3 bound-seeking SF fit is `NO_ORACLE`, not a direct
count-calibrated raw-intensity recovery.

`--frozen` makes the complete numerical dependency closure in `uv.lock` part of the operational
replay. The lock bytes are re-hashed immediately before execution, before an output directory or
fit stage is created. The case does
not claim bitwise portability across Python patch releases, operating systems, CUDA drivers, or
GPU models: fit values are tolerance-checked. Bi2Te3 mosaic, ordered-intensity, and render stages
reject CPU before geometry or output creation. The stored decoded-pixel hashes are a historical RTX-3060 oracle that has not yet been
requalified under the current lock; `--through render` either verifies them exactly or fails. To
perform that separate check, install the image dependency and request the stage explicitly:

```powershell
uv run --frozen --extra visualization python scripts/replay_staged_fit.py `
  examples/bi2te3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2te3-render --backend cuda --through render
```

## Quantitative one-state pixel diagnostic

The retained one-state CLI exercises native-pixel fixed or adaptive integration and emits an
external numeric diagnostic plus a linear/log figure:

```powershell
uv run python scripts/generate_bi2se3_continuous_detector.py `
  --output-dir C:\path\outside\the\repository `
  --integration-method adaptive_compiled `
  --relative-tolerance 1e-4
```

Omitted mosaic, layer-count, and shared-disorder arguments inherit the canonical YAML. Explicit
overrides are available for controlled convergence studies. This diagnostic intentionally uses one
incident state, the six physical `m=1` rods, and the upper root; it is not a second default model.

## Optional weighted Monte Carlo pixel mass

Use the same configured 1,000-state Bi2Se3 detector and stream weighted forward roots directly into
native pixels:

```python
from pathlib import Path

from rasim_next.pipeline.configured_simulation import (
    build_configured_simulation_inputs,
    build_source_averaged_detector,
    load_simulation_config,
)

root = Path.cwd()
config = load_simulation_config(root / "configs" / "bi2se3_simulation.yaml")
inputs = build_configured_simulation_inputs(config)
detector = build_source_averaged_detector(inputs)
estimate = detector.sample_native_pixel_mass(
    draws_per_source_state=40,
    seed=20260728,
)
```

This uses 40 independent mosaic draws for each fixed `ki` row, hence 40,000 source-stratified
draws. `estimate.image_A2` is weighted raw pixel mass in `A2`; `visible_hit_count` is a root-work
ledger, not a photon-count image. Increase the draw count and compare independent seeds or a refined
latent oracle when quantitative stochastic accuracy matters.

## Matched continuous detector and angle views

Render the same continuous detector density on native detector coordinates and canonical
`(phi, 2theta)` coordinates:

```powershell
uv run python scripts/render_continuous_angle_comparison.py `
  --output C:\path\outside\the\repository\bi2se3-detector-angle.png
```

The default evaluates two 3,000 × 3,000 center-sampled fields for one nominal source state, every
configured physical rod, and every retained root. The angle panel displays the normalized
`I=S/N`, not `S` alone, so its values retain the detector-density units and shared color scale.
Every positive-strength nominal `alpha=0` exact-integer-L marker is labeled on both panels; these
landmarks are not inferred raster maxima. Increase `--source-sample-count` only when the additional
runtime of the full incoherent ensemble is intended.

## Interactive Monte Carlo detector viewer

Open the native-pixel detector view with sample/detector pose controls:

```powershell
uv run --extra visualization python scripts/interactive_detector_viewer.py
```

The default Bi2Se3 view uses the configured 1,000 incident-wavevector states, 49 Monte Carlo mosaic
draws per state, detector seed `20260728`, and the full 3,000 x 3,000 native detector. The two
numerical controls are **incident-ray samples** (`N_ray`) and **mosaic draws per ki state** (`M`). A
source row contains origin, direction, wavelength, and empirical weight; `M` controls only the
stochastic detector estimator. While a control moves, the last settled image remains visible.
Releasing a changed control starts the latest requested native-pixel render in a background thread.
The fixed detector seed uses common source-index substreams across pose changes, reducing visual
flicker while preserving the declared estimator. Press `R` to render, `0` to reset, or `Q` to close.

Every geometry slider is a zero-based correction to the configured geometry. The labels use the
original RA-SIM and manuscript vocabulary while distinguishing mechanical goniometer controls from
effective end-pose controls:

- detector pitch is the current column-axis correction, equal to `-delta gamma` in the original
  RA-SIM sign convention; detector yaw is the row-axis `delta Gamma` correction;
- detector and sample in-plane rotations are distinguished as `delta chi_D` and `delta chi_S`;
  the latter is locally `-delta psi` in RA-SIM's sign convention;
- detector-normal distance is `delta D_n`; detector column/row translations remain physical
  millimetre translations and are only marked as coupled to manuscript beam-center `x0/y0`;
- goniometer-axis pitch `delta alpha` is RA-SIM `delta cor_angle`, and goniometer-axis yaw
  `delta psi_g` is RA-SIM `delta psi_z`. These reorient the sole configured commanded axis while
  preserving its motor angle and LAB pivot; they are not direct sample rotations;
- effective incidence and sample tilt are `delta theta_i` and `delta delta` (the latter was RA-SIM
  `chi`, labelled **Sample Pitch**);
- sample tangent translations remain `delta x_S/delta y_S`; sample-normal translation is
  `delta n_S = -delta z_S` in the original RA-SIM sign convention.

Detector rotations are about the configured detector reference-coordinate point. The two
goniometer controls reconstruct the pivoted commanded motion using the RA-SIM axis
`(cos(alpha) cos(psi_g), -cos(alpha) sin(psi_g), sin(alpha))`; consequently they have exactly zero
effect when the commanded motor angle is zero. Nonzero goniometer offsets require exactly one
configured axis because the current schema does not name motor roles in a multi-axis tuple.
Remaining sample rotations and translations are effective local end-pose corrections applied after
that mechanical motion. The viewer does not relabel unmatched tangent translations as manuscript
parameters or add the independent beam offset `z_B`. These manual forward controls also do not make
`(alpha, psi_g)` separately identifiable from one measured image; that requires multiple commanded
goniometer angles and remains outside the current fitting contract.

Every view samples the declared mosaic distribution for all selected incident-wavevector states,
enumerates every elastically reachable physical `(h,k)` rod and retained Ewald root, and sums the
weighted raw mass directly into the exact native `[row, column]` pixel owner. The displayed array is
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`; nearest-pixel rendering and the logarithmic
color scale do not change or normalize it. Values are raw `A2` mass estimates, not photon counts,
detector-coordinate quadrature, a PSF, or a calibrated detector response. Empty pixels received no
sampled deposit at the chosen draw count. Configurations that disable detector-visible `m=0` are
rejected because this viewer's contract is the all-`m` physical-rod catalogue.

The streaming Monte Carlo estimator is currently CPU-only. Override its sample count, draw count,
or detector seed explicitly when desired:

```powershell
uv run --extra visualization python scripts/interactive_detector_viewer.py `
  --ki-samples 1000 `
  --draws-per-ki 49 `
  --seed 20260728
```

## Position-free measured peak indexing

Discover peaks globally in the tracked detector-native OSC images, infer their reciprocal
integer-`L` identities, and emit the frozen cross-incidence selection manifest:

```powershell
uv run python scripts/index_bi2se3_osc.py --json
```

Discovery accepts no marker catalogue or predicted marker positions. It searches a tiled
`(2theta, phi)` chart, refines proposals on the native detector, and generates exact alpha-zero
anchors only after each discrete reciprocal label is frozen. An external 3,000 × 3,000 simulation
diagnostic may be supplied with `--simulation-diagnostic`; unresolved diagnostics are rejected
unless the non-accepted override is explicit.

## Joint geometry fitting for an OSC series

Fit one shared full-rank geometry correction to the frozen 5, 10, and user-authoritative 15 degree
Bi2Se3 observations, cross-validate integer `L={4,11}`, rerun the measured outer audit, and include
separate timing/memory evidence:

```powershell
uv run --frozen python scripts/fit_osc_geometry.py `
  configs/bi2se3_osc_geometry_fit.yaml `
  --heldout-integer-l 4 11 `
  --benchmark `
  --json
```

The manifest is the reusable boundary: each record declares an exact image ID, OSC path, and full
commanded-angle tuple. A different layered-hexagonal material uses the same command with its own
simulation configuration and image records. Unrelated materials or mounts are separate fit groups.
The optional `qualification_profile` names a frozen data-specific acceptance profile. The tracked
Bi2Se3 profile requires both `--heldout-integer-l 4 11` and `--benchmark`, and it fails closed if the
initial indexed-manifest hash changes. A generic manifest omits the profile: the command then reports
numerical run completion without applying or claiming the Bi2Se3 thresholds.
The primary fit always uses every frozen key; `--heldout-integer-l` requests a separate training
refit and held-out prediction report without changing that primary data set.
Every geometry coordinate is active by default. Repeat `--freeze-parameter NAME` to hold any
coordinate at its configured base value; at least one coordinate must remain active. For example,
to use calibrated detector tilts without refitting them:

```powershell
uv run --frozen python scripts/fit_osc_geometry.py `
  configs/bi2se3_osc_geometry_fit.yaml `
  --freeze-parameter detector_column_tilt_rad `
  --freeze-parameter detector_row_tilt_rad `
  --json
```

The JSON fit record lists canonical `fitted_parameter_names` and `fixed_parameter_names`. Beam
center and lattice constants are calibration/material inputs, not switchable fit coordinates.
The objective evaluates one ideal nominal incident state and no mosaic, structure intensity,
raster, or pixel integration. JSON reports the frozen manifest, pooled/per-image metrics, rank and
weak direction, active bounds, held-out predictions, independent-root audit, and frozen-candidate
measured audit. Acceptance relabels the unchanged selected native candidates; the separately
reported fresh global rediscovery is an operational chart/candidate-stability diagnostic and cannot
replace fitted observations.

## Recover the Bi2Se3 mosaic distribution without pixelizing the fit

Run the tracked deterministic 5, 10, and 15 degree synthetic recovery into a directory outside the
repository:

```powershell
uv run --frozen python scripts/recover_bi2se3_mosaic.py `
  --output-directory C:\path\outside\the\repository\mosaic-recovery
```

The case fixes the accepted nine-coordinate geometry, beam center, and lattice. Geometry marker
centers use one exact source-center, zero-divergence, mean-wavelength companion state, but that state
contributes no detector intensity. By default each incidence instead reduces the same 250 sampled
positions, directions, and wavelengths into one detector function before profile comparison. The
fit varies only Gaussian sigma, Lorentzian HWHM, and mixture probability. Every individual profile
receives an exact profiled nuisance amplitude, so absolute peak heights, structure-factor
amplitudes, and cross-reflection intensity ratios do not weight the answer.

The joint fit uses 32 profiles across all three datasets: the frozen 10/8/8 indexed nonzero
selection plus `00L={003,006}` at 5 degrees, `{006,009}` at 10 degrees, and `{006,009}` at 15
degrees. All six `m=0` profiles have raw OSC support and positive signal in the fixed top-exit
simulation. Raw-significant `003` at 10 and 15 degrees is recorded but not fitted because the
current forward evaluator predicts zero signal throughout those two profile windows. A frozen
kernel-certified geometry audit excludes 43 inverse-support boundary bins identically from truth
and every response component; all 32 profiles remain, with at least 38 of 41 bins apiece. General
topology-split cubature that retains those bins remains future work.

The three 3,000 x 3,000 PNGs visualize the configured truth forward model without planted nuisance
amplitudes; they are neither recovered-model renders nor fit inputs. The JSON, numeric diagnostic,
distribution comparison, and images are generated artifacts and must remain outside the repository.
Use `--skip-images` for the faster numerical recovery check.

To fit the same fixed geometry to the three detector-native Bi2Se3 OSC images, run:

```powershell
uv run --frozen python scripts/recover_bi2se3_mosaic.py `
  --observation-mode osc `
  --source-sample-count 250 `
  --skip-images `
  --output-directory C:\path\outside\the\repository\bi2se3-real-mosaic
```

The tracked `mosaic_fit_measured_policy.toml` binds the selection to the immutable case. It excludes
each weak profile as a whole when its central excess energy is less than five times the local
sideband scatter, and excludes both 10-degree `m=1,L=4` profiles as the paired secondary-lobe case.
No central-profile bin is intensity-masked. The surviving `m=0` set is `003/006` at 5 degrees,
`006` at 10 degrees, and `006/009` at 15 degrees. Every surviving peak still receives an
independent nonnegative amplitude, so the fit uses shape rather than absolute or cross-peak
intensity. The output is an effective common radial envelope: uncalibrated detector/source
resolution and remaining forward-model disagreement prevent interpreting it as a unique intrinsic
mosaic distribution or a statistical confidence interval.

## Recover fixed-position Bi2Se3 occupancies and directional displacement

After accepting the geometry and mosaic stages, run the deterministic three-incidence structure
recovery with the accepted external mosaic result:

```powershell
python scripts/recover_bi2se3_ordered_intensity.py `
  --source-sample-count 250 `
  --mosaic-result C:\path\outside\the\repository\bi2se3-real-mosaic\bi2se3_real_mosaic_fit.json `
  --output C:\path\outside\the\repository\bi2se3-sf\ordered_intensity_result.json `
  --json
```

The tracked case uses one shared 250-state source realization at 5, 10, and 15 degrees, the accepted
nine geometry corrections, and the supplied recovered mosaic. With the shown measured-mosaic
handoff, its exact fit-eligible selection is authoritative: `2/5/8 = 15` centers, including five
`00L` anchors, enter one joint fit. Running without `--mosaic-result` is the separate full-catalog
synthetic proof and retains `88/78/72 = 238` centers and six `00L` anchors. Each view produces one
incoherently source-summed detector function before any comparison. Bi and Se2 Wyckoff coordinates
remain exactly at their CIF values. The absolute proof
fits three occupancies plus `Ur/Uz`; the relative proof fixes `oBi=1`, reports the two Se/Bi ratios,
and solves one nonnegative scale per image analytically. The cached response uses the exact
occupancy quadratic, exact family `Qr`, and a 13-node `Uz` response certified at 12 interlaced
full-detector nodes. Its synthetic observations are selected-group point densities in `A^2/rad^2`,
not integrated peak masses or raw unseparated OSC counts.

To render the three raw OSCs beside three 3,000 by 3,000 source-averaged forward-model density
images after an accepted recovery, rerun with `--render-only`, both result JSON paths, and an
external `--render-directory`. The simulated PNGs sample the continuous detector function at pixel
centers; they are not pixel-integrated count predictions. Raw and simulated rows use separate
declared log transforms and are display-only, not intensity calibrated to each other.

```powershell
python scripts/recover_bi2se3_ordered_intensity.py `
  --render-only `
  --source-sample-count 250 `
  --mosaic-result C:\path\outside\the\repository\bi2se3-real-mosaic\bi2se3_real_mosaic_fit.json `
  --ordered-result C:\path\outside\the\repository\bi2se3-sf\ordered_intensity_result.json `
  --execution-backend cpu `
  --render-directory C:\path\outside\the\repository\bi2se3-250ki-images `
  --json
```

## Reference and observed data

- `examples/bi2se3/structures`: crystallographic inputs.
- `examples/bi2se3/observations`: compact detector/peak evidence.
- `examples/calibration/hbn`: detector calibration examples.
- `examples/pbi2`: stacking-transition examples.
- `reference`: immutable provenance and compact numerical reference packs.

Generated images, profiles, benchmark dumps, and diagnostics never belong under these directories or
elsewhere in the repository.
