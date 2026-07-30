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

## Publication reciprocal-to-detector mapping

Generate the reusable four-stage Bi2Se3 publication figure and its exploded projection schematic:

```powershell
.\.venv\Scripts\python.exe scripts/figures/render_bi2se3_publication_mapping.py `
  --output-directory C:\path\outside\the\repository\bi2se3-publication `
  --ewald-worker-count 4
```

The defaults use one mean incident beam at 10 degrees, Gaussian sigma 2 degrees, Lorentzian HWHM
0.2 degrees, and Lorentzian probability 0.1. The reciprocal panel takes 32 deterministic nodes from
the compiled, scale-resolved mosaic support, resolves the narrow Lorentzian center, extends to
nearly 180 degrees, and inserts every in-range integer-L finite-stack maximum and ±0.5/N shoulder
sample.

The Ewald panel evaluates the exact almost-everywhere intrinsic density at detector-native centers
and maps only the configured active-panel support onto the internal-film outgoing-direction sphere.
Every physical rod and every regular inverse preimage contributes, including regular nonzero
`m=0`; the collapsed direct `Q=0` root remains excluded by the reported positive top-exit Q gap.
The measure is `detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1`, where `sr` is
internal-film outgoing-`kf` solid angle. Detector coordinates supply only the visibility
parameterization: the field has no source, optical, attenuation, detector-coordinate Jacobian, or
detector-solid-angle factor. Positive values below the displayed eight-decade window use the
under-range color, exact positive fold caustics use the over-range color, and the faint unpainted
wireframe provides whole-sphere context. Spawned CPU processes evaluate deterministic row batches;
`--ewald-worker-count 1` selects the equivalent serial path.

To update only this Ewald figure without recomputing the reciprocal or either detector mapping,
add `--only-ewald`. This writes the standalone PNG/PDF and a focused provenance manifest:

```powershell
.\.venv\Scripts\python.exe scripts/figures/render_bi2se3_publication_mapping.py `
  --only-ewald --ewald-worker-count 8 `
  --output-directory C:\path\outside\the\repository\bi2se3-ewald
```

To regenerate only the exploded sphere-to-plane projection while preserving its established
camera, layout, colors, and guide styling, use `--only-schematic`. This skips reciprocal-space
sampling and the 1440 × 1440 standalone detector maps. The two detector textures are evaluated
directly at `--schematic-detector-cell-count`, while the Ewald patch still uses the spawned CPU
workers:

```powershell
.\.venv\Scripts\python.exe scripts/figures/render_bi2se3_publication_mapping.py `
  --only-schematic --ewald-image-size 720 --ewald-worker-count 8 `
  --schematic-detector-cell-count 480 `
  --output-directory C:\path\outside\the\repository\bi2se3-projection
```

The two planar panels independently evaluate the complete continuous detector-coordinate density on
1440 × 1440 center-sampled grids at the configured reference pose and at that pose plus an
explicitly illustrative 20-degree row-axis increment. For the default YAML the reference detector
tilt is zero. The increment is not a fitted calibration. Use `--tilt-column-deg` and
`--tilt-row-deg` to choose a different illustrative increment.

PNG and mixed vector/raster PDF outputs, four standalone panels, and a JSON provenance manifest are
written by default. PDF text, labels, and axes remain vector while dense intensity layers are
rasterized at the requested DPI. The detector panels are center-sampled displays of
`raw_detector_coordinate_density_A2_per_px2.v1`; they include the detector-coordinate Jacobian and
do not apply detector solid angle again. The exploded schematic textures the detector-visible Ewald
patch over a faint whole-sphere wireframe. Full-mode rendering linearly block-averages each
independently evaluated detector field to at most 480 × 480 cells for tractable 3D rendering; the
schematic-only route evaluates those textures directly at that reduced resolution. It never
averages in log space. The detector densities are evaluated at
their physical poses, but the exploded sphere/plane positions and scales are diagrammatic; only
their orientations are retained. The guides show a positive-`Qz`, exit-valid common-ray subset and
are not a shared reciprocal/metre coordinate construction. Separate
colorbars declare the sphere (`A^2/sr`) and detector (`A^2/px^2`) measures. Existing case files are
never silently replaced: use `--case-name` for another parameter set or pass `--overwrite`
intentionally.

## Final reciprocal-space figures

Generate the two final Bi2Se3 reciprocal-space views with one command:

```powershell
.\.venv\Scripts\python.exe scripts/figures/render_bi2se3_reciprocal_intensity.py `
  --output-directory C:\path\outside\the\repository\bi2se3-reciprocal
```

The first output shows the zero-mosaic reciprocal cylinders colored by the incoherent family sum
of the configured 52-layer structure strength. The second shows the continuous reciprocal-space
density produced by the normalized mosaic measure and the same structure strength; it is an
axisymmetric annular finite-volume field, not a collection of representative rotated cylinders.
Both use the physical hexagonal families `m=0`, `m=1`, and `m=3`. There is no physical `m=2`
family in this indexing convention.

Defaults retain the requested 10-degree incidence in configuration provenance and use Gaussian
sigma 2 degrees, Lorentzian HWHM 0.2 degrees, and Lorentzian probability 0.1. Because these are
intrinsic crystal-axis reciprocal-space plots, rigidly changing the beam/sample incidence does not
change their distribution; incidence does affect the Ewald and detector figures above. Rod
deposition is distributed across spawned CPU processes; set `--worker-count 1` for the
deterministic serial path. The continuous calculation is cached as a provenance-bound external
NPZ. Pass `--cache` to reuse a matching final-form cache or `--recompute` to replace it. Legacy
pre-consolidation caches are rejected because their axial SF integration was not converged. The
final 32,000-node axial rule gives at most 0.54% normalized-L1 and 0.64% peak error against a
200,001-node binned oracle for the three displayed families after smoothing. Fixed,
mass-conserving display smoothing is reported in the figure caption in radial and axial bins.

For smooth interactive inspection of the final precomputed Ewald-surface textures, run:

```powershell
.\.venv\Scripts\python.exe interactive/ewald_sphere_viewer.py --stride 1
```

The viewer keeps only positive-sample-`z` values, draws sphere coordinates aligned with the
incident wavevector, and provides reset plus sample-`x`, sample-`y`, and sample-`z` view controls.
Its textures are embedded so camera interaction does not rerun the expensive mappings.

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
  --output-directory C:\external\bi2se3-replay --backend cuda --through ordered_intensity
uv run --frozen python scripts/replay_staged_fit.py examples/bi2te3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2te3-replay --backend cuda
```

`--through geometry`, `--through mosaic`, and `--resume` retain strict stage order and scientific
revision chaining. A run may stop after any stage, but a later stage requires its verified immediate
predecessor; stages cannot be skipped by silently refitting or substituting case defaults. Resume
also requires the exact per-stage execution runtime recorded in the stage artifact. Each v2 stage
hashes only its declared inputs, configuration, expected result, and tolerances, so changing a later
stage does not invalidate a verified earlier checkpoint. Geometry uses the ideal source center,
zero divergence, and mean wavelength.
Mosaic and ordered intensity each use the case's identical 250-state realization and compare the
single fully reduced detector function per incidence; neither fit constructs 3,000 x 3,000 images.
Bi2Te3 can additionally render the six native images with `--through render`. Bi2Se3 render is
disabled because the available images bind a retired mosaic gate.

The frozen replay targets are:

- Bi2Se3 current verified terminal, ordered intensity. Geometry has eight shared corrections plus one common
  incidence delta, RMS/max `1.55894/5.65079 px`; `delta_theta_i=0.4197204 deg` gives effective angles
  `5.4197204/10.4197204/15.4197204 deg`. The measured mosaic fit recovered
  `(sigma_G, HWHM_L, eta)=(1.3228757 deg, 0.48989795 deg, 0.44809620)` with objective
  `1.68025891` from 15 profiles including five `m=0`. With Bi fixed as the relative-occupancy
  gauge, the real-OSC transferred-amplitude ordered fit gives
  `(Se1/Bi, Se2/Bi, Ur, Uz)=(1.19007892, 1.13233073, 0, 0)` and objective `9.71621227`.
  This is an unqualified boundary result: overall relative-residual RMS is `0.804828`, and the
  ten `m=1` profiles have RMS `0.984540`. It is not a calibrated physical occupancy result.
- Bi2Te3 geometry: detector tilts fixed, seven active coordinates, RMS/max
  `7.17509/15.07787 px`; measured mosaic `(1.0340984 deg, 0.66004834 deg, 0.34468382)` from 33
  profiles including three `m=0`; relative SF parameters
  `(Te1/Bi, Te2/Bi, Ur, Uz)=(1.0, 0.84487145, 0.1, 0.0)` from 31 profiles.

Exact verification covers input bytes, decoded OSC values, identity lists (including `m=0`),
source revision, masks, classifications, stage lineage, and decoded display pixels. Fit floats use
the declared absolute tolerances; PNG container bytes, paths, timings, memory, and device strings
are not scientific identity. The Bi2Te3 mosaic and ordered stages are current-lock CUDA-qualified;
the render has the separate historical qualification below. Both measured mosaic fits are
model-limited effective envelopes. Both ordered fits are `NO_ORACLE` transferred-amplitude
estimates, not direct count-calibrated raw-intensity recoveries.

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

All supported live viewers are indexed in [`interactive/README.md`](../interactive/README.md).

Open the native-pixel detector view with sample/detector pose controls:

```powershell
uv run --extra visualization python interactive/detector_viewer.py `
  --execution-backend cuda `
  --presentation-backend opengl
```

The default Bi2Se3 view uses the configured 1,000 incident-wavevector states, 49 Monte Carlo mosaic
draws per state, detector seed `20260728`, and the full 3,000 x 3,000 native detector. The two
numerical controls are **incident-ray samples** (`N_ray`) and **mosaic draws per ki state** (`M`). A
source row contains origin, direction, wavelength, and empirical weight; `M` controls only the
stochastic detector estimator. Geometry and draw controls coalesce at 5 ms cadence into latest-only
full-native previews with draw prefixes 1, 4, and 8; release appends the exact requested settled draw
count without restarting an accepted prefix. A newer revision cancels unfinished work, and a stale
frame cannot publish. Source-count changes still commit on release because they rebuild the source
bundle. The fixed Philox seed preserves source/draw prefixes across pose and draw-count changes,
reducing visual flicker while preserving the declared estimator. Press `R` to render, `0` to reset,
or `Q` to close.

Every geometry slider except absolute `theta_i` is a zero-based correction to the configured
geometry. The labels use the original RA-SIM and manuscript vocabulary while distinguishing
mechanical goniometer controls from effective end-pose controls:

- detector pitch is the current column-axis correction, equal to `-delta gamma` in the original
  RA-SIM sign convention; detector yaw is the row-axis `delta Gamma` correction;
- detector and sample in-plane rotations are distinguished as `delta chi_D` and `delta chi_S`;
  the latter is locally `-delta psi` in RA-SIM's sign convention;
- detector-normal distance is `delta D_n`; detector column/row translations remain physical
  millimetre translations and are only marked as coupled to manuscript beam-center `x0/y0`;
- goniometer-axis pitch `delta alpha` is RA-SIM `delta cor_angle`, and goniometer-axis yaw
  `delta psi_g` is RA-SIM `delta psi_z`. These reorient the sole configured commanded axis while
  preserving its motor angle and LAB pivot; they are not direct sample rotations;
- effective incidence is displayed as absolute `theta_i` from 0 to 20 degrees and converted to a
  configured-pose delta only at the viewer boundary; sample tilt remains `delta delta` (RA-SIM
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

CUDA execution and OpenGL presentation are separate explicit choices. CUDA keeps compiled physics
and full-native image buffers resident between compatible geometry revisions; the OpenGL path
uploads every native pixel as R32F and applies only the log color transform in its shader. Neither
path silently falls back. Use the complete full-native CPU/Matplotlib proof path explicitly when
needed:

```powershell
uv run --extra visualization python interactive/detector_viewer.py `
  --execution-backend cpu `
  --presentation-backend matplotlib
```

Override source count, settled draw count, or detector seed explicitly when desired:

```powershell
uv run --extra visualization python interactive/detector_viewer.py `
  --ki-samples 1000 `
  --draws-per-ki 49 `
  --seed 20260728 `
  --execution-backend cuda `
  --presentation-backend opengl
```

Programmatic progressive callers use
`detector.compile_monte_carlo_sampler(execution_backend="cuda", seed=...)`, then
`advance_preview_to(1)`, `advance_preview_to(4)`, and `advance_to(requested)`. Preview arrays are
leased presentation buffers and must be consumed before the sampler's next operation; only
`advance_to` returns the immutable authoritative float64 result. Select `"cpu"` instead for the
bounded software execution path.

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
  --fit-incidence-angle-delta `
  --freeze-parameter sample_normal_x_tilt_rad `
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
coordinate at its configured base value; at least one shared coordinate or the common delta must
remain active. `--fit-incidence-angle-delta` activates one scalar added to every commanded angle;
it is never independent by image and requires `sample_normal_x_tilt_rad` to be frozen because both
occupy the nominal incidence-axis gauge. For example,
to use calibrated detector tilts without refitting them:

```powershell
uv run --frozen python scripts/fit_osc_geometry.py `
  configs/bi2se3_osc_geometry_fit.yaml `
  --freeze-parameter detector_column_tilt_rad `
  --freeze-parameter detector_row_tilt_rad `
  --json
```

The JSON fit record lists canonical `fitted_parameter_names`, `fixed_parameter_names`, combined
`jacobian_parameter_names`, the common delta, and every commanded/effective incidence pair. Beam
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

The default standalone case fixes its legacy nine-coordinate geometry, beam center, and lattice.
The staged replay instead passes its verified `geometry.json` into this same runner, which extracts
the bound position revision, shared corrections, common incidence delta, and effective angles
atomically. Geometry marker
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
  --position-artifact C:\path\outside\the\repository\positions\geometry.json `
  --source-sample-count 250 `
  --skip-images `
  --output-directory C:\path\outside\the\repository\bi2se3-real-mosaic
```

Here `geometry.json` is the v2 stage artifact written by the earlier position-only staged replay;
raw correction values and an unverified revision cannot be substituted for it.

The tracked `mosaic_fit_measured_policy.toml` binds the selection to the immutable case. It excludes
each weak profile as a whole when its central excess energy is less than five times the local
sideband scatter, and excludes both 10-degree `m=1,L=4` profiles as the paired secondary-lobe case.
No central-profile bin is intensity-masked. The surviving `m=0` set is `003/006` at 5 degrees,
`006` at 10 degrees, and `006/009` at 15 degrees. Every surviving peak still receives an
independent nonnegative amplitude, so the fit uses shape rather than absolute or cross-peak
intensity. The output is an effective common radial envelope: uncalibrated detector/source
resolution and remaining forward-model disagreement prevent interpreting it as a unique intrinsic
mosaic distribution or a statistical confidence interval.

## Fit real-OSC Bi2Se3 relative structure parameters

After the verified position and mosaic stages, resume the portable case through ordered intensity:

```powershell
uv run --frozen python scripts/replay_staged_fit.py `
  examples/bi2se3/experiment/staged_fit_replay.toml `
  --output-directory C:\external\bi2se3-replay `
  --backend cuda --through ordered_intensity --resume
```

The tracked `ordered_intensity_fit_measured.toml` transfers all 15 identity-matched nuisance peak
amplitudes extracted from the three physical-pixel OSC projections. It multiplies each by the
corresponding baseline 250-source continuous peak-center response and jointly fits `Se1/Bi`,
`Se2/Bi`, `Ur`, and `Uz`, with one analytic scale per incidence. The simulated distribution is
never detector-rasterized; measured detector pixels enter only through the upstream exact-overlap
profile amplitudes. Geometry, the common incidence-angle delta, mosaic, atomic positions, and the
52-layer nearly-perfect 2H state remain frozen. The stage may be resumed independently, but its
verified geometry and mosaic predecessors must already exist or be supplied.

The implementation keeps preparation and fitting as separate callable boundaries:
`prepare_measured_ordered_inputs` validates and constructs the fixed upstream state, while
`fit_prepared_bi2se3_measured_ordered_document` consumes that prepared state and returns the
standalone fit artifact. The response compiler and inverse solver remain reusable core functions.
The present end-to-end parameter mapping is intentionally Bi2Se3 quintuple-layer-specific; a
material-neutral structural-coordinate interface is later work, not an implied result of this fit.

The present result is deliberately fail-closed as
`MODEL_LIMITED_REAL_OSC_STRUCTURE_ESTIMATE_NO_ORACLE` with adequacy
`UNQUALIFIED_BOUNDARY_HIGH_RESIDUAL`. Do not interpret its normalized
`(Bi,Se1,Se2)=(0.84028,1,0.95148)` representative as absolute occupancy; it is only the admissible
gauge representation of the two fitted ratios.

## Run the synthetic fixed-position Bi2Se3 structure proof

After accepting the geometry and mosaic stages, run the deterministic three-incidence structure
recovery with the accepted external mosaic result:

```powershell
python scripts/recover_bi2se3_ordered_intensity.py `
  --source-sample-count 250 `
  --mosaic-result C:\path\outside\the\repository\bi2se3-real-mosaic\bi2se3_real_mosaic_fit.json `
  --output C:\path\outside\the\repository\bi2se3-sf\ordered_intensity_result.json `
  --json
```

This standalone command is the synthetic identifiability and response-compiler proof, not the
real-OSC transferred-amplitude fit above. Its tracked case uses one shared 250-state source
realization at 5, 10, and 15 commanded degrees,
the exact position state embedded in the supplied mosaic artifact, and the supplied recovered
mosaic. The ordered runner reconstructs the recorded effective angles and rejects a missing or
incompatible position revision. With the shown measured-mosaic
handoff, its exact fit-eligible selection is authoritative: `2/5/8 = 15` centers, including five
`00L` anchors, enter one joint fit. Running with the explicit `--synthetic-truth-proof` flag instead
of `--mosaic-result` is the separate full-catalog synthetic proof and retains `88/78/72 = 238`
centers and six `00L` anchors. Each view produces one
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
