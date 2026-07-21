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

## Interactive continuous detector viewer

Open a medium-resolution detector view with live sample/detector pose controls:

```powershell
uv run --extra visualization python scripts/interactive_detector_viewer.py
```

The viewer starts with 25 requested incident-ray phase-space samples and 128 continuous
detector-coordinate display samples per axis for the settled field. The two numerical controls are
named **incident-ray samples** (`N_ray`) and **display samples per axis** (`N_disp`): a source sample
contains origin, direction, and wavelength, while the display count is only an ephemeral view of
the continuous function. While a pose control moves, one nominal incident ray samples the unbinned
function on a fixed 32 x 32 grid for responsive feedback. Releasing a changed control starts the
requested incoherent source-average render in a background thread; the title always distinguishes
the nominal preview from the completed requested source average. Press `R` to render, `0` to reset,
or `Q` to close.

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

Every view evaluates all elastically reachable physical `(h,k)` rods and all retained Ewald roots,
including only the certified detector-visible nonzero `m=0` support. The inverse pullback starts at
the active panel, so no off-detector Ewald mesh is built. Values are center samples of
`raw_detector_coordinate_density_A2_per_px2.v1`; the smoothly interpolated raster is display-only
and is not detector-pixel integration, a PSF, or a normalized physical result.
Configurations that disable detector-visible `m=0` are rejected because this viewer's contract is
the all-`m` physical-rod catalogue.

The requested-render backend inherits the explicit YAML setting. Override it without fallback when
desired:

```powershell
uv run --extra visualization python scripts/interactive_detector_viewer.py `
  --source-samples 25 `
  --display-samples-per-axis 128 `
  --backend cuda
```

The earlier `--ki-samples` and `--raster-size` spellings remain command-line aliases; the canonical
names state what is actually sampled.

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

## Reference and observed data

- `examples/bi2se3/structures`: crystallographic inputs.
- `examples/bi2se3/observations`: compact detector/peak evidence.
- `examples/calibration/hbn`: detector calibration examples.
- `examples/pbi2`: stacking-transition examples.
- `reference`: immutable provenance and compact numerical reference packs.

Generated images, profiles, benchmark dumps, and diagnostics never belong under these directories or
elsewhere in the repository.
