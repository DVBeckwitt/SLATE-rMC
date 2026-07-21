# Example and reference policy

The repository is self-contained for proof. Worktrees use only tracked files.

## OSC

`examples/common/osc` proves binary parsing, both endian paths, high-range pixels, non-square shape,
clockwise orientation, inverse mapping, and pixel centers. The three Bi2Se3 files prove real 3000 by
3000 decoding. HBN and dark files support later detector calibration.

## Coordinate correction from legacy state

Original saved-state names were display oriented. In the supplied Bi2Se3 state:

```text
legacy center_x -> detector-native row
legacy center_y -> detector-native column
legacy background_detector_x -> detector-native row
legacy background_detector_y -> detector-native column
```

New files expose only `center_column_px`, `center_row_px`, `observed_column_px`, and
`observed_row_px`. The old names remain only in provenance columns.

## Structures

The R-3m legacy and VESTA CIFs must parse to equivalent structures. The expanded P1 file must yield
equivalent complex amplitudes. VESTA tables are parity references, not absolute truth.

## Reference pack

`reference/rasim_reference_v1.npz` contains compact stage intermediates for geometry, optics,
mosaic/Ewald, ordered/Parratt, stacking, and OSC. Its embedded manifest declares which legacy
outputs must match and where corrected implementations intentionally diverge.

## Canonical Bi2Se3 runtime inputs

`scripts/generate_bi2se3_detector_image.py` owns one pure `build_default_case_inputs()` boundary
for the exact source request, PCG64 seed/model provenance, transforms, detector calibration,
support model, and film thickness. The immutable `examples/bi2se3/experiment/forward_case.toml`
remains legacy provenance and is neither parsed as runtime configuration nor edited.

The legacy zero sample dimensions meant that finite footprint clipping was disabled. The runtime
maps that meaning to `unbounded_plane.v1` with absent width/length; zero is not a dimension
sentinel. This is `CORRECTED` relative to the former script-only finite rectangle, with the first
possible divergence at `geometry.footprint_acceptance`. Nothing under `examples/` or `reference/`
is regenerated to encode the correction.

## Continuous detector mosaic selection

The continuous Bi2Se3 detector generator accepts Gaussian sigma, Lorentzian HWHM, and the
Lorentzian mixture probability directly in degrees. For a pure 1-degree Gaussian mosaic:

```powershell
python scripts/generate_bi2se3_continuous_detector.py `
  --gaussian-sigma-deg 1 `
  --lorentzian-hwhm-deg 0 `
  --eta 0 `
  --layers 52 `
  --stacking-epsilon 0.001 `
  --integration-method adaptive_compiled `
  --worker-count 32 `
  --allow-unresolved-diagnostic `
  --output-dir C:\path\outside\the\repository
```

`--eta 0` selects a pure Gaussian and `--eta 1` selects a pure Lorentzian. Values strictly between
zero and one mix the two normalized densities. Every component with positive probability must
have a positive width; an inactive component may use zero width. The existing defaults remain a
5-degree Gaussian sigma, 2-degree Lorentzian HWHM, eta 0.1, seven layers, and ideal 2H. The example
uses 52 quintuple layers (49.636 nm from the CIF repeat) and the shared rich-parent disorder law
with parent probability 0.999 and four alternative probabilities of 0.00025 each. The diagnostic
flag permits an explicitly labelled, non-accepted image because the current zero-transverse-width
rod model contains detector-visible caustics; it does not claim quadrature convergence. Layer count
must be at least one, and stacking epsilon must be finite and lie in `[0, 1]`. This generator fixes
the finite-stack strength normalization to `FINITE_TOTAL`.

## One-file configured continuous simulation

`configs/bi2se3_simulation.yaml` is the strict, editable runtime declaration for the continuous
Bi2Se3 views. It owns the source, all four rigid transforms, detector, mosaic, finite-2H structure
factor, rod policy, numerical display controls, and three independent output switches. Run it with:

```powershell
uv run --frozen --extra visualization python scripts/run_configured_simulation.py configs/bi2se3_simulation.yaml
```

The default selects every physical rod in the elastic-reach union at the shortest realized
wavelength, preserves a stable master catalog while each source state evaluates only its reachable
subset, and sums all retained roots. This naturally includes detector-visible kinematic `m=0` and
every higher family that can contribute; unreachable or off-panel contributions remain zero. The
top-exit detector patch supplies the physical `m=0` direct-root support gap. Full-shell `m=0` and
dimensionless Parratt/composite reflectivity remain excluded from the raw Bragg mass.

The reciprocal-space and Ewald arrays are finite display samples of continuous callables. The
reciprocal view labels its color as latent `(alpha, beta, u)` intensity, not Cartesian `d^3Q`
density, and declares the nominal source wavelength used for that representative Bragg field.
The Ewald view uses one explicitly nominal mean incident state because wavelength-varying source
states do not share one Q-space Ewald surface. Its color is the intrinsic mosaic times per-rod SF
times one Ewald coarea factor; detector geometry is used only as a visibility mask, so no exit
optics or detector Jacobian enters that artifact.

The detector is the exact common sum domain: each state retains its wavelength, origin, `ki`, exit
refraction, attenuation, and structure factor before the continuous fields are summed. The default
60-pixel, order-2 macrobin image is explicitly a fast, non-quantitative fixed-quadrature preview of
that callable. It is intentionally suitable for interactive visualization, not quantitative fitting
or detector-mass convergence; change the two YAML detector quadrature controls for a different
preview, and use a separately convergence-certified integration path for fitted observables.

All PNGs plus one compressed `.ra_diag.npz` are written outside the repository. Change each
`outputs.<name>.enabled` value independently. Relative CIF and output paths are resolved against the
YAML file, not the process working directory; unknown, missing, duplicate, merged, aliased, or
non-finite configuration values are rejected.
