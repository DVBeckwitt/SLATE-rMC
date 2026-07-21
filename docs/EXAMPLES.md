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
  --integration-method adaptive_compiled `
  --worker-count 32 `
  --allow-unresolved-diagnostic `
  --output-dir C:\path\outside\the\repository
```

`--eta 0` selects a pure Gaussian and `--eta 1` selects a pure Lorentzian. Values strictly between
zero and one mix the two normalized densities. Every component with positive probability must
have a positive width; an inactive component may use zero width. The existing defaults remain a
5-degree Gaussian sigma, 2-degree Lorentzian HWHM, and eta 0.1. The example flag permits an
explicitly labelled, non-accepted diagnostic because the current zero-transverse-width rod model
contains detector-visible caustics; it does not claim quadrature convergence.
