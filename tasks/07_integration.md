# T07: native-detector integration (historical)

Status: `READY_CONTINUOUS_SUCCESSOR`.

The original sampled candidate/selection/deposition task was superseded by the continuous
detector-native implementation accepted on `main` through `0167f6dac79a66d79a94d358c041665bcb73ed38`.
It is retained only as a dependency marker for later planning.

Current authority is `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/RESULT_MEASURE.md`, and
`docs/DOVETAIL_MATRIX.md`. No work is authorized from the retired T07 algorithm. In particular,
do not restore candidate pools, sampled scattering events, selectors, detector-hit batches,
depositors, raster grids, or Ewald-sphere objects.

Accepted boundaries are the continuous `MosaicBraggSpace`, `ContinuousEwaldCoating`,
`DetectorEwaldMeasure`, and `SourceAveragedDetectorEwaldMeasure` APIs with detector-coordinate
density and explicitly named pixel or preview-macrobin measures.

Contract API v11 also admits the distinct optional
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1` terminal defined by D035. It samples only the
declared latent mosaic law, enumerates analytic roots, and streams weighted mass into exact native
pixel owners. It does not authorize any retired candidate, event, hit-table, selector, bilinear
depositor, or count-calibration contract.
