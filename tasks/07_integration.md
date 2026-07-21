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
