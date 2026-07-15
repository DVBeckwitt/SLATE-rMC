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
