# Bi2Se3 example

This example supplies the three detector-native OSC files, the R-centered Bi2Se3 configuration,
and the measured-region and five-coordinate structure-fit recipes. The production state retains
epsilon zero, while the shared RichEpsilon evaluator also proves bounded nonzero 3R departures.
Continuous
coordinates are `(column_px, row_px)` and arrays are indexed `[row, column]`; OSC orientation is
converted exactly once at input.

The current workflow is:

1. fit the shared geometry, one common incident-angle offset, and zero-sum image trims;
2. optionally run the tightly bounded lattice-sensitivity stage;
3. combine position, lattice, and the tracked hash-bound `experiment/fixed_mosaic.json` into one
   fixed checkpoint;
4. prepare measured mixed-chart regions and calibrate the shared radial background;
5. either run the v7 A (Wyckoff positions), B (outer-chalcogen vacancy fraction), C (sample-Q
   envelope), and joint chain, or run the tracked v8 explicit-start joint-only policy;
6. evaluate continuous full-branch profiles and render the measured detector ROI.

Crystallographic site ADPs are fixed in the atomic amplitude. The fitted sample-Q envelope is a
separate `exp(-U_r Q_r^2-U_z Q_z^2)` intensity factor. The model is never rasterized or smoothed;
native pixels remain only for measured counts and detector display. Their piecewise-constant count
field is integrated over the same continuous rectangles with full propagated covariance. See
`docs/EXAMPLES.md` for the complete resumable commands and `docs/VALIDATION.md` for the historical
pre-cleanup model-limited result and current admission requirements.
