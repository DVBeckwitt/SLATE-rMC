# Self-contained examples

These inputs are tracked so a clean clone can prove the forward subsystems and later staged
fitting without access to the original RASIM repository, manuscript checkout, or user paths.

- `common/osc` contains tiny non-square big- and little-endian RAXIS files for exact decoding
  and orientation tests.
- `bi2se3` contains the supplied structures, three compressed measured OSC images, a sanitized
  detector-native experiment configuration, legacy peak observations converted to explicit
  `(column_px, row_px)`, and VESTA-parity structure-factor references.
- `calibration/hbn` contains the supplied compressed calibrant and dark OSC images plus a compact
  calibration reference stripped of full derived images and absolute paths.
- `pbi2` contains 2H, 4H, and 6H structures and declared transition-model parameters.

Compressed `.osc.gz` files may be streamed directly or materialized to an external cache. Do not
decompress them into the repository. Example data are scientific inputs, not diagnostics.
