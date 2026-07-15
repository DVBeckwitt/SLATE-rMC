# Bi2Se3 example

This example supports OSC decoding, detector-native geometry, refraction/attenuation, mosaic and
Ewald events, ordered rods, reflectivity, rod-family selection, and the later geometry, mosaic,
and ordered-intensity fit sequence.

The saved legacy state used `background_backend_rotation_k = 3`, equivalent to one clockwise
array rotation. It also named detector-native row as `x` and detector-native column as `y`.
The sanitized files never use that naming. Continuous coordinates are `(column_px, row_px)` and
arrays are indexed `[row, column]`.

The measured OSC files are gzip-compressed to keep the repository modest. A proof may stream them
through `gzip.open`; it must not create uncompressed copies under the repository root.
