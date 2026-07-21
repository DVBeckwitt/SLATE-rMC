# Permanent test policy

The compact suite is organized by scientific boundary:

- `test_core_coordinates.py`: units, frames, immutable source/incident contracts, transforms, and
  external diagnostics.
- `test_geometry_optics.py`: canonical geometry, detector rays/tilts, refraction, attenuation, and
  incident transport.
- `test_mosaic_ewald.py`: mosaic normalization, physical rods, continuous Bragg density, analytic
  Ewald roots, elastic closure, and exact-family reduction.
- `test_ordered_reflectivity.py`: CIF amplitudes, finite ordered stacks, and named specular models.
- `test_stacking_transition.py`: finite stacking correlations, parent limits, and normalization.
- `test_integration.py`: continuous detector inverse mapping, caustics, native-pixel integration,
  source averaging, YAML construction, compiled kernels, CUDA parity, and end-to-end factor
  ownership.

Retain a test only when it protects a distinct scientific invariant, public contract, accepted
reference comparison, or integration boundary. Broad parameter sweeps, image snapshots, benchmarks,
private implementation checks, and superseded runtime behavior belong in external proof artifacts or
are deleted after use.
