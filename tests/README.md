# Permanent test policy

The compact suite is organized by scientific boundary:

- `test_dream_rsi.py`: development-workflow isolation, immutable history, prefix-only
  replay, bounded policy execution/repair, policy selection, and global installation.
  This exercises no numerical physics and makes no model-service calls.

- `test_core_coordinates.py`: units, frames, immutable source/incident contracts, transforms, and
  external diagnostics.
- `test_geometry_optics.py`: canonical geometry, detector rays/tilts, refraction, attenuation, and
  incident transport.
- `test_mosaic_ewald.py`: mosaic normalization, physical rods, continuous Bragg density, analytic
  Ewald roots, elastic closure, and exact-family reduction.
- `test_ordered_reflectivity.py`: CIF amplitudes, finite ordered stacks, and named specular models.
- `test_stacking_transition.py`: finite stacking correlations, parent limits, and normalization.
- `test_fitting.py`: exact tagged-site geometry, detector-function objectives, rank checks, and
  blind bounded recovery, including independently switchable coordinates and named ordered bounds.
- `test_integration.py`: continuous detector inverse mapping, caustics, native-pixel integration,
  source averaging, YAML construction, compiled kernels, CUDA parity, and end-to-end factor
  ownership.
- `test_reciprocal_profiles.py`: material-neutral layered reciprocal coordinates, explicit
  detector-side/sideband membership, immutable payloads, and separate finite-bin signal/measure
  accumulation, plus continuous chart cubature, native-count projection with full covariance, and
  fixed-background anchor conditioning.
- `test_matched_regions.py`: material-neutral joint matched-region nuisance profiling and structure
  recovery with one dataset scale shared across families.
- `test_fixed_experiment.py` and `test_compose_fixed_experiment_cli.py`: strict modular position,
  lattice, and provided-mosaic composition, exact image/angle binding, and immutable physics reuse
  across incidence views.
- `test_osc_lattice_cli.py` and `test_fixed_lattice.py`: data-only lattice promotion gates and the
  immutable retained/accepted full-basis handoff into mosaic and structure construction.

- `test_native_execution.py`: exact completed raw recovery after a partial out-of-order batch.
- `test_native_search.py`: scaled bounded TRF, calibrated multistarts, exact nuisance-scale
  profiling, discrete/profile ownership and lower unfinished evidence.
- `test_native_fitting.py`: Bi/Pb site/mixture bindings, response dependency reuse, exact
  line masses, frozen proposal validity and continuous native pixel/batch equivalence.
- `test_native_workflow_cli.py`: hash-verified portable frozen-input adoption and complete
  image checkpoint/resume with explicit unqualified-candidate status.

Retain a test only when it protects a distinct scientific invariant, public contract, accepted
reference comparison, or integration boundary. Broad parameter sweeps, image snapshots, benchmarks,
private implementation checks, and superseded runtime behavior belong in external proof artifacts or
are deleted after use.
