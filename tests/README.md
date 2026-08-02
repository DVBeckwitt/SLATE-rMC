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
- `test_fitting.py`: exact tagged-site geometry, detector-function objectives, rank checks, and
  blind bounded recovery, including independently switchable coordinates and named ordered bounds.
- `test_staged_fit_replay_cli.py`: strict portable Bi2Se3/Bi2Te3 cases, exact input identities,
  complete locked numerical runtime closure, pre-execution backend/dependency rejection, stage
  order/revision chaining, nested-path/loaded-case immutability, pre-persistence tolerance checks,
  and runtime-aware artifact-complete resume.
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
- `test_layered_quintuple_regions_cli.py`: exact A/B/C/joint lineage, implementation/cache identity,
  parameter-scaled identifiability, fit-conditioned policy, checkpoint integrity, and render
  publication contracts for the material adapter.
- `test_osc_lattice_cli.py` and `test_fixed_lattice.py`: data-only lattice promotion gates and the
  immutable retained/accepted full-basis handoff into mosaic and structure construction.

Retain a test only when it protects a distinct scientific invariant, public contract, accepted
reference comparison, or integration boundary. Broad parameter sweeps, image snapshots, benchmarks,
private implementation checks, and superseded runtime behavior belong in external proof artifacts or
are deleted after use.
