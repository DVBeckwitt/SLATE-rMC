# T02: geometry and optics

Branch: feat/geometry-optics

Start from PROOF_BASE_SHA.

## Goal

Produce correct OSC decoding, rigid instrument geometry, sample and detector intersections,
entrance and exit refraction, scalar local-field amplitudes, and absorption/attenuation for
individual rays and event batches.

## Owned paths

    src/rasim_next/io/osc.py
    src/rasim_next/geometry/
    src/rasim_next/optics/
    tests/test_geometry_optics.py
    this task handoff

Shared contracts, schemas, dependencies, reference evidence, examples, and other task files are
read-only.

## Scope

T02 owns:

- strict OSC signature, endian, dimension, payload, and high-range decoding;
- the single clockwise OSC-to-detector-native orientation boundary;
- explicit active, column-vector rigid transforms and ordered axis/pivot motion;
- finite sample-plane and rectangular anisotropic-detector intersections;
- continuous detector coordinates in (column_px, row_px) order;
- a shared complex-normal-wavevector branch for incident/exit refraction and decay;
- scalar interface amplitudes and manuscript uniform-depth attenuation;
- aligned incident/outgoing/hit batches, first-failure status, and optional public traces.

T02 does not own reciprocal rods/events, structure amplitudes, Parratt recursion, stacking,
detector deposition, branch selection, fitting, caking, GUI work, or acceleration frameworks.

## Required corrections

- Preserve the full rigid sample origin; never overwrite one transformed coordinate.
- Use the canonical OSC orientation exactly once.
- Use 2*k1z/(k1z+k2z) and |t_in*t_out|^2, not the old s/p power average.
- Use the manuscript uniform-depth attenuation average, not unconditional full-thickness entrance
  and exit damping.
- Treat any finite nonzero detector-point distance as a valid inverse ray; do not impose an
  arbitrary 1e-12 m rejection threshold.

## Permanent proof budget

The permanent tests protect five distinct boundaries:

1. OSC endian/high-range/orientation and malformed-input behavior.
2. Rigid instrument rotation order and pivots.
3. Sample/detector statuses, anisotropic coordinates, edges, and the close-detector regression.
4. Complex-k refraction, scalar amplitude, no-exit validity, and attenuation equations.
5. Batch identity, source/factor preservation, joins, first failure, and trace stages.

The proof command contains only:

- immutable-pack integrity and the seven T02 legacy classifications;
- synthetic and real OSC comparisons;
- direct line-plane and rigid-transform comparisons;
- detector round-trip and covariance invariants;
- independent complex-root, scalar-amplitude, quadrature, and critical-limit equations;
- one public transport-contract case;
- three compact boundary-convergence checks;
- seventeen bounded, fixture-sensitive coordinate/geometry/optics error injections;
- one equivalent-work scalar/vector benchmark with explicit numeric output and measured allocation
  bytes.

Broad generated corpora, private proof imports in tests, duplicate fixtures, full parameter sweeps,
and benchmark artifacts are intentionally not retained.

## Commands

    python -m compileall -q src
    ruff check src/rasim_next/io/osc.py src/rasim_next/geometry src/rasim_next/optics tests/test_geometry_optics.py
    pytest -q tests/test_geometry_optics.py
    pytest -q
    python -m rasim_next.proof geometry-optics --json
    python -m rasim_next.proof references --json
    git diff --check

## Handoff

Status: BLOCKED pending the reviewed shared tolerance policy. The implementation and all local
scientific gates pass; the proof command must not return READY while it consumes only a
branch-local tolerance candidate for shared-pack comparisons.

The exact commit SHA is reported externally because a commit cannot embed its own hash.

### Public APIs

- OSC: read_osc, OscImage, OscMetadata, OscFormatError.
- Instrument: AxisRotation, InstrumentConfiguration, CompiledInstrument, compile_instrument.
- Geometry: SampleIntersection, intersect_sample_ray, DetectorProjection, DetectorRay,
  project_detector_ray, detector_coordinate_to_ray.
- Transport: IncidentTransportResult, EventTransportResult, build_incident_states,
  transport_scattering_events.
- Optics: IncidentMode, ExitMode, solve_incident_mode, solve_exit_mode, mode_decay_constant,
  uniform_depth_attenuation, path_attenuation, scalar_optical_weight.

### Proof state

- Frozen base: 812f896fde5b8365ff5c218fc606df674ad7dcad.
- Reference pack SHA-256:
  e958703426ebea7a3fd62a8bb52447f9a5a8d7d5d4ad0eb0ce3b3706bbca1f06.
- Provisional branch-local tolerance SHA-256:
  866c40e1e68a3d2c08049d14f7531b78119047d5fb93ad06f17fb349082c1503.
- Eleven compact analytic/reference/interface/error-injection checks pass.
- Three boundary-convergence checks pass.
- Seventeen assigned coordinate/geometry/optics mutations are detected at their first affected
  scientific stage.
- The five permanent tests pass.
- A 512-ray independent-scalar-versus-vector benchmark agrees exactly in sample points, normal
  wavevectors, and statuses; its maximum complex-amplitude difference is 2.23e-16. The latest
  cleanup runs retained 86,528 numeric output bytes and stayed below a 0.5 MiB incremental
  allocation peak.

### Legacy classifications

- MATCH: osc.real.bi2se3, osc.synthetic.non_square, geometry.line_plane,
  optics.external_exit.
- CORRECTED: geometry.sample_origin_nonrigid at geometry.instrument_transforms;
  optics.interface_fresnel at measurement.optical_weight; optics.depth_attenuation at
  optics.uniform_depth_attenuation.
- NO_ORACLE: geometry.detector_rectangular_anisotropic,
  geometry.global_rigid_covariance, optics.critical_limit.

All T02-owned PHY-IO-*, PHY-GEO-*, PHY-OPT-001 through PHY-OPT-010, and PHY-THK-001/002 ledger
rows are attached to the compact classifications.

### Limitations and integration requests

- Before acceptance, publish a reviewed versioned shared stage-tolerance artifact, make T02 load
  it, and record its hash. The current candidate table is reproducible but has no proof-base
  provenance, so it cannot be frozen by this branch after inspecting its results.
- Exact-zero internal exit normal uses the documented +normal limiting side.
- Multilayer distorted fields, deposition, caking, and fitting remain outside T02.
- The tracked Bi2Se3 forward_case.toml uses zero sample width/length while the public geometry API
  requires explicit positive finite dimensions. T07 must replace those legacy values with physical
  finite dimensions or adopt a reviewed explicit unbounded representation; zero must not become a
  silent sentinel.
- Production transport currently consumes shared trace value types from the bootstrap proof
  namespace. Move those types to a shared non-proof layer when shared contracts next change.

### Branch reconciliation

The stopped validation worktree preserved correctness fixes absent from the divergent local
feat/geometry-optics ref, including the close-detector inverse-ray correction. Do not merge the
older divergent ref. Integrate the final cleanup SHA reported with this handoff, then deliberately
repoint or recreate feat/geometry-optics only after review.
