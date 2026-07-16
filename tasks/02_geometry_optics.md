# T02: geometry and optics

Branch: `feat/geometry-optics`

Start from `PROOF_BASE_SHA`.

## Goal

Produce correct OSC decoding, rigid instrument geometry, sample and detector intersections, entrance and exit refraction, scalar local-field amplitudes, and absorption/attenuation for individual rays and event batches.

## Owned paths

```text
src/rasim_next/io/osc.py
src/rasim_next/geometry/
src/rasim_next/optics/
tests/test_geometry_optics.py
this task's execution-plan and handoff sections
```

## Read-only inputs

- bootstrap core, contracts, trace schema, and proof CLI
- immutable tracked reference pack and examples
- in-repository source/equation ledger
- explicit `MaterialOptics` fixtures supplied through the shared contract

## Forbidden

- Miller indices, rod catalogs, or event generation
- structure factors, Parratt recursion, or stacking
- detector deposition or image accumulation
- branch and `Qr` selection
- fitting or acceleration frameworks
- edits to shared contracts, dependencies, or the reference manifest

## Source and equation map

Tracked locations:

```text
Original paths resolve under reference/legacy_source/.
Manuscript equation labels resolve under reference/manuscript/.
```


Inventory: `PHY-IO-*`, `PHY-GEO-*`, `PHY-OPT-001` through `PHY-OPT-010`, `PHY-THK-001`, `PHY-THK-002`.

Original:

```text
ra_sim/io/osc_reader.py:18-66
ra_sim/gui/_runtime/runtime_session.py:558-565
ra_sim/simulation/diffraction.py:285-436, 1682-1946, 2190-2316
ra_sim/simulation/intersection_analysis.py:80-615
```

Manuscript:

```text
geometry figures and modelling-methods ray construction
SI refraction equations near eq:si_ktz_solution_lambda
eq:si_scalar_transmission
eq:si_entry_exit_transmission
eq:si_transmission_intensity
eq:si_kappa_if_lambda
eq:si_imkz_pathlength_weight
eq:si_abs_complex_kz
eq:si_full_optical_weight_lambda
```

## Mandatory tasks

### GEO-01: OSC binary reader

Implement signature, version-dependent endian, dimensions, payload validation, high-range pixels, raw metadata, and canonical detector-native conversion. Reuse bootstrap orientation functions.

### GEO-02: instrument compilation

Compile user-facing goniometer, sample, crystal, and detector parameters into explicit `target_from_source` rigid transforms with declared pivots and order.

### GEO-03: sample intersection and footprint

Intersect rays with the sample plane. Return parallel, backward, outside-footprint, and valid statuses. Footprint acts on source probability mass, not structure intensity.

### GEO-04: detector geometry

Support rectangular detectors, independent row/column pitch, arbitrary pose, continuous forward ray-to-hit, and inverse coordinate-to-ray.

### GEO-05: entrance and exit modes

Use the bootstrap normal-wavevector function and branch selector. Conserve tangential wavevector. Reject non-propagating ambient exits rather than fabricating an angle.

### GEO-06: optical weight

Use scalar entrance and exit amplitudes and the manuscript uniform-depth attenuation average. Also expose path-length attenuation only when an explicit scattering depth/path is supplied. Do not implement full multilayer distorted fields.

### GEO-07: public trace

Return `IncidentStateBatch`, `OutgoingWaveBatch`, and `DetectorHitBatch` plus first-failure status. Emit the shared stage IDs.

### GEO-08: proof and benchmark

Compare to analytic equations and the immutable pack. Benchmark equivalent ray batches without adding a production backend.

## Expected corrections

- remove the non-rigid `P0_rot[0]=0` behavior
- centralize the OSC orientation mapping
- replace the old s/p power-transmission average with the scalar field coefficient
- replace old unconditional full-thickness entrance-plus-exit attenuation with the manuscript uniform-depth average for the current reference model

Each correction must identify the first divergent trace stage.

## Required proof

- OSC non-square markers, high-range pixel, and inverse mapping
- transform inverse, composition, pivots, order, and global rigid covariance
- sample and detector analytic intersections
- pixel-ray-pixel round trip
- tangential conservation and dispersion residual
- propagating, critical, evanescent, and no-exit cases
- `n=1` scalar coefficient limit
- attenuation zero-decay and thick-film limits
- legacy individual-ray cases with first divergences
- convergence near parallel planes, critical angle, and detector edges

## Overnight completion rule

`READY` means the mandatory manuscript/reference slice, analytic proof, immutable-pack comparison, convergence, and public contracts all pass. Record unimplemented generalization and acceleration as an explicit extension backlog. Do not mark a scientifically required reference case as optional merely to finish.

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/io/osc.py src/rasim_next/geometry src/rasim_next/optics tests/test_geometry_optics.py
pytest -q tests/test_geometry_optics.py
python -m rasim_next.proof geometry-optics --json
python -m rasim_next.proof references --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the common instrument, coordinate, material-optics, or trace contract cannot represent the correct result. Do not modify it in this branch.

## Execution plan

State: local recovery complete; central blockers remain.

- Objective: recover only GEO-01 through GEO-08 by preserving exact incident first failures,
  using canonical trace stages, and making proof status respect clean-tree state.
- Non-goals: no event generation, new physics, new public API, corpus growth, integration,
  fitting, caking, GUI, acceleration, dependency, or shared-contract change.
- Affected files: `geometry/transport.py`, `geometry/proof.py`,
  `tests/test_geometry_optics.py`, and this execution-plan/handoff.
- Interfaces: public signatures and shared batch shapes remain unchanged. Missing shared event
  status, production-neutral trace values, and versioned tolerances are contract requests.
- Tests: modify the existing transport invariant only; use one-shot proof/schema/clean-tree checks
  for proof-only corrections; run every original T02 command plus the full suite.
- Risks: event failures cannot retain a reason from a boolean-only shared contract; proof acceptance
  cannot be frozen without centrally reviewed tolerances.
- Acceptance: incident failures remain exact through outgoing/hit results; all emitted stages are
  canonical; dirty science never reports PASS; clean scientific proof passes; owned diff and
  repository residue checks pass.

## Handoff

Status: BLOCKED. The bounded GEO-01 through GEO-08 recovery is locally correct, but three shared
merge requirements remain unresolved.

The exact commit SHA is reported externally because a commit cannot embed its own hash.

### Recovery

- Starting HEAD: `4a2388004f39d0775f42b6d3deeed376320daa80`.
- Incident first-failure codes now remain exact through outgoing-wave and detector-hit results.
  The existing transport regression also proves lossless `RESIDUAL_EXCEEDED` propagation.
- Proof-only OSC stages now use the declared `osc.raw_array` ID; no `osc.raw_counts` stage remains.
- Scientific pass plus a dirty tree reports `BLOCKED`; scientific pass plus a clean tree reports
  `PASS`. Scientific failure reports `FAIL`.
- Public APIs and shared batch shapes are unchanged. No shared contract, dependency, reference,
  example, root configuration, or other workstream file was edited.

### Local gates

- The focused regression failed before the status fix and passes afterward.
- Compile, lint, format, five permanent T02 tests, and the ten-test full suite pass.
- Core proof passes all five checks; reference proof passes all seven checks.
- Geometry proof passes 11/11 scientific checks, 3/3 convergence checks, and 17/17 required
  mutations. Its dirty-tree decision is `BLOCKED` as required; the clean post-commit result is
  reported externally.
- All 20 proof comparison-stage call sites use IDs declared by `docs/TRACE_SCHEMA.md`.
- The 512-ray equivalent-work benchmark agrees exactly in sample points, normal wavevectors, and
  statuses. Maximum complex-amplitude error is 2.23e-16; retained numeric output is 86,528 bytes;
  incremental tracemalloc peak remains below 0.5 MiB.
- No new branch-local tolerance artifact was added. The pre-existing embedded candidate was
  exercised only to preserve provisional evidence; it is not accepted proof authority. Shared-pack
  acceptance remains blocked until the reviewed shared artifact exists.

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

### Central requests

1. Add an aligned, lossless first-failure status to the shared `ScatteringEventBatch`, require
   `valid == (status == VALID)`, and bump its shared schema/version. T03 must supply the event
   reason; T02 must copy it only after an incident remains valid.
2. Move `TraceRecord`, `Measure`, and `QuantityKind` from the proof namespace to one
   production-neutral shared core module. Keep tolerance and comparator machinery proof-only.
3. Publish a reviewed, versioned shared stage-tolerance artifact with canonical stage keys, scale
   semantics, and a stable hash. T02 must load that shared artifact before reference-pack acceptance.
