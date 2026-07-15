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

State: NS

## Handoff

Status:

Commit SHA:

Public APIs:

Proof summary:

Legacy classifications:

First divergences:

Convergence:

Benchmark and peak memory:

Known limitations:

Contract requests:
