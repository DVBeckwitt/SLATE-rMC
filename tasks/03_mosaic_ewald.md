# T03: mosaic and Ewald events

Branch: `feat/mosaic-ewald`

Start from `PROOF_BASE_SHA`.

## Goal

Produce equal-mass seeded source samples and correct reciprocal-space candidates with physical mass, event Jacobian, and elastic residual; do not select or render events.

## Owned paths

```text
src/rasim_next/sampling/
src/rasim_next/reciprocal/ewald.py
src/rasim_next/reciprocal/events.py
src/rasim_next/reciprocal/proof.py
tests/test_mosaic_ewald.py
this task's execution-plan and handoff sections
```

## Read-only inputs

- `IncidentSampleBatch`, `IncidentStateBatch`, `RodCatalog`, and `ScatteringEventBatch`
- immutable tracked reference pack and examples
- in-repository source/equation ledger

## Forbidden

- reciprocal-lattice or rod-catalog construction
- detector geometry or pixels
- CIF parsing, structure intensity, Parratt, or stacking
- interface optics
- weighted candidate selection or detector deposition
- fitting or acceleration frameworks
- shared contract or dependency edits

## Source and equation map

Tracked locations:

```text
Original paths resolve under reference/legacy_source/.
Manuscript equation labels resolve under reference/manuscript/.
```


Inventory: `PHY-SRC-*`, `PHY-MOS-*`, `PHY-REC-002` through `PHY-REC-009`.

Original:

```text
ra_sim/simulation/mosaic_profiles.py:15-63
ra_sim/simulation/diffraction.py:206-273, 565-1668,
2019-2600, 3113-3293
```

Manuscript:

```text
eq:mosaic_two_component_maintext
mosaic orientation-density equations in the supplement
modelling-methods Bragg-sphere and Ewald-sphere construction
detector bandwidth sum and wavelength-dependent geometry
```

## Mandatory tasks

### MOS-01: source samples

Implement explicit seeded randomized Latin-hypercube Gaussian sampling for position, divergence, and wavelength: independent dimensions, antithetic pairs, an odd central ray, equal empirical row mass, and reproducible count/seed. Use the direct `scipy.special.ndtri` dependency instead of a local inverse-normal approximation. Never multiply samples by their generating PDF. Keep deterministic Gauss–Hermite only as an oracle.

### MOS-02: orientation distribution

Implement independent Gaussian-like core width, Lorentzian-like tail width, and mixture weight. Normalize probability under the declared spherical measure without evaluating a singular point density at the pole.
Record `mosaic.wrapped_line_density` as `rad^-1`, frame-free `PROBABILITY_DENSITY`.

### MOS-03: dense independent oracle

Implement a transparent all-candidate support oracle using direct equations. It must not call the public candidate builder or render an image.

### MOS-04: Ewald support

Construct every physically valid candidate for each incident state and rod. A localized/adaptive builder is an extension unless proved against the all-candidate oracle. Handle two-root, tangent, no-root, and specular-family cases with real phase wavevectors.

### MOS-05: candidate mass

Calculate each candidate's `reciprocal_weight` as its wrapped-mosaic/Jacobian mass for T07's complete pool. Do not select candidates, assign per-selected-event mass, or raster pixels here.

### MOS-06: candidate contract

Emit candidate-aligned full sample-frame `Q`, exact sample-normal projection, `L`, outgoing film phase wavevector, `reciprocal_weight`, residual, source/orientation/rod IDs, exact status, and matching validity. Do not include source PDF weight, structure intensity, optics, solid angle, selection, or deposition.

### MOS-07: convergence and benchmark

Compare the candidate builder to the all-candidate oracle across narrow, broad, tail, tangent, and bandwidth cases; record construction convergence, timing, and peak memory before any localized replacement.

## Required proof

- fixed-seed source support, equal empirical row mass, antithetic pairing, and odd central ray
- source histograms recover declared distributions/correlations and preserve polarization-state IDs
- orientation normalization and azimuthal periodicity
- zero-width limit and finite pole behavior
- exact elastic residual
- tangent and no-root status
- all-candidate versus public support and total candidate mass
- legacy candidate cases with classifications and construction convergence

## Overnight completion rule

`READY` means the mandatory manuscript/reference slice, analytic proof, immutable-pack comparison, convergence, and public contracts all pass. Record unimplemented generalization and acceleration as an explicit extension backlog. Do not mark a scientifically required reference case as optional merely to finish.

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/sampling src/rasim_next/reciprocal/ewald.py src/rasim_next/reciprocal/events.py tests/test_mosaic_ewald.py
pytest -q tests/test_mosaic_ewald.py
python -m rasim_next.proof mosaic-ewald --json
python -m rasim_next.proof references --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the common incident-state, rod-catalog, event, or probability-measure contract is insufficient. Do not construct rods or detector pixels in this branch.

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
