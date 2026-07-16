# T03: mosaic and Ewald events

Branch: `feat/mosaic-ewald`

Start from `PROOF_BASE_SHA`.

## Goal

Produce deterministic source/orientation sampling and correct reciprocal-space scattering events with explicit probability mass, event Jacobian, and elastic residual.

## Owned paths

```text
src/rasim_next/sampling/
src/rasim_next/reciprocal/ewald.py
src/rasim_next/reciprocal/events.py
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
- branch and `Qr` selection
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

Implement deterministic spatial, directional, wavelength, and declared polarization-state samples with explicit joint or independent correlation semantics and integrated probability mass. A unity polarization assumption must be explicit metadata, not absence of data.

### MOS-02: orientation distribution

Implement independent Gaussian-like core width, Lorentzian-like tail width, and mixture weight. Normalize probability under the declared spherical measure without evaluating a singular point density at the pole.

### MOS-03: dense independent oracle

Implement a transparent dense support calculation using direct equations. It must not call the public localized solver.

### MOS-04: Ewald support

Construct the valid support for each incident state and rod. The overnight public reference may use the dense deterministic construction when it is converged; a localized/adaptive production solver is an extension unless completed and proved. Handle two-root, tangent, no-root, and specular-family cases. Use real phase wavevectors.

### MOS-05: deterministic quadrature

Integrate orientation probability and any required geometric Jacobian into `reciprocal_weight`. Do not use secondary random or quantile resampling in proof mode.

### MOS-06: event contract

Emit event-aligned internal `Q`, `Qz`, `L`, outgoing film phase wavevector, weight, residual, IDs, and validity. Do not include source weight, structure intensity, optics, solid angle, or deposition.

### MOS-07: convergence and benchmark

Compare the public method to the dense oracle across narrow, broad, tail, tangent, and bandwidth cases. When the public method is the dense reference, record its convergence and baseline timing. Any localized/adaptive method must agree before it replaces the reference path.

## Required proof

- source and spectrum weight normalization
- preservation of declared source correlations and polarization-state IDs
- orientation normalization and azimuthal periodicity
- zero-width limit and finite pole behavior
- exact elastic residual
- tangent and no-root status
- dense versus public integrated event mass
- no quantile-resampling bias
- legacy support and event cases with classifications
- convergence of total mass, event centroid, and selected quantiles

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

State: local recovery complete; central proof-path ownership and the shared tolerance/measure artifact remain blocking.

1. Preserve the correct public source, mosaic, Ewald, and count-then-fill event implementation.
2. Fix MOS-01 metadata preservation under the existing source contracts.
3. Reduce MOS-02..07 evidence to one compact oracle matrix, one real refinement, seven proof-only controls, and one sparse measured benchmark.
4. Run the original and repository-wide proof gates without committing unowned proof-path work. Complete.

## Handoff

Status: BLOCKED pending central ownership of `src/rasim_next/reciprocal/proof.py` and a reviewed shared stage-tolerance/result-measure artifact.

Commit SHA: `9ef8d9a7149b69c1f0c7ab13c528f905ec0340fc` (recovery changes intentionally uncommitted).

Public APIs: deterministic joint/independent source compilation with preserved polarization/correlation metadata, wrapped axisymmetric mosaic quadrature, continuous-rod Ewald roots, and count-then-fill event/status assembly.

Proof summary: the compact T03 suite passes 4/4 and the full suite passes 9/9. Compile, Ruff, documentation, core proof, reference proof, and `git diff --check` pass. Schema-v1 validation passes under an explicitly schema-only clean-guard substitution; the real and poisoned-Git T03 proof commands correctly reject the dirty checkout. The compact proof retains five named oracle regimes, seven detected controls, and one refinement. `proof.py` is 1,083 lines, `+487/-191` versus HEAD (net `+296`, within the PM ceiling).

Legacy classifications: `mosaic.ewald_intersection` is `MATCH`; `mosaic.legacy_density` is `CORRECTED`; deterministic source and continuous-rod events are `NO_ORACLE`.

First divergences: the corrected legacy density first diverges at the public wrapped-line-density calculation, before `mosaic.probability_measure`; central approval of the stage name is pending. The `MATCH` and `NO_ORACLE` cases have no divergence stage.

Convergence: alpha-cell levels 4, 8, and 16 preserve total reciprocal mass `2.5467472306046544`; the maximum successive selected-qz-quantile change decreases from `0.0058602352443966055` to `0.0014933815676112516 A^-1`. Acceptance remains blocked until the shared tolerance artifact exists.

Benchmark and peak memory: 4,096 attempted lines produce 3 events and 1 suppressed direct root, with every event/status field matching the 65-node independent dense oracle. Public time is `2.385948400013149 s`; dense-oracle time is `0.0718578000087291 s`. Returned numeric output is `139,603 B`; traced live/peak/temporary-working memory is `175,131/184,280/9,149 B`. The eliminated maximum-root event preallocator alone was `786,432 B` for this fixture.

Known limitations: `UNITY_APPROXIMATION` preserves declared polarization IDs but adds no polarization physics; no localized/adaptive accelerator or generic SO(3) sampler. The shared `IncidentSampleBatch` constructor still accepts a truthy non-string correlation value when called directly; T03 public compilers reject it locally.

Contract requests: centrally amend T03 ownership to include `src/rasim_next/reciprocal/proof.py`; approve a shared wrapped-line-density trace stage; publish a reviewed versioned shared stage-tolerance/result-measure artifact and require its hash before shared-pack acceptance. Integration must squash the two existing branch commits onto `812f896fde5b8365ff5c218fc606df674ad7dcad`; history is not rewritten locally. Recovery changes remain intentionally uncommitted until proof-path ownership is resolved.
