# T03: mosaic and Ewald events

Branch: `feat/mosaic-ewald`

Proof base: `812f896fde5b8365ff5c218fc606df674ad7dcad`

## Goal

Produce deterministic source/orientation samples and film-side reciprocal-space events with explicit
probability mass, one coarea Jacobian, and an unsquared elastic residual.

## Owned paths

```text
src/rasim_next/sampling/
src/rasim_next/reciprocal/ewald.py
src/rasim_next/reciprocal/events.py
src/rasim_next/reciprocal/proof.py
tests/test_mosaic_ewald.py
this task file
```

## Read-only interfaces

- `IncidentSampleBatch`, `IncidentStateBatch`, `RodCatalog`, and `ScatteringEventBatch`
- `RigidTransform` and the shared frame identifiers
- immutable tracked reference pack and physics ledger

## Scope

- deterministic explicit-joint and independent source samples
- named `manuscript_axisymmetric_v1` folded-alpha orientation probability
- continuous-rod Ewald roots and branch-local statuses
- deterministic film-side event/status assembly
- compact analytic, tracked-reference, dense-oracle, and sparse-memory proof

## Non-goals

- rod construction or structure intensity
- entrance/exit optics, attenuation, polarization physics, footprint, or solid angle
- detector projection, caking, or `2theta/phi` mapping
- fitting, acceleration frameworks, GUI, CLI expansion, or generic SO(3) sampling
- shared contract, dependency, reference-pack, example, or dispatcher changes

## Interfaces and conventions

- `reciprocal_basis_Ainv` stores crystal-frame column basis vectors.
- `sample_from_crystal` is `CRYSTAL -> SAMPLE`; reciprocal vectors use its rotation only.
- `IncidentStateBatch.k_film_phase_sample_Ainv` is already the real film-side phase wavevector.
- T03 performs no entrance refraction or attenuation and returns film-side outgoing phase wavevectors.
- Wavelength is joined from `IncidentSampleBatch` by `incident_sample_id`.
- `reciprocal_weight = orientation_probability_mass / abs(kf_hat dot d_hat)` exactly once.
- Exact tangents emit no arbitrary finite-weight event; direct-beam roots are counted and suppressed.
- `polarization_state_id="unity_scalar"` is an explicit `UNITY_APPROXIMATION`, not a Stokes model.

## Compact acceptance evidence

- Source masses, correlations, ordering, complex rejection, and the 262,144-row allocation cap.
- Folded Gaussian/Lorentzian direct-alpha mass, atom-plus-continuous measure, and uniform azimuth.
- Stable line-origin-invariant two/tangent/no-root roots, direct suppression, residual, and unclipped
  Jacobian.
- Event IDs, order, frames, wavelength join, film-side ownership, and factor boundary.
- A 16-attempt sparse fixture with three events, checked field-for-field against an independent
  residual-scan/bisection oracle. Large memory/timing fixtures remain disposable handoff evidence.
- Immutable-pack hash plus tracked legacy density and two-sphere event invariants.

## Commands

```powershell
$env:PROOF_BASE_SHA = "812f896fde5b8365ff5c218fc606df674ad7dcad"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
uv run --frozen --group dev python -m compileall -q src
uv run --frozen --group dev ruff check src/rasim_next/sampling src/rasim_next/reciprocal/ewald.py src/rasim_next/reciprocal/events.py src/rasim_next/reciprocal/proof.py tests/test_mosaic_ewald.py
uv run --frozen --group dev pytest -q
uv run --frozen --group dev python -m rasim_next.proof mosaic-ewald --json
uv run --frozen --group dev python -m rasim_next.proof references --json
uv run --frozen --group dev python -m rasim_next.proof core --json
git diff --check
uv run --frozen --group dev python tools/check_docs.py
```

## Handoff

Status: locally merge-ready when the commands above pass on the clean containing commit. The exact
commit SHA and runtime-dependent raw proof JSON are reported in the external handoff.

Legacy classifications:

- `mosaic.ewald_intersection`: `MATCH`; no divergent stage.
- `mosaic.legacy_density`: `CORRECTED`; first divergence is `mosaic.probability_measure`.
- `mosaic.deterministic_source`: `NO_ORACLE`; no divergent stage.
- `mosaic.continuous_rod_events`: `NO_ORACLE`; no divergent stage.

Permanent tests retained: four combined public-behavior tests protect source sampling, spherical
mosaic measure, stable Ewald roots/Jacobian/direct suppression, and sparse event ordering plus the
T02/T03 interface boundary. Broad sweeps, fabricated mutations, trace scaffolding, allocator spies,
and repeat benchmark dumps are not retained.

Known limitations: no localized/adaptive acceleration or generic SO(3) sampler. Polarization remains
the explicit `UNITY_APPROXIMATION`. T02 exclusively owns exit refraction and detector projection.

Integration request: publish a versioned shared result-measure and tolerance contract before
cross-workstream integration. T03 retains only analytic and machine-precision proof gates and does
not add a shared or branch-local tolerance framework.
