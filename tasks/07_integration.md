# T07: native-detector integration

Branch: `feat/integration`

Begin only after T06 approves the branch SHAs and any shared-contract changes are separately reviewed.

## Goal

Compose the four proven subsystems into one correct native-detector forward model without duplicating equations in orchestration code. Produce the reusable states required by later selection and fitting.

## Owned paths

```text
src/rasim_next/measurement/
src/rasim_next/render/
src/rasim_next/pipeline/
tests/test_integration.py
this task's execution-plan and handoff sections
```

## Mandatory integration order

1. Merge ordered rod catalog and prove it can feed mosaic under synthetic incident states.
2. Merge geometry incident states and feed mosaic event generation.
3. Feed event-aligned queries to ordered intensity.
4. Feed outgoing film wavevectors to geometry exit transport and continuous detector hits.
5. Assemble source, reciprocal, population, model, optical, footprint, polarization, solid-angle, and deposition factors exactly once.
6. Substitute stacking intensity under the same event contract.
7. Add phase mixtures only after single-phase mass conservation passes.
8. Run the tiny end-to-end legacy and independent cases.

## Required implementation

- one factor ledger matching `docs/RESULT_MEASURE.md`, including explicit polarization and phase/parent population semantics
- exact detector solid angle when required by the model measure
- mass-conserving non-square bilinear or exact pixel deposition
- detector PSF as a separate identity-default operator, if implemented
- distinct rod and phase sums without max normalization
- pure specular outputs kept separate from off-specular events until a declared detector composition step
- immutable compiled instrument, source, incident-state, rod-catalog, event, hit, and detector-response states
- proof traces containing all integration-stage factors
- profiling of full simulation and repeated intensity-evaluation workloads

## Tiny end-to-end case

Use a small non-square detector, explicit source samples, one wavelength, nonzero sample rotation, tilted detector, nonzero refraction, nonzero attenuation, one or two rods, deterministic mosaic quadrature, ordered intensity, and one stacking substitution. Record continuous hits, deposition indices and weights, total event mass, total detector mass, peak pixel, and selected pixel values.

## Merge gates

- all subsystem proofs still pass after each merge
- one canonical equation per operation
- one OSC conversion
- event IDs survive every stage
- no factor is duplicated or omitted
- total deposition mass is conserved under the declared clipping rule
- ordered and stacking use the same event-aligned interface
- legacy `MATCH` and `CORRECTED` cases are explained by stage traces
- no fitting, selection, caking, `2theta/phi`, or GUI code exists yet

## Commands

```bash
python -m compileall -q src
ruff check src tests
pytest -q
python -m rasim_next.proof all --json
python -m rasim_next.proof tiny-end-to-end --json
git diff --check
```

## Performance output

Record the dominant runtime and memory stages for:

- full forward rendering
- repeated ordered intensity evaluation with fixed event geometry
- repeated stacking intensity evaluation with fixed event geometry
- geometry-invalidating reevaluation of selected rays/rods

Recommend one production acceleration path. Do not implement a large backend framework in this task.

## Execution plan

State: NS

## Handoff

Status:

Integrated branch SHAs in order:

Proof summary:

Legacy classifications:

Mass and convergence evidence:

Compiled states:

Performance profile:

Recommended production path:

Remaining limitations:
