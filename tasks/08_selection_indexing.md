# T08: post-integration selection and indexing

Status: FUTURE. Begin only after T07 passes.

Branch: `feat/selection-indexing`

## Goal

Create stable rod-family, reflection-group, and physical branch identities and associate measured detector-native observations without embedding identity changes inside an optimizer.

## Owned paths

```text
src/rasim_next/selection/
tests/test_selection.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/utils/calculations.py:48-62,90-117
    ra_sim/gui/geometry_q_group_manager.py:1197-1329
    ra_sim/fitting/caked_geometry_objective.py:188-228

manuscript
    sections/refinement_workflow.tex:29-59
    2D_Supplemental/SI_failure_modes.tex:691-748
```

## Required work

1. Preserve immutable `rod_id`, `phase_id`, reciprocal-cell revision, and symmetry provenance.
2. Define exact radial-family identity. For hexagonal cells use integer `m=h^2+hk+k^2`; for general cells use a declared exact reciprocal-metric key.
3. Define an ordered reflection-group key that adds discrete `L` or equivalent out-of-plane identity.
4. Define branch from signed wrapped reciprocal azimuth in the declared sample/crystal in-plane basis, with explicit sign mapping and deadband.
5. Represent applicable `00L` cases as `branch_id=None` with `COLLAPSED_00L`.
6. Project candidates through the integrated forward model and associate measured peaks or ROIs using explicit gates and uncertainty.
7. Reject ambiguous associations or mark them unused.
8. Write an immutable, hashable `SelectionManifest` containing every relevant revision and candidate decision.
9. Implement re-index auditing between fit runs. Never change identity during an objective evaluation.
10. Characterize original-RASIM `m`, `(m,L)`, `Qr`, and signed-phi behavior under the canonical coordinate conventions.

## Proof

- exact family identity under input-order changes
- no collapse caused by rounded `Qr`
- branch invariance under detector display orientation
- expected transformation under a declared in-plane basis reversal
- deadband and `00L` behavior
- synthetic measured-to-predicted associations
- ambiguity rejection
- selection-manifest hash stability
- original-RASIM `MATCH` or `CORRECTED` classifications with first divergence

## Commands

```bash
python -m compileall -q src
ruff check src/rasim_next/selection tests/test_selection.py
pytest -q tests/test_selection.py
python -m rasim_next.proof selection-indexing --json
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the integrated event and rod contracts do not expose enough frame, identity, or projection metadata. Request the smallest shared-contract change rather than inferring identity from display pixels.

## Execution plan

State: FUTURE

## Handoff

Status:

Commit SHA:

Selection schema version:

Public APIs:

Legacy classifications:

Proof summary:

Known limitations:
