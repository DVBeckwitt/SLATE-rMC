# T00: bootstrap

Branch: `chore/bootstrap`

## Goal

Create the minimal greenfield package and shared proof spine needed by all four physics worktrees. Do not implement the four subsystem models.

## Owned paths

```text
pyproject.toml
uv.lock or one equivalent committed lockfile
.codex/setup.sh
src/rasim_next/core/
src/rasim_next/io/orientation.py
src/rasim_next/proof/
src/rasim_next/__init__.py
tests/test_core_coordinates.py
tools/check_docs.py
this task's execution-plan and handoff sections
```

## Required implementation

1. `src/` package layout and narrow imports.
2. Runtime dependencies limited to NumPy, Gemmi, and XrayDB. Development dependencies limited to pytest and Ruff. SciPy and acceleration dependencies are deferred.
3. Shared contracts in `docs/CONTRACTS.md` with shape and value validation.
4. Shared `RigidTransform` point/vector operations, inverse, composition, and explicit pivot rotation.
5. Shared complex square-root branch selection, normal-wavevector function, and scalar interface amplitude.
6. OSC raw-to-detector-native index and array mapping, including the exact inverse.
7. Common validity codes, event IDs, model versions, measures, and trace records.
8. `python -m rasim_next.proof <name> --json` dispatcher.
9. `python -m rasim_next.proof references --json` for tracked pack, example hashes, and schema versions.
10. Trace comparator that identifies the first failing stage.
11. External-only atomic diagnostic writer producing one `.ra_diag.npz` with one JSON manifest.
12. Synthetic no-physics plumbing that passes IDs and arrays across every contract boundary without importing future modules.
13. Documentation link, task index, TOML, and YAML validation.

## Synthetic plumbing case

Use trivial values to prove this contract path:

```text
IncidentSampleBatch
    -> IncidentStateBatch
    -> RodCatalog
    -> ScatteringEventBatch
    -> RodQueryBatch
    -> EventIntensityResult
    -> OutgoingWaveBatch
    -> DetectorHitBatch
    -> PixelContributionBatch
    -> detector-native image
```

The plumbing test proves shapes, units, IDs, validity handling, and factor ownership. It contains no physical model.

## Proof

- transform round trips and composition
- complex branch propagating and evanescent cases
- scalar coefficient `n -> 1` limit
- non-square OSC marker forward and inverse mapping
- array `[row,column]` versus continuous `(column,row)` distinction
- diagnostic path rejection for every repository descendant
- one-file external diagnostic creation
- proof CLI emits one JSON object and correct exit status
- setup succeeds from a clean environment
- synthetic plumbing preserves event IDs and factor fields

## Commands

```bash
./.codex/setup.sh
python -m compileall -q src
ruff check src tests tools
pytest -q tests/test_core_coordinates.py
python -m rasim_next.proof core --json
python tools/check_docs.py
python -m rasim_next.proof references --json --allow-missing-pack
git diff --check
```

## Stop conditions

Stop `BLOCKED` if one shared contract cannot represent the synthetic full path without branch-specific fields. Record the smallest change. Do not implement a physics subsystem to work around the contract.

## Execution plan

State: NS

## Handoff

Status:

Commit SHA:

Contract API version:

Trace schema version:

Proof summary:

Dependencies and lock hash:

Known limitations:

Contract requests:
