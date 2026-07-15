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

State: READY

Evidence:

- Owned paths are those listed above; all physics domains, `measurement/`, `render/`, `pipeline/`,
  tracked evidence, schemas, root contracts, and other tasks are forbidden.
- Cover PHY-IO-003 through PHY-IO-005, PHY-SRC-005, PHY-REC-009, PHY-MEA-001, and PHY-PHA-002.
  Inspect only the tracked OSC source location and `osc.synthetic.non_square`; implement no physics.
- Retain one implementation each of the rigid transform, complex normal branch, scalar interface
  amplitude, clockwise OSC mapping, T02--T05 boundary dataclasses, trace comparator, external
  diagnostic writer, proof dispatcher, and one no-physics event flow.
- Prove transform identities, direct clockwise enumeration, dispersion/decay signs, equal-medium
  amplitude, event-ID/factor conservation, and first-stage detection for the assigned mutations.
- Done requires the frozen environment and every T00 command to pass, one compact permanent test
  module, no subsystem code or unused scaffolding, and a clean coherent commit.

Commands run:

- `uv sync --frozen --group dev`
- `uv run --frozen --group dev python -m compileall -q src tests tools`
- `uv run --frozen --group dev ruff check src tests tools`
- `uv run --frozen --group dev pytest -q tests/test_core_coordinates.py`
- `uv run --frozen --group dev python -m rasim_next.proof core --json`
- `uv run --frozen --group dev python -m rasim_next.proof references --json --allow-missing-pack`
- `uv run --frozen --group dev python tools/check_docs.py`
- `git diff --check`

Remaining work:

- None for T00.

Contract or dependency issue:

- None. The lock resolves from public package indexes; SciPy is transitive through XrayDB, not a
  direct runtime dependency.

## Handoff

Status: READY

Commit SHA: this handoff is committed with the T00 implementation; the immutable proof-base SHA is
recorded after T01.

Contract API version: 4

Trace schema version: 4

Proof summary:

- Four focused tests pass for the coordinate, transform, shared-wave, contract-flow, trace, and
  external diagnostic requirements.
- The synthetic one-event flow preserves its stable event ID and applies each of eight declared
  factors exactly once. It includes no subsystem physics.
- All seven assigned T00 mutations are detected at their named first divergent stage.
- Five proof runs took 0.290408 s total (58.082 ms mean); traced peak memory was 2,426,246 bytes.

Dependencies and lock hash: NumPy, Gemmi, and XrayDB at runtime; pytest and Ruff for development;
`uv.lock` SHA-256 `71ae0bfcd7b8c59006191d45f84521497a8866f003fe05a078ce7f3fff8e4394`.

Known limitations:

- No geometry, Ewald, structure, reflectivity, stacking, rendering, or fitting implementation is
  present.
- Per the launch-scope correction, deposition remains proof-local; no speculative T07 public
  contract is part of the four-worktree interface.

Contract requests: none.
