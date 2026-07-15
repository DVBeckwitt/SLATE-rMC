# Codex prompt: ordered rods and reflectivity

## Goal

Implement T04 completely in this worktree. Produce raw complex structure amplitudes, complete rod identities, material optics, finite ordered intensity, and separate specular outputs.

## Context and startup


Before editing:

```bash
uv sync --frozen --group dev
python scripts/verify_seed.py
python -m rasim_next.proof references --json
test -z "$(git status --porcelain)"
if test -z "$(git branch --show-current)"; then git switch -c feat/ordered-reflectivity; fi
```

Record the starting `HEAD` as the branch proof-base SHA. It must be identical across all four worktrees.

Use Plan mode first and write the no-edit plan in the assigned task. The main agent is the only writer. Read-only subagents may inspect the tracked source snapshot, manuscript extracts, and equation ledger, derive equations independently, or challenge proof cases. Do not change shared contracts, dependencies, root guidance, or the reference manifest. Stop `BLOCKED` rather than weakening a proof gate.


Read `tasks/04_ordered_reflectivity.md` and every referenced shared document, tracked manuscript extract, and tracked legacy-source file.

## Work

Implement only owned paths. Preserve raw physical scale and individual rods. Use the shared complex-wave and interface functions. Do not implement off-specular multilayer fields, detector logic, mosaic events, stacking, or fitting.

## Verify

Run the assigned mutations in `docs/ERROR_INJECTION.md` and confirm the expected first failing stages. Run every T04 command, compare raw intermediates before old normalization, rerun permanent tests from the tracked examples, remove temporary files, and review the diff.

## Done when

Make one coherent commit, complete the T04 handoff, and end exactly `READY`. End exactly `BLOCKED` for unresolved species, occupancy, displacement, or shared-contract semantics.
