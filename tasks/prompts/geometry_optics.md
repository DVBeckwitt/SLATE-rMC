# Codex prompt: geometry and optics

## Goal

Implement T02 completely in this worktree. Produce the smallest correct geometry/optics subsystem and prove it against analytic cases and the immutable reference pack.

## Context and startup


Before editing:

```bash
uv sync --frozen --group dev
python scripts/verify_seed.py
python -m rasim_next.proof references --json
test -z "$(git status --porcelain)"
if test -z "$(git branch --show-current)"; then git switch -c feat/geometry-optics; fi
```

Record the starting `HEAD` as the branch proof-base SHA. It must be identical across all four worktrees.

Use Plan mode first and write the no-edit plan in the assigned task. The main agent is the only writer. Read-only subagents may inspect the tracked source snapshot, manuscript extracts, and equation ledger, derive equations independently, or challenge proof cases. Do not change shared contracts, dependencies, root guidance, or the reference manifest. Stop `BLOCKED` rather than weakening a proof gate.


Read `tasks/02_geometry_optics.md` and every referenced shared document, tracked manuscript extract, and tracked legacy-source file.

## Work

Implement only the owned paths. Complete the mandatory tasks before optional generalization. Record every legacy case as `MATCH`, `CORRECTED`, or `NO_ORACLE`, with the first divergent trace stage for corrections. Do not implement full multilayer distorted fields.

## Verify

Run the assigned mutations in `docs/ERROR_INJECTION.md` and confirm the expected first failing stages. Run every T02 command, inspect the proof JSON, rerun permanent tests from the tracked examples, remove temporary files, and review the diff.

## Done when

Make one coherent commit, complete the T02 handoff, and end exactly `READY`. End exactly `BLOCKED` only for a precise shared-contract or dependency limitation.
