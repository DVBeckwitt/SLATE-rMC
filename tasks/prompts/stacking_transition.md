# Codex prompt: stacking transition

## Goal

Implement T05 completely in this worktree. Produce finite transition-matrix intensity and prove it against direct sequence enumeration and direct pair sums.

## Context and startup


Before editing:

```bash
uv sync --frozen --group dev
python scripts/verify_seed.py
python -m rasim_next.proof references --json
test -z "$(git status --porcelain)"
if test -z "$(git branch --show-current)"; then git switch -c feat/stacking-transition; fi
```

Record the starting `HEAD` as the branch proof-base SHA. It must be identical across all four worktrees.

Use Plan mode first and write the no-edit plan in the assigned task. The main agent is the only writer. Read-only subagents may inspect the tracked source snapshot, manuscript extracts, and equation ledger, derive equations independently, or challenge proof cases. Do not change shared contracts, dependencies, root guidance, or the reference manifest. Stop `BLOCKED` rather than weakening a proof gate.


Read `tasks/05_stacking_transition.md` and every referenced shared document, tracked manuscript extract, and tracked legacy-source file.

## Work

Implement only owned paths. Keep full and reduced representations, typed parameter models, explicit normalization, and event-aligned outputs. Do not parse CIFs, construct optics, render images, or implement fitting.

## Verify

Run the assigned mutations in `docs/ERROR_INJECTION.md` and confirm the expected first failing stages. Run every T05 command, inspect direct-oracle agreement and legacy classifications, rerun permanent tests from the tracked examples, remove temporary files, and review the diff.

## Done when

Make one coherent commit, complete the T05 handoff, and end exactly `READY`. End exactly `BLOCKED` only for a precise shared amplitude/population/normalization contract issue.
