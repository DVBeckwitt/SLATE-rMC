# Worktree launch

## Historical spine

The bootstrap/reference spine and T02--T05 physics worktrees are complete, merged, and retired.
Their tracked task files and reference evidence remain provenance only. Do not recreate their
branches, relaunch their prompts, or restore their sampled event/raster runtime.

## Start new work

Every write-heavy task starts from the approved current `main` in one short-lived `codex/`
worktree. First verify that the main checkout is clean and record its exact commit:

```powershell
git -C C:\path\to\SLATE-rMC status --short
$baseSha = git -C C:\path\to\SLATE-rMC rev-parse main
git -C C:\path\to\SLATE-rMC worktree add `
  -b codex/<task-name> `
  C:\path\to\SLATE-rMC\build\worktrees\<task-name> `
  $baseSha
```

Inside the new worktree:

1. Read `AGENTS.md`, the current contracts, and the task-specific prompt or plan.
2. Confirm `git merge-base --is-ancestor main HEAD` succeeds before editing.
3. Use the repository virtual environment or create an isolated one with the frozen lockfile.
4. Keep one writer. Subagents may perform bounded read-only derivations and reviews.
5. Write diagnostics and generated figures outside the repository.
6. Finish with one coherent commit and a clean worktree.

The tracked `reference/` pack and cited legacy snapshot are immutable evidence. Production code and
permanent tests never import or execute the legacy snapshot.

## Review and integrate

Before merging, run the focused tests, compact suite, formatting, lint, registered proofs,
`scripts/verify_seed.py`, and an independent read-only review. Then merge from the clean main
checkout without rewriting history:

```powershell
git -C C:\path\to\SLATE-rMC merge --ff-only codex/<task-name>
git -C C:\path\to\SLATE-rMC status --short
```

If `main` advanced and is no longer an ancestor, stop and reconcile explicitly in the feature
worktree before integration. Remove a retired worktree or branch only after its accepted commit is
reachable from `main` and no task still uses it.
