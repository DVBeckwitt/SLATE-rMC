# Codex prompt: geometry and optics

## Goal

Implement T02 completely in this worktree. Produce the smallest correct geometry/optics subsystem and prove it against analytic cases and the immutable reference pack.

## Context and startup


Before editing:

```powershell
$ErrorActionPreference = "Stop"
$expectedBranch = "feat/geometry-optics"

if ([string]::IsNullOrWhiteSpace($env:PROOF_BASE_SHA)) {
    throw "PROOF_BASE_SHA must name the verified proof-base commit."
}
$expectedSha = $env:PROOF_BASE_SHA.Trim()
$actualBranch = (git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to read the current branch." }
if ($actualBranch -ne $expectedBranch) {
    throw "Expected branch $expectedBranch; found $actualBranch."
}
$actualSha = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to read HEAD." }
if ($actualSha -ne $expectedSha) {
    throw "Expected HEAD $expectedSha; found $actualSha."
}
$status = @(git status --porcelain)
if ($LASTEXITCODE -ne 0) { throw "Unable to read Git status." }
if ($status.Count -ne 0) { throw "Worktree must be clean before setup." }

uv sync --frozen --group dev
if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }
uv run --frozen --group dev python scripts/verify_seed.py
if ($LASTEXITCODE -ne 0) { throw "Seed verification failed." }
uv run --frozen --group dev python -m rasim_next.proof core --json
if ($LASTEXITCODE -ne 0) { throw "Core proof failed." }
uv run --frozen --group dev python -m rasim_next.proof references --json
if ($LASTEXITCODE -ne 0) { throw "Reference proof failed." }
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
