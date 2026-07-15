# T09 Codex prompt

## Goal

Implement the minimal reusable fitting contracts and invalidation layer without forward physics.

## Context

Read `AGENTS.md`, `docs/FITTING_ROADMAP.md`, `docs/CONTRACTS.md`, `docs/DOVETAIL_MATRIX.md`, `docs/VALIDATION.md`, `docs/PERFORMANCE.md`, and `tasks/09_fit_foundation.md`. Inspect the original-RASIM and manuscript locations cited there. Use Plan mode first, record the plan in the task file, and then continue without waiting for confirmation.

## Constraints

- Work only on branch `feat/fit-foundation` from the accepted dependency SHAs.
- Edit only paths owned by T09.
- Reuse the accepted forward APIs. Do not copy equations into fitting code.
- Keep identities and deterministic samples frozen inside one objective run.
- Do not add GUI, caking, or `2theta/phi` code.
- Stop `BLOCKED` rather than silently changing shared contracts or upstream results.

## Work

Complete every mandatory item and proof gate in `tasks/09_fit_foundation.md`. Use a compact independent synthetic oracle and held-out case. Record provenance, conditioning or identifiability, wall time, peak memory, and known limitations.

## Verify

Run the applicable selection or fitting controls in `docs/ERROR_INJECTION.md` and confirm they are rejected for the expected reason. Run the exact commands in `tasks/09_fit_foundation.md`, inspect `git diff --check`, and verify no diagnostic or generated file exists under the repository root.

## Done when

Make one coherent commit, fill the task handoff, report the commit SHA and proof state, and end exactly `READY`. When a mandatory gate cannot be satisfied, record the smallest blocker and end exactly `BLOCKED`.
