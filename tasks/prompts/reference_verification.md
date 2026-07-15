# Codex prompt: verify the tracked proof reference

## Goal

Implement T01 so the repository proves its immutable reference pack and examples from a clean clone.

## Context

Read `AGENTS.md`, `docs/OSC_COORDINATES.md`, `docs/TRACE_SCHEMA.md`, `docs/VALIDATION.md`,
`reference/README.md`, `examples/README.md`, and `tasks/01_reference_verification.md`.

## Constraints

- Use Plan mode first.
- Never modify tracked files under `reference/` or `examples/`.
- Do not require the original repository, manuscript, GUI state, or absolute user paths.
- Do not write diagnostics or decompressed OSC files under the repository.
- Keep the permanent coordinate/reference proof compact.

## Done when

`python -m rasim_next.proof references --json`, the assigned tests, seed verification, and mutation
controls pass. Make one coherent commit and end exactly `READY`, or end `BLOCKED` with the smallest
precise missing bootstrap contract.
