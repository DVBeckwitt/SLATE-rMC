# Codex prompt: bootstrap

## Goal

Implement T00 from the greenfield repository seed. Create the common numerical and proof spine, not a physics subsystem.

## Context

Read `AGENTS.md`, `PLANS.md`, `docs/SCOPE_AND_PHASES.md`, `docs/CONVENTIONS.md`, `docs/OSC_COORDINATES.md`, `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/DOVETAIL_MATRIX.md`, `docs/TRACE_SCHEMA.md`, `docs/VALIDATION.md`, and `tasks/00_bootstrap.md`.

## Constraints

- Use Plan mode first and record the plan in T00 before editing source.
- Keep dependencies and public APIs minimal.
- Implement the shared transform, wave-mode, interface, OSC mapping, trace, proof, diagnostic, and synthetic-plumbing requirements exactly once.
- Do not implement geometry models, Ewald physics, structure factors, Parratt recursion, or stacking.
- Diagnostics must be external and single-file.

## Verify

Run the applicable coordinate, trace, comparator, and reference-pack negative controls from `docs/ERROR_INJECTION.md`. Run every command in T00. Recreate the environment from the committed lockfile. Review the diff for accidental subsystem code and generated files.

## Done when

Make one coherent commit, update the T00 handoff, report the commit SHA and proof summary, and end exactly `READY`. If a common contract cannot support the synthetic pipeline, record the smallest problem and end exactly `BLOCKED`.
