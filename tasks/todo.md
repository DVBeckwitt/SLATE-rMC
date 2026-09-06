# Repository-wide simplification checklist (2026-09-06)

Acceptance and historical provenance: [plan.md](plan.md).

- [x] Isolate the current-main worktree and complete read-only subsystem consumer audits.
- [x] Remove dead adapters/helpers and verify focused subsystem tests.
- [x] Consolidate identical arithmetic and validation without changing scientific conventions.
- [x] Complete active-document reconciliation and independent review.
- [x] Pass the full suite, formatting/lint, documentation checks, and parity benchmark.

Final commit gate: refresh and verify the seed inventory, then confirm all registered proofs on
the clean committed candidate. Integrate only while preserving the main checkout's active fit
dependencies. The accepted SHA and final proof state belong to the handoff.

## Historical obligations, not new work authorization

The archived checklist left PAR-01, FIT-01, SRC-01, GEO-FIT-01, NOM-01, and PERF-01 unchecked.
Their scope, dependencies, and proof obligations are retained in the historical table in
[plan.md](plan.md). Checkpoint K acceptance does not mark those downstream entries completed.
Reconcile them with current contracts before scheduling or implementing any follow-up.
