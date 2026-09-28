# Change plans

Keep each change small enough to explain and review. A plan records:

1. The user objective and the fitting/physical capability it supports.
2. Owned paths, dependencies and behavior that must be preserved.
3. The smallest implementation or deletion that achieves the objective.
4. External checks justified by the actual change, and what they cannot establish.
5. Completion evidence: commit, clean state, line reduction and remaining limitations.

Use current `AGENTS.md` and `docs/VALIDATION.md`. Historical task phases do not trigger
test suites, proof campaigns or new fitting runs. Use one writer and read-only reviewers.
Prefer removing duplicated work to adding orchestration or compatibility layers.

## Native desktop simulation and fitting interface

The consolidated [desktop UI grand plan](docs/DESKTOP_UI_PLAN.md) records the accepted
interface scope, OSC-style marginal profiles, beam-center estimation, initial-value controls, staged fitting,
performance targets and independent audit findings. Section 11 is the executable build sequence:
inspection, independent simulator, geometry routes, experiment views, prepared native fitting,
new-acquisition preparation and release integration, each with dependencies and acceptance checks.
It is the current plan for this UI work; implementation has not started. Its checklist is kept in that
same document. Historical `tasks/plan.md`, `tasks/todo.md` and fitting roadmaps retain
their original evidence and do not schedule additional scientific work.
