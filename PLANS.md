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
performance targets and independent audit findings. Section 11 provides the delivery roadmap,
provisional effort ranges, ownership, first build cycle, risks and the single 46-task backlog.
The first cohesive researcher release combines inspection, reciprocal/experiment editing,
independent simulation and hBN calibration. Later packages extend geometry routes, prepared native
fitting, new-acquisition preparation and daily use, each with dependencies and acceptance checks.
Milestones group capabilities without blocking independent tasks. Simulator completion covers both
configured and native physical-input routes; new-acquisition preparation has a separate scientific
decision point before its implementation can be estimated.
Implementation is in progress; root accepted 25/46 conditions after the R8 review, with 21 remaining.
Five of 16 original assignments are accepted (01/02/07/08/09); 11 remain incomplete. U02b stays
unaccepted because its B002 Open gate failed. Native Run/indexed adoption/stage-result import and
new-acquisition numerical preparation remain unavailable; exact-lock, scientific and performance
deferrals remain explicit. The checklist and reviewed scopes remain in that same document.
Delivery verification follows the user's nominal-proof policy in
[VALIDATION.md](docs/VALIDATION.md#desktop-implementation-verification); existing fitting validation
and qualification states remain authoritative. Historical `tasks/plan.md`, `tasks/todo.md` and fitting roadmaps retain
their original evidence and do not schedule additional scientific work.
