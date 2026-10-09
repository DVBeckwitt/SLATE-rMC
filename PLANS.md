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
The UI is integrated with the current core on main; its separate branch is retired.
The UI roadmap records historical acceptance and remaining gaps; integration does not
close them.
Native fitting Run, stage-result import, indexed adoption and new-acquisition
numerical preparation remain unavailable. The current scientific procedure and
publication figure/error-bar conventions remain in
[the fitting and figure guide](docs/FITTING_WORKFLOW.md). Historical plans are
[archived](docs/REPOSITORY_HISTORY.md) and do not schedule scientific work.
Software verification follows [VALIDATION.md](docs/VALIDATION.md); it does not
qualify a physical fit or resolve the recorded desktop performance gaps.
