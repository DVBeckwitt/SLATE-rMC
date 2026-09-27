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
