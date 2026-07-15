# Codex execution strategy

The plan follows current Codex guidance:

- Put durable repository rules in a concise root `AGENTS.md`.
- Put detailed mathematics and done conditions in task documents rather than duplicating them in prompts.
- Structure prompts as Goal, Context, Constraints, Work, Verify, and Done when.
- Use Plan mode for difficult tasks before source editing.
- Give each independent write-heavy task its own worktree and avoid concurrent edits to the same files.
- Use subagents mainly for read-heavy exploration, derivation review, and test-log analysis. Keep one writer per worktree.
- Review diffs and proof evidence before merging.

Official references:

- https://developers.openai.com/codex/learn/best-practices
- https://developers.openai.com/codex/agent-configuration/agents-md
- https://developers.openai.com/codex/environments/git-worktrees
- https://developers.openai.com/codex/long-running-work
- https://developers.openai.com/codex/subagents

## Why the serial spine is mandatory

Coordinates, complex square-root branches, event measures, and trace schemas are shared mathematical contracts. If four agents invent them independently, the parts may each look correct but fail when combined. Bootstrap and tracked reference verification therefore precede parallel work.

## Why legacy capture is separate

The agent implementing a subsystem must not choose or regenerate its own legacy evidence. One serial characterization step created the immutable pack, including intermediate values and environment metadata. The pack, manuscript extracts, and cited source snapshot are tracked and read-only. All branches use the same evidence and cannot edit it.

## Prompt size

Prompts remain concise. They point to:

- `AGENTS.md` for durable rules
- the assigned task for equations, tracked source/manuscript locations, proof cases, and writable paths
- shared documents for contracts and conventions

This reduces context duplication and conflicting instructions.

## Unattended task behavior

Each worktree must:

1. verify the exact common base and pack hash
2. write a no-edit plan in the assigned task
3. inspect the tracked manuscript extracts and tracked legacy-source files listed by the task
4. implement the smallest mandatory slice first
5. run proof and convergence commands
6. run the assigned negative controls and confirm the comparator catches them at the expected stage
7. stop as `BLOCKED` instead of changing shared contracts silently
8. make one coherent commit
9. return a compact handoff ending `READY` or `BLOCKED`
