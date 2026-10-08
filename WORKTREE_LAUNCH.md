# Isolated work and integration

1. Read the assigned scope and current `AGENTS.md`.
2. Inspect main, attached worktrees and uncommitted changes. Reuse a free isolated checkout
   at current main where possible; preserve paused or ongoing scientific work.
3. Keep `main` and the authorized `codex/desktop-ui-implementation` branch. Preserve the
   active UI checkout. Work in detached HEAD for other isolated changes.
   Use the app's managed worktree tools where available.
4. Keep one coding writer. Reviewers may inspect but do not edit the same checkout.
5. Make one coherent change. Keep generated outputs and task-specific checks external.
6. Review the diff, perform the relevant checks in `docs/VALIDATION.md`, and commit.
7. Recheck primary main is clean and unchanged, then fast-forward it to the reviewed commit.
   Report commit, checks, limitations and production/development line changes.

Do not resume old task prompts, recreate retired branches or launch numerical campaigns as
an integration ritual. Archive a managed worktree only when no ongoing work needs it.
