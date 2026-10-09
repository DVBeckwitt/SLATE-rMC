# Repository history and retired interfaces

## Retrieve historical work

The pre-cleanup tree is `5ea105f20aad708fc0b497aa4ce55748dd7eaae7`. Its `tasks/`
directory preserves 72 historical plans, prompts and the status registry. Retrieve a
named record without restoring obsolete instructions or running its commands:

```powershell
git show 5ea105f:tasks/44_ewald_integration_research.md
git show 5ea105f:docs/FITTING_ROADMAP.md
```

The current process is [FITTING_WORKFLOW.md](FITTING_WORKFLOW.md). Runtime physical
contracts remain in [CONTRACTS.md](CONTRACTS.md); scientific acceptance remains in
[VALIDATION.md](VALIDATION.md). The retired roadmap and task campaigns do not authorize
new calculations. Immutable `reference/` and `examples/` inputs remain tracked.

## October 8, 2026 cleanup

Unused short-sequence enumeration, the full six-state stacking proof recurrence,
the source-averaged per-rod native-pixel proof terminal, and the one-source generator's
optional `--reference-diagnostic` comparison harness were removed. They remain available
at the revision above for historical inspection. The live reduced recurrence,
one-source integration, conditional native fitting, numerical qualification and GUI
rendering interfaces are retained. The generator still emits its result identities,
work accounting and explicit unresolved status; it no longer runs a reference comparison.

The redundant Codex execution/error-injection pages were folded into the current
project and assessment instructions. The obsolete seed inventory referred to a
nonexistent verifier; use README and the actual tracked tree instead.

Before retiring branches/worktrees, all 33 original worktree HEADs and named refs were
preserved in an external Git bundle. A separate verified archive contains 77 changed,
untracked or non-cache ignored files plus tracked/staged patches. The active UI working
files were excluded from that initial cleanup; the subsequent integration is recorded below.

Archive location on this workstation:
`C:/Users/Kenpo/.codex/visualizations/2026/10/01/01a0f7e3-277a-7070-9b34-7ded40ce414d/`

- `slate_repository_before_cleanup_20261008.bundle`: committed histories, including
  `codex/bite-intergrowth` at `30ee83715921f6ad3485f33d904d27fd257e33f4` and
  `codex/pea-pbi2-n6-n8-fit` at `73cca82f735594af6420af1f39a9114e276188e1`.
- `slate_retired_worktree_changes_20261008.zip`: exact worktree files, patches and manifest.
- `slate_repository_cleanup_archive_20261008.json`: original paths, heads and file hashes.

Use `git bundle verify <bundle>` and `git bundle list-heads <bundle>` before recovery.
Restore to a separate checkout; do not apply archived changes over main or a recovery checkout.
Unmerged scientific features are preserved history, not implicitly accepted into main.

## Desktop integration and main-only cleanup

Merge `410ff7fb7db1df962e2d1526c6d20803b877862b` integrated the desktop branch and
its seven pending UI files with the current numerical core. The pending files were
preserved with hashes in the external `ui_merge_snapshot_20261008` directory before
integration. The original checkout retains those exact files in detached HEAD.

The UI branch contained no commits absent from main and was deleted at the user's
request. Only the local `main` branch remains; protected detached checkouts are not
branches and retain recovery material. The prior branch-preservation exception is
retired. Current fitting, background, qualification and figure conventions remain
in their authoritative documents, and native desktop fitting limitations remain explicit.
