Read `tasks/OVERNIGHT_RUNBOOK.md` before launching T02 through T05.

# Task sequence

```text
T00 bootstrap
    -> T01 immutable tracked reference verification
    -> PROOF_BASE_SHA
    -> T02 geometry/optics ┐
       T03 mosaic/Ewald    ├ parallel
       T04 ordered/Parratt │
       T05 stacking        ┘
    -> T06 read-only parallel review
    -> T07 native-detector integration
```

All four forward worktrees start from the same `PROOF_BASE_SHA`, use the same in-repository reference pack, and write disjoint paths.

Post-integration work begins only after T07:

```text
T08 selection/indexing ───────────────┐
T09 fit foundation -> T10 instrument calibration
                                     ├-> T11 sample geometry fit
                                     └   (uses stable T08 selection)
T11 -> T12 mosaic fit -> T13 ordered intensity fit -> T14 stacking fit
T14 -> T15 caking and reciprocal-space transforms, deferred
```

T08 may be implemented while T09/T10 are developed, but a final sample selection manifest is generated from accepted instrument calibration before T11. Each future task uses its own worktree and one writer.

Task status is authoritative only in `tasks/index.yaml`.
