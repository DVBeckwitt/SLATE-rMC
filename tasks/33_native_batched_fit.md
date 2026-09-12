# Batched native fitting predictions

The all-parameter Se fit now executes, but its source/geometry finite differences
require roughly 100 seconds per candidate. Continue that run on its unchanged source
while adding the smallest optional execution hook for independent predictions.

This worktree starts from approved main `e6b36b5` and fast-forwards the reviewed
prerequisite `3ca80c1`. The main checkout's inherited untracked B4 scripts are preserved.
Only the main agent writes; reviewers are read-only.

## Plan and ownership

1. Add optional `predict_many` to `fit_native_parameters`. SciPy generates the existing
   bounded finite-difference points. The hook receives complete physical parameter
   vectors and returns aligned raw native predictions. Objective calculation, profiled
   scale, calibration, guards, optimizer history and callbacks execute serially.
2. Require SciPy 1.16 or newer only when the hook is requested. Keep the current dependency
   range and default execution. No derivative equation, optimizer or physical model is added.
3. Extend the compact public search regression to compare serial/batched search and reject
   invalid batch output. Time equivalent real native predictions in isolated processes;
   never share mutable response owners or XrayDB sessions between concurrent predictions.
4. Use a measured benefit to advance the real all-parameter fit. Complete focused tests,
   ordinary handoff checks and one coherent commit. Avoid additional physics proof sweeps.

Owned files: `fitting/native_search.py`, `tests/test_native_search.py`, this task/index,
the public fitting documentation and file manifest. Parallel process orchestration belongs
to the external fitting experiment. Every worker receives explicit state and returns arrays;
only the parent writes diagnostics. No scientific state is stored in module globals.

PHY-FIT-025 remains the numerical-acceptance authority. Failed or absent qualification
still prevents a selected physical estimate. No legacy classification or physical equation
changes. The existing all-39-coordinate fit remains active during implementation.

## Handoff

The optional batch hook is implemented. One permanent public test protects identical
bounded multistart candidates, calibrated scores, fixed-coordinate mapping and serial
callback history. It also rejects malformed batches, unsupported SciPy and incorrectly
shared batch predictors across different discrete choices. Independent review identified
the discrete-choice ownership hazard and older-SciPy test compatibility; both are fixed
and the review is cleared. No new dependency, optimizer or derivative equation is added.

`native_batched_prediction_benchmark.ra_diag.npz` compares two complete Se predictions,
including the actual source-divergence stencil. Serial and isolated-process predictions
are bitwise identical; prediction, contrast and objective discrepancies are zero.
Cold end-to-end time is 222.321 seconds serial and 206.385 seconds with two processes,
a measured ratio of 1.0772. The parent peak working set is 1489575936 bytes; worker peaks
are 1009750016 and 1012473856 bytes. These are per-process peaks, not an aggregate peak.
The original fit ran concurrently throughout, and standard tests overlapped the parallel
leg. This is a modest gain under mixed load, not a clean scaling benchmark or evidence
for fourfold acceleration. The hook is optional; default fitting remains serial.

The prediction benchmark predates only the subsequent fitting-wrapper ownership guard;
none of its invoked detector, structure, source or projection code changed. Its provenance
hashes describe the startup snapshot rather than the later complete branch snapshot.
The final source passes 511 permanent tests in 606.03 seconds, formatting and lint.
All six registered precommit proofs pass. `native_batched_fit_proofs.ra_diag.npz` retains
the results; the final inventory and two clean-worktree wrappers finish the handoff gate.
No new physics sweep, temporary package test or generated repository file is retained.

A separate geometry-only audit of the permitted source/instrument search range found
combined corners reaching Q = 5.01990714 inverse angstroms. The next external fit uses
common domains [0, 5.1] with the existing support guard. This encloses examined cases
with margin, not a certified bound over every continuous combination. It changes only
the numerical proposal. All physical rods, native observation memberships and free
parameters remain present. Physical fitting and source/integration qualification continue.
