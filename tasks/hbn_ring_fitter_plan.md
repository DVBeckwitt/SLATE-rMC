# T16 plan: minimum cohesive hBN ring fitter

Status: PLANNED, `BLOCKED` at Checkpoint A pending accepted T10.

Specification: `docs/HBN_RING_FITTER_MINIMUM_SPEC.md`

This plan and `tasks/hbn_ring_fitter_todo.md` are the feature-specific execution artifacts. Shared
planning files remain generic and do not duplicate this detailed T16 breakdown.

## Goal

After T10 has accepted one general detector-calibration API and proof, add the smallest useful hBN
observation frontend: canonical OSC/dark preprocessing, five fixed rings, precise place/replace,
one T10 solve, result inspection, and accepted-result export.

## Fixed planning decisions

- Session persistence remains deferred.
- Beam center is estimated by T10; the first UI has no separate center-pick mode.
- T16 depends on completed, accepted T10, not merely a draft contract.
- The copied `feat/hbn-ring-fitter` branch is evidence only and is never merged or copied.
- Exactly two hBN-specific production modules and two permanent hBN test functions are allowed.
- The hBN-specific production cap is 800 physical Python lines across both modules.

Changing one of these decisions requires updating the specification and this plan before coding.

## External prerequisites

These prerequisites are owned by existing tasks and are not reimplemented here:

```text
T07 native-detector integration
    -> T09 fit foundation
        -> T10 source and detector calibration
            -> accepted CalibrantObservation
            -> accepted detector-ring residual and optimizer
            -> accepted DetectorCalibrationResult and strict writer
            -> accepted synthetic recovery, held-out ring, identifiability, and proof
                -> T16 hBN frontend
```

T16 stops `BLOCKED` if T10 cannot expose predicted ring support, fitted-versus-fixed parameter
metadata, rank/conditioning, rejection reasons, and accepted-result export without a private hBN
schema.

## Dependency graph

```text
Task 1 documentation/task boundary
    -> Task 2 fixed hBN ring model
        -> Task 3 canonical image preparation
            -> Task 4 observation state and local snap
                -> Task 5 T10 observation adapter
                    -> Task 6 optional launcher
                        -> Task 7 minimal image workspace
                            -> Task 8 precision interaction wiring
                                -> Task 9 solve integration
                                    -> Task 10 result overlay/export
                                        -> Task 11 user and task documentation
                                            -> Task 12 final proof and deletion gate
```

All implementation is sequential. Tasks 2--5 share `hbn_fitter.py` and `test_fitting.py`; Tasks
6--10 share `hbn_gui.py`. Read-only review may run concurrently, but no second writer is useful.

## File strategy

Prefer modification and deletion. The only planned new production files are:

```text
src/rasim_next/hbn_fitter.py
src/rasim_next/hbn_gui.py
```

Modify the existing T10 test module, dependency metadata, lock file, and existing documentation.
Do not add a session module, calibration-artifact module, controller framework, hBN test module,
configuration file, nested project, launcher batch file, or compatibility adapter. Temporary tests
and smoke artifacts are deleted before their owning task completes.

## Task 1: Publish the optional-tool boundary

**Description:** Record T16 as a post-T10 optional top-layer tool before source work. Resolve the
three specification questions with the fixed decisions above, state that T10 owns all fit physics
and export, and provide one concise execution prompt. This is a documentation-only contract gate.

**Exact intended behavior:** Repository planning must show that the numerical core never imports
the hBN GUI, T16 cannot begin before accepted T10, and no copied legacy branch or private result
format is allowed.

**Tests and verification:**

- `python tools/check_docs.py`
- `git diff --check`
- Inspect `git diff --name-status` and confirm this task changes documentation/task metadata only.

**Dependencies:** Human approval of the specification and this plan; accepted committed baseline.

**Files likely touched:**

- `docs/HBN_RING_FITTER_MINIMUM_SPEC.md` (modify status and resolved decisions)
- `docs/SCOPE_AND_PHASES.md` (modify)
- `docs/ARCHITECTURE.md` (modify)
- `tasks/index.yaml` (modify)
- `tasks/prompts/hbn_ring_fitter.md` (add; required execution prompt)

**Acceptance criteria:**

- [ ] T16 is registered as `FUTURE`, depends on T10, and has no dependency from the core back to UI.
- [ ] Scope and architecture name T10 as the sole calibration-physics owner and keep GUI imports optional.
- [ ] Documentation checks pass with no production, dependency, example-data, or reference change.

**Estimated scope:** Medium, 5 files.

## Checkpoint A: Documentation gate

- [ ] Human approves the resolved first-release decisions.
- [ ] T10 is accepted and exposes every prerequisite named above.
- [ ] Create `feat/hbn-ring-fitter` from the approved clean main; do not reuse the copied branch.
- [ ] Worktree is clean before Task 2.

## Task 2: Define the fixed hBN ring model

**Description:** Add the GUI-free module with immutable hBN lattice/ring values and strict fixed
input validation. Compute the five d-spacings and `2theta` values from one hexagonal formula; do
not store a second hand-maintained angle table.

**Exact intended behavior:** The module exposes ordered `(002)`, `(100)`, `(101)`, `(102)`, and
`(004)` identities, explicit `a_A`, `c_A`, and `wavelength_A`, positive row/column pitches, and
rejects nonfinite values, invalid Bragg domains, duplicate rings, or changed order.

**Tests and verification:**

- Extend only `test_hbn_native_preprocessing_and_observation_state` in `tests/test_fitting.py`.
- Independently evaluate the hexagonal d-spacing identity in the test.
- `pytest -q tests/test_fitting.py -k hbn_native_preprocessing_and_observation_state`
- `ruff check src/rasim_next/hbn_fitter.py tests/test_fitting.py`

**Dependencies:** Checkpoint A.

**Files likely touched:**

- `src/rasim_next/hbn_fitter.py` (add)
- `tests/test_fitting.py` (modify)

**Acceptance criteria:**

- [ ] The five stable ring identities and their order are exact and immutable.
- [ ] D-spacings and angles match an independent scalar calculation at predeclared tolerance.
- [ ] Invalid lattice, wavelength, pitch, ring identity, and Bragg-domain inputs fail explicitly.

**Estimated scope:** Small, 2 files.

## Task 3: Prepare one canonical hBN image

**Description:** Add exposure-pair loading through the accepted OSC boundary and produce one
finite detector-native display/search image without retaining duplicate derived full-resolution
arrays.

**Exact intended behavior:** Call `read_osc` once per path, require equal detector-native shapes,
compute `log1p(max(hbn_counts - dark_counts, 0))` in `float32`, retain source paths/hashes and
shape, and perform no later rotation, flip, transpose, or display-coordinate conversion.

**Tests and verification:**

- Extend `test_hbn_native_preprocessing_and_observation_state` with an asymmetric non-square OSC/dark fixture.
- Prove exact clockwise orientation, subtraction/clipping/log output, dtype, finiteness, immutability, and mismatch rejection.
- `pytest -q tests/test_fitting.py -k hbn_native_preprocessing_and_observation_state`
- `ruff check src/rasim_next/hbn_fitter.py tests/test_fitting.py`

**Dependencies:** Task 2.

**Files likely touched:**

- `src/rasim_next/hbn_fitter.py` (modify)
- `tests/test_fitting.py` (modify)

**Acceptance criteria:**

- [ ] Both inputs use the existing reader and the output stays detector-native `[row,column]`.
- [ ] The prepared image is one immutable finite `float32` array with explicit provenance.
- [ ] Shape, format, nonfinite, and path failures produce informative exceptions without fallback.

**Estimated scope:** Small, 2 files.

## Task 4: Add deterministic observation state

**Description:** Add immutable ring points plus one small authoritative state transition path for
placement, replacement, per-ring undo, reset, active-ring selection, and solved-result
invalidation. Add deterministic local snapping without full-image work.

**Exact intended behavior:** Every observation retains ring HKL, raw full-resolution
`DetectorCoordinate`, accepted coordinate, snap decision, and positive uncertainty/quality when
available. A bounded local patch proposes a deterministic snap; a failed quality gate retains the
raw point. Placement and replacement share one transition and increment the observation revision.

**Tests and verification:**

- Extend `test_hbn_native_preprocessing_and_observation_state` with accepted/rejected snap cases,
  replacement without append, per-ring undo, reset, cross-ring isolation, and revision invalidation.
- Inject row/column swap, half-pixel shift, stale result, and fitting raw instead of declared snapped coordinates.
- `pytest -q tests/test_fitting.py -k hbn_native_preprocessing_and_observation_state`
- Record local preview timing on a representative patch; do not commit benchmark output.

**Dependencies:** Task 3.

**Files likely touched:**

- `src/rasim_next/hbn_fitter.py` (modify)
- `tests/test_fitting.py` (modify)

**Acceptance criteria:**

- [ ] State transitions are deterministic, immutable, ring-safe, and invalidate solved state exactly once.
- [ ] Snap rejection is visible and preserves the raw coordinate; no silent candidate is accepted.
- [ ] Snap/preview examines only a bounded patch and performs no full-image copy.

**Estimated scope:** Small, 2 files.

## Task 5: Adapt hBN points to T10

**Description:** Convert the fixed ring model and current observation revision into T10's accepted
`CalibrantObservation`. Do not call a legacy fit, create an hBN result type, or reproduce T10's
ring equation.

**Exact intended behavior:** Use the accepted snapped coordinate when the snap is declared valid
and otherwise the retained raw coordinate; preserve HKL/d-spacing, covariance or uncertainty,
dataset/source hashes, detector shape, fixed pitches, wavelength, and observation revision. T10's
result and writer pass through without aliases or sign conversion.

**Tests and verification:**

- Add the second and final permanent test,
  `test_hbn_frontend_uses_t10_calibration_boundary`.
- Use scripted nonzero-tilt observations to prove exact adapter fields, one T10 call, held-out-ring
  prediction, rejection propagation, and accepted-result round trip.
- `pytest -q tests/test_fitting.py -k "hbn_native or hbn_frontend"`
- In a clean interpreter, import `rasim_next.hbn_fitter` and assert Tk/Matplotlib are absent from `sys.modules`.

**Dependencies:** Task 4 and accepted T10 API/result writer.

**Files likely touched:**

- `src/rasim_next/hbn_fitter.py` (modify)
- `tests/test_fitting.py` (modify)

**Acceptance criteria:**

- [ ] Adapter output is a general T10 observation with exact ring, coordinate, uncertainty, and provenance identity.
- [ ] No hBN fit equation, result class, serializer, or GUI import exists in the headless module.
- [ ] Exactly two permanent hBN test functions exist and both pass with T10's proof slice.

**Estimated scope:** Small, 2 files.

## Checkpoint B: Headless boundary

- [ ] Tasks 2--5 pass focused tests and Ruff.
- [ ] T10 synthetic recovery, held-out-ring, free-roll, pitch/distance, and wavelength/distance controls pass.
- [ ] Headless import loads neither Tk nor Matplotlib.
- [ ] `hbn_fitter.py` remains narrow enough that no package directory or third module is justified.
- [ ] Read-only scientific review approves coordinates, identifiable parameters, and factor ownership.

## Task 6: Add one optional launcher

**Description:** Add the minimal GUI module and package metadata for one command. Matplotlib is the
only optional dependency; Tkinter availability is checked at runtime.

**Exact intended behavior:** `hbn-ring-fitter --help` parses and exits without opening a window;
normal launch lazily imports Tk/Matplotlib; missing optional support reports the exact install or
runtime remedy. Importing `rasim_next` and `rasim_next.hbn_fitter` remains unchanged.

**Tests and verification:**

- Extend `test_hbn_frontend_uses_t10_calibration_boundary` with headless import and CLI-help checks.
- `uv sync --frozen --group dev --extra hbn`
- `uv run --frozen --extra hbn hbn-ring-fitter --help`
- `ruff check src/rasim_next/hbn_gui.py tests/test_fitting.py`

**Dependencies:** Checkpoint B.

**Files likely touched:**

- `pyproject.toml` (modify)
- `uv.lock` (modify mechanically from the declared optional dependency)
- `src/rasim_next/hbn_gui.py` (add)
- `tests/test_fitting.py` (modify without adding a third hBN test)

**Acceptance criteria:**

- [ ] One command and one optional `hbn` extra exist; no alias, nested project, or core dependency is added.
- [ ] `--help` is noninteractive and missing-GUI errors are actionable.
- [ ] Headless imports remain GUI-free with and without the optional extra installed.

**Estimated scope:** Medium, 4 files.

## Task 7: Show the minimal image workspace

**Description:** Build the smallest window that loads or selects the two OSC paths, prepares the
image through Task 3, shows it in detector-native orientation, and exposes fixed inputs plus one
active-ring selector.

**Exact intended behavior:** CLI paths prefill the selectors; dialogs are optional. Loading is an
explicit action, not a hidden watcher. The figure uses the standard Matplotlib toolbar, labels
axes as detector column/row pixels, and clears observation/result state only after a new valid pair
is accepted.

**Tests and verification:**

- Re-run both permanent hBN tests and CLI-help check.
- Temporary hidden-window smoke may verify construction and teardown, then must be removed.
- Manual smoke with the tracked hBN/dark example verifies file selection, orientation, labels,
  toolbar, active ring, and load-error messages.

**Dependencies:** Task 6.

**Files likely touched:**

- `src/rasim_next/hbn_gui.py` (modify)

**Acceptance criteria:**

- [ ] A valid pair displays once in canonical orientation with explicit fixed inputs and active ring.
- [ ] Invalid paths or mismatched shapes preserve the last valid state and show one clear error.
- [ ] No file watcher, hidden config, image duplicate, or scientific equation is added to callbacks.

**Estimated scope:** Extra small, 1 file.

## Task 8: Wire precision placement

**Description:** Translate canvas press/move/release and nearest-point replacement into the
headless transitions from Task 4. Reuse one artist-update path for new and replacement previews.

**Exact intended behavior:** Left press starts a local precision preview, motion displays raw and
candidate snap, release commits once, and replacement changes one selected point without append.
Per-ring undo/reset call the same headless state API. Toolbar pan/zoom, out-of-axes events, wrong
buttons, and canceled gestures never mutate observations.

**Tests and verification:**

- Re-run `test_hbn_native_preprocessing_and_observation_state` for every transition and mutation.
- Use a temporary pointer-translation test only while developing; remove it if the pure-state test
  and manual smoke cover no distinct permanent boundary.
- Manual smoke verifies preview artists, replacement, undo/reset, toolbar exclusion, cancellation,
  and view restoration.
- Record p95 event-to-preview latency; no repository-local benchmark file.

**Dependencies:** Task 7.

**Files likely touched:**

- `src/rasim_next/hbn_gui.py` (modify)

**Acceptance criteria:**

- [ ] One gesture commits at most one observation and replacement never changes ring length.
- [ ] Every excluded/canceled event leaves observation and result revisions unchanged.
- [ ] Preview copies no full image and records p95 latency at or below 100 ms on the reference machine.

**Estimated scope:** Extra small, 1 file.

## Checkpoint C: Observation workflow

- [ ] The scientist can load, choose a ring, place, replace, inspect raw/snap, undo, and reset.
- [ ] Both permanent hBN tests pass; no third permanent test or temporary artifact remains.
- [ ] GUI callbacks contain no ring geometry, fitting, uncertainty, or coordinate transformation.
- [ ] Current hBN-specific production line count is recorded and remains on track for 800 lines.

## Task 9: Connect the T10 solve

**Description:** Send the current observation revision and fixed inputs through the Task 5 adapter
to T10. Present accepted and rejected results without introducing a second optimization or result
model.

**Exact intended behavior:** Solve is enabled only for T10-valid support. It calls T10 once,
records the input revision, shows rejection reason/rank/conditioning or accepted center,
beam-intersection distance, beam direction, residual, and fixed roll/pitch metadata. Any later
input or point change clears the current result and disables export.

**Tests and verification:**

- Extend `test_hbn_frontend_uses_t10_calibration_boundary` without adding a new test function.
- `pytest -q tests/test_fitting.py -k "hbn or detector_calibration"`
- `python -m rasim_next.proof instrument-calibration --json`
- Manual smoke covers accepted solve, rejected solve, input mutation, and point mutation.

**Dependencies:** Checkpoint C and accepted T10 proof.

**Files likely touched:**

- `src/rasim_next/hbn_gui.py` (modify)
- `tests/test_fitting.py` (modify)

**Acceptance criteria:**

- [ ] The GUI invokes only the accepted T10 solve/result and exposes its rejection/identifiability evidence.
- [ ] Result revision equals the current observation/fixed-input revision before export is enabled.
- [ ] Free roll, fitted pitch, fitted wavelength, and any private hBN result are absent.

**Estimated scope:** Small, 2 files.

## Task 10: Render and export the accepted result

**Description:** Render predicted ring support and residual state from T10 output, then delegate
export to T10's strict accepted-result writer.

**Exact intended behavior:** Raw points, accepted points, predicted rings, center, and residuals
are visually distinct and share detector-native coordinates. Export is available only for the
current accepted revision, writes to a user-selected external path, and performs no alias, sign,
unit, frame, or schema conversion in GUI code.

**Tests and verification:**

- Re-run `test_hbn_frontend_uses_t10_calibration_boundary` for exact result/writer pass-through.
- Manual smoke writes to an external temporary path, reloads through the T10 reader, and compares
  the accepted fields exactly; delete the artifact afterward.
- Inject detector-normal reversal, pitch swap, and stale-revision export; each must fail at its
  declared boundary.

**Dependencies:** Task 9; T10 must expose predicted ring support and strict writer/reader.

**Files likely touched:**

- `src/rasim_next/hbn_gui.py` (modify)

**Acceptance criteria:**

- [ ] Overlays derive only from authoritative observation state and T10 predictions.
- [ ] Rejected or stale results cannot be exported.
- [ ] An exported accepted result round-trips through T10 with no GUI-specific schema or conversion.

**Estimated scope:** Extra small, 1 file.

## Checkpoint D: Smallest useful release

- [ ] Load-observe-correct-solve-inspect-export works end to end.
- [ ] Synthetic recovery and held-out-ring evidence meet frozen tolerances.
- [ ] All coordinate/sign/identifiability/stale-result injections fail at the intended stage.
- [ ] Human accepts the workflow without session persistence or center-pick mode.

## Task 11: Document only the shipped surface

**Description:** Update existing user and example documentation after behavior is final. Record the
optional dependency, one command, T10 ownership, tracked example workflow, coordinate boundary,
limitations, and no-parity claim. Update the task prompt and handoff; add no second design history.

**Exact intended behavior:** Documentation describes the current shipped interface in timeless
language. It never presents the copied branch, legacy NPZ, roll, or a complete detector pose as an
accepted output.

**Tests and verification:**

- `python tools/check_docs.py`
- `uv run --frozen --extra hbn hbn-ring-fitter --help`
- Follow the documented tracked-example workflow manually through accepted-result reload.
- `git diff --check`

**Dependencies:** Checkpoint D.

**Files likely touched:**

- `docs/EXAMPLES.md` (modify)
- `README.md` (modify)
- `tasks/hbn_ring_fitter_plan.md` (rename to `tasks/16_hbn_ring_fitter.md`, then modify execution state/handoff only)
- `tasks/prompts/hbn_ring_fitter.md` (modify only if the accepted commands differ)

**Acceptance criteria:**

- [ ] User docs contain one install/launch/workflow path and explicit detector-native semantics.
- [ ] Limitations name fixed pitch/wavelength/roll, no session, no auto-discovery, and no legacy compatibility.
- [ ] Documentation/task checks pass without duplicating the specification or GUI inventory.

**Estimated scope:** Medium, 4 files.

## Task 12: Run the final proof and footprint gate

**Description:** Run the complete handoff gate, measure the final footprint, and identify anything
not required by the accepted user job, public boundary, two permanent tests, or documentation.
This task makes no edits: every failure or residue item reopens the atomic task that owns its file,
where deletion or repair occurs before this gate is rerun.

**Exact intended behavior:** The final branch contains two hBN modules, two permanent hBN tests,
one optional dependency, no generated output, and no copied/legacy/nested implementation. The
optimized interaction path agrees with the proof path and the worktree ends clean after one
coherent commit.

**Tests and verification:**

```text
uv sync --frozen --group dev --extra hbn
python -m compileall -q src
ruff check src/rasim_next/hbn_fitter.py src/rasim_next/hbn_gui.py tests/test_fitting.py
pytest -q tests/test_fitting.py -k "hbn or detector_calibration"
pytest -q
python -m rasim_next.proof instrument-calibration --json
python tools/check_docs.py
uv build
uv run --frozen --extra hbn hbn-ring-fitter --help
git diff --check
```

Also run the manual desktop smoke, assigned error injections, wall-time and peak-memory capture,
line count, import graph check, source/provenance audit, and `git status --short`.

**Dependencies:** Task 11.

**Files likely touched:**

- None. Any required edit reopens its owning Task 1--11 and stays within that task's listed files.
- No proof, benchmark, diagnostic, or generated file may be added under the repository root.

**Acceptance criteria:**

- [ ] All specification acceptance criteria and the project Definition of Done pass.
- [ ] hBN-specific production is exactly two modules and at most 800 physical Python lines; only two permanent hBN tests remain.
- [ ] One optional dependency remains, all proof/performance evidence is recorded externally, and the final intended commit is clean.

**Estimated scope:** Medium verification gate; zero files changed.

## Final Definition of Done

Every task clears its focused acceptance criteria plus the standing project bar:

- runtime behavior, error paths, focused tests, and existing tests pass;
- names and structure reveal intent; no duplicate physics, dead code, debug output, or unrelated refactor remains;
- lint/format and relevant proof commands pass;
- public behavior and architectural decisions are documented in existing authoritative files;
- untrusted paths and strict result loading are validated;
- no repository-local diagnostics, screenshots, build output, temporary tests, or benchmark files remain;
- the human reviews the checkpoints and final diff before merge.

## Handoff record

At completion record: commit SHA, accepted T10 API/revision, public hBN API/command, exact changed
files, physical Python line count, retained permanent tests and why, synthetic recovery,
held-out-ring result, error-injection stages, manual smoke, p95 preview latency, wall time, peak
memory, limitations, legacy classifications, and minimum next request. End exactly `READY` or
`BLOCKED`.
