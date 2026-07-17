# Spec: minimum cohesive hBN ring fitter

Status: accepted planning baseline; runtime feature `BLOCKED` pending accepted T10.

This specification was derived from committed `main` and the reviewed historical
`feat/hbn-ring-fitter` tip (`18c6da69067aa05d9814f4a5f0281c3cc27e89a2`). Uncommitted files in either
worktree were intentionally outside its evidence base. The copied workbranch was retired after
its intended workflow and provenance were captured here and in the T16 plan.

## Problem statement

How might we let a diffraction scientist turn an hBN exposure and matching dark exposure into a
scientifically explicit detector calibration, while preserving accurate manual ring observation
and adding as little code and dependency weight as possible?

The user is a scientist who can identify the five known hBN rings but needs local precision help,
visible correction, a trustworthy physical solve, and an export usable by the staged fitting
workflow. Success is one complete desktop job: load, observe, correct, solve, inspect, and export.
Legacy control parity is not the job.

## Refinement and decision

### Directions considered

| Direction | User value | Feasibility | Footprint and risk | Decision |
|---|---|---|---|---|
| Merge the committed copied tool unchanged | Immediate working GUI and CLI | High | Adds 9,012 lines, including 6,648 production Python lines, seven direct runtime dependencies, a second OSC reader, a second configuration system, and copied legacy modules | Reject |
| Add a narrow vendored-tool exception | Same behavior with an explicit isolation boundary | High | Still contradicts the greenfield no-copy rule, retains ambiguous legacy coordinates, and creates a GPL boundary that the root project has not accepted | Reject |
| Recreate complete legacy GUI parity | High migration fidelity | Medium | Reintroduces session migration, two frontends, advanced controls, duplicate outputs, and a large interaction surface before the calibration core exists | Reject for the first release |
| Implement only headless T10 calibration | Correct and smallest numerical result | High | Does not give the scientist an interactive way to create observations | Insufficient by itself |
| Add a thin hBN observation UI after T10 | Completes the scientist's job and reuses the accepted solver | High after T10 | Two hBN-specific modules, one optional GUI dependency, no duplicate physics or artifact | **Recommended** |

### Recommended direction

T10 remains the sole owner of detector-ring geometry, optimization, identifiability, result
provenance, and instrument revision. The hBN feature is a later optional frontend that:

1. reuses the canonical OSC reader and detector-native coordinate types;
2. defines the five fixed hBN rings and gathers quality-bearing ring observations;
3. adapts those observations to T10's general `CalibrantObservation` contract;
4. calls the accepted T10 detector-calibration API; and
5. displays and exports the T10-owned result without interpreting legacy tilt fields.

The reviewed historical tip is behavior/provenance evidence recorded by this specification and
the workbranch archive. Its copied implementation was not merged and is not an implementation
base. No production or permanent-test code may be copied from it or from
`reference/legacy_source`.

## Planning handoff status

- **Feature:** the minimum scope, interfaces, proof obligations, 12 atomic tasks, and execution
  checklist are complete; no hBN runtime code, dependency, API, test, or CI workflow was added.
- **Blocker:** committed `main` has no accepted T10 calibration API, result writer, predicted-ring
  support, or proof. T16 must remain blocked rather than create private calibration physics.
- **Bug/error:** no runtime defect was fixed or introduced; the failed prerequisite is a feature
  dependency, not a suppressed test or known numerical error.
- **Migration/deprecation:** none. The retired copied workbranch has no supported migration path
  into the greenfield package.
- **Launch:** documentation is reviewable, but the feature is not implemented or shippable.

## Assumptions to validate

- [ ] Five fixed hBN reflections `(002)`, `(100)`, `(101)`, `(102)`, and `(004)` cover the first
  user workflow; validate against the tracked hBN example and one scientist walkthrough.
- [ ] Known row and column pitch, wavelength, and hBN lattice constants can be fixed inputs; T10
  must reject attempts to trade pitch or wavelength against distance in this first scope.
- [ ] Precision place/replace with a local snap is the only custom interaction needed beyond the
  standard Matplotlib zoom/pan toolbar; validate in one desktop smoke session.
- [ ] A session file is convenience rather than a release blocker; validate by timing one complete
  five-ring observation pass before adding persistence.
- [ ] Tkinter plus optional Matplotlib is available on the supported local Windows environment and
  keeps the headless package import free of GUI imports.
- [ ] T10 can expose one stable detector-calibration request/result and strict export before the
  hBN branch begins. If it cannot, the hBN work stops rather than inventing a private solver or
  format.

## Objectives

1. Complete one standalone load-observe-correct-solve-inspect-export workflow.
2. Preserve canonical arrays `[row, column]` and continuous coordinates
   `(column_px, row_px)` from OSC decoding through export.
3. Preserve both the raw pointer coordinate and the accepted snapped coordinate, with a named snap
   decision, uncertainty or quality measure, and stable ring identity.
4. Use the general T10 physical detector-ring solver; do not fit a second hBN-specific ellipse or
   projective model.
5. Report only identifiable calibration quantities. Powder rings determine beam center, distance
   to the beam-plane intersection, and two components of beam direction in detector coordinates;
   they do not determine absolute detector roll about the beam.
6. Keep GUI code optional, lazy, and outside the numerical import path.
7. Limit hBN-specific production code to two modules and at most 800 physical Python lines total.

## Smallest useful scope

### Included

- One command: `hbn-ring-fitter`.
- hBN and dark `.osc` or `.osc.gz` selection through `rasim_next.io.osc.read_osc`.
- Exact shape validation, detector-native subtraction, and one finite `float32` log-display array.
- Fixed hBN ring definitions from explicit `a`, `c`, and wavelength inputs, with the accepted
  legacy constants only as visible defaults.
- Explicit fixed detector row and column pitches.
- One active-ring selector for the five rings.
- Precision placement with raw/snap preview, snap gating, and full-resolution coordinate storage.
- Nearest-point replacement through the same placement path, per-ring undo, and reset.
- Automatic invalidation of a solved result after any observation or fixed-input change.
- Conversion to T10 `CalibrantObservation`, one T10 solve call, and display of predicted rings,
  residuals, fitted center, distance, beam direction, rank, and conditioning.
- Export through the T10-owned detector-calibration result writer.
- One lazy optional Matplotlib dependency; Tkinter remains a Python runtime capability.

### Explicitly not included

- The copied `tools/hbn-ring-fitter` project, `ra_sim` namespace, nested lock file, nested
  configuration, copied license file, or root batch launcher.
- A second `hbn-fit` command or interactive workflow.
- Legacy version-3 NPZ bundle compatibility, click-profile JSON, background TIFF, overlay PNG, or
  tilt-corrected PNG.
- Session save/resume in the first release.
- Suggested missing-angle targets, point-count growth, advanced refinement tunables, warm-start
  policy controls, or exact widget-layout parity.
- Automatic ring discovery, generic calibrant authoring, or unknown-ring indexing.
- Fitting wavelength, lattice constants, detector pitch, detector roll, source parameters, sample
  pose, or sample structure.
- A private calibration JSON schema, simulator importer, compatibility facade, GUI framework,
  plugin system, registry, or serialization framework.
- OpenCV, scikit-image, PyYAML, pyparsing, or any other new core runtime dependency.

## Scientific model and ownership

T10 owns the physical residual. For an observation at detector coordinate `(column_px, row_px)`,
let `b` be the unit incident-beam direction expressed in detector `(column,row,normal)` axes,
`d_m` the distance along the beam to its detector-plane intersection, and `p_column_m` and
`p_row_m` the fixed pitches. With beam center `(c0,r0)`, the ray from the sample to the observed
pixel is proportional to

```text
d_m * b
+ [(column_px - c0) * p_column_m,
   (row_px - r0) * p_row_m,
   0]
```

T10 compares the angle between that normalized ray and `b` with the known ring `2theta`. One
physical equation therefore produces both residuals and overlay curves. The hBN frontend supplies
ring identities and observations; it does not implement this equation.

The first fit varies only `(c0, r0, d_m)` and the two independent components of `b`, with
`b_normal > 0`. It holds pitch, wavelength, lattice constants, and roll fixed. A full
`lab_from_detector` transform may be minted only when an independent roll/lab-axis declaration is
available. Otherwise the result remains an explicit partial detector calibration.

## Interfaces

### Existing interfaces reused unchanged

- `read_osc(path) -> OscImage`
- `DetectorCoordinate(column_px, row_px)`
- T10 `CalibrantObservation`
- T10 instrument-revision and `FitResult` provenance conventions

### T10 prerequisite interface

Before hBN implementation, T10 must freeze one narrow public interface equivalent to:

```python
@dataclass(frozen=True, slots=True)
class DetectorCalibrationResult:
    accepted: bool
    reason: str | None
    beam_center: DetectorCoordinate | None
    beam_intersection_distance_m: float | None
    beam_direction_detector: NDArray[np.float64] | None
    fixed_detector_roll_rad: float | None
    detector_shape_rc: tuple[int, int]
    detector_row_pitch_m: float
    detector_column_pitch_m: float
    rms_residual_px: float | None
    parameter_rank: int
    condition_number: float | None
    instrument_revision: str | None
    provenance: Mapping[str, object]
```

The writer accepts only an accepted result; a rejected result carries a reason and may omit fitted
values. The exact class placement belongs to T10. The semantic requirements do not: fitted versus
fixed quantities, units, detector basis, normal sign, rank, conditioning, rejection reason, and
provenance must be explicit. The T10 writer/reader must reject unknown versions, nonfinite values,
invalid shapes or pitches, non-unit beam direction, ambiguous coordinate metadata, and a claimed
measured roll.

### hBN frontend interface

`src/rasim_next/hbn_fitter.py` exposes only GUI-free values and functions equivalent to:

```python
@dataclass(frozen=True, slots=True)
class HbnRingPoint:
    hkl: tuple[int, int, int]
    raw_coordinate: DetectorCoordinate
    snapped_coordinate: DetectorCoordinate
    snap_sigma_px: float | None
    snap_accepted: bool


def prepare_hbn_image(hbn_path: Path, dark_path: Path) -> HbnImage: ...
def snap_hbn_point(
    image: HbnImage,
    ring: HbnRing,
    raw: DetectorCoordinate,
) -> HbnRingPoint: ...
def as_calibrant_observation(...) -> CalibrantObservation: ...
```

Names may sharpen during T10 contract review, but there is no separate hBN fit result. The GUI
module owns widgets, pointer translation, artists, and messages only. It calls the headless adapter
and T10 API and lazily imports Tk/Matplotlib.

### Launch interface

```text
hbn-ring-fitter [--hbn PATH] [--dark PATH]
```

No other command alias is retained. `--help` must work without opening a window. Missing optional
GUI support produces one actionable error.

## Project structure and affected files

Only this specification file changes now. Future implementation is limited to the following.

### T10 prerequisite, owned and reviewed before hBN

| Path | Intended change |
|---|---|
| `src/rasim_next/fitting/detector.py` | General physical ring residual, fit request/result, identifiability, and strict result export. |
| `tests/test_fitting.py` | Generic detector-recovery, held-out-ring, rejection, and artifact tests. |
| `docs/CONTRACTS.md` | Freeze the general calibration observation/result semantics only if T10 requires more than the current declared contracts. |
| `tasks/10_instrument_calibration.md` | Record the accepted API and proof. |

### hBN companion branch

| Path | Treatment |
|---|---|
| `pyproject.toml` | Modify: one optional `hbn` extra and one `hbn-ring-fitter` entry point. |
| `src/rasim_next/hbn_fitter.py` | Add: fixed-ring definitions, image preparation, snapping, observations, and T10 adapter. |
| `src/rasim_next/hbn_gui.py` | Add: minimal Tk/Matplotlib frontend only. |
| `tests/test_fitting.py` | Modify: two unique hBN frontend tests; no new test module. |
| `docs/SCOPE_AND_PHASES.md` | Modify: permit the optional companion only after accepted detector calibration; core GUI remains deferred. |
| `docs/ARCHITECTURE.md` | Modify: record a top-layer optional UI depending on `io` and `fitting`, never the reverse. |
| `docs/EXAMPLES.md` | Modify: document the tracked hBN/dark workflow and authoritative coordinate boundary. |
| `README.md` | Modify only when shipped: one concise launch and boundary note. |
| `tasks/index.yaml` | Modify: add one post-T10 hBN companion task. |
| `tasks/hbn_ring_fitter_plan.md` | Add during task planning; rename to `tasks/16_hbn_ring_fitter.md` when T16 is registered for execution. |
| `tasks/prompts/hbn_ring_fitter.md` | Add later during task planning: concise execution prompt. |

No hBN branch change is expected in `core`, `geometry`, `pipeline`, `selection`, ordered/stacking
physics, `docs/DOVETAIL_MATRIX.md`, the reference pack, or example data.

## Code style and boundaries

- Use frozen slotted dataclasses, small pure array functions, `Path`, explicit units, and explicit
  detector coordinate names.
- Keep one authoritative state object in the GUI. One placement function handles both new and
  replacement observations; do not duplicate preview/render paths.
- Keep numerical arrays immutable outside a short local calculation.
- Do not cache through module globals or hide mutable scientific state in callbacks.
- GUI callbacks may validate, call an API, update state, and render; they contain no ring geometry,
  fitting, uncertainty, or coordinate-conversion equation.
- Always reuse `read_osc` and T10. Ask before changing either public contract or adding a
  dependency. Never copy or execute legacy source in production or permanent tests.

## Commands

Future branch verification uses the accepted environment and exact commands:

```text
uv sync --frozen --group dev --extra hbn
python -m compileall -q src
ruff check pyproject.toml src/rasim_next/hbn_fitter.py src/rasim_next/hbn_gui.py tests/test_fitting.py
pytest -q tests/test_fitting.py -k "hbn or detector_calibration"
python -m rasim_next.proof instrument-calibration --json
uv build
uv run --frozen --extra hbn hbn-ring-fitter --help
git diff --check
```

One manual Windows desktop smoke run covers dialogs, precision placement, replacement, toolbar
exclusion, solve, overlay, and export. It produces no repository-local artifact.

## Testing strategy

### T10 prerequisite proofs

T10 must independently prove:

- recovery on a non-square detector with unequal pitches, offset center, nonzero two-axis beam
  direction, and known roll gauge;
- held-out-ring prediction;
- exact rejection of underdetermined pitch/distance, wavelength/distance, and free-roll requests;
- parameter rank, conditioning, active bounds, and multi-start consistency;
- strict result round-trip and instrument-revision invalidation.

### Permanent hBN tests

Add exactly two behavior-level tests to `tests/test_fitting.py`:

1. **Native preprocessing and observation state**: an asymmetric non-square OSC/dark case proves
   the one clockwise conversion, subtraction, raw/snap coordinates, snap rejection fallback,
   point replacement, and stale-result invalidation.
2. **Frontend-to-T10 boundary**: fixed hBN rings and scripted observations produce the expected
   `CalibrantObservation`, call the accepted T10 API, retain ring/provenance identity, and export
   without importing Tk or Matplotlib through the headless module.

Do not retain screenshots, pixel snapshots, pointer recordings, full-size derived arrays, broad
sweeps, copied legacy outputs, or tests of private widget layout.

### Required error injections

External proof must detect the first affected stage for:

- wrong OSC rotation, transpose, row/column swap, or half-pixel shift;
- swapped row/column pitch;
- reversed detector-normal or beam-direction sign;
- fitting raw points when accepted snapped points were declared;
- changing an observation while retaining a solved result;
- duplicate/missing hBN ring identity;
- estimating roll or pitch from an underdetermined ring-only request;
- GUI callback reimplementing a value that disagrees with T10 output.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Legacy raw/display coordinates leak into the new path | Plausible but rotated calibration | Reuse `read_osc` once, use typed detector coordinates, and inject asymmetric coordinate errors. |
| Powder rings are asked to identify roll or pitch and distance simultaneously | Numerically good but physically non-identifiable result | Fix pitch and roll, expose beam direction directly, report rank/conditioning, and reject underdetermined requests. |
| Snap chooses a nearby feature | Biased calibration hidden by a clean overlay | Retain raw and snapped points, gate the snap, show both, allow replacement, and retain per-ring residuals. |
| hBN code duplicates T10 physics | Divergent calibration equations | Make accepted T10 API a hard prerequisite; stop if missing. |
| GUI imports contaminate headless use | Core import or deployment failure | One optional extra, lazy GUI imports, and an import-boundary test. |
| Full-resolution images make interaction slow | Unusable pointer feedback and high memory | Keep one derived float32 display array, restrict snap work to a local patch, copy no full image during pointer events, and profile before optimizing. |
| Legacy-parity requests grow the branch | Recreates the 6,648-line tool | Treat every deferred feature as a new reviewed scope; enforce module, dependency, test, and LoC caps. |
| Copied GPL code contaminates a clean reimplementation | Licensing and provenance ambiguity | Use only documented behavior and independent equations; copy no source, tests, comments, or internal structure. |

## Acceptance criteria

The smallest useful release is accepted only when all of the following are true:

1. T10's detector-calibration API and proof are accepted before hBN code begins.
2. The hBN branch contains no file from `tools/hbn-ring-fitter`, no copied `ra_sim` module, no
   nested environment/configuration, and no production/permanent-test import of legacy source.
3. hBN-specific production code consists of exactly two new modules and no more than 800 physical
   Python lines total; exceeding the cap requires scope reduction and renewed approval.
4. Matplotlib is the only new dependency and is optional. Importing `rasim_next` or
   `rasim_next.hbn_fitter` without the extra does not import or initialize GUI libraries.
5. Both OSC inputs pass through the canonical reader exactly once, shapes must match, arrays remain
   detector-native, and every stored point is full-resolution `(column_px,row_px)`.
6. The user can load an exposure pair, choose any of the five fixed rings, place or replace a
   precision observation, inspect raw versus snapped state, undo/reset, solve, inspect residuals,
   and export the accepted T10 result.
7. Any observation, wavelength, lattice, or pitch change invalidates the prior result and disables
   export until a successful re-solve.
8. The fit recovers center, beam-intersection distance, and two beam-direction degrees of freedom
   within predeclared synthetic tolerances and predicts a held-out ring. Pitch, wavelength, and
   roll remain fixed and are recorded as such.
9. The result never represents detector roll as measured from powder rings. A complete instrument
   transform is emitted only when independent roll/lab-axis provenance is supplied.
10. Exactly two permanent hBN tests protect the coordinate/observation boundary and the hBN-to-T10
    boundary; all other GUI and performance evidence is temporary or manual.
11. Pointer preview performs no full-image copy and meets a recorded p95 event-to-preview latency
    of at most 100 ms on the branch reference machine.
12. Compile, lint, focused and full permanent tests, T10 proof, package build, headless import,
    `--help`, manual desktop smoke, error injections, wall time, peak memory, and `git diff --check`
    pass with a clean intended diff.

## Resolved first-release decisions

1. Session persistence remains deferred.
2. Precision place/replace and undo are sufficient; T10 estimates beam center, so no separate
   center-pick mode is included.
3. T16 waits for completed, accepted T10 rather than a draft contract slice.

Implementation remains blocked until T10 satisfies the prerequisite and work begins in an
isolated worktree created from the approved `main` baseline.
