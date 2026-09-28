# Native desktop UI grand plan

Status: accepted feature scope; planned implementation, not delivered functionality.
Updated: 2026-09-28. Repository baseline inspected: `32af6af`.

Build one local desktop application for inspecting detector images, understanding experimental
geometry, performing staged fitting and running independent simulations. This document consolidates
the conversation's interface decisions and all requested additions. The checklist below is the
single implementation backlog for this work, linked from [PLANS.md](../PLANS.md).

Current [architecture](ARCHITECTURE.md), [contracts](CONTRACTS.md),
[conventions](CONVENTIONS.md), [result measures](RESULT_MEASURE.md) and
[assessment policy](VALIDATION.md) remain authoritative. This plan introduces no scientific model,
fit qualification, numerical tolerance change or permission to resume historical fitting campaigns.

## 1. Application and workflow

Use a native PySide6 desktop shell with two workspaces: **Fit experiments** and **Simulator**.
Keep files and calculations local. Support single images, multiple incident angles, multiple
specimens/mounts and an optional powder calibrant such as hBN. A sample-only experiment is allowed;
report the calibration assumptions and identification limits associated with its available data.

The fitting workspace has a project/acquisition browser on the left, linked scientific views in the
center, a parameter/selection inspector on the right, and a collapsible result/job area below.
Panels are resizable and can be enlarged individually; remember the layout. Use keyboard navigation,
visible focus, explicit units, high-DPI text and shapes/labels in addition to color.

Default stage order:

1. Import and inspect images; assign source, specimen, mount, calibrant, dark and masks.
2. Geometric fitting, including calibrant/detector calibration and sample/goniometer alignment.
3. Mosaic fitting with the declared upstream geometry and source state.
4. Ordered structure/intensity fitting under the current native observation contract.
5. Stacking-disorder fitting when the material binding supports it, with an ordered control.
6. Review, qualified downstream handoff and reproducible export; optional joint refinement only
   through a supported, explicitly configured current fitting recipe.

Stages organize the interface; they do not create new optimizers or revive retired staged engines.
Inspecting an unqualified candidate is allowed. Using it as a qualified upstream result is not.
Keep upstream revisions visible and mark affected downstream results stale after an input edit.

## 2. Import, acquisition management and comparison

- Import one or many supported detector files by file picker or drag/drop. OSC is the initial
  format; further formats need explicit readers. Show thumbnails and load full images lazily.
- Group by instrument setup, specimen/mount and acquisition. Store stable IDs, file hashes,
  commanded incidence, exposure, detector calibration and material/calibrant assignments.
  Incidence is an explicit declaration, never inferred authoritatively from a filename.
- Associate dark/background and masks explicitly. Retain original counts and the correction ledger;
  a display transformation does not change the observation used by a fit.
- Detect incompatible shape/calibration, missing files, duplicate IDs and unassigned metadata before
  fitting. Show which record needs repair while keeping other images inspectable.
- Provide a thumbnail filmstrip, keyboard previous/next, and two-image comparison. Link pixel zoom
  and pan when detector calibrations are compatible; otherwise state the comparison frame.
  Shared display limits are optional and visibly locked. Exposure-normalized comparisons require
  an explicit mode and units; independent automatic rescaling must not imply intensity agreement.

## 3. Detector inspection and fitted overlays

Keep the detector image central, in its canonical native orientation, with linear/log display,
contrast limits, saturation indication, pan, box zoom, fit-to-image and native-pixel inspection.
Provide a magnified selected region without replacing the main context view.

Overlay layers are independently selectable:

- Fitted peak/landmark positions as hollow markers and fitted powder rings as thin curves.
- Observed feature positions, residual connectors, feature/reflection labels, fit regions and masks.
- Distinct styles for fit observations, held-out validation, excluded observations and ambiguous
  assignments; show overlap identities together without duplicating a physical observation.
- Optional uncertainty ellipses or ring bands only when supported by qualified covariance evidence.
  State whether they describe measurement, parameter propagation or prediction uncertainty.

Fitted locations come from the saved result's canonical predictor and ring geometry, not a new
ellipse fit or a detector-image maximum. Nominal exact-L landmarks are not necessarily intensity
maxima. Draft geometric predictions, optimizer candidates and selected results have distinct labels.
An unfitted image has no fitted overlay. Stale saved fits are hidden by default and can be explicitly
shown with dashed styling and their old revision; their saved positions never move with a draft edit.
Only compare residuals against the observations bound to that result.

Click a peak/ring, or choose it in the keyboard-accessible feature list, to inspect its stable ID,
reflection/rod/branch identities where defined, observed and predicted coordinates, signed residual
components, residual magnitude in pixels, uncertainty evidence and inclusion status. Highlight its
corresponding reciprocal-space feature and geometric ray when defined. Offer zoom to feature.
Label decluttering may hide text but never silently remove observations or change their fit weights.

The cursor reports `(column_px, row_px)`, underlying intensity and units, mask/saturation status,
and available scattering angle/reciprocal coordinates. Name the frame, external-air versus
internal-film Q, and draft/saved geometry revision. Invalid mappings display unavailable with a
reason; do not substitute a plausible coordinate.

## 4. OSC-style horizontal and vertical marginal profiles

The detector panel includes a horizontal profile above or below the image and a vertical profile
along its right edge. Their position axes align exactly with detector columns and rows, including
the image's row direction. The vertical plot places intensity on its horizontal axis. Both plots
remain linked to image zoom and pan without changing their integration support implicitly.

### Interaction

- A movable crosshair selects any detector point. Hover follows the cursor; click pins it so the
  user can inspect elsewhere. Numeric column/row fields and arrow-key stepping provide precise,
  accessible placement. Unpin restores cursor following.
- Show a shaded horizontal band through the selected row and vertical band through the selected
  column. Each has an independent pixel width with draggable boundaries and exact numeric entry.
  A one-pixel band gives an ordinary row/column line cut.
- Default to band-integrated **sum**. Offer **mean per valid pixel**, full-detector projections and
  integration within an explicit selected ROI. The selected mode, bounds and effective valid-pixel
  support are visible. Panning alone never switches full-image integration to the viewport.
- Profiles update while moving the crosshair or bands. Allow a pinned reference profile for
  comparison. Selecting a fitted feature can center the crosshair while preserving chosen widths.
- Add user-drawn straight-line profiles with an explicit sampling/interpolation and strip-width
  policy; keep their display sampling distinct from native-pixel count integration.
- When compatible data exist, optionally compare measured, model and residual profiles using the
  same support, mask, units and fit revision. A geometry-only fit supplies positions, not an
  invented intensity profile. Uncalibrated simulation mass is not labeled measured counts.

### Numerical meaning

Let `I[r,c]` be the selected underlying native data plane, and `M[r,c]` its Boolean inclusion mask
after declared invalid-pixel policy. For horizontal row band `[r0,r1)` and vertical column band
`[c0,c1)`, with bounds clipped explicitly to the detector and any selected ROI:

```text
H[c] = sum over r in [r0,r1) of M[r,c] * I[r,c]
V[r] = sum over c in [c0,c1) of M[r,c] * I[r,c]
NH[c] = sum over r in [r0,r1) of M[r,c]
NV[r] = sum over c in [c0,c1) of M[r,c]
```

Implement masking without `0 * NaN` propagation. Mean mode divides H/NH and V/NV only where support
is positive. Zero-support bins are missing, not zero intensity. Sum mode does not compensate for
masked pixels. Display actual clipped band bounds and support near edges/gaps. Raw counts sum as
counts; corrected data retain their declared units and signed values. Never sum log-transformed,
colormapped, clipped, resized or float32 presentation pixels. Use overflow-safe accumulation for
integer counts and float64 for corrected values. A log axis explicitly omits nonpositive plotted
values while retaining them in stored profiles and linear/signed views.

Native marginal profiles are inspection observables. They do not replace the scientific fitting
objective or reciprocal/angle-region projector. Any later use as fit observations needs its own
declared support and covariance boundary. Error bars are optional evidence, not an independent
Poisson assumption applied to dark-subtracted or correlated data.

## 5. Masks, feature review and reciprocal-space inspection

Provide rectangular, polygon and brush mask tools, mask import, undo/redo and visible excluded-pixel
overlays. Represent beamstops, detector gaps, saturated pixels and user exclusions with reasons.
Display-layer visibility and fit inclusion are separate controls. Selection, dark/background,
mask and assignment changes create new observation revisions. Freeze them for a running fit;
edits made during that run belong to a new draft. Do not silently censor poor residuals or introduce
intensity-dependent masks into a recipe whose observation contract prohibits them.

Feature review supports accepting/rejecting candidates and correcting assignments before fitting.
Retain original discovery, edited value, reason and provenance; validate a new frozen manifest
through current selection ownership. Flag overlaps/ambiguous assignments and unsupported manual
admissions. Fits never reassign branch identities during optimization.

For every acquisition, show approximate reciprocal-space coverage from the active geometry and
link selections to detector features. Identify geometry-only previews and their approximation.
Use shared geometry/measurement functions; show validity/coverage and gaps explicitly. Offer saved
versus draft comparison. Reciprocal display bins, sparse meshes and label limits cannot modify
native fit observations, remove physical rods or redefine the displayed signal's measure.

### Beam-center estimation tool

Provide **Choose beam center** from the detector toolbar and the detector/beam callout in the
experiment view. The small approximately Gaussian spot is a candidate wherever it occurs in an
hBN or sample image; neither the image midpoint nor the brightest diffraction peak is an automatic
beam-center assignment. Let the user select an ROI anywhere, click an approximate center, or enter
native `(column_px, row_px)` coordinates directly. Optional local suggestions remain proposals.

Offer three explicit estimate sources:

1. **Direct-spot estimate:** fit an elliptical 2D Gaussian with a constant or explicitly selected
   sloping background inside the user ROI, using original native values and the declared noise
   treatment. Seed from local background-corrected moments where usable; permit unequal widths
   and orientation. Report subpixel center, widths, background, residual map/profiles, valid-pixel
   coverage, bound contacts and supported center uncertainty. Positive widths/amplitude and ROI
   limits are validated. The Gaussian describes the local spot, not the scattering physics.
2. **Calibrant/sample geometry estimate:** obtain center from the current physical hBN ring model
   or an explicitly supported sample-landmark geometry fit when enough independent constraints
   exist. Reuse the existing predictor and report coupling to distance, tilt, wavelength and sample
   alignment. A ring's fitted ellipse center need not be the direct-beam intercept on a tilted
   detector. Unsupported or rank-deficient sample data cannot produce a qualified center.
3. **Manual estimate:** click/drag a separate center marker or type coordinates; retain its source
   as a user estimate with no invented uncertainty. A direct beam outside the active panel can be
   represented as an off-panel numeric estimate when the geometry contract permits it.

Exclude declared invalid/saturated pixels without treating them as zero. If a spot is clipped,
beamstop-obscured, saturated, weak, overlapped, non-Gaussian or unstable under a modest ROI change,
show those limitations and leave the estimate unqualified. Do not move it onto a nearby reflection
to manufacture success. A fit to unsaturated tails may remain an explicitly approximate seed.
Overlay its center/contours and show the two marginal profiles through it so the user can review
the proposal before **Use as initial estimate**. A normal inspection crosshair never edits geometry.

Record acquisition, ROI, mask/correction revision, method, fitted local parameters and quality
evidence. The observed direct-spot center and the geometric incident-beam intercept are separately
named until the chosen acquisition/beam-path assumptions justify equating them; a specular or
sample-scattered spot is not automatically the transmitted/direct beam.

Compare estimates across hBN and sample images without forcing agreement. Applying a center to a
shared setup is explicit, shows affected acquisitions and validates matching detector/beam settings;
otherwise retain it per acquisition as a proposal. The geometry owner converts the chosen intercept
to its reference-coordinate/rigid-pose convention once; no extra image shift or duplicate beam-center
parameter is added to the residual path. A seed, a fixed calibration value and a weighted calibration
observation are different roles. Adding an observation requires an admitted covariance model, and
derived estimates cannot be counted again as independent evidence from the same pixels/rings.

Run local spot estimation on a small ROI in a cancellable worker after the ROI settles or the user
requests it. Reuse the loaded image and prepared profiles; moving an ordinary cursor never launches
a Gaussian fit, full-image search or full geometry fit. Superseded proposals cannot apply themselves.

## 6. Interactive experiment geometry

Render beam, sample/holder, goniometer axes and pivots, and detector with the selected actual image
and optional fitted overlays on its plane. Keep detector texture, masks and overlay coordinates
aligned with the 2D view. Distinguish schematic hardware from dimensions supplied by the experiment;
this is a geometry editor, not a hardware-control or collision-clearance system.

Show editable values near their physical objects, with leader lines and an expandable inspector:

| Object | Controls and context |
| --- | --- |
| Beam | Direction, source/reference position and supported source size/divergence settings; wavelength/spectrum in the inspector |
| Goniometer | Commanded motor angle, declared axis direction/misalignment, LAB pivot and supported corrections, ordered motion chain |
| Sample | Mount/crystal orientation, local rotations, normal displacement and identifiable translations, sample normal and beam footprint |
| Detector | Distance, column/row translations, column/row tilts, in-plane rotation, declared pivot/reference point, shape/pitch and center calibration |

Click an object or its label to select it and animate the camera to a useful close-up; include the
associated axis/pivot in the framing. Suppress unrelated callouts, keep context breadcrumbs and
provide Back to experiment. Camera presets include front, side, beam direction and detector normal.
Orbit/pan/zoom and reset camera never modify physical parameters. Respect reduced-motion settings.

Selecting a parameter highlights its axis or pivot. Use rotation arcs and translation arrows plus
exact numeric fields, units, bounds, positive-rotation direction and fine adjustment. Separate
camera gestures from parameter drags; one drag becomes one undo action. Show the coordinate triad
and reference frame. Edits apply complete canonical rigid transforms about declared pivots.
Keep effective reduced sample corrections distinct from separately measurable motor errors.

All supported simulator parameters remain inspectable. A fit can activate only coordinates admitted
by its model, data and gauge constraints. A fixed reference or unidentifiable parameter is explained
and cannot acquire a fabricated fitted standard error. Display which acquisitions a shared edit
affects before committing it; preview, apply, undo and revert preserve saved results.

### Initial estimates in the experiment view

Every experimental coordinate admitted by the chosen fit exposes an **Initial estimate**, unit,
scope, allowed bounds and fixed/fitted status through its physical callout and the synchronized
parameter table. This includes beam center, distance, detector tilts/roll/shifts, incident-angle
zero/trim, sample orientation/offsets, and goniometer axes/pivots where supported. Source parameters
have beam callouts; nonspatial model parameters use the inspector. Display dependent/derived
quantities read-only and explain gauges instead of creating redundant adjustable coordinates.

Allow typed values, constrained scene handles, configuration import, compatible prior-result
values and the reviewed beam-center proposal as starting values. Show the source of each estimate,
restore defaults or a saved seed set, and preview its detector/reciprocal consequences. Applying
an estimate does not start optimization. Snapshot the complete starting vector, fixed values,
scopes and bounds at **Fit**; retain it separately from live draft values and returned fit values.
Provide initial/current/fitted comparison and undo/revert without overwriting saved results.

Validate units, finite values, physical domains and active-set/gauge compatibility before launch.
An initial estimate is neither a prior penalty nor an uncertainty interval. A user may change
bounds independently within the core's admitted physical range; silently clipping a bad seed or
activating an unsupported parameter is prohibited.

## 7. Parameters, statistics and uncertainty

Keep ownership scope separate from fitting status: a shared instrument parameter can be fixed or
fitted. Provide scope groups **shared instrument/setup**, **specimen/mount**, and **per image**;
provide an independent **fixed/fitted** control and a fixed-input view. Never allow a scope selector
to change the physical ownership admitted by the fitting contract.

Parameter rows show value, unit/frame, starting value, allowed bounds, affected images, fixed
reference reason and supported uncertainty evidence. Show residual RMS/max and counts per image,
ring/reflection and combined fit where meaningful, along with actual objective/weighting, rank,
conditioning, correlations/weak directions, active bounds and held-out performance. Fixed inputs
show provenance rather than fitted uncertainty. Correlated marginal group scores are not additive.

Keep optimizer termination, numerical qualification, identifiability and physical/model limitations
as separate statuses. A current result can still be unqualified. Missing covariance means
uncertainty unavailable; local sensitivity/SVD diagnostics are not posterior confidence intervals.
Parameter selection can request a small bounded sensitivity preview showing feature motion, with
perturbation size, held-fixed assumptions and invalid/branch-changing cases. Run it on demand in a
worker, label linearized previews and do not fabricate uncertainty bands from hard parameter bounds.

## 8. Independent simulator and results

Expose all parameters supported by the selected current simulator, grouped into source/spectrum,
instrument, specimen/crystal, mosaic, ordered structure, stacking/disorder, optics/specular and
numerical execution. Include search, units, defaults, valid ranges and parameter descriptions.
Show unsupported combinations explicitly. Material presets configure supported bindings; a CIF
alone does not establish a disorder law, acquisition, background or generic full-image capability.

Provide a responsive geometric/qualitative preview and a separate requested quantitative run.
Report execution backend, observable measure, numerical settings, progress, cancellation and
qualification. Reuse detector/profile/geometry views and allow explicitly copying an experiment
snapshot into a simulation draft or adopting a compatible result into a new experiment draft.
Never silently alter an accepted experiment while exploring the simulator.

Before/after comparison locks image/support/units and distinguishes input, candidate and selected
results. Save/reopen a project with numeric configuration, file identities, acquisition grouping,
masks/assignments, drafts, selected result references and view layout. Validate file hashes and
schema when reopening; relocation requires verification. Do not use executable serialized objects.

Export detector figures with optional overlays, profiles with support/units, feature tables,
parameter tables and the fit/configuration provenance needed to reproduce them. Preserve original
inputs. Generated results and diagnostics remain external to the repository; existing result
writers and the one-external-`.ra_diag.npz` diagnostic contract remain authoritative. Exported
figures/tables are user-requested result products, not a second hidden diagnostic format.

## 9. Small implementation architecture

Reuse `interactive/detector_viewer.py` for its current full-native presentation, progressive worker,
input coalescing and invalidation behavior; reuse the geometry consumers in
`interactive/ewald_sphere_viewer.py`. These tools stay usable during incremental extraction.
Extract only helpers shared by actual consumers, into narrow optional presentation modules.
Keep GUI imports/device initialization out of the numerical package's normal import path.

| Concern | Existing authority / proposed boundary |
| --- | --- |
| OSC bytes and orientation | `src/rasim_next/io/osc.py`; once-only clockwise conversion |
| Physical geometry and Q | `src/rasim_next/geometry/`, configured geometry-only context and canonical detector mapping |
| Calibrant and crystal fitting | `fitting/hbn.py`, `joint_geometry.py`, `indexed_series.py`, `joint_geometry_handoff.py` |
| Beam-center proposals and starting values | Native ROI spot estimator in a narrow selection helper; existing calibration model for ring/sample estimates; presentation controller for reviewed seed application |
| Observation identities | `selection/`, `fitting/native_observations.py`, existing measurement projectors |
| Mosaic, ordered and disorder inference | `NativeRefinementModel`, `native_workflow`, `native_search`, `native_accuracy`, current material bindings |
| Simulation | `pipeline/configured_simulation.py` and supported native/Monte Carlo execution owners |
| Presentation/session | Small explicit project controller, immutable job snapshots, bounded owned caches, Qt widgets and worker results |

Use PySide6/Qt and the existing optional OpenGL image path first. PyQtGraph is a candidate for
linked axes, profiles and ROIs only if a small integration comparison shows that it removes more
code/maintenance risk than it adds; it is not currently a dependency. Keep any addition in the
visualization extra. Do not add a browser shell, web service, generic plugin system, second physics
engine or interchangeable backend framework for this UI.

Widgets stay on the GUI thread; workers send results through queued delivery and never mutate
widgets. This follows the [Qt worker-object guidance](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).
If PyQtGraph is selected, configure native row-major interpretation and explicit display levels;
its [ImageItem documentation](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html)
is an implementation reference, not evidence that a particular workload meets latency targets.

Keep immutable scientific/result snapshots separate from mutable camera, selection and display
state. Jobs bind acquisition ID and data/mask/calibration/model revisions plus a request generation.
Reject stale publications, including A-to-B-to-A selection races. A completed fit for an older draft
may remain inspectable as a labeled historical result, but cannot replace the current selection.

## 10. Responsiveness and bounded resource use

Responsiveness is an acceptance requirement. The following are proposed engineering targets,
not measurements or guarantees already achieved. Start with the existing 3000 x 3000 native
detector, one powder plus three sample images, and a stress case of twenty lazily loaded images.
Record CPU, GPU, RAM, backend, screen scale and cold/warm state; record disk decode/JIT separately.

| Warm interaction | Target on the declared reference machine |
| --- | --- |
| Pan, zoom, crosshair and camera feedback | 60 fps target; p95 frame time <= 33 ms |
| Two native band profiles following a crosshair | p95 input-to-visible profiles <= 50 ms after profile preparation |
| Selecting an already prepared image | p95 visible image/overlay/profile update <= 150 ms |
| Parameter edit, start/cancel command acknowledgment | <= 100 ms; expensive computation has independent progress |
| Superseded preview work | Cancellation observed at bounded checkpoints; aim <= 100 ms on the reference workload |
| UI while loading/fitting/rendering | No synchronous heavy compute or I/O on the GUI thread; no stale result publication |

Do not promise fixed full-fit or full-simulation completion time. Show initialization/progress and
keep navigation responsive. A missed target requires measured diagnosis; do not silently reduce
scientific support, change tolerance or relabel an approximation as a completed result.

### Profile and image work

- Read/hash/decode once per acquisition revision in a worker. Cache only the selected/compared
  native planes plus a bounded working set; prefetch nearby thumbnails with low priority.
  Avoid a full image copy, file read or whole-image reduction on each mouse event.
- For one-pixel/narrow bands start with direct native slices. If timings justify it, prepare one
  summed-area table for signal and one for valid support for each active data/mask revision.
  Four-corner differences produce a whole horizontal/vertical profile in O(width + height), after
  O(width * height) preparation. Choose the direct path or prefix path explicitly by workload.
  Verify prefix subtraction against direct reductions, including large backgrounds/small signals.
- A 3000 x 3000 float64 plane is about 68.7 MiB. Float64 signal plus uint32 support prefix tables
  cost about 103 MiB per plane, in addition to image/texture storage. Set and expose a bounded cache
  budget; do not allocate these tables for every loaded image. Check support-counter overflow for
  larger arrays. Evict only reconstructible buffers, never source data or accepted results.
- Update only the crosshair, band handles and two 1D curves when the cursor moves. Cache full-image
  marginals. Key profile reuse by data, mask, correction, bounds and reduction mode; log/contrast
  changes do not recompute native profiles. Mask edits prepare a new generation atomically.
- Preserve full-native image textures and authoritative precision. Float32 GPU frames are
  presentation only. Reuse textures/buffers and change shader levels/colormaps without rerunning
  physics. Share owned texture data between 2D and 3D where the Qt context permits it; otherwise
  upload once per data revision to each context, never once per camera movement.
- Reuse vector overlay items and profile curves. Clip offscreen drawings and labels without
  changing observation membership. Screen-pixel min/max envelopes may reduce drawn 1D vertices
  while exact profile values, extrema and exports remain available.

### Jobs and invalidation

- Extend the existing latest-only preview scheduling; coalesce input to the next paint tick,
  retain at most the running request and newest pending preview per view, and drop superseded
  progress frames. Never launch one worker per mouse move or parameter control.
- Separate priority interactive previews from explicit fit/quantitative jobs. Bound CPU workers,
  BLAS threads and GPU ownership so a fit cannot starve navigation. Use a dedicated process where
  required by Python-heavy work or thread-unsafe numerical/database resources; queue numeric
  snapshots, not live mutable models. No generic job-service layer is needed.
- Use cooperative cancellation; never force-terminate a thread holding scientific/device state.
  Keep GUI/OpenGL context affinity and the existing thread-confined sampler contract. A leased
  preview frame must be consumed/copied safely before the sampler reuses it.
- During parameter dragging, immediately update scene transforms and inexpensive geometric
  landmarks. Update selected-image reciprocal coverage at a bounded cadence. On release, request
  its settled preview, then other visible images. Hidden panels need no paint/evaluation until used.
- Preserve existing full-native progressive Monte Carlo prefixes, source/rod/root support and final
  requested draw count. Source-count changes commit on release; previews never qualify a fit.
  CPU/CUDA and presentation choices remain explicit, with no silent fallback.

| Change | Work that should become dirty |
| --- | --- |
| Camera, selection, overlay visibility | Presentation and selection only |
| Contrast/log/colormap | Presentation shader/axes only; original data and fits unchanged |
| Crosshair or band | Native profile query and local overlays only |
| Beam-center ROI/estimate or initial parameter value | Local proposal or canonical geometry draft only; full fitting starts explicitly |
| Observation mask/dark/assignment | Profile cache and observation manifest; dependent results become stale |
| Detector rigid pose | Canonical detector projection, Q mapping, markers and dependent results; reuse incident transport where the current contract permits |
| Sample/goniometer/source/material | Corresponding canonical transport/geometry/physics dependencies; no blanket projection-only shortcut |
| Structure/mosaic/numerical settings | Only the responses and execution state whose declared keys depend on that change |

## 11. Implementation sequence and acceptance checklist

All entries below are unimplemented. Paths name likely ownership, not a mandate for one module per
task. Keep each slice to a few files; subdivide before broad changes. Dependencies are task IDs.
One writer owns edits. The numbered checks are focused acceptance work for that slice, performed
externally where code is needed, then removed under [VALIDATION.md](VALIDATION.md).

| ID / delivery slice | Dependencies and likely paths | Acceptance and verification |
| --- | --- | --- |
| [ ] U01: project/acquisition shell | None; `interactive/slate_app.py`, narrow project-state helper, `interactive/README.md` | Import one/many OSCs with explicit grouping/angles; reject duplicate or incompatible metadata; verify native coordinates on a non-square input and clean close/reopen with existing I/O. |
| [ ] U02: native detector inspection | U01; app, shared optional detector presenter | Pan/zoom/native pixels, contrast and lazy filmstrip work; switching images never mixes IDs; inspect full-native orientation/precision and warm frame timing. |
| [ ] U03: marginal profiles | U02; narrow profile helper and detector panel | Both aligned profiles follow/pin crosshair and independent bands; sum/mean/full-image/ROI modes handle masks, edges, signed/nonfinite data and zero support; compare with direct reductions and measure prepared cursor latency. |
| [ ] U04: masks and regions | U03; detector panel, project-state helper | Rectangle/polygon/brush masks have undo and reasons; observations remain frozen during a run; verify profile/fit invalidation and separate display/inclusion controls. |
| [ ] U05: saved fits and feature inspection | U02/U04; result binding, overlays/inspector | Peaks/rings, feature list, labels, held-out/excluded states and residuals match saved IDs/coordinates; reject mismatched provenance; verify unfitted/current/stale/candidate states and unavailable uncertainty. |
| [ ] U05a: beam-center proposals | U03/U04/U05; narrow ROI estimator, detector tool and calibration consumer | Manual and Gaussian ROI proposals plus supported hBN/sample-derived estimates retain source/reliability; shared application is explicit and unsupported spots remain unqualified; check known off-center/non-square spots, background, saturation/occlusion, rejected stale proposals and exact seed handoff. |
| [ ] U06: linked comparison and profiles | U03/U05; panel layout/profile controls | Two images support compatible linked pan/limits, magnifier and straight-line cuts; model/profile comparisons use matched measures/support; verify pinning, exposure labels and before/after selection. |
| [ ] U07: reciprocal geometry preview | U05; canonical geometry consumer, reciprocal panel | Every image has on-demand draft/saved coverage; linked feature selection and cursor Q use the declared frame; verify valid/off-panel cases against current geometry APIs and stale-job rejection. |
| [ ] U08: textured experiment scene | U02/U05; scene helper, app | Beam/sample/goniometer/detector use canonical poses and image/overlays; click-to-zoom, context return and camera presets work; verify detector texture corners and that camera actions preserve parameters. |
| [ ] U09: physical parameters and initial estimates | U05a/U07/U08; parameter metadata/inspector and scene | Callouts/handles and table edit one complete initial vector with scopes, units, bounds, provenance and gauges; launch snapshots preserve initial/current/fitted values; verify beam-center seed mapping, nonzero compound rotations, one-drag undo and dependent invalidation. |
| [ ] U10: geometry fitting jobs | U04/U05/U09; job/controller and current geometry fit boundaries | Single/multi-image and calibrant/sample workflows call supported owners; results preserve fixed/local/shared diagnostics and qualification; verify cancellation/revision races with small controlled jobs, not full scientific campaigns. |
| [ ] U11: supported later fitting stages | U10; current native plan/result binding and stage panel | Mosaic/ordered/disorder controls reflect actual material support and ordered controls; hash-bound handoffs and stale-stage handling work; verify nominal/candidate/selected results and unsupported cases using current records. |
| [ ] U12: independent simulator | U02/U03/U09; simulator forms/current viewer worker | Every supported parameter is available with units/validation; preview/quantitative jobs and measures are distinct; verify copied experiment drafts stay independent and unchanged current CLI configurations still load. |
| [ ] U13: sensitivity and supported uncertainty | U05/U09/U10; result inspector and narrow worker task | On-demand parameter perturbations show motion and validity; covariance bands appear only with defensible evidence; compare a tiny supported case with canonical predictions and preserve unavailable/unidentified states. |
| [ ] U14: persistence and export | U06/U10/U11/U12; project I/O and existing result writers | Save/reopen restores bindings, edits and layout; altered/moved input files require verification; exported profiles, features and figures preserve native coordinates, support, units, fit status and provenance. |
| [ ] U15: responsiveness and release review | Each earlier slice measures its own risk; app, optional dependency declaration and documentation | Meet the declared latency/resource targets with simultaneous cursor/3D/fit activity; verify bounded memory, cancellation, failure recovery and optional GUI imports; record cold/warm limits and remove temporary checks. |

Checkpoints:

- [ ] After U03: real detector images and exact marginal profiles remain responsive; choose any
  optional plotting dependency from this measured slice before expanding the interface.
- [ ] After U05a/U06: beam-center, image/feature/mask review and comparison are useful with saved results, independently
  of launching a new optimizer; no fit observations are altered by presentation actions.
- [ ] After U10: the complete geometry workflow has one parameter state, canonical transforms,
  preserved gauge/qualification rules and a verified stale-job boundary.
- [ ] After U15: review the coherent diff, optional imports, relevant formatting/lint/build checks,
  user flows and remaining scientific limitations. Report actual measurements separately from targets.

## 12. Risks, limits and completion evidence

- Historical task timing is useful context, not a current guarantee. No timing claim in this plan
  has been measured for the combined UI. Profile before adding an acceleration or larger cache.
- Current native Bi models do not provide stacking-disorder coordinates. Generic CIF support does
  not imply a generic full-image simulator. Material-specific missing capabilities remain visible;
  expanding scientific support requires a separate scoped core change.
- Do not replace the reduced joint geometry gauge with independently fitted mechanical parameters
  merely because the scene can draw them. New pivots/axes or scope combinations need existing
  contract support and identification evidence before becoming fit controls.
- Masks/manual assignments must remain admissible for the chosen observation recipe. A flexible
  inspection tool does not authorize changing a running objective or hiding unfavorable data.
- A Gaussian-looking sample feature is not proof of the incident-beam intercept. Preserve the
  selected spot's physical interpretation and report disagreement with ring/sample calibration;
  a convenient starting guess cannot become an independent calibration constraint by relabeling.
- CPU-only and lower-memory systems keep the same scientific semantics with explicitly reported
  slower preparation/compute. Test the selected presentation backend rather than silently swapping it.
- Future scientific changes must name observables and independent evidence; UI/render checks alone
  cannot certify a fit, uncertainty, physical adequacy or numerical convergence.

The present deliverable is this documentation and its plan-index link. No production implementation,
dependency installation, fit, simulation, performance run or retained checking harness is part of
this planning change. Implementation completion requires evidence for every checked slice and one
coherent reviewed commit per delivered change, following the repository's main/worktree policy.
