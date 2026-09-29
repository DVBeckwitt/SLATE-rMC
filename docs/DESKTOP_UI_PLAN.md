# Native desktop UI grand plan

Status: accepted feature scope; staged implementation in progress.
Updated: 2026-09-28. Delivery breakdown incorporates the audited plan at `a80e995`.

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

### First successful session

The opening screen offers **Import images**, **Open project**, **Recent projects**, **Open example**
and **Simulator**. Importing an image must make its raw detector view available before demanding
every scientific setting. Use existing example acquisitions for a short optional guided tour;
do not add a second set of demonstration physics or test fixtures.

The default guided path is:

```text
Import images -> review roles and acquisition metadata -> choose instrument/material
-> inspect image and beam center -> set geometry starting values -> review fit readiness
-> run geometry fit -> inspect and choose a result -> continue or save/export
```

Users can leave setup to inspect images and return without losing their progress. The simulator
can be opened independently. Expert users can navigate directly to any inspectable stage;
execution still honors that stage's scientific prerequisites.

Start with the selected image and its profiles as the dominant view. Show the experiment scene
when setting geometry, reciprocal space when reviewing coverage, and results when evaluating a
fit. Keep optional panes collapsible and remember layouts rather than showing every panel at once.
Use one context-sensitive inspector and one clear next action per stage. Common controls appear
first; advanced sections and search expose every supported parameter without duplicating forms.

Every tool has a visible active mode: Inspect/select, Set beam center, Draw mask or Measure profile.
Escape returns to Inspect/select and cancels the current gesture. Clicking the image while masking
must not change the beam center. Give concise inline gesture hints, tooltips and keyboard equivalents;
use familiar names such as "shared across these images" with formal scope details available nearby.

### Guidance and repeat experiments

Save named instrument and experiment templates containing declared source/detector settings,
material/mount defaults and compatible starting values. Show where a value came from: file metadata,
template, manual entry, calibration or previous fit. Copy templates into project snapshots;
later template edits cannot alter old projects. Applying a calibration to another acquisition
requires matching setup/pose and an explicit affected-image review.

Show a compact readiness panel with concrete actions, such as **Enter angles for 2 images**,
**Choose a material**, **Review ambiguous peaks**, or **Resolve redundant active parameters**.
Clicking an item opens the relevant field or view. Disabled actions have a visible reason, not
only a hover tooltip. Missing metadata, unsupported capabilities and scientific failures remain
distinct. Do not replace rank/qualification evidence with a readiness percentage or confidence score.

Before starting a fit, show its images, parameter scopes, starting values, fixed references and
output destination together. During execution show the current activity, available progress and
Cancel; estimate remaining time only from defensible run information. Inspecting another image
must remain possible. On failure, preserve the draft and partial diagnostic evidence, explain the
condition in plain language and offer an appropriate edit/retry action with technical details
expandable. Offer resume only where the existing execution contract supports it.

At completion, summarize what changed, which images/features still disagree, and which conclusions
remain unresolved. Provide **Compare with initial**, **Name this result**, **Use as selected result**
when qualified, and **Continue to next stage** when prerequisites pass. Keep named attempts and
their inputs available; a later run never silently overwrites the user's selected result.

## 2. Import, acquisition management and comparison

- Import one or many supported detector files by file picker or drag/drop. OSC/OSC.GZ are the initial
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

### File import and metadata review

In this local desktop app, label the action **Import files**. Accept individual files, multiple
selections and a folder with a reviewable candidate list. Never recursively import an entire
folder tree without showing its scope. Expose the supported extensions in the file picker/drop
area; report unsupported files individually. Allow images to be added to an existing project.

The import review is a compact table with thumbnail, filename, role (sample/calibrant/dark/mask),
specimen/mount, commanded angle, exposure, shape, detector setup and status. Provide multi-select,
paste-from-spreadsheet metadata, an explicit CSV column mapping, and **Apply to selected** for shared
settings. Allow sorting/reordering without changing acquisition IDs. Header/filename suggestions
show their source and remain unconfirmed until reviewed; angles are never accepted silently.
Provide file pickers for supported material/CIF and configuration inputs as well as calibrant presets.

Decode and validate in the background with per-file progress/cancel. Keep successfully imported
images when another file is corrupt, missing or incompatible, and offer retry/relink on affected
rows. Exact content duplicates and repeated nominal angles are different conditions: duplicate
imports should be flagged; legitimate repeated exposures at the same angle remain representable.
Use declared acquisition identity when deduplicating, not filename alone.

Make the storage choice explicit: reference files in place by default, or copy them into the user's
external project data folder. Display source and project locations and show the size before a
large copy. Never modify originals. **Remove from project** removes a reference, not the source
file. Relinking a moved file verifies its identity and cannot substitute different image bytes.

Before fitting, the review indicates unresolved required fields. Raw image browsing still works.
Users must be able to import an hBN-only calibration, a sample-only image, a multi-angle sample
series or a mixed calibrant/sample project without following an irrelevant mandatory wizard branch.
Import support does not imply that every combination supports the same fitting recipe. The readiness
panel routes each combination to its actual fitting owner and names unavailable combinations.
In particular, the current automatic hBN calibration uses five declared hBN reflections and a
fixed Cu K-alpha wavelength. Its distance is calibrant-private. A different wavelength/calibrant
or a transferred sample distance needs an admitted core path, not a changed label in the UI.
The current OSC-series indexer also admits only one configured incidence axis. Displaying a
multi-axis goniometer does not make that indexing recipe support arbitrary motion chains.

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

For simulator data, distinguish the progressing presentation from a quantitative snapshot. The
current Monte Carlo preview supplies a leased float32 plane; it cannot supply these exact profiles.
Initially, exact simulator profiles/cursor values bind to the latest immutable float64 snapshot,
showing its revision and draw prefix. Until a matching snapshot exists, show **Awaiting quantitative
snapshot**; an older pinned profile remains visibly historical. Offer **Inspect this snapshot** to
hold a matching image and profiles together. Obtain snapshots in the worker at explicit inspection
or quantitative completion boundaries, never on every cursor event. A future worker-owned exact
reduction API is a separate measured extension, not a reason to silently sum presentation pixels.

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

Make **Find features -> Review -> Freeze for fit** an explicit workflow. Bind the acquisition,
instrument/material configuration, mask and discovery policy before calling canonical selection.
Retain discovered positions separately from reviewed inclusions/assignments. Publish one immutable
observation pack with its identities and revisions, and pass that exact pack into the fitter.
The current OSC-series convenience path reads images and supplies its own zero-edge mask; its
fitting wrapper reindexes internally. Add narrow input boundaries where needed to consume reviewed
data/masks, rather than bypassing review by calling that wrapper unchanged. hBN automatic tracing
and any preliminary refinement belong to preparation; after review, the fit cannot retrace points
silently. Unsupported manual edits remain inspectable proposals until admitted by selection.
Preserve qualification and post-fit audits when exposing this boundary. A direct core optimizer
result is a candidate until the applicable existing qualification checks pass. If the CLI owns
required orchestration, extract a narrow shared operation that accepts the frozen pack while
preserving the CLI path; do not promote optimizer success by omitting its qualification work.

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

Before enabling these controls, inventory each fitting owner's admitted coordinates, seed inputs,
bounds, fixed references and supported acquisition combinations. Current `fit_joint_geometry`
constructs its starting state from hBN and requires both Bi2Se3 and Bi2Te3 series; it does not accept
an arbitrary starting vector. Current hBN calibration accepts center/distance seeds but initializes
tilts internally and fixes its search bounds. Add narrow, explicit seed/bounds arguments where
needed, preserving defaults and gauge constraints. These are prerequisites for the promised
controls, not transformations hidden in widgets. Until supported, explain unavailable controls;
never display an editable value that the launched optimizer ignores. Use the hBN, indexed-series
and joint owners only for their respective admitted workflows.

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

Deliver configuration-load/run/save first, using an independent draft and the shared job lifecycle;
complete the supported parameter forms next. Neither needs fitted observations or the full 3D
editor. Later, connect scene handles and experiment-copy actions to the same parameter owner.

Cover the configured Monte Carlo and native physical-input routes explicitly. The native render
CLI requires observations and a saved fit, so it is not the entry point for an independent native
simulation. Bind an independent native draft through the existing physical model, detector and
integration owners; extract only fit-independent parameter binding where needed. Support admitted
surface/phase fractions, coherent repeats, thickness, fault and other model parameters without
manufacturing observations or a fit result. Preserve each route's input contract, output measure
and numerical status; simulation mass is not measured counts without an explicit calibration.

Later fitting stages have a separate preparation boundary. `scripts/prepare_native.py` currently
relocates and verifies existing frozen observation/background arrays and their provenance; it is
not a raw-image-to-native-observations builder. First expose supported prepared recipes. Preparing
new imported images then requires an explicit measurement projection, membership, dark/background
policy and covariance recipe using the current measurement/fitting owners. Ordinary side profiles
and geometry feature lists are not substitutes for those native intensity observations. Keep
unsupported combinations visible until that preparation path is implemented and scientifically
qualified; do not promise automatic progression merely because geometry has converged.

Before/after comparison locks image/support/units and distinguishes input, candidate and selected
results. Save/reopen a project with numeric configuration, file identities, acquisition grouping,
masks/assignments, drafts, selected result references and view layout. Validate file hashes and
schema when reopening; relocation requires verification. Do not use executable serialized objects.

Autosave editable project state atomically to the chosen external project location, visibly report
save status and offer recovery after a crash. Retain undo/redo for metadata, masks, feature edits
and initial values, with clear separation from camera history. An unsaved project can use an
explicit local recovery location until named. Bound recovery history and never write diagnostics
under the repository. Checkpoints for long calculations use the existing execution owner; restoring
an autosaved UI cannot imply that an interrupted solver is resumable.

Provide recent projects, Save as, named result history and **Duplicate experiment** for trying a
different supported hypothesis while sharing immutable image references. Warn about missing input
files on reopening and provide a guided relink action. A share/archive export offers a self-contained
copy of selected inputs, numeric project state and result products with file identities, estimates
its size, and preserves existing files. It never includes executable checkpoints or an implicit
cloud upload. Keep routine figure/table export a short selection of content, format and destination.

Export detector figures with optional overlays, profiles with support/units, feature tables,
parameter tables and the fit/configuration provenance needed to reproduce them. Preserve original
inputs. Generated results and diagnostics remain external to the repository; existing result
writers and the one-external-`.ra_diag.npz` diagnostic contract remain authoritative. Exported
figures/tables are user-requested result products, not a second hidden diagnostic format.

## 9. Small implementation architecture

Use `interactive/detector_viewer.py` as evidence for full-native texture presentation, progressive
work, input coalescing and invalidation. Its current opaque OpenGL child covers Matplotlib axes,
maps the entire image to the widget, and implements positive logarithmic display. It is not yet a
zoomable, layered detector panel. Extract only helpers shared by actual consumers and establish the
viewport/composition contract below before reusing it. Keep existing tools usable during extraction.

`interactive/ewald_sphere_viewer.py` is a fixed-wavelength, six-case illustration with embedded
textures and specialized geometry. Reuse suitable interaction ideas only; its formulas, positive-z
display cutoff and embedded images are not acquisition geometry or quantitative data authorities.
Build the experiment/reciprocal scene from canonical geometry APIs and retained drawing objects.
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
code/maintenance risk than it adds and meets the same rendering contract; it is not currently a
dependency. Make this decision in U00 before building the full interface. Keep any addition in the
visualization extra. Do not add a browser shell, web service, generic plugin system, second physics
engine or interchangeable backend framework for this UI.

Widgets stay on the GUI thread; workers send results through queued delivery and never mutate
widgets. This follows the [Qt worker-object guidance](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).
If PyQtGraph is selected, configure native row-major interpretation and explicit display levels;
its [ImageItem documentation](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html)
is an implementation reference, not evidence that a particular workload meets latency targets.

The detector panel owns one explicit native-pixel-to-viewport transform, shared by the image,
crosshair, mask, fitted overlays, pointer picking and both marginal position axes. Account for
pixel centers/edges, row direction, aspect ratio and device-pixel ratio without a second OSC
rotation. Provide signed/linear and positive-log modes with explicit invalid/nonpositive handling.
Use ordered layers within a compatible compositor, not Matplotlib markers hidden behind an opaque
GL child. Verify non-square corner/interior fiducials through pan, zoom, resize and DPI changes.

Use ordinary Qt widgets, direct signals, small typed snapshots and the existing I/O functions.
Introduce shared presentation helpers only for real consumers, and keep new numerical code limited
to the explicitly missing input/selection boundaries. A task ID is not a module or class. Prefer
the existing atomic JSON publisher for compatible project documents; no application database,
generic serializer, job service or parameter registry is needed. Judge dependencies by the code
and maintenance they remove as well as runtime and installation cost.

Define a documented repository launch command in the first shell slice. The current wheel does
not include `interactive/`; creating a script there is not an installed application entry point.
Assign project schema/version ownership to one small I/O helper from U01. Basic save/reopen and
atomic draft recovery arrive in U01b before editing tools; portable archives and complete result
exports follow later. Packaging a desktop installer is a separate delivery decision, not an assumed capability.

One explicit parameter description per supported binding supplies labels, units, scope, bounds and
editability to forms, callouts and saved drafts. Core constructors remain the validators. Keep
simulation configuration, geometry-fit inputs and native-intensity recipes as their actual distinct
contracts; write explicit mappings only for supported transfers. Do not invent a universal fitting
schema, introspection framework or registry. Save newly introduced masks, observations, seeds and
results in the slice that creates them, and export each slice's supported outputs at that point.

Keep immutable scientific/result snapshots separate from mutable camera, selection and display
state. Jobs bind acquisition ID and data/mask/calibration/model revisions plus a request generation.
Reject stale publications, including A-to-B-to-A selection races. A completed fit for an older draft
may remain inspectable as a labeled historical result, but cannot replace the current selection.

## 10. Responsiveness and bounded resource use

Responsiveness is an acceptance requirement. The following are proposed engineering targets,
not measurements or guarantees already achieved. Start with the existing 3000 x 3000 native
detector, one powder plus three sample images, two displayed together, and a stress case of twenty
lazily loaded images. U00 declares the reference CPU/GPU/RAM, presentation and execution backends,
screen refresh/DPI, maximum displayed overlay workload and numeric memory budgets before comparison.

| Warm interaction | Target on the declared reference machine |
| --- | --- |
| Pan, zoom, crosshair and camera feedback | Under sustained redraw demand on a 60 Hz reference display: p95 presented-frame interval <= 16.7 ms, p99 <= 33.3 ms; input-to-present p95 <= 50 ms |
| Two native band profiles following a crosshair | p95 input-to-presented profiles <= 50 ms for the declared prepared workload; preparation is measured separately |
| Selecting an already prepared image | p95 visible image/overlay/profile update <= 150 ms |
| Parameter edit, start/cancel/close acknowledgment | <= 100 ms, independently of actual job termination |
| UI during cold load, profile preparation, fitting and rendering | No observed GUI event-loop stall > 100 ms in the declared interaction window; no heavy synchronous compute/I/O or stale result publication |
| Superseded work and shutdown | State the longest noninterruptible phase; measure cancellation-request-to-safe-stop and resource-release time separately |

Do not promise fixed full-fit or full-simulation completion time. Show initialization/progress and
keep navigation responsive. A 33 ms cadence is approximately 30 fps, not evidence of a 60 fps pass.
Report lower-capability hardware separately with its measured limits. A missed target requires
measured diagnosis; do not silently reduce scientific support, change tolerance or relabel an
approximation as a completed result.
Frame-interval targets apply only while continuous interaction requires new frames. For isolated
events, measure input-to-present latency. Stop repainting when idle; do not maintain a permanent
60 Hz loop merely to satisfy a timing statistic.

### Retained plots and image presentation

- Upload an imported image only on cache admission or data revision. Camera, pan/zoom, contrast,
  crosshair and selection changes must cause zero image uploads and zero whole-image scans.
  Cache contrast statistics when admitting the plane; sliders change shader uniforms/colormaps.
  Keep full-native precision available independently of the float32 display texture.
- Keep 1D curve objects, position arrays, pens and axes alive. Cursor movement changes only band
  handles, crosshair and the two profile value buffers. Do not clear/rebuild figures, legends,
  colorbars or tick layouts for each event. Intensity autoscale is explicit and rate-limited;
  offer pinned limits. Basic plots use thin opaque lines without per-sample markers.
- Batch peak markers and ring polylines; keep label count bounded by visibility/selection. Start
  picking with straightforward vectorized distances over the visible feature set. Add a native-
  coordinate spatial index only if picking misses the declared workload's latency budget; rebuild
  it on geometry changes, not camera movement. Drawing culls never alter fit membership. Optional
  screen-pixel min/max envelopes preserve extrema, missing-data gaps and exact underlying values.
- Keep scene meshes, detector texture, overlay buffers and picking geometry alive. Camera motion
  changes view matrices; physical edits change only the affected transforms/geometry buffers.
  Use a schematic low-complexity goniometer, not unnecessary hardware mesh detail. Decorative
  complexity and label work are adjustable presentation costs, never scientific support changes.
- Budget uploads and upload staging as part of frame latency. Progressive images carry a data
  revision/draw prefix; present only the newest safe completed frame. Do not assume that a worker
  makes an OpenGL upload asynchronous. Add asynchronous staging only if measured upload stalls
  justify it, with explicit ownership and completion fences.
- Share owned textures between compatible 2D/3D contexts where supported; otherwise upload once
  per admitted data revision per context. Keep GL operations/context lifecycle on their declared
  threads, release resources on context loss and rebuild from owned data. Never use framebuffer
  readback to calculate cursor values or profiles.

If PyQtGraph is selected, its [PlotDataItem guidance](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotdataitem.html)
supports ndarray inputs, reused pens, thin opaque lines, clipping and optional display downsampling.
Do not disable finite checks when masked profiles contain missing values. Qt documents context
sharing, synchronization and potentially stalling readback in
[QOpenGLWidget](https://doc.qt.io/qtforpython-6/PySide6/QtOpenGLWidgets/QOpenGLWidget.html).
These mechanisms are options to measure, not evidence of achieved performance.

### Profile and image work

- Read/hash/decode on admission or a cache miss in a worker, reusing valid resident data. Avoid
  repeated reads/conversions while cached. Stream/validate import with bounded concurrency,
  declared size limits and per-file errors; a large folder is not a request to decode every image
  at once. Prefetch only nearby thumbnails at low priority. Never read a file on a cursor event.
- For one-pixel/narrow bands start with direct native slices. If timings justify it, prepare one
  summed-area table for signal and one for valid support for each active data/mask revision.
  Four-corner differences produce a whole horizontal/vertical profile in O(width + height), after
  O(width * height) preparation. Choose the direct path or prefix path explicitly by workload.
  Verify prefix subtraction against direct reductions, including large backgrounds/small signals.
- Cold decode, mask rasterization and prefix preparation are cancellable worker work, with visible
  **Preparing profiles** status. Use exact direct reductions while preparing only if their measured
  cost fits the budget. Otherwise retain an explicitly old/pending profile; never label an old-mask
  profile current. The crosshair and navigation remain interactive throughout preparation.
- Cache full-image marginals. Key profile reuse by acquisition/data, mask, correction, ROI/band
  bounds, reduction mode and quantitative snapshot/prefix. Contrast changes do not recompute them.
  Publish a new mask/profile generation atomically. Cursor coalescing must still show the final
  requested position when movement stops.

### One resource ledger

Declare numeric CPU resident/peak, GPU display and numerical-engine budgets in U00, including the
available-memory margin. Track their combined peak, not just one cache. A 3000 x 3000 float64 plane
is about 68.7 MiB; current OSC loading also holds raw and native int32 planes totaling about 68.7 MiB.
Signal float64 plus support uint32 prefix tables cost about 103 MiB per active plane, and a float32
display plane is about 34.3 MiB before any second staging or GPU copy. Check prefix-counter overflow
for larger arrays and numerical cancellation in signal differences.

The ledger includes decoded data, corrections/model snapshots, masks, prefix tables, textures per
context, upload staging, worker outputs, process-transfer copies, numerical workspaces, result
history and undo. Keep only selected/compared planes and a bounded nearby working set resident;
do not prepare prefixes for all twenty images. Bound mask undo using deltas/compact strokes with
periodic checkpoints and a visible undo limit. Keep older accepted results on disk with lightweight
metadata. Evict reconstructible RAM/GPU copies while preserving original files, project state and
accepted result artifacts. Reload asynchronously without changing their identities or values.

Bound both request count and payload bytes globally. Initially permit one running and one newest
pending interactive request per key, within one application-wide worker/resource limit; hidden
panels cannot each accumulate jobs. Keep at most one ready presentation frame plus the worker's
owned working frame per active producer. A leased buffer is consumed/copied before reuse, with
acknowledgment and all copies counted. Do not serialize whole images for every process command;
pass stable IDs/configuration and use a measured bounded data-transfer strategy.

### Jobs and invalidation

- Extend the existing latest-only preview scheduling; coalesce input to the next paint tick,
  apply the global limits above and drop superseded progress frames. Never launch one worker per
  mouse move or parameter control. Return small summaries unless the consumer requests a plane.
- Separate priority interactive previews from explicit fit/quantitative jobs. Bound CPU workers,
  nested BLAS threads and GPU compute ownership, leaving measured capacity for presentation. A
  background thread alone cannot prevent GPU compute from delaying drawing. Measure simultaneous
  work; use bounded existing compute batches/checkpoints without changing numerical support or
  accumulation semantics. Use a dedicated process where required by Python-heavy work or
  thread-unsafe numerical/database resources; queue numeric
  snapshots, not live mutable models. No generic job-service layer is needed.
- Use cooperative cancellation; never force-terminate a thread holding scientific/device state.
  Preserve the existing thread-confined sampler contract. Existing geometry optimizers need
  explicit safe cancellation boundaries; GUI wrapping alone does not add them. Report setup,
  compilation and optimizer phases that cannot yet be interrupted, and their observed stop times.
- Jobs have queued, running, cancel-requested, completed, canceled and failed states. Cancellation
  immediately invalidates publication; acknowledgment is not termination. Application close saves
  the draft, requests cancellation and remains responsive while safe shutdown releases worker,
  sampler and GL resources. Do not copy the current viewer's blocking close-handler thread join.
  No successful shutdown or resumable checkpoint is claimed while owned work remains active.
- During parameter dragging, update cheap scene transforms immediately. Canonical landmark/Q
  calculation may be expensive: schedule it with latest-only revisions and show **Updating geometry**
  rather than running it in the pointer handler. An old prediction remains clearly labeled or hidden.
  On release, request selected-image settled geometry, then other visible images. Hidden panels
  need no paint/evaluation until used; each visible panel shows the revision it represents.
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

### Measurement and decision gate

U00 is a thin rendering/interaction slice, not a new retained benchmark suite. Compare the current
optional presentation components and at most one plotting alternative using equivalent native
images, profiles, masks, overlay density and numerical work. Record actual versions, hardware,
workload, budget and outcomes. Choose the smallest path that meets both alignment and responsiveness
requirements; remove discarded code and temporary external checking scripts.

Keep U00 limited to its two-image/profile/overlay/textured-plane prototype and a bounded background
workload. It does not build mask history, the full goniometer editor or fitting workflows early.
Apply the following protocol only to scenarios available in the current slice: U04 adds mask/undo
journeys, U08 scene/context handling, U10 optimizer cancel/close, and U15 the combined twenty-image
journey. Reuse delivered application components; do not build a separate reusable checking framework.

Use at least three 30-second repeatable interaction windows per declared scenario: pan/zoom,
crosshair/band movement, mask editing, two-image comparison, textured 3D orbit, and those gestures
during cold preparation or a bounded representative background job. Include rapid A-B-A selection,
at least ten repeated image switches, context recreation and cancel/close races. Measure event to
paint/composition completion (not merely dispatch), frame intervals, p50/p95/p99 and longest GUI
stall; state the measurement boundary rather than claiming physical monitor latency. Report
time-to-first-image/profile and cold decode/compile/preparation separately from warm interaction.

Diagnose misses by file decode, array reductions/copies, CPU plotting, texture upload, GPU compute
contention and composition. Memory must stabilize under the declared twenty-image journey and
repeated mask/history operations. Verify exact final values, orientation, leases and stale rejection
alongside latency; a speed gain cannot offset a scientific or coordinate failure. Each later slice
checks only its new risk against this contract; U15 integrates the results, not the first measurement.

## 11. Delivery plan and acceptance checklist

This is the implementation plan and sole task checklist. Eight tasks are accepted; 38 remain. Existing
U identifiers are retained, with smaller lettered slices where the previous task was too broad.
Dependencies govern execution; milestones group completion criteria and are not serial barriers.
Use the delivery order below instead of waiting for every row in the preceding milestone.
The main agent is the only writer; independent reviewers can inspect completed changes.

### Delivery roadmap and provisional effort

The packages below are a delivery view of the same 46 tasks, not another backlog or a new set of
milestone IDs. Each task appears in one package; its detailed row below remains authoritative.
Build in this order by default, taking individual tasks early when their dependencies are met.
The M0-M7 sections group capabilities and do not impose a second execution order.

Effort ranges are low-confidence planning judgments for one experienced implementation owner,
expressed in focused engineering weeks of five working days. They are not measured agent runtime,
calendar commitments or claims about existing functionality. They include the slice's focused
verification and documentation, but exclude waiting for data/feedback and new scientific models.
Re-estimate after U00 and the first detector/profile delivery using actual completed work. The
largest uncertainty is new-acquisition native preparation; do not assign it a delivery date before
its measurement recipe and independent comparison are established.

| Delivery package | Existing tasks, in dependency order | Usable outcome and completion gate | Provisional effort |
| --- | --- | --- | --- |
| Foundation | U00, U08a, U01, U01c, U01a, U01b | Select one measured rendering path; launch the native shell; open a real image asynchronously; save/reopen and recover project state. Declare memory budgets and confirm orientation, late-job rejection and responsive close. | 1-2 weeks |
| Detector reader | U02, U03, U02a, U14b | Pan/zoom native images with exact horizontal/vertical bands, browse several acquisitions, assign metadata and export figures/data. Verify alignment, support/counts and the applicable latency targets. | 1-2 weeks |
| Visual experiment editor | U09a, U07, U08, U09 | Edit initial values and see per-image reciprocal coverage plus the textured beam/sample/goniometer/detector scene. Click-to-zoom, camera return, callouts and constrained handles share canonical state and undo. | 1-2 weeks |
| Independent simulator | U12, U12a, U12b, U12c, U09b | Both configured and admitted native routes load/edit/run/save without a fitted experiment. Cover supported parameters, exact quantitative inspection and explicit experiment-to-draft transfer. | 1-2 weeks |
| First geometry release | U04, U05a, U08b, U05b, U05, U10, U05d | Review masks/rings, propose a beam center, pass exact seeds/bounds to hBN fitting and inspect overlays, residuals, statistics and qualification status in the existing views. Save and reopen the complete journey. | 2-3 weeks |
| Extend geometry routes | U05c, U08c, U10a, U08d, U10b | Add admitted sample-only and joint workflows with frozen observations, per-image/shared/fixed scopes and validated execution boundaries. Keep unsupported materials and underdetermined fits explicit. | 1-2 weeks |
| Prepared native fitting | U11, U11a, U11b, U11c, U11d | Open existing prepared experiments; run supported mosaic, ordered and disorder stages; adopt compatible indexed geometry while preserving provenance and qualification. | 1-2 weeks |
| New-acquisition native preparation | U11e, U11f, U11g | Qualify one raw-image-to-observation recipe, then deliver preparation/review and stage handoff without handwritten manifests. Show invalidation after upstream changes. | About 1 week to specify the recipe; estimate implementation after U11e |
| Complete daily use | U04a, U06, U13, U02b, U14, U14a, U15 | Finish brush/imported masks, comparison/cuts, sensitivity, templates, portable archives and repeated-use recovery. Complete the combined performance/usability journey on declared hardware. | 1-2 weeks |

The first cohesive researcher release includes the first five packages: approximately 6-11 focused
engineering weeks under these assumptions, to be revised after early measurements. Earlier reader,
scene and simulator deliveries are independently usable. This first release does not complete
sample/joint fitting, later native stages or the daily-use refinements; those remain accepted scope.
No defensible full-scope completion date exists yet because U11f/U11g depend on the preparation
recipe. A repository launch is the initial distribution target; no standalone installer is promised.

A first geometry fit can use numeric inputs while individual handles are being completed. Preview
editing never requires a completed fit; fit-specific controls activate only after their execution
mapping is verified. U05d can expose an hBN-derived center as soon as that result exists, adding
sample-derived proposals only when an admitted sample result becomes available. Prepared native
fitting can use a valid existing recipe without waiting for a new geometry fit. Completing that
workflow does not establish preparation of newly imported data.

### Ownership, priorities and progress reporting

- Product/scientific owner: the researcher supplies representative inputs when available, resolves
  scientific intent and gives feedback on completed journeys. Existing tracked examples allow
  independent work to continue; missing evidence blocks only the capability that requires it.
- Implementation/delivery owner: the main agent owns task ordering, code, integration, scope and
  evidence. Work on one implementation slice at a time; do not create competing writers or treat
  parallelism as an assumption in the effort estimates.
- Review role: read-only review checks the actual diff and scientific boundaries when commissioned.
  A document review is not a substitute for a measured interaction or qualified numerical result.

Priority is the first complete inspection -> geometry editing -> simulation/calibration journey,
with persistence and responsiveness in every slice. Then extend supported routes and complete
repeat-use refinements. Move brush, comparison or template tasks forward when user feedback
justifies it and their dependencies are ready; retain the same task IDs and scope.

Use this checklist as the single status record. At a delivery update, report the current task,
completed behavior/commit, focused checks, remaining blocker and next task; report a blocked
scientific route separately from independent UI progress. Record a completion entry when checking
a task. Reforecast at package boundaries rather than reporting unchecked rows as percentage done.
New requests are mapped to an existing task or added here with their dependency and effort impact;
do not silently expand the first researcher release.

### First build cycle

This cycle delivers the foundation package, not the whole application. Use three reviewable
checkpoints; each may require several small commits under the task-delivery rules below.

| Checkpoint | Work | Concrete handoff |
| --- | --- | --- |
| Choose the rendering path | U00 and the bounded U08a map | One baseline and at most one justified alternative compared with two real images, exact bands, overlays and a textured plane. Record hardware, cold/warm timing, peak memory and coordinate correctness; retain only the chosen live components. |
| Open the first image | U01, U01c, U01a | Documented launch, file picker/drop, asynchronous OSC decode, native image presentation and useful error states. Verify corrupt-file behavior, stale completion and responsive cancellation/close. |
| Preserve the work | U01b | Save/reopen and draft recovery preserve acquisition identity and references, with atomic writes and actionable missing-file handling. Record the foundation's actual effort and remaining limitations. |

The next cycle completes U02/U03/U02a/U14b: the production viewport, exact marginal profiles,
multi-image metadata and inspection exports. Do not turn the rendering decision into a full
application prototype or scientific benchmark campaign. Keep the comparison finite, then build
on the selected components.

### Delivery risks and decision rules

| Risk | Owner and decision rule |
| --- | --- |
| Plotting or background work causes lag or excessive memory | Implementation owner measures U00, then each new interaction against section 10. Fix the observed reduction/copy/upload/queue cost before adding acceleration or features on the same failing path. Unrelated work may continue. |
| A control has no valid numerical binding or the fit is not identifiable | Scientific and implementation owners retain the explicit capability limit. Enable only verified seeds/scopes; permit inspection of unqualified results while enforcing existing downstream admission rules. A drawable mechanical degree of freedom is not automatically a fitted parameter. |
| New raw-data preparation needs unimplemented scientific support | Resolve the admitted recipe and independent evidence at U11e, then split/re-estimate its implementation. Deliver existing prepared workflows and independent simulation in the meantime. |
| UI scope or architecture grows faster than usable outcomes | Delivery owner keeps one writer, one backlog, direct state/signal bindings and the existing numerical owners. Additional dependencies or abstractions require a demonstrated need; finish the current user journey before adding another subsystem. |

Each package must demonstrate its user journey, save/reopen introduced state, preserve coordinate
and scientific contracts, reject stale work, and meet the applicable resource/interaction targets.
The first reader package cannot establish behavior for later unbuilt features; U15 checks the combined
application. Acceptance is evidence about delivered behavior, not a routine request for permission
to continue already authorized work. At the time of this planning update, implementation had not started.

The first geometry-user walkthrough must combine import, profiles, starting-value edits, reciprocal
coverage, the textured scene, a supported geometry fit, overlays/statistics and save/reopen. A fast
image reader alone is an intermediate delivery, not completion of the requested geometry interface.

### M0 — settle expensive decisions first

Likely ownership: this document, existing optional viewer components and visualization dependencies.
Keep the rendering comparison finite: one baseline, at most one alternative, and one focused repair
pass before recording the remaining bottleneck and a scoped next step. Preserve chosen live
components; remove discarded alternatives and all temporary checking code.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [x] U08a: initial capability map | None | Map reader/presentation contracts and the first configured simulator's physical fields, units and output measure. Identify both simulator routes and known fit limitations without enumerating every future control. Extend this same map before each corresponding form/fitter task: required inputs, seeds/bounds, scopes, qualification and safe-stop limits must be verified before that route is enabled. |
| [x] U00: rendering decision | None | Use two existing native images, exact band profiles, batched overlays and a textured plane with bounded background work. Declare reference hardware/versions and numeric CPU/GPU budgets; compare alignment, cold/warm interaction and peak memory under section 10. Choose one compositor/plot path before expanding features; neither a full fit inventory nor an optimizer is needed. |

#### U00/U08a implementation checkpoint (2026-09-28)

For this supervised implementation, the user requested `codex/desktop-ui-implementation` in the
isolated `desktop-ui-plan` checkout. Keep the branch for review; do not fast-forward `main` at
these checkpoints. This is a task-specific exception to the usual one-local-branch handoff.

Reference machine for the U00 decision: Intel Core i9-13900K (32 logical CPUs), 64 GiB RAM,
NVIDIA GeForce RTX 3060 (12 GiB, driver 572.47), Windows, 1920x1080 at 74.99 Hz and 1.0 device
pixel ratio. Python 3.13.13, NumPy 2.2.6, PySide6/Qt 6.10.2 and Matplotlib 3.10.3 are installed.
The selected presentation backend is Qt OpenGL 3.3/R32F with QPainter overlays/profiles; no
PyQtGraph dependency is admitted. The measured presentation process uses three 3000x3000 textures
(two detector panels and one schematic plane). Its fixed upload allocation is about 103 MiB of
GPU R32F storage; actual driver allocations are not inferred from that figure.

Initial resource ledger for this reference machine: presentation resident CPU <= 1.5 GiB,
presentation peak CPU <= 2.0 GiB, GPU display textures/staging <= 512 MiB, concurrent numerical
engine <= 4 GiB RAM, and at least 8 GiB available system RAM retained. The 3000x3000 two-plane
OSC raw/native int32 decode costs about 68.7 MiB per acquisition (34.3 MiB per plane) before the
panel's owned native int32 and display float32 copies (34.3 MiB each per displayed image). No
prefix tables or twenty-image cache are allocated in this slice. The one-thread background check is bounded to
one 1024x1024 array; it is not an optimizer or a scientific forward calculation.

U08a's initial capability map is intentionally limited to the reader and first configured route:

| Field / result | Authority | Units, frame, measure and current limit |
| --- | --- | --- |
| OSC bytes, raw indices and native counts | `io/osc.py`, `io/orientation.py` | One clockwise conversion at decode: native `[row, column] = [raw column, raw height - 1 - raw row]`; counts remain int32. A presentation widget receives native data only. |
| Detector texture, coordinates and bands | `interactive/detector_panel.py` | Column/row pixel centers map through one view rectangle; exact bands reduce original int32 with int64 sums or corrected real data with float64 sums. Float32 R32F and display levels are presentation only; support excludes masked and nonfinite pixels. |
| Configured material, source and spectrum | `pipeline/configured_simulation.py` `SimulationConfiguration` | CIF/phase identity; LAB origin in m; transverse axes and mean direction dimensionless, spatial sigma m, divergence sigma rad, wavelength and lines Å with probabilities, correlation, polarization and source sampling. YAML angle fields in degrees are converted by the core constructor path to radians internally. |
| Configured instrument and specimen | same config, `geometry` compiled instrument | Declared active LAB/goniometer/sample/crystal/detector transforms, detector shape `[row,column]`, pitch and positions m, reference coordinate `(column_px,row_px)`, film thickness Å, path medium and attenuation m^-1. Only existing rigid-transform and detector mapping owners calculate geometry. |
| Configured scattering and numerics | same config and configured pipeline | Mosaic widths/quadrature (degree-valued YAML to internal radian measures), structure model/layers/normalization, physical `(h,k)` rods, populations, polarization and worker/backend/quadrature settings. These are distinct physical and numerical fields; no controls are enabled by this map. |
| Configured outputs | `run_configured_simulation.py`, configured pipeline | Reciprocal/Ewald display densities are Å² rad^-2; detector macrobin quadrature is raw Å² per macrobin, and native center density is Å²/px². The separate Monte Carlo native pixel mass is raw Å². A preview is not a converged fit observable. |
| Progressive native preview | `source_averaged_detector.py` | `MonteCarloDetectorPresentation.image_A2` is a leased float32 plane valid only until the sampler's next operation; `MonteCarloDetectorPixelMass.image_A2` is immutable float64 quantitative mass. The panel copies its input before retaining it. The existing viewer's `SimpleQueue` and blocking `join()` do not supply the new app's job/shutdown contract. |
| Other simulator / known fit boundary | `fitting/native_input.py`, `scripts/render_native.py`, `native_workflow.py` | `NativeFitPhysics.detector` is the physical entry point for an independent supported native draft; `render_native` binds prepared observations/fit records and is not that draft. Rank/covariance and numerical qualification limit fit promotion; new acquisition preparation and native Bi disorder support remain separate admitted work. U12c maps only supported independent native inputs later. |

For the later geometry controls, hBN's current `_fit_observations` fits two detector tilts (rad),
native center `(column_px,row_px)` and a calibrant-private distance (m) with fixed bounds and
automatic ring selection; it does not yet expose arbitrary reviewed seeds/bounds or cooperative
optimizer cancellation. The reduced joint fit starts from the hBN calibration, requires Bi2Se3
and Bi2Te3 indexed series, keeps named mechanical references fixed, and omits unobserved PbI2
local coordinates. Neither route's drawable degrees of freedom imply independently fitted scope.
Their GUI seed and safe-stop boundaries remain U08b/U08d, not controls enabled by U08a.

The retained OpenGL child in `detector_viewer.py` proved texture feasibility but hides Matplotlib
layers and has no shared native pan/zoom overlay transform. The chosen U00 component is a separate
Qt detector pane that owns one transform and keeps exact profile arrays apart from the display
texture. It does not claim U01's launch, project identity, job ownership or shutdown behavior.

U08a's bounded map above is complete in this checkpoint's enclosing commit. Its later route
extensions remain dependencies of U08b/U08c/U08d and U12c, not finished form specifications.
U00 remains open because the 60 Hz cadence and one GUI-stall target were missed. One Qt/GL baseline
and one focused repair were used; no alternative package or scientific simulation was run.

U00 measured two tracked 3000x3000 native OSC images (hBN 5 m and Bi2Te3 5 m/5°), 4,000 points
per image, two exact one-pixel band profiles and a third textured schematic detector plane. The
background case used one numerical thread repeatedly reducing a 1024x1024 array; another Python
numerical process was active on the machine (roughly 1.3-1.5 GiB working set and rising CPU time).
The reference screen ran at 74.99 Hz. After short setup probes and one aborted exploratory run with
flawed event labeling/pan drift, the corrected run used three 30-second windows per scenario with
15 ms input requests. Active comparison time stayed below the 20-minute cap. Times below are
milliseconds from synthetic Qt event dispatch to the matching generation's `paintGL` completion or
Qt `frameSwapped` after top-level composition/buffer swap; they do not measure physical scanout.
Horizontal and vertical QPainter profile paint completion is measured separately. Intervals count
only matched presented generations; unpresented requests are listed instead of assigned a later
frame's timestamp.

| Scenario, 3 x 30 s | Input-to-swap p95 across repeats | Swap interval p95 across repeats | Other evidence |
| --- | --- | --- | --- |
| Bounded pan/zoom, one panel | 20.05-20.15 | 20.32-20.40 | 1,996-1,999 requests/window, all presented; first-window longest GUI timer overrun 102.07 ms, later maxima 18.14/15.87 ms. |
| Cursor and exact profiles, one panel | 13.35-19.93 | 26.68-27.05 | Both profile paints p95 7.59-17.99; ~1,997 requests/window, all presented; native cursor covered columns 1-2999 and rows 1-2998. |
| Two images updated per input | 17.17-19.90 | 17.64-20.40 per panel | 3,682-3,768 panel requests/window, 0-2 unpresented; both profile paints p95 13.75-14.67. |
| Schematic plane orbit | 1.06-1.11 | 16.05-16.11 | ~2,000 requests/window, all presented. This is a textured presentation plane, not canonical geometry. |
| Cursor during bounded background work | 13.32-13.35 | 26.68 | Both profile paints p95 7.92-8.03; all ~2,000 requests/window presented; longest GUI timer overrun 17.22 ms. |

Cold OSC decode was 99.96/90.93 ms; first image/plane composition after `set_image` was
169.86/140.85/107.01 ms respectively. Peak presentation-process RSS across measured windows was
591.6 MiB, within the declared CPU budget. Each texture uploaded once from admission through all
camera, cursor, profile and orbit windows: `[1,1,1]` before and after. GPU driver allocation and
uncontended timing were not measured; total GPU memory use during the run was affected by unrelated
activity. The small direct probe verified non-square OSC orientation, exact int64 accumulation of
native int32 counts, float64 masked/nonfinite bands (including clipped edges), integer-overflow rejection,
positive-log replacement, and distinct framebuffer corner values (red 212 versus 38). An overlay
pixel changed from `#444c4d` to `#64f5eb` at the native `(column,row)` marker. Reparenting the GL
view recreated the context and caused one expected reupload; ten A-B cursor pairs caused none.
Rejected unsupported integer replacement preserved the prior image revision and profile values.

The direct integer bound rejects int64/uint64 planes conservatively; even-width bands include one
more higher-index pixel than lower-index pixel before independent edge clipping. A pan with
unchanged profile values retains the prior profile generation, so missing profile-paint samples in
that scenario are not interpreted as fast updates. Float32 presentation leases cannot enter the
quantitative band API. The profile/cursor result is inspection only and establishes no fitting
observable, convergence or scientific adequacy. Context recreation was checked on the small
probe; twenty-image memory, mask history, full scene geometry, asynchronous stale-job rejection,
close/cancel races and real monitor scanout remain for their dependent tasks.

The compositor path is retained without PyQtGraph because the measured direct Qt components
aligned the probe and kept uploads stable. The original 15 ms supply windows missed the 16.7 ms
p95 interval target and observed one over-100-ms GUI gap. The supervised cadence follow-up below
investigates those misses; this original evidence is retained rather than relabeled.

#### U00 cadence follow-up (2026-09-28)

The supervised follow-up used the same i9-13900K/RTX 3060, 74.99 Hz screen, two native 3000x3000
OSC images, 4,000 overlays per image, exact bands and visible schematic plane. An unrelated Python
numerical process remained active at about 0.95 GiB working set with rising CPU time; total GPU
use was about 2.2 GiB and 29-36% during spot checks, not attributable to this presentation alone.
The 10 ms Qt timer was a *request*, not the observed event rate. With one panel, its callback
interval p50 was 13.31-13.34 ms and p95 13.64-14.54 ms; with both panels updating per callback,
the pre-repair callback p95 slowed to 16.95-17.92 ms. This explains why the original 15 ms supply
could alias against 74.99 Hz refresh and show a 26.68 ms p95 cursor interval without proving a
slow `paintGL` path. The original data still demonstrate a miss under that request pattern.

The pre-repair source SHA256 was `15a47328697ba763ff8a322bbefe32de994f33bcdb5fd755017cf5e050ae5f4e`.
A 10-second controlled pan/cursor probe found matched-frame interval p95 14.03/14.11 ms and
`paintGL` p95 about 0.66 ms. The unchanged-source three 30-second windows at the 10 ms setting
gave pan p95 intervals 14.12-14.22 ms, cursor 14.18-14.48 ms, and cursor with bounded background
work 13.45-14.34 ms. Their FIFO paint-to-swap attribution did not prove that two paints could not
precede one composition, so those interval labels remain provisional. Their directly measured
two-image callback p95 16.95-17.92 ms and horizontal/vertical QPainter paint costs of
3.27-3.40/3.06-3.18 ms p95 are valid work-count evidence. Four profile paints per two-image
update consumed the available callback interval; `paintGL` stayed below 0.7 ms p95. The old
two-image frame-interval p95 of 16.92-17.87 ms is an exploratory estimate, not the basis for a
claimed speedup. No second backend or dependency was added.

One measured repair uses a screen-pixel min/max envelope for each visible contiguous valid profile
segment when its exact native samples outnumber twice the available display pixels. Each displayed
bucket retains both extrema; mask/nonfinite gaps break polylines, and isolated valid bins draw a
point. The exact int64/float64 source profile and support arrays are unchanged. A focused rendered
check retained a positive spike, a negative trough, an unbridged gap and an isolated valid point.
The dense-path timing source SHA256 was
`841d1cb31465c87d2f27a58113ee5d1f869eb6884fa92d9be0650dc1ffe685ed`; the enclosing
commit adds only the isolated-point edge branch afterward (source SHA256
`57e78c4edd18bcf1484c7394272028af9c11a1930249bf457c67f0f667d24808`). That branch was
verified directly, not separately benchmarked.

The repaired two-image path had three 30-second windows with 2,251 timer callbacks and 4,502
panel requests each. The tracer matched every request to the newest completed paint before its
composition: 4,502 presented, zero superseded paints/profiles and zero pending paints in every
window. Each top-level composition also emitted a `frameSwapped` from the plane without a new
plane paint; those 2,251 extra signals per window were counted separately. Horizontal/vertical
profile paint completion was paired to the next composition for the same view generation.

| Repaired two-image window | Actual callback p50/p95/p99 (ms) | Composed frame interval p50/p95/p99 (ms) | Input-to-composition / presented-profile p95 (ms) | GUI callback gap max (ms) |
| --- | --- | --- | --- | --- |
| 1 | 13.324 / 14.351 / 14.720 | 13.330 / 14.327 / 14.709 | 14.133 / 14.133 | 16.609 |
| 2 | 13.313 / 14.242 / 14.676 | 13.322 / 14.219 / 14.703 | 14.047 / 14.047 | 18.554 |
| 3 | 13.337 / 14.189 / 14.614 | 13.328 / 14.175 / 14.608 | 14.008 / 14.008 | 22.525 |

Repaired horizontal/vertical profile paint costs were 1.45-1.48/1.27-1.28 ms p95, and image
`paintGL` 0.66-0.67 ms p95. The three frame interval maxima were 16.60, 18.53 and 22.13 ms;
all p99 values were below 33.3 ms. The two-image 16.7 ms p95 frame and 50 ms response/profile
targets are supported under this actual 74.99 Hz supply. Peak presentation RSS was 513.7 MiB,
and image/plane uploads remained `[1,1,1]`. Cold OSC decode was 116.09/91.73 ms and initial
image/plane composition after admission 151.23/120.38/80.54 ms, separately from warm windows.
No scientific simulation, optimizer or fit ran. Focused checking code/results were external and
removed after recording this evidence. Active profiling/interaction time was about 470 seconds,
within the new eight-minute cap; no further campaign was launched.

U00 remains open. The repaired two-image window supports its frame, response, profile and GUI-gap
targets, but the earlier single-panel/background FIFO timing is provisional and the first pan
window's over-100-ms heartbeat gap was not separated from setup/window-transition time. Its
99.462 ms reported *lateness* would be a 109.462 ms callback gap if it occurred wholly within the
window. The repaired tracer resets heartbeat at each boundary, but was used only for two-image
windows under the remaining budget. Thus the no-GUI-stall claim for the complete U00 workload is
not established. The smallest next decision is a separately budgeted, correctly labeled
single-panel/background interaction check if the supervisor requires that gate before U01; do not
weaken the target or reuse the exhausted budget.

#### U00 final interaction gate (2026-09-28)

The final source SHA256 was `45a583b083ba1454af72666e803fed9d69ad2d244c98d7759f20869552e47866`.
The only further production change retained the immediate offscreen sample on either side of a
visible profile interval within each contiguous valid run. A two-sample line crossing the entire
plot now remains visible even when neither sample center is inside it. Focused horizontal and
vertical rendering checks also verified the exact source arrays, signed extrema, an unbridged
invalid gap and an isolated valid point. The earlier envelope work and its physical values are
unchanged.

On the same 74.99 Hz reference setup, the final tracer ran three 30-second active windows each
for bounded pan/zoom, one-panel cursor with both exact bands, cursor with a bounded 1024x1024
NumPy reduction job, and schematic plane orbit. The 10 ms Qt timer yielded the measured callback
cadence below. Each `paintGL` generation was matched to the *latest* completed paint before that
view's `frameSwapped`; paints superseded before composition, swap signals with no new paint, and
unmatched requests were counted separately. Both profile paints were paired to a subsequent
composition of the same panel. Times end at Qt composition/swap completion, not physical monitor
scanout. The table gives the **largest per-window** p50/p95/p99/max tuple across three repeats,
in ms; it is not a pooled percentile.

| Scenario | Requests -> composed; profiles composed, per 30 s window | Actual callback p50/p95/p99/max | Composed frame interval p50/p95/p99/max | Input-to-composition p50/p95/p99/max | Longest GUI callback gap |
| --- | --- | --- | --- | --- | --- |
| Pan/zoom | 2,249-2,250 -> all; n/a | 13.329 / 14.145 / 15.123 / 20.634 | 13.336 / 14.071 / 15.020 / 20.597 | 13.223 / 13.973 / 14.741 / 20.431 | 20.797 |
| Cursor and exact bands | 2,249-2,250 -> all; all | 13.332 / 13.545 / 13.711 / 16.407 | 13.335 / 13.420 / 13.468 / 14.121 | 13.209 / 13.337 / 13.388 / 13.981 | 16.375 |
| Cursor plus background | 2,248-2,250 -> all; all | 13.332 / 13.610 / 13.879 / 26.618 | 13.335 / 13.459 / 13.654 / 26.654 | 13.184 / 13.326 / 13.518 / 26.325 | 26.619 |
| Schematic plane orbit | 2,279-2,298 -> 2,249-2,250; n/a | 13.331 / 13.513 / 13.645 / 14.946 | 13.335 / 13.420 / 13.460 / 14.005 | 13.232 / 13.348 / 13.396 / 13.807 | 15.083 |

For cursor and background windows, the presented horizontal-plus-vertical profile response had
the same p50/p95/p99/max tuples as their input-to-composition columns. Every panel request and
profile pair in those windows was presented, with zero superseded or pending paints/profiles.
Every window ended on the exact requested native cursor coordinate; coverage reached columns and
rows 1-2999. Orbit ran slightly faster than the available composition cadence, so 30-48
intermediate requested generations per window were coalesced before paint; no completed paint was
superseded or left pending. A separate 10-second orbit check confirmed its final requested
generation was presented (752 requests, 751 composed). This coalescing is not counted as 100%
request presentation. The previous 109.462 ms inferred GUI gap remains historical unclassified
evidence; the new active-window callback gaps were measured from a reset clock and did not exceed
26.619 ms.

A 10-second two-image smoke on this source presented all 1,500 panel requests and both exact
profile pairs, with zero superseded or pending paints/profiles. Its callback p50/p95/p99/max was
13.305/14.339/15.041/27.152 ms; per-panel composed interval was
13.330/14.288/15.084/26.816 ms; input-to-composition and presented-profile response was
13.111/14.111/14.802/26.737 ms; the longest GUI callback gap was 27.359 ms. The preceding
three 30-second two-image windows supply the sustained dense-path evidence.

Cold OSC decode was 114.60/106.77 ms; first image/plane swap after admission was
161.37/121.42/83.87 ms. The initial transition cost 101.20 ms; later transition costs were
9.51-11.43 ms, measured outside active windows. Peak presentation RSS was 516.4 MiB and texture
uploads remained `[1,1,1]`. All active scenarios met the 16.7 ms p95 and 33.3 ms p99 frame
limits, 50 ms p95 interaction and presented-profile limits, and no observed >100 ms GUI stall on
this 74.99 Hz hardware. U00 therefore selects the existing direct Qt/OpenGL image, QPainter
profile and Qt composition path. Future mask, project, context-recreation and fitting interactions
retain their own delivery gates. No scientific fit or physical detector calculation was inferred
from these presentation measurements. Temporary checking scripts and raw outputs were external
and removed after this record; no reusable benchmark harness or new dependency was retained.

### M1 — a useful detector reader

Likely ownership: `interactive/slate_app.py`, narrow optional project/I/O/presenter/profile helpers,
`src/rasim_next/io/osc.py` only if a measured boundary change is necessary, and
`interactive/README.md`. Do not allocate one module per row by default.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [x] U01: shell and project identity | U00 | Provide a documented repository launch, the two workspaces, stable acquisition IDs and explicit empty/loading/error states. Widgets contain no scientific model state; normal numerical imports remain GUI-independent. |
| [x] U01c: shared job lifecycle | U01 | Introduce bounded worker ownership, queued/running/cancel-requested/terminal states, generation rejection and responsive close. Exercise a small loading/preparation operation and late completion; acknowledgment and safe stop remain distinct. Numerical owners gain only their own later cancellation hooks. |
| [x] U01a: first file import | U01c | File picker/drop imports one OSC/OSC.GZ asynchronously through the existing orientation boundary. Show native counts before scientific metadata is complete; verify tracked non-square inputs, corrupt-file handling and no second rotation. Accepted after review repair. |
| [x] U01b: save, reopen and draft recovery | U01a | Own one versioned numeric project schema with file identities and atomic save/autosave. Verify an interrupted save, moved/missing source, relink identity and close/reopen; originals remain unchanged and solver resume is never implied. Accepted after focused review. |
| [x] U02: detector viewport | U01a | Add pan/zoom, native pixels, signed/linear/log contrast and retained layers. Corner/interior fiducials, pointer and marginal axes remain aligned through resize/DPI changes; cursor/camera/contrast cause zero image uploads. Accepted after sustained native-loop review at `4c921c1`. |
| [x] U03: exact marginal profiles | U02 | Add follow/pin crosshair, independent bands, sum/mean/full-image/ROI modes and support labels. Compare small direct reductions at edges, gaps and signed/nonfinite values; measure preparation and warm latency. Add prefix caching only for a measured need and verify its subtraction error. Accepted after independent sustained native-shell review at `14c9799`; first/new ROI remains synchronous. |
| [x] U02a: multi-file import and metadata | U01b/U02 | Add lazy filmstrip, folder candidate review, roles, angle/exposure/material/CIF inputs and bulk metadata mapping. Mixed valid/corrupt files preserve successes; duplicates differ from repeated exposures; incomplete metadata does not block inspection. |
| [ ] U04: masks and regions | U03/U01b | Add rectangle/polygon masks with reasons, a persistent revision and bounded undo. Publish mask/profile generations atomically; mask display visibility and fit inclusion stay separate. |
| [ ] U04a: brush and imported masks | U04 | Add brush gestures and supported mask import with native orientation/shape validation. One gesture is one compact undo item; repeated strokes and rebuilds stay within the resource budget. |
| [ ] U06: comparison and cuts | U03/U02a | Display two images with compatible linked pan/limits, pinning, magnifier and explicit straight-line sampling. Keep exposure/units visible and masks/support matched; later result bindings reuse this view. |
| [x] U14b: inspection export | U03/U01b | Export the current detector figure and exact profile values/support/units to an external destination. Reopen exported values and compare to the named data revision; unrelated fitting stages are not prerequisites. |

U02a is split into sequential subchanges without adding task IDs: (1) strict acquisition
metadata, provenance and backward-compatible project persistence; (2) bounded per-candidate
admission, retry/relink/cancel and selected-image ownership through the existing global job;
(3) UUID-based review table, bulk mapping, material/reference pickers with a bounded canonical
configuration-reader snapshot, filmstrip and explicit
metadata export; (4) focused integrated checks and resource accounting. Each implementation
commit touches at most five files. The sustained qualification gate remains open after these
implementation commits.

The remaining U02a source closure is split within the same task: (5) full
folder-candidate review and an external metadata-export boundary; (6) retained
standalone/dependent reference identities, bounded reopen checks and visible
current-state statuses; (7) focused schema/I/O/UI checks and resource accounting.
U09a owns undo/redo for these metadata mutations together with later physical
and initial-value edits. U02b still owns guided relocation, templates and copy
storage. These are explicit later acceptance items, not completed U02a features.

The later geometry work follows these ownership boundaries. U05b and U05c use
U09a's shared feature-review undo state for accepted ring and sample indexing edits;
neither fit entry point silently regenerates a frozen observation pack. U09 may bind
fit-specific handles only after U08b, U08c or U08d verifies the corresponding
launched parameter vector and scope. U05d may offer an hBN-derived center after an
admitted U10 result; a sample-derived proposal requires an admitted U10a result.
These are conditional follow-through items, not acceptance of those tasks.

#### U02a import and metadata implementation checkpoint (2026-09-28; review pending)

Commits `6aae90f`, `79bd9a3`, `a1e3980`, `d4d02b5`, `3977f05`, `9adcf87` and
`5d53f5e` and `3b58abb` implement the
U02a slices. The project document now retains acquisition role, specimen/mount,
commanded incidence in radians, exposure in seconds, detector setup, material and
calibrant identifiers, dark/mask associations, explicit unknowns, provenance and
reference path/hash pairs. Version-one documents retain strict defaults. A single
global worker admits bounded OSC candidates independently, records exact decoded
hash duplicates as distinct acquisitions and permits cancel, retry and removal.
The read-only review table, filmstrip, bounded bulk mapping, unconfirmed filename-angle
suggestions, canonical CIF/configuration reference validation and external CSV
export are connected to the native shell. The configuration reader parses the
same bounded YAML bytes used for its digest and bounds/hashes its referenced CIF
snapshot for revision identity. It does not structurally parse that nested CIF;
the standalone CIF picker uses the canonical crystal reader. No metadata entry
starts a fit or qualifies a calibrant.

Focused external checks used `PYTHONPATH=interactive;src`, native
`QT_QPA_PLATFORM=windows` and `python -u -B`. A mixed candidate check observed
three valid admissions, one missing source and one unsupported extension;
retry admitted a fourth row. A separate four-row small-source journey
(`exec-45ea049c`) saved and reopened exact JSON and checked selected reopened
fields before normal close in 0.471 s. The reference/configuration checks compared the
canonical physics and rendering revisions from default and bounded-snapshot
readers; the nested CIF limit rejected an oversized reference. Transactional
metadata checks exercised signed finite angles, paired reference path/hash
updates and both CSV association row orders. The CSV check (`exec-b7a84528`)
checked UUID, hash-kind and degree headers plus selected values, not every
field's exported value. A controlled cold selection/removal check
(`exec-f37d5490`) reached `REMOVE_IN_FLIGHT_OK` in 0.215 s: removal
invalidated the running load, and the other acquisition remained. Its harness
stopped its timer before asynchronous window shutdown and needed forced process
exit, so that command supplies no normal-close proof. The earlier queued
cancel/retry and canceled-open checks (`exec-bd6e848f`) passed before a
cold-load setup assertion. A follow-up controlled check (`exec-76445537`)
found a Qt item-API error in its own
harness before reaching the product workflow. No such failure is counted as
product evidence. The temporary check code is not retained.

Read-only review then found two admission gaps: individually valid metadata
could grow a project beyond the 1 MiB document limit, and malformed angle
proposals could reach confirmation. Both are now checked before UI publication.
A focused 128-acquisition check (`exec-d4d85fb3`) began from a valid document
and confirmed oversized mapped metadata is rejected while the prior project
remains. A focused selection check (`exec-e99f7288`, before its unrelated
selected-context setup failure) confirmed candidate-only rows yield no admitted
target and multi-selection survives table refresh. Malformed or nonfinite
proposal values are rejected at metadata construction. A later pure near-cap
check (`50f74d`) proved a 128-row document remained readable without an active
view while new admissions reserve 4,096 bytes. A fully populated valid native
view with ROI, both intensity limits, flags, native bounds and maximum finite
double-precision encodings added 837 canonical JSON bytes (`642418`), below
that margin. The table's `NoEditTriggers` setting was checked in a tiny native
window (`80bbd1`). No per-cursor document serialization was added.

Ruff format and lint passed for the touched Python files; an offline source and
wheel build passed. Built artifacts and the temporary saved project were removed
from the external scratch location. These software checks establish packaging
and interaction behavior only, not scientific fitting adequacy.

The final correctness closure made oversized multi-file picker choices report an
actionable limit without admitting partial rows, displayed a stale reference
retry notice when a queued batch changed the project, accepted long read-only
CSV columns while enforcing 256-character mapped edits, and sorted angle and
exposure cells numerically with unknowns explicit. Tiny native checks
`c114d1` and `e2b21e` checked those UI paths, signed angles, 2/10 ordering,
unknowns, selected/current UUID preservation and persistent acquisition
reordering. `10c319` checked all introduced metadata CSV values, units,
provenance, proposals, revisions, order, UUIDs, reference paths/hashes and
decoded-source hash kind; a 2,990-character exported provenance cell was
ignored during an allowed-column remap, while an oversized mapped edit failed
atomically. `8140d1` wrote and reread a 3,759-byte canonical JSON document and
compared the complete metadata and selected UUID. These value checks do not
independently revalidate the referenced CIF/configuration files.

Native `598248` kept the initially captured UUID target through a modal table
refresh and changed review selection. `db2de7` saved an immutable older
snapshot while a newer metadata edit remained in memory, compared disk and
memory, then closed normally. An A-B-A/project-switch checker first failed on
its own Python local-variable scope (`270a58`); corrected `f19412` verified the
final selected UUID, native image and exact full profile for A, then opened a
different project and verified its UUID, image and profile before normal close.
`8d2535` closed normally during a gated batch decode, observed owner drain and
no accepted acquisition or late image publication. These are tiny fixture
checks, not composition-latency qualification.
The final UI repair passed Ruff format/lint, Python imports and `git diff --check`;
the previous offline source/wheel build covered unchanged package
code, so it was not repeated for these two interactive files.

The one permitted 20-record smoke (`exec-c209cfdd`) used repeated explicit references to the
tracked 3000 x 3000 `hBN_calibrant_5m.osc.gz` and completed in 2.094 s within
3.508 s command wall time.
It confirmed 20 distinct UUIDs and identical decoded source hashes, a final
matching selected ID, exact saved JSON and normal drained close. It did not
reopen this 20-record document. Fourteen
selected-image publications comprised 12 cold and two resident selections;
publication latency ranged from 13.008 to 231.959 ms. These timings exclude
composition; there is no evidence for ten presented switches or matching
generation/hit-test evidence on each composition. Eight GL uploads were
observed because paints coalesced. The cache
held at most two prepared packs and thumbnails occupied 110,592 bytes; observed
RSS was 369.7 MiB and Windows peak working set 484.9 MiB. The selected detector
panel can retain a third distinct old plane after a selection, removal or Open,
and one worker result may coexist with these. Four worst-case 96 MiB prepared
packs total 384 MiB. A conservative concurrent CPU allowance includes the
64 MiB source-file admission maximum, although the reader streams it in 1 MiB
chunks; it also adds 32 MiB decoded bytes, a 48,000,000-byte raw int32 plane, two
12,000,000-byte Boolean high-range/positive masks and a 1 MiB read chunk;
the worker result already counts native/display planes and exact profiles.
Allow another 8 MiB for the one-entry U03 full/ROI profile state and 32 MiB for
up to 1,179,648 thumbnail bytes plus Qt icon copies. Reserve 32 MiB for the
bounded 3 MiB pending write queue, active serialized request, publisher's
parsed object and second JSON serialization; reserve 16 MiB for the bounded
loaded JSON bytes, project dataclasses and up to 384 small reference-status
records. Another 16 MiB covers table/other Qt items. Against the previously
observed 172 MiB shown-shell baseline, this accounted CPU subtotal is about
826 MiB, leaving about 710 MiB to the 1.5 GiB resident budget and about
1.19 GiB to the 2 GiB peak budget before allocator, driver and other
unmeasured overhead. The queue and file-size caps are enforced; these extra
Qt/serialization allowances are planning reserves, not measured hard bounds.
Reference rechecks send at most 384 binding triples within the existing 4 MiB
job-request cap and declare a 128 KiB result allowance inside the active-work
reserve; these caps are admission limits, not additional measured RSS.
A maximum admitted 12-million-pixel R32F texture is 48,000,000 bytes with an equally sized upload
copy, about 91.6 MiB combined and about 420 MiB below the 512 MiB GPU display
budget before driver allocation. The measured 3000 x 3000 hBN shape uses
36,000,000 bytes for each. Driver GPU allocation was not measured. This is a conservative overlap ledger,
not a measured maximum. The short repeated-source
smoke does not establish sustained twenty-image latency, memory or mixed-source
qualification. Native command walls, including the two focused admission
follow-ups and the final tiny native checks, conservatively totaled about
51.8 s of the original 60 s allowance;
the one permitted 20-record smoke is closed. The U02a checklist
remains open for independent review and the declared responsiveness gate.
Remaining qualification includes sustained final composition/hit-test matching,
mixed-source twenty-image behavior and presentation-latency distributions.
The tiny checks do not claim those gates or scientific fitting adequacy.

#### U02a reference and review source closure (2026-09-28; review pending)

Source commit `f1f6d68` closes these bounded implementation gaps; U02a remains
unchecked pending independent review and the stated sustained qualification.
The folder action now presents all direct candidates in a bounded scrollable
dialog, with each unsupported extension labeled and the folder/count/scope
shown before confirmation. Cancel retains the project and candidate queue.
Metadata CSV export rejects both local checkouts and aliases of the project or
OSC sources, including resolved and hard-link identities; no check wrote under
either checkout. Configuration binding stores its own YAML identity and the
canonical configured reader's resolved dependent-CIF path and bounded byte
SHA-256, separately from an independently selected standalone CIF. It replaces
or clears the dependent pair with its configuration. Schema 3 writes these
fields; strict schema 1 and 2 reads leave the absent dependent identity
explicitly unknown. The configuration reader hashes that bounded CIF snapshot
for revision identity but does not structurally parse the crystal.

Project Open still runs through the one global background owner. It reads each
distinct reference at most once under the 1 MiB picker limit, compares saved
byte hashes, and publishes per-acquisition verified, missing, changed,
unreadable or unverified status without changing saved metadata. Raw OSC
browsing remains available when a reference fails. Current states and
choose-again guidance appear beside historical binding provenance. Reopening
does not claim current structural/material validity from a matching hash.
An explicit recheck clears the current status for every acquisition bound to
that path (and the recorded dependent CIF when rechecking its configuration).
The observed digest then updates all matching bindings independently against
their saved hashes; failed, canceled and stale rechecks leave them unverified.
CSV now exports both dependent-reference columns (25 total against a declared
26-column parser cap); editable mapping remains limited to the existing fields
and the 64 KiB/128-row input bounds.

External pure check `07a8e6` used copies of the tracked Bi2Te3 configuration
and CIF plus the tiny non-square OSC. Two acquisitions shared the same
references; all six current checks initially verified. A changed dependent
CIF with unchanged YAML reported standalone/dependent changed and YAML
verified. Separate changed, missing and over-limit unreadable CIF and YAML
cases produced their named states while both OSC source checks stayed verified.
The schema-2 read kept two dependent checks unverified without deriving a
hash from disk; schema 1 retained empty metadata. CSV readback preserved 25
headers and the exact dependent SHA. Pure `70800f` rejected an oversized
dependent-reference candidate while the previous immutable project stayed
unchanged. Pure `322765` rejected destinations in main/worktree and external
hard-link aliases of project/OSC inputs. Tiny native `c8fc63` inspected all
23 folder candidates, the unsupported row, cancel invariants and reference
status labels. First replacement checker `3df72f` failed because its fixture
used a non-hex digest; corrected `f699b6` showed configuration A then B
atomically replacing the dependent pair while standalone CIF identity stayed
unchanged. These targeted checks consumed another 3.738 s of native command
wall time, raising the conservative aggregate to 55.512 s of the original
60 s allowance. No additional GUI run or twenty-record smoke is authorized
in this packet.
An external pure shared-reference check observed one digest across two
acquisitions and standalone/dependent bindings: matching hashes verified,
different saved hashes changed without mutation, and a canceled recheck
cleared every shared status to unverified. It also checked alias path matching
and shared-configuration status comparison. This proves the status transform,
not the remaining native interaction and latency gates.

The subsequent shared-reference source review found that the first status
repair compared filesystem paths repeatedly on the GUI thread. With 128
shared configurations, it could rescan 384 bindings for each duplicate
dependent path. The follow-up moves alias identity checks to the existing
reference worker: it captures the selected file identity from the open handle,
stats each distinct saved path once, and returns bounded immutable matching
acquisition/kind keys. GUI invalidation deduplicates path and prior file
identities into sets, then makes one update pass over at most 384 bindings without
filesystem access; publication compares each worker-matched binding against
its own unchanged saved hash. A previously shared alias that now points to a
different file remains unverified after the recheck. When the filesystem does
not expose a usable file identity, only equal recorded path keys propagate.
An external pure 128-acquisition check covered 384 bindings with 1,411
in-memory key operations, prohibited GUI-path filesystem calls, compared
verified/changed states and confirmed unchanged project metadata. Its bounded
worker check used two distinct alias stats, and a separate external hard-link
CIF check confirmed both paths matched one observed file identity. The first
worker fixture supplied 385 bindings and was correctly rejected by the 384
cap; the corrected fixture passed. No native GUI time or second twenty-record
smoke was used. Sustained U02a interaction and resource gates remain open.

A further snapshot review found that the canonical configuration reader's
dependent-CIF hash could be paired with a later file identity after an atomic
path replacement. The reference worker now takes one additional bounded
dependent-CIF read with bytes and identity from the same open handle, then
requires its SHA-256 to equal the canonical reader's hash. A mismatch rejects
the binding before any alias matches reach the GUI; saved hashes remain
unchanged. An external owned-copy check used the tracked Bi2Te3 YAML/CIF and a
real hard-link alias. The unchanged read returned the exact dependent hash,
same-handle identity and both dependent alias binding keys. In a controlled
replacement after the canonical load, old hash H1 and new file identity I2
were present, and the worker raised
`dependent CIF changed during configuration validation` instead of publishing
I2 with H1. Temporary external files were removed. This pure check used no
native GUI allowance and does not close the sustained U02a qualification gate.

#### U02a mixed-source qualification preflight (2026-09-28; stopped)

The separately frozen 20-acquisition, four-source reporter stopped during pure
input preparation, before its state-machine preflight or any native GUI run.
`python -B <external u02a_mixed_reporter.py> --prepare` wrote the deterministic
20-UUID project and manifest, then failed an exact CSV text assertion for a
mapped 7.25-degree value. The stored angle was
`0.1265363707695889` radians and CSV serialized it as
`7.2500000000000009` degrees; the mapped 360-second exposure stayed `360`.
The export follows the documented floating-point degree/radian conversion;
this checker expected the literal `7.25` string without allowing its numerical
round trip. No native window, 20-record smoke, fit or performance measurement
ran, and the new 130-second native allocation remains entirely unused. The
external reporter/input/manifest are retained for independent review; U02a
remains unaccepted and its sustained gates have no new evidence.

#### U02a mixed-source native qualification window 1 (2026-09-28; stopped)

After pure reporter repair, the frozen first native window ran once and failed;
windows 2 and 3 did not start. The child ledger charges **43.983 s of 130 s**;
the supervisory command took 44.797 s, the conservative whole-command charge.
The unused allocation is suspended. The active interval was 29.988 s,
readiness 1.169 s and drain 12.046 s, ending in a final-composition timeout.
All 27 table/tree/filmstrip selection hits selected their intended UUID, but only
the initial hBN UUID composed during active time (3 matched warm returns to that
same image; no matched cold switch). There were 192 queued/running import jobs, 169 completed
and 23 canceled. The first Bi2Te3 job completed and the same source was queued
again about 2 ms later; its first actual publication was 0.146 s after active
end. This is consistent with the live import guard comparing a load-start
revision against a revision also changed by cursor/view activity. That
mechanism is an inference from the trace and current code, not a proved repair.

For the initial visible hBN state alone, 84 intervals with at least one effective
input had p95 39.366 ms and p99 46.578 ms, above both 16.7/33.3 ms gates.
All 3,695 mouse requests left the native crosshair at (1500, 1500); 2,772
were no-ops and 923 changed only band widths every fourth callback. Effective
input spacing had median 32.110 ms, and each of the 84 intervals contained
exactly one such input. These measurements do not qualify sustained redraw or
establish a renderer bottleneck. Future classification requires the delivered
mouse handler to change the intended native query. Active-boundary and 10 ms
heartbeat samples independently record selected/visible source, admission,
image availability, cursor follow, plane and loading state. Every sampled ready
browsing span, including its head and tail, requires changed native cursor
requests with no gap over 16 ms; a ready telemetry gap over 16 ms also fails.
An isolated ready sample is unqualified.
Cold/loading/unavailable spans are excluded explicitly, with eligible, covered
and excluded durations and maximum gaps reported. The old intervals and misses
remain recorded; insufficient coverage is an additional qualification failure,
never a reason to discard slow frames or relax gates.
Ninety matched cursor/profile presentations had p95 14.595 ms; heartbeat maximum was
22.961 ms. Process sampled RSS maximum was 447,082,496 bytes and Windows peak
working set was 449,363,968 bytes; GPU driver allocation was unmeasured. The
final query targeted Bi2Te3 while hBN remained visible; later Bi2Te3 paint did
not satisfy the prematurely captured final identity, so no final save, reopen
or successful acknowledgment is claimed. The failure path observed a hidden,
drained owner about 4.402 ms after close request; its first matching visible
close acknowledgment was not recorded because the failure guard stopped paint
observation. That upper bound does not replace the missing acknowledgment.
External raw events SHA-256 `e088890ea7d37fd36558b57bb86b73334133f4c7e4d6de468e4d5b80d71f7520`
and summary SHA-256 `09ab1099021838798ad3caef54b1bd59496b941e2317258de8acde4a57c88f3f`
are retained for audit. U02a remains unaccepted; no native retry is authorized
by this checkpoint.

#### U02a source-binding and reporter correction (pure-only checkpoint)

The import result now compares the captured original OSC path and SHA-256 with
the current acquisition binding, while the existing project, selection,
generation and decoded-byte checks remain. Cursor/view and metadata revisions
no longer discard an unchanged-source plane; a relink keeps current metadata
and changes only the admitted source path. A focused external call to the actual
import methods covered unchanged binding after revision changes, changed path
and hash, removed row, changed project/selection, obsolete generation, wrong
decoded hash, same-byte relink and independent save revisions. The external
reporter dispatches a no-button move through the delivered Qt mouse handler,
checks the intended native query, requires continuous changed cursor supply,
and waits for selected-source admission before fixing the final query. Pure
stub checks covered sparse/missing and 8 ms supply, handler routing, delayed
admission, and fresh matching G/H/V composition. These checks do not establish
native delivery, twenty-image throughput, sustained frame latency, final
save/reopen or U02a acceptance; no GUI window ran in this correction packet.
A follow-up pure audit found that one cursor request in a 30 s otherwise ready
window could pass the prior interval-only supply check. The reporter now checks
whole sampled ready spans; actual-analyzer pure cases reject missing, initial
and trailing silence and 32 ms sparse supply, accept 8 ms supply, and separate
a bounded loading gap. The final query also explicitly requests one G/H/V
redraw after setting its fixed controls, including when those controls were
already equal. Pure actual-helper stubs require new post-request paints and
reject historical paints. No GUI window ran in this follow-up either.

#### U02a authorized native window 2 (2026-09-28; failed and stopped)

One fixed window-2 continuation verified the immutable failed-window-1 ledger,
then wrote its own one-use receipt before launch. The original ledger remains
unchanged. The reporter's internal interval from `NativeReporter.started`
through report work was 32.536 s, its controller took 33.356 s, and later
parent command metadata exposed 33.679 s of command execution. The command
remains conservatively charged **36.000 s**; the yielded PTY did not expose
exact whole-command wall time. With the prior 44.797 s charge, 80.797 s
of the original 130 s is charged and 49.203 s remains suspended. Readiness was
1.150 s, active browsing 29.992 s and drain 0.778 s. No preflight, retry or
window 3 ran.

All 20 UUIDs composed, 27/27 selections matched, and 2,009 active swaps were
coherent with no selected/visible historical swap. Six warm returns had p95
23.268 ms; 21 cold selections had p95 188.963 ms. Matched cursor/profile p95
was 14.228 ms. The 1,975 any-effective-input frame intervals had p95 14.666 ms
and p99 18.353 ms; the 1,948 covered interval subset had p95 14.515 ms and
p99 15.222 ms. These percentiles do not qualify sustained redraw: all 22
sampled ready browsing spans failed the 16 ms continuous-supply rule, yielding
0/26.501 s covered ready time and 3.491 s explicitly excluded. There were 60
ready-span cursor gaps over 16 ms (maximum 23.014 ms) and 72 ready telemetry
gaps over 16 ms (maximum 27.903 ms). The nominal 8 ms cursor timer delivered
2,497 changed inputs with median spacing 13.044 ms. The first ready telemetry
gap was 22.407 ms from active start. This is a demand-supply qualification
failure, not evidence of a renderer bottleneck. Heartbeat maximum 27.903 ms
remained below its separate 100 ms gate.

The captured two-UUID bulk metadata edit, final exact native/profile values,
ordinary Save, complete 20-record disk comparison, background reopen, new
publication and matching G/H/V, and normal drained close were recorded. The
bulk-edit, Save and close visible acknowledgments were 43.245, 4.702 and
1.157 ms. Sampled RSS peaked at 509,186,048 bytes and Windows peak working set
at 552,865,792 bytes; GPU driver allocation was not measured. The source
freeze was production `f527462` with pre-run docs HEAD `67ec8c4`. External
raw, summary and continuation receipt SHA-256 are respectively
`2feef12d26ba426b809f88b48dea13f37c749d9b0fca864b695e6eabda2e524d`,
`82b9a8636a465e68b4a4f1126f0c24044ce3e4a3b145dfdf5be2777b37616b89`
and `b16b81e39c502af17bc2732418649de3b5b42a3d6ade9826b29a3e7a49388ccf`
after the accounting-only receipt annotation. The original receipt hash was
`6b9a48b6814a820335edbcc6fe5ecb9ffe2ee9542b54a039c82be3361bafe8e6`.
The failed window 1 and its five immutable files remain unchanged. U02a is
unaccepted; the window-3 active-old-write/newer-edit gate remains unrun.

#### U02a prospective rendering-demand measurement correction (2026-09-29; pure only)

The frozen window-2 result above remains **FAILED** under its original
delivered-input and ready-telemetry 16 ms rule. Its raw and summary files were
not rewritten, and the correction does not infer a renderer bottleneck from the
observed supply gaps. The external reporter and manifest now name a separate
`rendering-demand-v2` rule for prospective measurements. The old delivered-input
coverage and sustained-frame subset remain reported as diagnostics.

The first pure packet tested the reducer and canonical matcher directly, not
their integration through `analyze()`. One matrix stopped at an incorrect
supersession expectation before its cold case; a later focused check covered
supersession and repeated-state ABA only. Its selfcheck/Ruff/AST pass did not
establish a positive full-analyzer case. Audit then found that normalized
swaps exposed `hash` while the reducer required `decoded_sha256`, so the
actual analyzer could never discharge pending demand. The audit also found
two extra hard limits: 50 ms on a continuous busy episode and 16 ms on ready
state samples. Both limits have now been removed from prospective qualification.

The corrected rule derives ready spans from active boundaries, heartbeats and
recorded selection, publication and loading transitions. Missing transition
snapshots fail qualification. Selected/visible, project, publication, decoded
source, native and display identity must agree. An effective cursor request
starts pending time until its own first exact coherent G/H/V presented swap.
Supersession carries outstanding demand; an older presentation still counts
for its response/frame observations but cannot discharge newer demand. The
report separates eligible, pending, undemanded, covered and excluded time,
request and composition coverage, superseded/unresolved counts, and maximum
idle, state and pending-episode durations. Only genuinely undemanded heads,
between-request intervals and tails have the 16 ms coverage ceiling. Each
ready span needs a composition and unresolved demand at a boundary fails.
State-sample and pending-episode maxima remain diagnostics. The 100 ms
heartbeat, cursor p95, fresh-frame p95/p99, resource, selection and
persistence gates remain separate.

A temporary external pure matrix constructed raw publication, input, completed
G/H/V paint, swap and state events and called the actual `analyze()` function.
Healthy 8 ms streams with 5 ms and 10 ms exact responses qualified demand,
with cursor p95 5/10 ms and fresh-frame p95 8 ms, without those performance
or heartbeat misses. The 10 ms case retained a 114 ms busy episode as a
diagnostic. A 25 ms response qualified demand with both 10 ms and 25 ms state
sampling; its 30 ms fresh-frame p95 failed the unchanged frame gate. A 200 ms
response qualified demand coverage but failed the existing cursor p95 gate.
Sparse 32 ms, no/head/tail and 30 s one-input supply, never-composed demand,
wrong hash/publication/native/display/profile identity, historical paint,
old-only supersession, selection ABA between heartbeats and missing transition
state all failed coverage. A valid cold separation with new-source demand
qualified each ready span. The temporary matrix was removed. The existing
pure selfcheck, Ruff lint/format, AST and JSON checks passed. No GUI or native
window, preflight, retry or window 3 ran; U02a remains unaccepted. At this
checkpoint, reporter and manifest SHA-256 were
`1ca5b9d1852529ae7a76cce569aad66ef95faf8c9ac41bb9b4a2f00e3c3c4925`
and `e64218a0010a7e3df14e7e2ee709232d1b86a993659228d7cdd9ed3e577784f3`.

#### U02a prospective ready-boundary correction (2026-09-29; pure only)

A further actual-analyzer audit found two boundary errors in that checkpoint.
An unchanged eligible save QUEUED/RUNNING status split one ready span and
censored pending demand. A genuine loading transition ended the old span at
the preceding heartbeat, dropping an input and its unresolved demand before
the known transition time. The earlier publish and COMPLETED callbacks still
observe loading; ready re-entry occurs at the existing post-result callback.

The external reporter now extends a ready span through observations with the
same eligible identity, cuts explicit pre-selection boundaries, and closes a
real departure at its recorded timestamp using the preceding ready binding.
The existing post-result record carries the scalar ready-state snapshot and
starts a new span when ready. Missing transition state fails qualification.
The strict cursor/composition matcher and all response, frame, heartbeat,
resource, final-state and persistence gates remain unchanged.

Focused raw-record calls to `analyze()` compared healthy 8 ms input with 5 ms
and 10 ms responses before and after unchanged save QUEUED/RUNNING/COMPLETED
and post-result observations. Both retained 40 ms eligible time, the same
cursor/frame distributions and zero unresolved requests. A request at 14 ms
without composition failed at the true 15 ms loading cut; a genuinely
undemanded 22 ms tail before a loading cut failed. A 12–18 ms ready interval
between post-result and selection was measured without a heartbeat: absent
demand failed, while a 2 ms exact response qualified that span. A valid cold
gap followed by new-source demand qualified two ready spans. Hidden selection
ABA and a missing post-result snapshot failed. The temporary matrix was
removed. The reporter's pure selfcheck, Ruff lint/format, AST and JSON checks
passed after updating its existing stub for the post-result snapshot. No GUI
or native window ran. Reporter and manifest SHA-256 are
`5ad38b470d2276912e8721390cdd15ccca689dc9edb1d88c96c6ee5baf4c17c8`
and `4b04be927d43c0dc3d65bdce0cd85ea5261ed5da11b35c34cfd890b78ce80937`.
Window 2 remains failed under its frozen rule; U02a remains unaccepted.

#### U02a window-3 admission assessment (2026-09-29; pure, blocked)

The original 130 s allocation has 44.797 s charged to window 1 and the
conservative 36.000 s window-2 charge. Its authoritative remainder is
**49.203 s**, suspended. The later 33.679 s window-2 command-execution
measurement does not reduce that charge or establish a worst-case bound.
The original `remaining_native_budget` rejects the failed prior ledger and
requires at least 50 s; its guard and the consumed one-use window-2 path stay
unchanged. No window-3 receipt or output was created.

The complete command would need to reserve interpreter/import and controller
admission work, child import and project construction, up to 8 s readiness
from `NativeReporter.started`, the full nominal 30 s active workload, the
unchanged 12 s post-active final-query/save/reopen/close deadline, then raw
and summary writing, analysis, parent validation and durable final receipt.
The reporter starts its readiness clock *after* child imports, its source-freeze check
and the input project copy. Its 12 s clock stops before report writing and
parent validation. Let `O` be all other whole-command time. A guaranteed
full-journey envelope would require `8 + 30 + 12 + O <= 49.203` seconds, or
`O <= -0.797` seconds. Since `O` includes positive work and has no existing
hard bound, this allowance cannot admit the unchanged journey. Faster prior
readiness/drain measurements do not remove the protected phase allowances.
A mere change to the original 50 s admission threshold would still give its
child at most `49.203 - 1 = 48.203` s under the existing timeout expression,
below the 50 s phase envelope before child startup or parent reporting.
A timeout that makes an attempt fail cannot turn this inequality into a
qualified observation.

A pure call to the actual budget helper with the immutable ledger and index 3
raised `prior native attempt failed; stop`; scalar accounting independently
returned `130 - 44.797 - 36 = 49.203`. The manifest input hash still matches
the immutable input. The internal `--child-window 3` route has no supervisory
budget or one-use receipt and is not an authorized admission path. No new
window-3 entry point, reservation or release values were prepared. There was
no child, QApplication, preflight, retry or native time. Windows 1 and 2
remain failed; the active-old-write/newer-final-edit window-3 journey and
three-window U02a qualification remain unrun and unaccepted.

#### U02a fresh-series release packet (2026-09-29; pure preparation, pending approval)

The external reporter and manifest now describe a prospective new 180 s allowance,
reserved as three separate 60 s whole-command attempts. Ordinals 1 and 2 repeat
the mixed browse, bulk metadata, final save and reopen journey into output indices
4 and 5; ordinal 3 records the active old-write/newer-final-edit journey into
output index 6. Each uses a fresh receipt, one-use child-start claim, project,
event log and summary. The original window-1/window-2 evidence and its 80.797 s
charged/49.203 s suspended ledger remain untouched. This proposal does not revive
that allowance.

Admission requires a separate release file written by the sole implementation
writer only after an explicit supervisor release message following user approval
of native time. The release binds the clean source freeze, external launcher,
reporter, manifest, fixed input and original budget/receipt hashes, journey mapping and
180/60 s limits. It does not exist in this preparation packet. The controller
rejects missing or altered release, duplicate/future outputs, a missing or
failed prior receipt, changed prior evidence and exhausted reservations before
launching a child. The external launcher starts its deadline before starting the
reporter process and gives that process at most 57 s for import, admission,
native child execution, analysis, hashes and its final receipt write. It reserves
3 s of the 60 s attempt for owned-tree shutdown and its own receipt work. The
reporter reserves the entire 60 s in an exclusive durable receipt before the
native child starts. If a started reporter exits unsuccessfully before that
reservation or the launcher stops it first, the launcher writes a conservative
failed 60 s receipt. Missing release or duplicate receipt/claim refusal before
process launch creates no charge. On timeout or interruption it stops only that
attempt's owned process tree; an exclusive fsynced child-start claim prevents a second
native child even if the first dies before project output. Timeout, incomplete
evidence and budget failure cannot be replayed. A candidate success remains
unqualified until the sole writer records the independent whole-command duration,
including launcher interpreter/import, admission, child and parent/reporting time.
A duration over 60 s fails the attempt and
stops the series. The internal child timeout is a second bound, not a substitute
for whole-command supervision. The watchdog starts after its own imports;
the external duration check includes those imports and the launcher's final
receipt flush/exit. The measured pre-finalization clocks in the receipt are lower bounds, not claims
of full command duration or hard real-time OS termination. No preflight or
replacement attempt is included.

Pure synthetic admission/accounting checks exercised release mismatch,
sequential and duplicate admission, prior failure, cumulative charge,
whole-command threshold, successful provisional receipt/confirmation,
child failure and timeout stop. A focused repair check then used the actual
launcher/reporter path with fake processes, exercised repeated and concurrent
child-start claims before output, and verified timeout, interruption, expired
clock and owned-tree termination paths. A final launcher check verified failed
nonzero exit before receipt, duplicate refusal, uncharged missing-release
refusal and the 3 s cleanup reserve after launch overhead. The temporary fixtures
created no native application or OSC decode and were removed after use. The reporter's
existing `--selfcheck` and source freeze are separate pure checks. Three fresh
qualified windows are still needed;
U02a remains unaccepted and U14b remains outside this release.

#### U02a fresh-series ordinal 1 (2026-09-29; failed and stopped)

After user approval and explicit supervisor release, the sole writer created the
exact source-bound external release and ran ordinal 1 once through the native
watchdog. It used journey 2 and output index 4. The worker-reported tool-wall
interval was **34.633 s** and the command exited 1; a separate root
`read_thread` command-execution record reports **34.440 s**. The controller
and launcher receipt clocks reported pre-finalization lower bounds of
34.252 and 34.293 s. Their different boundaries are unresolved and do not
change the charge. The reporter exited 2 after completing its native journey
and analysis. The durable receipt records
`controller_failed` with reason `child exited 2`. The entire **60.000 s**
attempt is charged against the new 180 s allowance. Ordinals 2 and 3 were not
started, and the failed ordinal was not confirmed or retried. The remaining
120 s of the new allowance is unspent but the first-failure series is stopped.
The original 80.797 s charged/49.203 s suspended ledger and all nine prior
immutable evidence files remain unchanged.

The application reached readiness in 1.305 s, browsed actively for 30.003 s,
and drained in 0.798 s. All 20 UUIDs composed; 27/27 selections matched; 1,994
active swaps were coherent, with zero incoherent or historical-display active
swaps. Six warm selections had p95 24.685 ms and maximum 24.919 ms; 21 cold
selections had p95 198.184 ms and maximum 212.824 ms. The 1,983 matched cursor
responses had p95 14.255 ms, p99 19.081 ms and maximum 24.478 ms. The 1,963
fresh-frame intervals had p95 14.686 ms, p99 21.449 ms and maximum 33.737 ms;
the 1,925 sustained-frame intervals had p95 14.449 ms, p99 15.741 ms and
maximum 24.930 ms. Raw swap intervals include loading: p99 169.194 ms and
maximum 214.315 ms. Heartbeat maximum was 28.424 ms. These separate frame,
cursor and heartbeat distributions met their frozen percentile gates.

The sole qualification miss was prospective `rendering-demand-v2`. Its 28
ready spans contained 26.552 s eligible time, 25.209 s pending demand and
1.343 s classified-undemanded time. The longest gap classified as undemanded
by the frozen cursor-only method was **19.215 ms**, over its 16 ms ceiling;
19 effective requests remained
unresolved at ready-span boundaries. Of 2,182 effective requests, 180 were
superseded and 1,983 composed, yielding 0.99051 request coverage. All spans
had composition, and no identity-invalid request was recorded. The older
delivered-input coverage remains diagnostic and showed zero covered time in
22 sampled ready spans; it is not the prospective failure rule. This evidence
shows a demand-coverage failure, not a proven renderer bottleneck.

The two-UUID bulk edit, final exact native/profile values, ordinary Save,
complete 20-record disk comparison, background reopen, new publication and
matching G/H/V, and normal drained close were recorded. The bulk-edit, Save
and close visible acknowledgments were 42.480, 3.851 and 1.042 ms. Resource
maxima were two cache entries, 184,320 thumbnail bytes, one queued write of
18,621 bytes, one pending job and eight job summaries. Sampled RSS reached
526,958,592 bytes and Windows peak working set 558,047,232 bytes; GPU driver
allocation was not measured. No reporter/product failure event or persistence
acknowledgment miss was recorded.

The frozen production source remains `f527462`; the pre-run docs HEAD was
`615c25a`. External release, raw, summary and failed receipt SHA-256 are
respectively `803035e421f00b819cd65775dc26c35e8de9791f3f7008a0c2ded005712c41b7`,
`08131a59932f54527d409fb0a7dfe707f77bb9df824451f08502d322d162d35b`,
`0db0ead71685f754b45da966fe8cafdc9d431ba7d1573925c895a053fdd08ab9`
and `58ce607407e812c1851e5107b34bdfea04aac0291fd07ad556d5b14ba07dccc6`.
The failed evidence remains external and immutable. U02a remains unaccepted;
U14b remains outside this series.

#### U02a prospective demand-classifier candidate (2026-09-29; pure review packet)

Independent read-only review of the failed output-4 trace found that all 19
frozen unresolved requests ended at explicit selections of a *different*
UUID, with post-dispatch `selected_after` confirming the target. Their pending
ages were 0.410-1.366 ms; none had an exact subsequent G paint and matching
swap before the cut. Five cuts were warm and 14 cold. Production intentionally
hides the old panel on these selections. The one 19.215 ms gap was between
publication 23 (event 7248) and its first cursor request (event 7256).
The source's initial strict completed G/H/V presentation and coherent swap
(event 7255) occurred inside that interval; the first cursor followed the
swap by 0.375 ms. The frozen result remains **FAILED**: these findings explain
why its cursor-only labels need prospective review, and do not retroactively
qualify the old run or establish continuous rendering demand.

One external `u02a_mixed_reporter_v3_candidate.py` copies the frozen reporter
and changes only its prospective analyzer, pure selfcheck and command entry
point. It is an **unreleased classifier candidate**, not a qualification
runner. Its entry point accepts only `--selfcheck`; all preparation, native,
release and confirmation command routes are disabled before calling their
implementations. The candidate counts an admitted publication as pending
initial-presentation demand through its first exact latest completed G/H/V
composition, intersected with recorded ready spans. Source presentation and
cursor responses stay separate. A later cursor request supersedes older
outstanding demand. Only a recorded different-UUID selection with matching
post-dispatch target can explicitly cancel the old request. The candidate
retains canceled request identity and age as a censored lower bound; it does
not count cancellation as composition or a fast response. Pending time, true
idle heads/tails, actual frame and cursor distributions, and unresolved
boundaries remain visible. A 50 ms p95 lower-bound failure is reported only
when completed latencies plus nonduplicated censored ages prove that limit
cannot pass. Every ready span with cursor requests needs cursor presentation
progress, so repeated cancellation and source publication alone cannot
qualify. The 16 ms true-idle limit and the unchanged frame, cursor, heartbeat,
acknowledgment, workload, identity, resource and persistence gates remain.

The candidate's one finite actual-`analyze()` synthetic matrix passed
confirmed selection cancellation without composition credit; same-UUID,
failed, unconfirmed, absent-selection and ordinary active-end pending
rejection; exact publication presentation followed by cursor demand;
post-presentation idle over 16 ms rejection; wrong buffer, publication,
generation, query and historical paint rejection; supersession of a late
initial paint by newer cursor demand; a 112 ms canceled pending age whose
censored p95 lower bound and raw swap tail remain visible; repeated
cancellation without cursor progress; and A-B-A stale-publication rejection.
The existing pure reporter selfcheck also passed. One initial matrix assertion
used an exact 112 ms floating-point comparison and failed at rounding; its
tolerance was corrected to 111.9 ms without changing the classifier.
Ruff lint/format and AST command-entry inspection passed. No candidate run
used the historical raw output to relabel it, and no GUI, decode, preflight,
native attempt or budget/release action occurred in this packet.

The first candidate checkpoint was 156,077 bytes, SHA-256
`ae2ebc67fb8cf0c73fce67a8d67109f540942c9a72aa11ed3c2a026f7c7935c4`.
All 18 pre-existing external files, including the failed raw, summary,
receipt and child claim and frozen reporter/manifest/watchdog/release, remain
byte-for-byte unchanged. The candidate needs independent supervisor review
before any release decision. U02a remains unaccepted, and the first-failure
native series remains stopped.

#### U02a delayed cursor-presentation correction (2026-09-29; pure only)

Independent review found one prospective classifier error: an exact cursor
presentation for an older request remained useful evidence even when a newer
cursor request was still pending. The first candidate counted only
presentations that cleared the current pending request, so a healthy delayed
stream could have ten exact current-identity presentations yet report zero
cursor progress. It correctly kept the newest request pending, but its
per-span and global composition admission then failed.

The same external candidate now counts distinct canonical exact cursor
presentations by presented swap ID for useful per-span and global progress.
`composed_requests` still counts only requests that clear current pending
demand; `cursor_presentations` reports the independent observed count. An
older presentation cannot discharge a newer request or acquire a second
terminal disposition after supersession. The original request partition,
source matching, censored ages, true-idle 16 ms ceiling and separate
response, frame, heartbeat and persistence gates remain unchanged.

One added actual-`analyze()` synthetic pair used a publication at 1 ms,
initial presentation at 2 ms, eleven cursor requests at 8 ms intervals
through 88 ms and ten exact presentations 10.5 ms after their requests.
With confirmed different-UUID selection at 94 ms, it recorded ten cursor
presentations, zero cleared-current requests, ten superseded requests, one
explicit cancellation, zero unresolved boundaries and qualified demand
coverage; cursor p95 stayed under 50 ms, frame p95/p99 under 16.7/33.3 ms,
and maximum true idle gap under 16 ms. The paired ordinary active end left
the newest request unresolved and failed despite those ten older
presentations. Existing repeated cancellation with no actual cursor
presentation still failed. The focused matrix, existing pure selfcheck,
Ruff lint/format and AST pure-only command-entry check passed. No native
or historical-real-window candidate execution occurred.

The revised candidate is 158,004 bytes, SHA-256
`a90e1dab2ecbe0e33f32956388b6b9f91f10515be50ee1d337279180fe86d5a4`.
All 18 original external files remain unchanged. Failed window 4, its
60 s charge and the 120 s unspent stopped allowance remain as recorded;
the original 80.797 s charged/49.203 s suspended ledger is also unchanged.
U02a remains unaccepted pending supervisor review.

#### U02a v3 fresh-series proposal (2026-09-29; pure preparation, pending new approval)

Independent review accepted the prospective delayed-presentation classifier
repair above as a **pure preparation result**. It did not change the failed
output-4 verdict or authorize native time. Three fresh passing windows are
still needed; the third must cover the active 220 ms old-write/newer-final-edit
journey, final Save, disk comparison, reopen and normal close.

The external proposal consists of only three new files in the existing
evidence root: `u02a_mixed_reporter_v3.py`,
`u02a_mixed_manifest_v3.json` and
`u02a_fresh_watchdog_v3.py`. The reporter copies the accepted pure
classifier candidate and changes the fresh-series output mapping to
**7/8/9** for journeys **2/2/3**, the manifest/watchdog/release paths,
the matching native-output admission, and the release binding to include
the preserved stopped output-4 receipt and original fresh release hashes.
Its CLI enables only the existing reviewed `--fresh-window`,
`--fresh-child` and `--confirm-fresh` routes plus pure `--selfcheck`;
legacy preparation, original-window and window-2-resume routes remain
unavailable. The pure candidate itself remains unchanged and selfcheck-only.
The new watchdog changes only the reporter/release filenames, output mapping
and displayed usage. The new manifest preserves the fixed input hash,
20 records, 27-selection sequence, source identities and frozen limits,
while naming prospective v3 classification and the new proposal.

This is a proposed **new 180 s whole-command allowance**, three separate
60 s reservations at most, one nominal 30 s active journey each. It is
**PENDING explicit new user approval and later supervisor release**. The
release filename `u02a_fresh_release_v3.json` is declared but **absent**.
The existing launcher still allows at most 57 s for reporter work and
reserves 3 s for cleanup; the protected phase envelope remains up to 8 s
readiness, nominal 30 s active work and up to 12 s post-active work inside
the full 60 s command, including startup, analysis and receipt. Independent
whole-command timing/confirmation and conservative 60 s charge remain.
Stop on first failure; no preflight, retry, replacement or automatic
extension is included. The old 80.797 s charged/49.203 s suspended ledger
and the failed earlier new-series 60 s charge/120 s suspended remainder
remain separate and cannot be borrowed or reclaimed. Outputs 5 and 6
remain absent.

One focused pure check imported the actual new reporter and watchdog
without GUI or OSC decode, compared their 7/8/9 and 2/2/3 mappings with
the parsed manifest, verified the exact source/input/new-file and
preserved-ledger hash binding, and confirmed the new release was missing.
Reporter attempt, child and confirmation entry functions and watchdog
entry all rejected the missing release before a reservation, claim or
process launch; outputs 7-9 remained absent. AST inspection found only
the reviewed fresh-series routes in the reporter CLI. The changed
reporter's existing pure selfcheck and focused prospective matrix passed;
Ruff lint/format and JSON checks passed. No native window, release,
budget, receipt, claim, project, raw log or summary was created.

The prospective reporter is 159,236 bytes, SHA-256
`9e8ecb806877c8b98414053a835a2163d6acdaf684fcc32b94cf2156df2895fb`;
manifest 12,282 bytes,
`19832b7c8607f06dfb44afffc0dfe92ab8624982a368272000c7e0fdaed630a2`;
watchdog 6,810 bytes,
`eecddf34f09a4b67f1b230ea4183452050bbc5daf74ed46a7e6e5c817d696732`.
All 19 pre-existing external files, including the accepted pure candidate
and all frozen failed evidence, remain byte-for-byte unchanged. This
proposal awaits supervisor review before a single concrete runtime
decision is put to the user; U02a remains unaccepted and U14b is outside
the proposal.

#### U02a v3 fresh-series native qualification (2026-09-29; supervisor review pending)

The supervisor released the exact v3 packet above under a new, separate
180 s whole-command allowance. The sole writer created
`u02a_fresh_release_v3.json` from the reporter's actual
`fresh_release_binding(source_freeze())` at clean documentation HEAD
`e18897110dc1244091dac496696d399ba33e33df`. Its SHA-256 is
`d9e0c407daf08cf07d0cabdae256ac50f68be23dc3a980699afb2e4cbf16ef3d`.
The binding fixes the 7/8/9 outputs, 2/2/3 journeys, 180/60 s limits,
source and input hashes, all three v3 file hashes, and the original and
previously stopped budget evidence. The earlier failed verdicts and ledgers
remain unchanged. Outputs 5 and 6 remain absent.

For each ordinal 1, 2 and 3, a separate outer Python `perf_counter()`
bracketed the complete command `python -B <external
u02a_fresh_watchdog_v3.py> <ordinal>` through process exit, including
launcher imports, startup, readiness, 30 s active work, post-active save,
reopen, reporting, cleanup and final flush. After examining each provisional
receipt and summary, `python -B <external u02a_mixed_reporter_v3.py>
--confirm-fresh <ordinal> <whole-command-seconds>` wrote its confirmed
whole-command duration. No preflight, decode replay, retry, replacement,
extra GUI command or source/reporter/manifest edit ran during measurement.
All three attempts returned 0 and were conservatively charged 60 s each;
the new series is fully charged at 180 s, with no reclaim of unused time.

| Output / journey | Whole command (s) | Ready / active / drain (s) | Warm select p95 / cold select p95 (ms) | Cursor p95 / sustained fresh frame p95 / p99 (ms) | Heartbeat max / visible ack max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 / 2 | 34.198 | 1.148 / 29.997 / 0.796 | 22.463 / 189.894 | 14.472 / 14.616 / 16.043 | 28.720 / 44.294 |
| 8 / 2 | 34.168 | 1.126 / 30.008 / 0.808 | 22.687 / 189.023 | 14.112 / 14.511 / 15.135 | 27.306 / 43.703 |
| 9 / 3 | 34.236 | 1.160 / 30.000 / 0.807 | 23.479 / 189.879 | 14.134 / 14.467 / 15.280 | 26.361 / 45.612 |

Each summary recorded 27/27 matched selections, coherent active swaps
(2004, 2014, 2013), zero incoherent/loading/historical active swaps, no
reporter failure or missing evidence, and a normal drained close. The
prospective v3 demand classifier qualified all three: exact coherent
composition coverage was 1.0, every ready span had composition and cursor
progress, no unresolved boundary or invalid identity remained, and maximum
genuinely undemanded ready gaps were 12.557, 10.867 and 11.328 ms under
the frozen 16 ms limit. Sustained fresh-frame p95 stayed below 16.7 ms
and p99 below 33.3 ms; cursor p95 stayed below 50 ms, heartbeat maximum
below 100 ms and all visible acknowledgments below 100 ms. Cold selection
timing is retained separately from the frozen warm-selection 150 ms gate.

Every run retained a matching project, raw event stream, summary, child
claim and confirmed budget receipt. Final reopened G/H/V matched the
strict latest completed publication 29 for Bi2Te3, selected UUID
`9028630f-acc4-581c-bb03-24063f4899be`, source SHA-256
`6f00b27802e6419ad79d9ec38441f3cd051b0becaba399f351caeb9a0e451c26`,
sample `(0,0)=14`, horizontal sum 77988 and vertical sum 73153, with
20 source checks and a second full-document disk comparison after reopen.
Output 9 additionally records the old revision 2828 write becoming active
at event 9863, the newer revision 2835 edit at event 9872 during that
write, and the old-write receipt at event 9988 about 220 ms later. Final
Save persisted revision 3794 with a full current-document comparison;
reopen produced the newer publication and current G/H/V before normal
close. Both edited acquisition records retained `mount=reviewed-newer` on
disk. The stale old write did not replace the final document.

Peak cache entries were 2 each; thumbnails 184,320 bytes; write queue
depth 1 and bytes at most 18,621; pending jobs 1; retained job summaries
8. Sampled RSS maxima were 484.6, 502.5 and 473.8 MiB and Windows peak
working sets 529.7, 524.3 and 531.0 MiB, within the frozen resource
limits. GPU driver allocation was not measured; the existing source-derived
display/upload planning bound is the applicable GPU check. Receipt hashes
match all retained raw streams and summaries. These are native UI
qualification observations, not numerical fitting validation. U02a awaits
independent supervisor evidence review and is not automatically accepted;
U14b remains outside this work.

#### U02a acceptance clarification (2026-09-29)

U02a was accepted at `407716ee1c0bdec2b55ff574d87cfe03496701a7`
after independent review of all three new raw streams, source identity,
current G/H/V generations and buffers, saved documents, receipts and bounds.
The checklist is now **9 of 46 accepted, 37 remaining**. Earlier failed
windows retain their original verdicts; the new v3 series is closed with
180 s conservatively charged. The table above labels the sustained-demand
subset; the actual full fresh-frame gate retained all 1979/1986/1983
intervals and measured p50/p95/p99/max respectively as
13.291/14.834/18.836/27.854 ms, 13.296/14.554/17.845/27.065 ms, and
13.286/14.589/18.372/31.231 ms. Its 41/31/40 intervals above 16.7 ms
remain included, and none exceeded 33.3 ms. The accepted gate is the full
fresh-frame p95 below 16.7 ms and p99 below 33.3 ms in every window.

In output 9, the old-write receipt followed the newer edit by 220.084 ms
and the observed active-write event by 244.520 ms. The final complete
20-record disk document, SHA-256
`405bfdc75d5e4e48f65e0327b774568b0698c2610fd7e2f7b3823c048a9d036e`,
matched before and after reopen. Only the two intended rows retained
`mount=reviewed-newer`; publication 29's latest G/H/V and exact native
profile values were current at reopen. Memory had 129/129/130 samples,
with maximum active sample gaps 0.289/0.289/0.291 s. Late-window RSS
ranged 404.95-421.29, 404.32-420.09 and 406.10-420.95 MiB; drained
RSS was 343.42, 342.14 and 356.95 MiB. These samples do not establish
full memory reclamation or GPU driver allocation. U02a acceptance is for
the declared detector UI workflow, not scientific fitting.

#### U14b inspection export implementation (2026-09-29; review pending)

The detector panel now offers **Export figure + profiles** for the current
visible OSC acquisition. It captures the displayed OpenGL viewport and
overlays with the two existing Qt profile plots and native-coordinate axes;
the paired CSV serializes the canonical current `BandProfiles` vectors, not
the framebuffer, plot decimation, contrast or clipped display values. The
rectangular CSV has metadata records naming project/acquisition UUIDs,
decoded-source SHA-256 and scope, native shape, source-bound panel data
revision, effective band/ROI bounds, profile measure, units, viewport DPR
and PNG hash. Its profile records include every horizontal native column
and vertical native row, exact integer or round-trip float64 value, integer
support and an explicit missing flag. Zero support is missing even when a
sum is zero; means at zero support are `nan`.

The chooser requires a new external PNG path and derives a distinct
`.profiles.csv`. The existing Git/project/source/reference alias guard is
shared with metadata CSV export; both destinations must be new. Figure
capture is limited to 16 MiB raw, CSV to 3 MiB, and the immutable paired
request to the existing 4 MiB `JobOwner` limit. A single worker writes and
reopens both files, returns their hashes, and removes its newly created
files on failure. Before and after the destination dialog and synchronous
capture, the shell checks the selected/visible acquisition, native data
revision, current query key, canonical profile buffers and viewport state.
Later selection changes do not relabel or invalidate the immutable write
request. Success is shown only after the completed write receipt; failure
is visible. No new job pool, numeric reducer or physics path was added.

Focused external pure `python -B -` checks wrote and reopened one PNG/CSV
pair. Independent expected records covered signed integer sums, values
above 2^53, float64 means, nonfinite input exclusion, zero support,
unequal horizontal/vertical lengths, provenance and unit fields. An
existing destination retained its original bytes; missing-directory, Git
checkout and protected hardlink-alias destinations were rejected. The
first checker used an
`int64` input where the existing conservative full-band overflow bound
correctly refused a two-row reduction; changing only the tiny fixture to
`int32` let the unchanged reducer and export pass. This was a checker
fixture error, not a product failure. Pure artifacts are external:
`u14b_pure_export.png` and `u14b_pure_export.profiles.csv`.

The initial focused native command ran an external temporary journey script under
an independent `subprocess.run(..., timeout=55)` whole-command bracket;
it returned 0 in **1.428 s**, within the new 60 s aggregate allowance.
No older U02/U03/U02a qualification run occurred.
The real Windows Qt shell admitted tracked 3000x3000 hBN and Bi2Te3 OSCs.
Before export, hBN used custom 1.6 zoom, mean per valid pixel and 7-row/
5-column bands. Dialog cancellation wrote nothing. Changing acquisition
during destination choice rejected the request; changing to Bi2Te3 while
the worker wrote left the exported hBN snapshot unchanged. The result
report's final `viewport_scale=fit` describes this later Bi2Te3 selection,
not the asserted custom hBN capture.

The exported 549x262 physical-pixel PNG reopened with sampled detector
and both profile-plot pixels matching the captured Qt layers. The separate
437,316-byte CSV reopened with all 6,000 native bins exactly matching the
captured float64 means/support and the named hBN source hash
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`
at panel data revision 3. The PNG was 15,589 bytes, SHA-256
`e48c86ac195a8396a8f63a5e34a995de3bc61eb48839c74f8f0ed5c60a5be907`;
CSV SHA-256 was
`9b1b9c80f954abdd396910ee86285a357f417591c75a4da4922e8a8b14b6e0f8`.
Capture/encoding took 26.935 ms on the GUI thread; dispatch through the
completed write and receipt took 49.505 ms total. Process RSS was 379.0 MiB
before and 384.7 MiB after; these endpoints are not a sampled peak or a proof of
full reclamation. The shell closed normally. The external evidence is
`u14b_native_export.png`, `u14b_native_export.profiles.csv` and
`u14b_native_result.json`; the temporary journey code was removed.
These checks establish this inspection export path, not scientific fitting
or all future display sizes. U14b awaits independent supervisor review.

The follow-up review found that the initial journey pumped events manually,
so its timings do not establish input-to-visible completion latency under
`QApplication.exec()`. The retained image also shows the right-hand plot
content only near the top of the detector. The plot now requests vertical
expansion in the shared grid row and avoids drawing a tick label over
`row_px`; its native-coordinate transform and reducer are unchanged. Ruff
lint/format and import checks passed for this source change. Its settled
layout has not been confirmed in a native run.

One follow-up checker launched the production `main()` with the normal
Fusion/Segoe UI stylesheet and an event-driven export action. The external
`u14b_followup_launch.py` bounded the command to 25 s; it timed out after
**25.058 s**, producing no checker result, PNG or CSV and no stdout/stderr.
The external checker and command record are retained as
`u14b_followup_journey.py` and `u14b_followup_command.json`. The checker
has no phase log before `app.exec()` returns and its normal close path can
enter the unsaved-project dialog, so this timeout does not identify the
stage at which it stopped. No second follow-up launch was made. The
visible-ACK/heartbeat and extent checks remain unqualified; U14b is still
pending review, and the prior native file/value proof retains its narrower
scope.

The next single native recheck used a corrected external checker with
phase/error records, an explicit Discard choice only for its new project,
and a normal asynchronous close observation. A source-extracted pure check
of first-error and close decisions passed before launch. The production
`main()` reached the settled 1280x800 Fusion/Segoe UI 10 shell at DPR 1,
admitted tracked hBN, then stopped before pressing Export: the detector
viewport was 526x180 logical pixels, while the vertical profile plot was
50x50. The vertical `QSizePolicy.Expanding` request in `b7695d0` did not
make this grid item fill the detector row. The checker preserved that first
assertion, selected Discard for its own project, observed accepted close,
hidden window and idle owner/queues, and exited nonzero. The child command
took **1.307 s**; the full tool boundary took **1.470 s**. No new PNG/CSV,
painted completion ACK, or active export heartbeat sample was produced.
This is a concrete layout failure, not an export-latency measurement.

New external evidence is `u14b_recheck_journey.py`,
`u14b_recheck_launch.py`, `u14b_recheck_pure_check.py`,
`u14b_recheck_pure_result.json`, `u14b_recheck_phases.jsonl`,
`u14b_recheck_result.json` and `u14b_recheck_command.json`. The protected
SHA-256 manifest/checker verified all 46 older U02a/U14b evidence files
unchanged. No additional native launch or production edit followed this
failed recheck. The source layout remains unqualified and U14b unaccepted.

The next layout repair kept the detector and vertical profile at matching
180-pixel minima and made the center pane vertically scrollable. The first
native launch exposed a concrete width constraint: the old action
rows still made the center content 1196 pixels wide inside a 604-pixel
viewport. The detector and vertical plot were both 180 pixels high, but the
right plot was horizontally offscreen; no export was attempted. The two
action rows and the detector's widest control row were then reflowed,
preserving access to the filmstrip, review table and all actions without
forcing the shell larger or shrinking the detector. This
first failed command took **1.303 s** internally and **1.398 s** across the
tool boundary; `u14b_layout_result.json` and `u14b_layout_phases.jsonl`
retain the measured rectangles and first error.

The second and final native layout launch passed. In the production 1280x800
Fusion/Segoe UI 10 shell at DPR 1, the center content and viewport were both
604 pixels wide, with zero horizontal and 677 pixels vertical scroll range.
The 510x180 detector, 510x50 horizontal profile and 50x180 vertical profile
were entirely visible, shared their respective native axes and did not
overlap; the lower filmstrip, review table and action rows remained inside
scrollable content. The checker used tracked hBN with custom 1.6 zoom,
7-row/5-column bands and mean per valid pixel. A stub chose the new external
filename; the actual export button, production event loop and worker then
wrote the figure and exact profile CSV.

The 560x262 physical-pixel PNG reopened with nine sampled pixels from each
captured detector/profile layer matching and cyan curve pixels present in
every third of both profile extents. Visual inspection showed the native
column and row axis labels and ticks readable. The CSV reopened with all
6000 native bins, values, support and missing flags exactly matching the
captured canonical vectors, plus the current hBN decoded-source hash and
panel data revision 1. PNG SHA-256 was
`aefe56542175a7bb07ade9f267284064e439fcfd46b7f08defc392677795645e`;
CSV SHA-256 was
`6c51c9d60625b75bd83948d70c8e633375d0fd633954ab87e65968c3222776f2`.
Capture/encoding took **20.738 ms**; input to completed worker write took
**43.967 ms**. The completion message's status-bar paint region was observed
before the conservative after-paint ACK at **49.709 ms** from button input.
The 10 ms heartbeat had six active samples through normal drained close,
with a **23.117 ms** maximum gap. RSS endpoints were 305.28 MiB before
export and 304.03 MiB after close; these do not bound peak memory. The
successful child command took **1.463 s**, the full tool boundary **1.555 s**.

Both layout launches used **2.953 s** of the distinct 60 s allowance and
exhausted its two-launch count. The second checker and output are external
`u14b_layout2_*` files. SHA-256 manifests verified the 55 prior files before
the first launch and all 62 prior files before/after the second, with no
overwrites. This single event-loop action verifies the declared export and
layout scope; it supplies no latency distribution or scientific-fit
qualification. Handled write exceptions remove newly created pair files,
but the pair is not claimed crash-atomic. U14b still awaits independent
supervisor acceptance.

U14b was accepted by the supervisor on 2026-09-29 at
`6aabcc2f0b7100de57168d881faf114d21897190`. The accepted count is
10/46; the historical pending statements above record their earlier review
state.

#### U09a numeric parameter implementation (2026-09-29; review pending)

The project document now has schema 4 with one bounded numeric initial-value
draft. It keeps the exact admitted configuration bytes, configuration and
dependent CIF hashes, acquisition/source identity, field-level proposed values,
units, provenance and a monotonic draft revision. Schema 1-3 projects still
open. The shell loads an admitted configuration on the existing background job
owner, then shows supported source, detector and first-axis initial values in
explicit display units. Each edit re-enters the canonical configured reader;
the beam direction, detector rotation and native shape explain why they remain
read-only here. No fitter, reciprocal preview or result is launched.

One bounded session history records metadata and numeric edits as changed
fields, with 32 actions and a 256 KiB total cap. Undo/redo preserves unrelated
acquisitions, increments metadata/draft revisions and marks altered reference
identities unverified. A frozen launch snapshot retains the exact configured
object and proposal revision even when later draft edits occur. Freezing checks
current source/reference validation and on-disk configuration/CIF hashes;
snapshots are transient and cannot be mistaken for a saved fit.

Focused external checks passed for the nondefault Bi2Te3 configuration:
canonical edit/round trip, invalid-domain and read-only rejection, immutable
freeze, field-level metadata undo/redo with an unrelated edit, reference-status
invalidation, worker cancellation and history cap. The first native event-loop
journey completed edit, rejection, undo/redo, freeze, save and reopen, with a
16.5 ms painted loading ACK, but its check script ran all clicks and an OpenGL
capture in one callback, causing a 251.9 ms heartbeat gap. The second journey
sent actions on separate timer turns and passed: 17.2 ms painted loading ACK,
68.3 ms maximum heartbeat gap through normal close, and exact proposal
round-trip after save/reopen. Whole-command times were 3.424 and 3.106 s,
using both launches and 6.530 s of the distinct 60 s U09a allowance. All 72
older external evidence files remained SHA-256 identical. These checks verify
this UI/state boundary; they do not qualify a scientific fit or benchmark
every display size. A final inspector-only scroll adjustment and bounded worker
memory-accounting correction received static and focused pure checks after the
second launch; the two-launch cap prevented another native check. U09a awaits
independent supervisor acceptance.

#### U09a focused repair (2026-09-29; review pending)

The independent review found three state defects after `d09fe794`: a rejected
numeric Load could clear another job's routing, persisted proposals admitted
unsupported fields/units/domains, and reloading numeric values erased metadata
undo. Load now checks busy/open/close state before claiming the numeric job and
rolls back only its own failed submission. The one Qt-free numeric field catalog
validates saved proposals at document admission. The existing project-load
worker reconstructs the complete configured object when the matching source,
configuration and dependent CIF are verified; missing/changed references keep
the draft unavailable. A same-identity reload retains current proposals and
history. Replacing a baseline drops only obsolete draft changes from undo/redo
with corrected byte accounting, preserving metadata actions.

Focused external checks passed for the actual shell methods: a rejected Load
during a save left its receipt routable, own submission failure rolled back,
same-identity reload preserved metadata undo and edited values, and replacement
removed obsolete draft redo only. Project admission rejected negative wavelength,
wrong stored unit and unsupported field; schemas 1-3 and a valid schema-4
project reopened. The actual project reader and load worker rejected the saved
negative-wavelength file, and the shell's failed-open handler mapped that worker
failure to its visible error state in a focused non-GUI check.

The new native allowance used two distinct `slate_app.main()` journeys with
whole-command launcher times **3.421940 s** and **26.236326 s** (29.658266 s
aggregate of 60 s, two-launch cap exhausted). Tool-call wall times were
3.547 and 26.377 s, a different boundary. The first journey painted load,
edit, undo and save at 9.9, 39.0, 27.6 and 11.7 ms, but asserted the invalid
open state before its pending operation resolved; its whole-journey heartbeat
maximum was 164.2 ms. The second painted those controls at 10.3, 39.3, 28.4
and 12.7 ms, with a 37.9 ms maximum heartbeat from numeric Load through normal
hide/drain and a 156.2 ms cold-import whole-journey maximum. Its invalid-open
observer timed out because it ran after the shell cleared the failed job kind;
the script therefore did not prove the visible invalid-open transition in a
native run. This remains an explicit evidence limit, not a successful native
claim. Both journeys exercised U02a metadata undo retention, valid save/reopen,
snapshot independence and final styled 1280x800 controls; the inspector's
selection details were scrollable in the first run. All 82 protected external
files remained SHA-256 identical. No fit or forward calculation was run. U09a
still awaits independent supervisor acceptance.

#### U09a retained-draft completion check (2026-09-29; review pending)

The subsequent review found that a saved numeric draft opened with missing
references could become editable when the references were restored, although
only a new empty baseline had entered the canonical configured reader. A
field-valid positive wavelength can still violate the complete discrete-line
centroid constraint. Project opening now reports whether that exact saved draft
was fully validated. The editor admits only the validated project/draft pair;
reference invalidation clears that receipt. Background Load validates the
retained proposals and revision, and rejects a result if the captured draft
changed before delivery. Editing and freezing require the receipt. Revert can
remove unavailable proposals without silently replacing them. Idle dispatch
also refreshes the Load button after failed or canceled Open.

Focused external checks covered missing-reference admission, restored-reference
rejection of a positive but centroid-inconsistent proposal, valid retained
admission, the actual numeric worker, exact-draft stale receipt rejection, and
the failed-Open handler followed by its idle dispatch. A first new native
journey stopped at reference recovery because its external fixture rewrote YAML
line endings and changed the bound SHA-256; the product correctly kept Revert
unavailable for that mismatched identity. Its whole heartbeat maximum was
119.8 ms during shell startup, before the cold import request. The corrected
fixture copied the exact configuration bytes. The second production
`slate_app.main()` journey used Fusion, Segoe UI 10 and 1280x800. It waited for
the initial status paint to settle, recorded 206.7 ms startup separately,
started its heartbeat before cold import, and kept it running through normal
close and job drain. The longest gap was **43.3 ms**, at edit. Load, edit,
undo, save, recovery Load and invalid-Open error painted in **16.6, 33.1,
27.1, 11.8, 11.2 and 37.7 ms** respectively. It proved that invalid retained
values stayed unavailable after reference recovery, failed validation, could
be reverted and revalidated, and that failed Open preserved the current
project/draft while re-enabling Load. The error observer was first checked
with the active job kind already cleared, then bound to the exact Open
generation in the live journey. Whole-command times were 4.3355 and 4.3682 s
(8.7038 s aggregate) within the separate two-launch, 60 s allowance. All 97
previously protected external files remained SHA-256 identical. This is UI and
document-state evidence, not numerical fit qualification. U09a awaits
independent supervisor acceptance.

The independent review accepted U09a at
`4d609dff32a903a425b36f810599f7d264bf4826`. The accepted checklist count
is 11/46; the review-pending notes above preserve the earlier evidence state.

#### U07 reciprocal preview candidate (2026-09-29; native qualification failed)

The shell now requests nominal geometry coverage for one selected, source-verified
acquisition through the existing bounded job owner. The worker checks exact
configuration/dependent-CIF identities, native shape and any declared single
commanded angle, then uses the configured geometry-only context and canonical
detector-coordinate evaluator. It returns a 13-by-13 sample-frame reciprocal
mesh for the bound reference baseline and a separately validated proposed
draft when present. Invalid cells and connecting mesh edges are omitted. The
view projects `(|Q_parallel|, Qz)` in sample-frame inverse angstroms; its labels
state that it is a one-ray geometry approximation with no intensity or fit.
The pointer and selected native coordinates report both internal-film and
external-air sample-frame Q, or an explicit unavailable status. No feature
pack exists yet, so none is drawn or linked. Two retained previews are keyed by
project, acquisition, source, references, shape, declared angle and exact
numeric draft; selection epochs and request hashes reject late A-B-A results.
Configuration/CIF reads and the maximum 338 detector grid evaluations happen
on the worker; cursor events reuse the retained mapping and do no file read.
The measured two-state result payload was 65,314 bytes under a 256 KiB cap.

Focused external checks passed for baseline/draft difference, nondefault
1.61 Å wavelength, non-axial incident direction, seven-degree rigid pose,
canonical internal/external Q at a continuous detector coordinate, valid and
backward native corners, off-panel and nonfinite coordinates, a valid 1980 by
3000 non-square detector, malformed identity/shape, separate acquisition IDs,
exact stale A-B-A rejection, and a two-acquisition project whose source and
references verified on reopen. A failed first pure fixture had non-tangent
transverse source axes; a second had a detector dimension incompatible with
the configured macrobin size. Both were corrected without changing numerical
contracts. All 116 earlier external files remained SHA-256 identical.

The **first and only** production `slate_app.main()` native window in the
authorized three-launch series failed after a conservative 56.676 s whole
command. Its initial status paint settled in 121.54 ms, the 10 ms heartbeat
then ran continuously through normal close/drain, and the maximum observed
gap was 80.696 ms. The start ACK painted in 41.072 ms. The matching A coverage
paint arrived 127.255 ms after request; that interval includes background
geometry preparation and is not a start-ACK failure. The checker timed out in
`cursor_demand` without recording a matched final cursor-label paint, so no
cursor latency, sustained frame interval, edit, A-B-A native or cancel/close
qualification can be claimed. The detector cursor label is below the view in
a scrollable center pane; the checker did not record whether it was visible,
whether mouse coordinates matched or whether paint was suppressed. A focused
post-failure check measured the retained canonical cursor mapping at median
0.160 ms and p95 0.253 ms over 100 calls, which does not establish native
paint latency. The smallest product change publishes the same Q and unavailable
status to the always-visible status bar; a pure check passed, but its paint is
unverified. A future native observer must watch that visible status region,
preflight one final coordinate, and retain dispatch/paint counts on timeout.
The native series stopped at its first failure; no remaining launch allowance
was used. U07 remains unchecked pending native qualification and independent
review. No fit or forward intensity image was run.

#### U07 correction and stopped native packet (2026-09-29)

Independent review found that saved-only coverage raised from a strict zip of
one map and two colors during Qt painting. The painter now pairs only available
maps and colors. It also found that a stationary pointer could leave old Q in
the detector readout when a numeric draft or reference invalidated the map.
Preview identity changes now clear that Q while retaining native detector
coordinates and counts; they replace the status-bar pointer message only when
it still owns that message. The status bar uses a compact film/sample and
air/sample Q line that fits the declared 1280-pixel window in a focused font
check. The inspector retains the full frame and unit labels. External actual-
method checks rendered saved-only, saved-plus-draft, unavailable and invalid
coverage, and checked the invalidation method with a stationary pointer,
current revision republishing and preservation of unrelated status text. Draft
and reference changes use that method through the shared preview refresh. An
observer preflight checked exact coordinate/revision/generation matching and failure
counters. These checks do not qualify native paint latency.

The first of a **new** three-launch correction packet ran the production
`slate_app.main()` window and failed in its early exact-cursor preflight. One
native point was dispatched, but the observer saw zero cursor handler events
and zero completed matching status paints. It stopped after a 4.939 s whole
child command; the start ACK painted in 39.746 ms, A coverage in 126.519 ms
including worker preparation, maximum 10 ms heartbeat gap was 71.007 ms, and
the window closed and drained normally. The checker recorded a visible region
for the detector widget but did not establish that its input point intersected
the enclosing center scroll viewport or was hit-testable at the window surface.
A focused offscreen `DetectorTextureView` check did deliver the same QTest mouse
move as a cursor event. These observations do not distinguish a clipped native
input point from another native routing issue. The correction series stopped
on the first failure; launches 2 and 3 were not used. No sustained cursor,
prepared-switch, A-B-A, cancel or close performance gate was reached. U07
remains unchecked. A new observer needs center-viewport intersection and
hit-test evidence before dispatch, then exact handler and post-paint counters.

#### U07 cached-preview correction and stopped finish packet (2026-09-29)

A direct cached A-to-B preview change could still leave A's Q in the inspector
pointer label. The shared invalidation helper now clears that label on every
preview identity change. A focused external actual-method check covered cached
A-to-B and preview-to-None transitions with a stationary pointer, retained
native coordinates/counts and an unrelated save status. The accepted saved-only
painting, canonical mapping and identity checks were not reopened.

The revised external observer supplies a `QMouseEvent` to the production
detector widget through `QApplication.sendEvent`, matching the accepted U02a
input route. Its pure checks covered exact unavailable Q text, post-delivery
paint generation and old-A paint callbacks after B and newer A. In the first
native finish launch, the center viewport and window hit target contained the
input point; the widget received one MouseMove, emitted native `(1200,1200)`,
and the current acquisition/preview/epoch Q text matched an independent
retained mapping at a completed status-bar paint. The compact status text
measured 449 px within the status bar's 1280 px visible region. The full
inspector pointer text was populated, but the observer's immediate
`ensureWidgetVisible` check found its label rectangle not wholly inside the
inspector viewport. It did not retain the two rectangles or test a later layout
turn, so horizontal clipping, vertical clipping and layout timing remain
unresolved. The possible full-readout visibility defect is not qualified away.

That first launch failed at this readability gate and stopped the entire new
series. Its child command took 3.069 s, with start paint 41.905 ms, cold A
coverage paint 127.647 ms including worker preparation, maximum heartbeat gap
81.456 ms and normal close/drain. Launches 2 and 3 were not used. No 30-second
mixed cursor window, prepared switching, A-B-A, cancel or close performance
gate was reached. U07 remains unchecked; the accepted checklist count is
11/46. A later observer should capture label and viewport rectangles after a
separate layout turn, then repair any demonstrated clipping before the
sustained performance packet.

#### U07 measured inspector overflow and stopped visibility packet (2026-09-29)

One new, separately bounded production-main launch retained the inspector
geometry after `ensureWidgetVisible` and a later event turn. The native pointer
again arrived at `(1200,1200)` and produced the exact current Q at a token-
matched completed status-bar paint. The full inspector label was **268 px wide
in a 245 px viewport**, mapped from x = -11 through x = 257. The horizontal
scrollbar was at 30 of 61; the label's 68 px text height fit its 68 px content
rect, and its y = 495..563 exactly fit the viewport height. The later geometry
was unchanged. This establishes horizontal clipping from an oversized inspector
content minimum, not a delayed scroll or vertical wrapping failure. The
label's post-exposure paint was not credited, so full-readout visibility failed.

The existing inspector keeps its resizable scroll. Its numeric selector now
uses a bounded contents-width hint; the five numeric actions occupy one column,
and the final action is labeled `Freeze` with its full meaning in the tooltip.
An external Fusion/Segoe UI 10 focused width check measured the widest revised
control plus inspector margins at 242 px, below the 245 px native viewport.
This is a layout repair, not native validation of its result. No second native
launch was authorized in this packet. U07 remains unchecked at 11/46 accepted.

The external observer also now accounts for the entire 30-second demanded-frame
window, including its head, tail and final detector-frame drain; a two-frame
window with a 29.991 s empty tail is rejected. Changed selected-image
publications expect their own texture upload. The initial prepared-switch
observer checked detector paint and bound horizontal/vertical profile state;
that was insufficient because profile state changes before profile paint.
Those sustained and switching paths were not reached in this failed launch.
The child command took 3.908 s (4.406 s tool
transcript), start paint 48.368 ms, cold A coverage paint 147.992 ms including
worker preparation, maximum heartbeat gap 67.366 ms, and normal close/drain.
No scientific fit or forward intensity image was run.

The observer's cursor move also triggered the application's ordinary project
autosave, changing the protected two-acquisition fixture during this launch.
Its autosaved bytes were retained as a distinct external visibility artifact;
the protected fixture was reconstructed and restored byte for byte to its
recorded SHA-256 `d99a9b13348315f3a6dfbba8ced61ef896fa3e2ef1e50df0a9e1280cab7ba271`.
Any future native packet must open an exact-byte disposable copy of that fixture
so autosave cannot change the protected original.

#### U07 full-width and three-surface visibility packet (2026-09-29; native qualification failed)

A complete focused inspector-control width check under the production
Fusion/Segoe UI 10 font and button style found further horizontal minima:
`Load numeric draft` 264 px, `Map selected geometry` 303 px and an unwrapped
explanatory label 598 px. The numeric and reciprocal headings also exceeded
the 209 px content budget. The controls retain their actions and full tooltip
meaning with the captions `Load draft` and `Map geometry`; the three labels
now wrap. The widest measured minimum of all inspector controls is the numeric
selector at 206 px, or 242 px including 36 px margins.

One separately authorized native production-main launch opened an exact-byte
disposable project copy. The actual inspector Q label was 207 × 85 px, fully
visible at x = 19..226 and y = 478..563 in the 245 × 563 px viewport after
an independent layout turn and a completed later paint. The horizontal scroll
range was zero. The native pointer, production handler, current canonical Q,
and status-bar paint matched the same token. The 30.020 s mixed cursor/profile
window had 2,253 matched detector frames, complete 19.465 ms head and 1.174 ms
tail, frame p95 15.320 ms and p99 16.688 ms, cursor p95 16.112 ms, no image
upload, and maximum heartbeat gap 51.756 ms.

The observer was corrected to require completed queued paint receipts from the
selected detector and both visible profile plots for the same publication,
generation, value buffers and unique switch token. A focused actual-method
check rejects detector-only, one-profile, stale-token and changed-publication
receipts. In the native launch, ten prepared A/B selections each had matching
detector/H/V receipts and one image upload; their maximum post-paint latency
was 24.914 ms. The rapid final A also had matching receipts, one upload and a
canceled stale generation. These are one measured repeat window, not U07
acceptance.

The launch **failed** at `cancel_wait` on its 52 s internal deadline. No cancel
button paint acknowledgment was recorded; the final reciprocal job reached
`completed` without a `cancel-requested` event. The retained observer did not
record button enabled state at the attempted click, so the exact reason it did
not cancel remains unresolved. Failure handling closed and drained the window,
but ordinary close/hide timing was not reached or qualified. Whole child command
was 53.275 s under the 60 s cap. No retry or substitute launch was made.
The original 7,177-byte fixture retained SHA-256
`d99a9b13348315f3a6dfbba8ced61ef896fa3e2ef1e50df0a9e1280cab7ba271`;
only the disposable copy changed through ordinary autosave. U07 stays unchecked
at 11/46 accepted. No fit, forward intensity image or scientific comparison
was run.

#### U01 shell and identity checkpoint (2026-09-28)

Launch from the repository root with
`uv run --extra visualization python interactive/slate_app.py`. The native PySide6 shell opens an
unsaved local project, a project/acquisition browser, detector placeholder, inspector and the two
named workspaces. File import, project opening/saving and simulation are visibly unavailable; no
placeholder data, worker or calculation starts. The experiment status view has explicit empty,
loading and error presentations for the upcoming import lifecycle. The shell owns only transient
selection/display state; its immutable, Qt-free project records carry schema version 1, project and
acquisition UUIDs, source path and SHA-256 identity. Adding records does not deduplicate matching
paths or bytes. Renaming/reordering preserves acquisition IDs and current selection by ID rather
than row or filename. The schema owner is `interactive/project_state.py`; atomic save/reopen and
recovery remain U01b.

A temporary external PySide6 check launched and closed the real window, switched both workspaces,
rendered empty/loading/error states and verified two independent IDs for identical path/hash,
rename/reorder and selection retention. The actual styled 1280x800 shell was visually inspected.
An identity follow-up corrected the selection handler to read selected rows: Qt can retain a
current item after `clearSelection()`. A focused real-Qt check selected an acquisition, cleared
selection, selected the project root, then renamed/reordered and restored an acquisition by UUID;
the ID and inspector matched the selected row at every step.
An entry-point check executed `interactive/slate_app.py` via `runpy` and closed the window normally.
The documented `uv run` form requires the preceding dependency sync; a `--no-sync` attempt in this
checkout found no prepared environment and was not used as launch evidence. Normal numerical imports
remain GUI-independent; importing the shell does not import `rasim_next` or initialize a device.
No OSC file, physics, fit, asynchronous worker, persistence or latency campaign was exercised.
Temporary screenshots and checking code were removed after review. U01c owns the next job-state
and late-result boundary; U01a then connects real OSC import to this shell.

#### U01c bounded job lifecycle checkpoint (2026-09-28)

`interactive/job_lifecycle.py` owns the shell's single application-wide worker thread, one newest
pending request, one replaceable progress message, one ready result slot and eight retained
terminal summaries. The GUI polls the slot every 20 ms; the worker touches no widget and has no
unbounded command, result or progress queue. Request arguments are currently immutable bytes,
text or paths; admitted byte views are copied to owned bytes after their `nbytes` is checked,
so a caller can release the view and a small slice cannot retain a large backing allocation.
Workers use plain functions without captured state. A request declares at most 4 MiB of argument
bytes and 96 MiB of expected result;
the worker result declares and is checked against both its estimate and the 96 MiB global cap.
Flat bytes, byte views (`nbytes`), text/path encodings and objects exposing integer `nbytes` have
a checked lower bound. Nested result ownership and temporary decode peaks remain the specific
producer's responsibility, not a claimed generic deep-size calculation. U01a must preflight OSC
source/decoded size and count every retained buffer against the existing CPU ledger.

Each request freezes project/acquisition IDs, data/mask/calibration/model revisions and an
application-monotonic generation. New requests cancel the active work and replace the one pending
slot. Selection changes invalidate the generation even before another request starts. Cancellation
immediately suppresses progress/result publication, including ignored cancellation, late success,
late failure, equal-input A-B-A and synchronous status-signal reentrancy. The states are queued,
running, cancel-requested, completed, canceled and failed. Summaries contain only short messages,
identity and safe-stop time; no traceback or result array is retained in history. Shell Cancel and
close request cancellation without a GUI-thread join. Close remains visibly pending until the
worker exits and resources have been released; no successful close or resumable work is claimed
before then. A selected-acquisition change leaves a visible stopping state that clears after the
obsolete worker safely exits. Project persistence remains U01b, real OSC loading U01a and
numerical cancellation hooks stay with their later owners.

A focused follow-up corrected three lifecycle edges without repeating the timing campaign. Cancel
now detaches its pending request and marks the captured old active worker canceled before emitting
terminal signals; a listener that submits newer work from the pending-canceled signal cannot have
that new request erased by the outer cancel. Byte views are snapshotted once after byte-limit
validation, including a non-byte typed view and a one-byte slice of a larger backing buffer.
Selection invalidation immediately disables Cancel and updates the status bar, then clears the
waiting message when obsolete work actually stops. A temporary real-Qt check covered those three
cases via `python <external>/u01c_repair_check.py`; its script was removed after the result.

A finite external check used a 64 KiB file read/hash operation, controlled worker barriers and a
250 ms noninterruptible preparation section. It covered normal success/failure, pre-start and
during-work cancellation, replacement/coalescing, late success/failure, A-B-A and selection
invalidation, request/result byte rejection (including non-byte memory views and mutable input),
COMPLETED-listener cancellation, reentrant newest-request retention and 13 repeated outcomes with
history capped at eight. In the shown native shell, Cancel dispatch returned in 0.163 ms and its
changed status label reached a Qt paint event in 1.629 ms; safe stop/release took 262.205 ms.
Close dispatch returned in 0.065 ms, its waiting label painted in 0.426 ms and safe stop/release
took 260.790 ms. The longest 10 ms GUI heartbeat gap during the close wait was 11.091 ms; no
result was published after close. These are GUI dispatch/Qt paint boundaries, not monitor scanout.
The command was `python <external>/u01c_lifecycle_check.py`; Ruff format/lint, numerical/shell
import separation and diff checks were run separately. The checking script and input file were
removed. No full image, fit, scientific validation, dependency or reusable harness was added.

#### U01a first OSC import checkpoint (2026-09-28)

The shell's **Import OSC** picker and one-local-file drop submit a bounded path/texture-limit
request to the U01c owner. The worker calls the canonical `io/osc.py` reader with 64 MiB source,
32 MiB decoded-stream, 12-million-pixel and per-axis admission limits. The axis limit is the
smaller of 16,384 pixels and the actual detector OpenGL context's `GL_MAX_TEXTURE_SIZE`, queried
once before the shell is shown and checked again on panel adoption. It reads gzip in 1 MiB chunks
after checking header dimensions,
checks the source file identity/size/mtime before and after, and hashes the exact decoded header
and payload. The sole clockwise `raw_to_detector_native` conversion stays in the reader. The
existing unlimited reader call remains available to non-interactive callers. High-range int32
decoding now uses in-place masked NumPy operations with the same values.

The worker prepares a read-only native int32 plane, a separate read-only float32 texture plane,
display levels and exact center bands. The panel adopts those buffers on the GUI thread without
another full image copy, scan or profile reduction. At the 12-million-pixel cap, decoded content
is at most 24,006,000 bytes. For any admitted shape, native plus display planes and four int64
profile arrays are bounded by `8*pixels + 16*(rows + columns)` bytes. The 16,384-axis cap bounds
this to 96,524,288 bytes plus the 6,000-byte header allowance, below U01c's 96 MiB result cap.
The largest simultaneous decode/prepare arrays are the decoded content, raw and native int32
planes, Boolean high-range mask and float32 display plane, at most about 180,000,000 bytes
(172 MiB) plus bounded band-reduction temporaries, a 1 MiB read chunk and Python/Qt overhead.
One image is resident in the panel; selecting an older acquisition reloads it through the same owner
and verifies its decoded SHA-256. Importing another file creates a new immutable acquisition UUID.
Angles and calibration remain explicitly unknown. Failed/canceled replacement retains the prior
usable image; changed source bytes reject reload and require a new import. Generation and UUID
checks reject replaced, canceled, selection-obsolete and closing results.

Focused checks used `python -B -` with external scratch inputs, removed on completion. Both
tracked 7x11 big/little endian fixtures retained exact raw/native counts and the documented
corner/interior mapping; the independent unsigned-16-bit high-range expression matched both
fixtures and the tracked 3000x3000 hBN gzip image. The SHA-256 matched the decoded stream.
Short header, wrong/extra plain or gzip payload, bad signature, source/decoded/pixel overlimits,
cancellation and changed-during-load were rejected. In real Qt, the picker and one-file drop imported;
multiple drop explained its limit; a failed new path, immediate cancel and A-B-A replacement
left the prior image intact or published only the newest result. Older selection reloaded under
the original UUID without duplication; an injected in-memory hash mismatch refused to replace
the resident plane. Close during hBN import drained with no late publication.
Native corners/interior values and widget pixel-center round trips aligned on the non-square
fixture. One cold hBN shell probe reached worker result in 0.248 s and first panel `paintGL` in
0.267 s; native+display retained planes were 68.66 MiB and RSS at result was 279.84 MiB. Its
10 ms heartbeat's longest gap was 139.97 ms. That first probe started import immediately after
`show()` and quit inside the first `paintGL` signal; it did not separate shell startup from import
or wait for composition. It was an ambiguous responsiveness miss, not an accepted cold-load gate.

The narrow U01a review repair rejected a crafted 1x12,000,000-pixel header at the axis check
before payload allocation (6 KiB file; 17,914 traced Python bytes peak). A controlled delayed
import followed by clearing selection hid the old A plane immediately while retaining A's exact
profiles; B never published, and reselecting A restored aligned display/profile identity. A
forced 1,024-pixel active-context cap reached the worker header check and rejected a 3000x3000
image before decode. The measured context reported a 32,768-pixel GL texture limit, so the
application's 16,384-pixel cap applied on this machine.

A phase-attributed probe without startup context preparation showed a 107.39 ms longest startup
heartbeat gap, then a 38.10 ms longest gap when import began after the shell had settled. It
measured worker preparation at 145.86 ms, `initializeGL` at 8.26 ms, texture upload at 6.10 ms
and panel `paintGL` at 9.76 ms. The final path prepares the actual detector context before the
shell is shown, which also supplies the active GL dimension limit. One final cold process probe
measured construction to 384.22 ms (including 354.68 ms context preparation), first exposure
at 452.09 ms, empty-status paint at 456.43 ms and usable shell at 477.92 ms. Import was requested
250 ms after that ready point. The hBN worker ran for 138.35 ms; the result arrived at 144.91 ms
from request, panel paint at 153.77 ms, exact horizontal/vertical profile paints at 157.83/158.74
ms and matching panel swap at 162.55 ms. The longest 10 ms GUI heartbeat gap from request through
swap was 11.76 ms; no synthetic repaint was added. Startup construction precedes the heartbeat
timer and is reported as startup latency, not a measured responsive interval. These few probes
are not a latency distribution or monitor-scanout measurement. Software/I/O checks do not establish
fitting or scientific adequacy. No permanent checker, test fixture, simulation, fit or full-image
timing campaign was added.

One final cold-process hBN check closed the memory evidence gap without repeating interaction
profiling. Windows `psutil.Process.memory_info().peak_wset` reported **316.03 MiB** from process
creation through the matching panel swap and both exact profile paints. This OS peak working-set
counter includes Python, Qt/OpenGL startup, the worker, and display admission; it is not a NumPy
allocation peak or GPU-memory measurement. RSS was 171.95 MiB after shell creation/show and
294.88 MiB 50 ms after composition, when the worker's temporary decode buffers had been released.
RSS was 291.11 MiB after closing with the Python window object still referenced, so that last
value does not claim complete object reclamation. Windows private commit at composition was
1,072.36 MiB; its Qt/driver attribution was not decomposed. The one displayed 3000x3000 hBN
native/display pair retains 72,000,000 bytes plus 96,000 profile bytes. While replacing a prior
same-sized resident image, the new producer's largest decoded/raw/mask/native/display arrays add
about 135,000,000 bytes, plus the 1 MiB read chunk and bounded band temporaries. The pair and
producer therefore account for about 208,000,000 bytes (199 MiB) before Python/Qt overhead. At
admitted maxima, an old resident result is at most 96,524,288 bytes and new producer arrays at
most about 180,000,000 bytes (172 MiB), before the small chunk/bands and runtime overhead. One
hBN R32F texture is 36,000,000 bytes; allowing one equally sized upload/staging copy gives a
72,000,000-byte display estimate,
not a measured driver allocation. The original at-result RSS of 279.84 MiB remains an at-result
sample, not the peak counter. The three phase/peak probes used 1.324 + 1.088 + 0.886 = 3.298 s
of measured active process time; all narrow review diagnostics together used under 30 s of
reported tool wall time, including startup and tool overhead, within the 180 s allowance.

#### U01b project persistence checkpoint (2026-09-28; accepted)

The optional desktop shell now owns one versioned `.slate.json` document containing project and
acquisition UUIDs, ordered source references and decoded-stream SHA-256 identities, the selected
workspace and the detector view. JSON admission is strict and bounded to 1 MiB and 128 acquisitions.
The document contains neither source pixels nor solver state. A named project autosaves to its file;
an unnamed project autosaves to one UUID-named draft in the external local recovery location. That
location admits at most 32 drafts and 32 MiB. Explicit Save/Save As, Open, Recover Draft, rename,
reorder and hash-matched Relink actions are visible in the shell. The existing atomic JSON publisher
provides a flushed temporary file, replacement and cleanup. Writes are serialized as immutable
snapshots through the existing job owner, and save status names only the completed revision. The
750 ms autosave debounce coalesces queued autosaves. Up to eight pending writes/3 MiB are admitted
behind one active request; a full queue is visible and explicit writes are never silently replaced.

Reopen validates source references independently. Missing, changed and unreadable OSC files retain
their acquisition identities and appear with actionable status; a relink requires identical decoded
bytes. The selected source reloads on demand. Saved view state remains available while a source is
missing or loading; empty projects do not inherit a prior resident image's view. Project switching
rejects a delayed open if current edits changed meanwhile. Selection and import transitions supersede
deferred work without canceling an active durable write.

Focused task-owned external checks covered schema/version/path/duplicate-key/size rejection; actual
Qt import, Save As, close and reopen with stable IDs, order, names and crosshair; unnamed draft
recovery with Save/Discard/Cancel; moved/missing source and matching/mismatching relink; serialized
delayed writes and autosave of a newer edit; injected atomic replacement failure preserving old
bytes and removing its temporary file; delayed-open edit retention; missing-source view retention;
empty-project capture after a previous image; A-B-A selection during a save; A-active/B-pending/C
replacement; and a corrupt deflate source alongside a valid source. Scratch files were external
and removed. These checks establish local project I/O and shell transitions only, not numerical
fitting adequacy. Portable archives, later editable scientific state and solver resume are outside
U01b; no solver resume is implied by a saved draft.

The task-owned checks ran as `python -B -` inline commands from this checkout, with
`PYTHONPATH=interactive;src`, native `QT_QPA_PLATFORM=windows` for Qt checks, and a temporary
directory under the external Codex visualization scratch root. Inputs were the tracked 7x11
big/little-endian OSC fixtures, a deliberately malformed deflate gzip source, and small generated
project JSON documents. The final delayed-open/missing-view check used the shell's top-level
`project_state` imports and waited for the written path, matching receipt, drained owner and empty
queue. The actual unnamed-draft Discard path deleted its UUID-named file. A Save As path through an
existing `child/../UUID.slate.json` alias was rejected before publication; the draft and another
saved file remained byte-identical, with no temporary file left.

One native-window persistence timing run delayed each atomic publisher by 250 ms and made a second
edit while the first Save As was active. Save dispatch returned in 0.192 ms and its status reached a
Qt paint event in 1.058 ms. Close dispatch returned in 0.100 ms and its waiting status reached a
Qt paint event in 0.760 ms. Final close and worker release took 469.696 ms after the close request;
the on-disk project contained the final accepted name. The longest gap in a 10 ms GUI heartbeat
during the write was 13.819 ms. Both status-paint and heartbeat observations met the 100 ms bound.
Qt paint events are not monitor scanout, and this single small-project run is not a latency
distribution or a full-image performance claim. No numerical fit or physical observable was run.

Initial check attempts did not all pass: offscreen Qt had no detector OpenGL context, and sandboxed
temporary-file permissions prevented a scratch write. Native Qt with an explicit external scratch
root resolved those setup limits. One combined assertion failed because the check imported
`interactive.project_state` while the shell loaded `project_state`, giving duplicate class identity;
a later wait also treated a pre-existing saved revision as a completed Save As receipt. The corrected
top-level-import/path-and-drain check passed. An A-B-A check initially waited in an unhandled
Save/Discard dialog; it was stopped and rerun with the intended Discard choice. An alias check first
caught the expected exception through the duplicate module class and exited nonzero; the consistent
import rerun passed. All task-owned scratch directories from these attempts were removed. Ruff
lint/format, module parse/import and Git diff checks passed after the production edits.

#### U02 detector viewport checkpoint (2026-09-28; frame cadence gate open)

The existing Qt/OpenGL panel now has pointer-anchored wheel zoom, drag pan, box zoom, Fit and
physical-device-pixel 1:1 controls. Fit, 1:1 and box have keyboard shortcuts; arrows pan the
focused image. The one native-to-viewport transform drives image pixels, marker/crosshair overlays,
pointer picking and both marginal position axes. Off-image pointer events clear the readout instead
of clamping to an edge. The readout uses the original native count, names `column_px` and `row_px`,
and leaves Q/angles and physical saturation unknown. Image, crosshair and external marker layers
are independently visible in that order; fitted-result controls remain unavailable until a fit
exists. New image admission clears old markers/cursor and resets the camera. No extra backend or
dependency was added.

Linear, zero-centered signed and positive-log display use validated numeric limits without
changing native values or exact profile arrays. The float32 display shader marks nonfinite values
and log-nonpositive values with a checkerboard. Scientific-notation entry retains finite float64
limits that the float32 shader can distinguish; unsupported bounds reject before mutation. The
worker computes extrema and minimum positive count once on OSC admission; interaction handlers
do not reduce the whole image. Project schema version 1 now saves contrast mode, layer visibility,
scale mode and the measured DPR. The explicit legacy version 1 field set still opens. Its absent
DPR remains unknown, so old logical pan is not scaled as though measured at DPR 1; current drafts
scale recorded pan between DPRs. Native 1:1 derives its scale from current DPR even if logical
viewport size does not change.

Focused `python -B -` native checks used both tracked 7x11 endian OSCs and 2x3 float64 planes.
Actual framebuffers matched the original corner/interior values through drag pan, wheel and box
zoom, resize and separate DPR 1/1.5 Qt launches; a DPR 2 launch reopened legacy pan `(60,20)`
unchanged. A patched `devicePixelRatioF` with a real Qt DPR-change event at unchanged logical size
kept one physical pixel per detector pixel and converted physical pan without uploading. This is
a focused DPR-only transition simulation, not a physical monitor move. Framebuffer marker and
crosshair centers, exact cursor counts and off-image behavior matched the shared transform.
Tiny signed/nonfinite planes retained native values and exact profiles through contrast changes;
invalid controls and an unrepresentable display plane left prior state intact. Actual framebuffers
mapped low/mid/high red values to `[0,127,255]` for `[0,1e-20]` and `[-3e38,3e38]`. Adjacent
float32 bounds `1.300000184382815e20` and `1.3000002723437452e20` mapped to opposite display
endpoints. Underflowed positive-log and signed limits rejected atomically. A current DPR 1.5
Save As/reopen retained source/project IDs, native 1:1, signed contrast, visibility and pan.

The changed-path workload was one named existing 3000x3000 image,
`examples/calibration/hbn/hBN_calibrant_5m.osc.gz`, with 4,000 external grid fiducials and two
exact marginals. The temporary external command was
`python -u -B <task-scratch>/u02_viewport_probe.py 30 3` with `QT_QPA_PLATFORM=windows` and
`PYTHONPATH=interactive;src`; the three windows continuously mixed synthetic Qt pointer, wheel,
drag-pan and contrast-control events at an actual approximately 13 ms request timer cadence.
Input time was paired to the newest completed `paintGL` generation before `frameSwapped`, then to
that composition signal. Intermediate requests without paint and paints superseded before swap
were counted separately. Profile paint signals were counted, but no new composed-profile latency
was inferred from their count. The boundary is Qt composition, not monitor scanout.

| 30 s window | Callbacks / requests / composed; coalesced | Callback interval p50/p95/p99/max ms | Composed interval p50/p95/p99/max ms | Input-to-swap p50/p95/p99/max ms | Longest 10 ms heartbeat gap |
| --- | --- | --- | --- | --- | --- |
| 1 | 1959 / 1958 / 1957; 1 | 14.836 / 22.418 / 24.085 / 45.396 | 13.914 / 28.528 / 29.167 / 56.183 | 12.254 / 16.834 / 17.622 / 43.505 | 45.951 ms |
| 2 | 1935 / 1935 / 1919; 16 | 13.486 / 26.729 / 27.105 / 31.295 | 13.351 / 26.710 / 28.455 / 45.987 | 11.779 / 13.721 / 16.981 / 29.441 | 31.294 ms |
| 3 | 1928 / 1928 / 1897; 31 | 13.463 / 26.810 / 27.175 / 35.889 | 13.349 / 26.701 / 26.814 / 42.389 | 11.667 / 12.076 / 12.315 / 30.646 | 35.890 ms |

All completed paints reached composition; coalesced/unpresented counts were 1/16/31 and no
completed paint was superseded. Each final requested generation composed, then the 750 ms debounced
autosave wrote its matching contrast, native cursor and unchanged source hash. One image texture
upload remained one through every window; instrumented cursor/camera/contrast reductions touched
no full native plane. Horizontal and vertical profile paint counts were 2188/2136/2099 per window.
Process RSS was approximately 241-246 MiB around the windows and Windows peak working set was
266.22 MiB, including shell/import setup; GPU allocation was not measured.

One targeted repaint repair skipped marginal-axis updates when the native viewport rectangle was
unchanged and avoided rewriting unchanged control text. A single comparable 30 s follow-up had
1959 requests, 1937 composed, 22 coalesced/unpresented, 1852 paints per marginal, one upload,
no full-plane reductions, 263.16 MiB peak working set and final autosave. Its composed-frame
interval p50/p95/p99/max was 13.350/26.678/26.730/35.545 ms; input-to-swap was
11.678/12.142/13.693/23.905 ms. Cursor/wheel/pan/contrast dispatch p95 was
0.547/0.330/0.414/0.299 ms, while callback interval p95 was 26.530 ms and the longest
heartbeat gap 27.836 ms. The reduced profile paints did not restore 60 Hz composition. All four
windows met the 50 ms p95 input-to-composition, 33.3 ms p99 interval and 100 ms GUI gap limits,
but **all missed the 16.7 ms p95 composed-frame interval target**. The extra interval is beyond
the measured direct input dispatch; its split among Qt scheduling, OpenGL composition and the
remaining profile/marker work is unqualified. No further performance repair or replay was run.

The completed active timing windows totaled 122.027 s. One earlier setup `app.exec()` attempt
stalled and was stopped after roughly 35 s wall time; conservatively counting all of it keeps the
packet under the 180 s cap. An initial scan counter incorrectly treated zero-copy `np.asarray` as
a full-image scan; the corrected counter checked actual reductions. One small repaint assertion
encountered a queued profile paint from admission and was not used as evidence. The task-owned
scratch project and checking script were removed after extracting these results. U02 stays open
for the measured frame-cadence miss; these checks do not qualify a scientific fit.

#### U02 native-event-loop attribution checkpoint (2026-09-28; diagnostic only)

The prior U02 timing packet was closed. A separate 30 s aggregate active-window allowance was
used only to attribute its frame miss. The committed source was `a5d645f41089f70be7f5be7ddbe1081dca373454`;
there was no production rendering edit. An external `python -u -B <task-scratch>/u02_native_loop_probe.py
<worktree>` launched the existing `ShellWindow`, imported
`examples/calibration/hbn/hBN_calibrant_5m.osc.gz` through the native import worker, and ran
the window with `QApplication.exec()`. The decoded OSC SHA-256 was
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`.
An external recovery root was task-owned. After the 3000x3000 image and 4,000 grid markers were
ready, Qt timers supplied pointer, wheel, drag-pan and contrast-control events with a 13 ms
requested interval. The visible screen was Sceptre C27 (2), 1920x1080 logical pixels, DPR 1,
74.99 Hz. During the measured window, sampled whole-machine CPU was median 15.6%, p95 34.5%,
maximum 53.6%, with about 31.3 GiB available RAM at finalization. These samples do not isolate
the presentation process's CPU share.

The first 15 s native-loop window completed, but the temporary reporter raised while formatting
a NumPy interval array; its counters were not emitted or retained. The process was stopped, that
reporter-only error was corrected, and the remaining 15 s yielded one usable diagnostic window.
The temporary reporter also failed to exit after printing its second result on `app.quit()`;
the owned process was stopped and its empty external recovery folders and script were removed.
This exit behavior was not attributed to the production app. No further window was run.

| Native `app.exec()` window, 15 s | p50 / p95 / p99 / max (ms) |
| --- | --- |
| Actual interval between 13 ms requested input callbacks | 13.279 / 21.804 / 25.157 / 30.989 |
| Input callback dispatch duration | 0.205 / 0.443 / 0.599 / 1.138 |
| Matched composed-frame interval | 21.249 / 28.412 / 40.283 / 51.113 |
| Input to matched `frameSwapped` | 11.704 / 15.923 / 17.387 / 20.854 |
| Detector `paintGL` duration | 0.663 / 0.974 / 1.258 / 5.910 |
| Completed detector paint to matching `frameSwapped` | 1.433 / 11.750 / 12.045 / 12.385 |
| Horizontal / vertical profile `paintEvent` p95 | 1.172 / 0.760 |

There were 1,122 callbacks, 707 new paint-request generations, 704 detector paints/compositions,
three unpresented generations and 418 duplicate or unmatched `frameSwapped` signals without a
new detector paint. Both marginals painted 551 times, the texture uploaded once, and the final
generation was composed with signed contrast and native cursor `(column_px=60, row_px=1655)`.
Dispatch p95 by event was 0.344/0.444/0.497/0.474 ms for cursor/wheel/pan/contrast. A 100 ms
heartbeat observed p95 119.930 ms and maximum 123.388 ms, corresponding to at most 23.388 ms
over its requested interval; this is not a measure of a >100 ms GUI stall. Profile paint
durations were measured, not matched presented-profile latency. Timing ends at Qt composition,
not monitor scanout. The reconstructed input sequence produced only 707 requests from 1,122
callbacks, so this is not a like-for-like throughput comparison with the earlier manual-pump
windows that requested a paint on almost every callback.

The 28.412 ms p95 is the interval between *changed* composed frames under only 707 new
requests in 15 s, or about 47 requests/s. It does **not** establish a continuous-demand
rendering miss or a causal effect of `QApplication.exec()`. Direct dispatch and measured paints
were short; paint-to-swap includes ordinary refresh waiting and is not a compositor-cost
measurement. The exact share of input supply, Qt scheduling, instrumentation and remaining
viewport work was unqualified. The controlled pair below addresses this measurement gap;
this sparse-demand diagnostic neither supersedes the earlier 26.7-28.5 ms misses nor completes
U02.

#### U02 controlled loop-driver pair (2026-09-28; diagnostic only)

A new 25 s aggregate active allowance covered one failed 9.999 s window and two corrected
windows of 7.511/7.499 s, totaling 25.009 s; Qt timer scheduling exceeded the cap by 9 ms.
No further active run was made. The first native window generated 776 requests but its
external OpenGL timing hook had been installed after widget construction: it recorded zero
detector paints despite 758 swap signals, so its frame result was discarded. A small-array
preflight then verified empty/nonempty percentile calculation, all four input events, OpenGL
hook invocation and the task-owned close path before the corrected full-image windows.

The unchanged production source was `a5d645f41089f70be7f5be7ddbe1081dca373454`.
An external Python script with SHA-256
`3b46bc9a84afc263b7cd4f0780e05cf07a2fd9b2ca54968843cc93a68bb84f2a` used
`python -u -B <task-scratch>/u02_pair.py native|manual <worktree> <task-scratch>/STREAM.json
7.5`, with `QT_QPA_PLATFORM=windows` and `PYTHONPATH=interactive;src`. Both separate processes
used the existing shell, production Fusion style/font/stylesheet, the named hBN 3000x3000 OSC
(decoded SHA-256 `137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`),
4,000 grid markers, exact marginal profiles, DPR 1 and the same 74.99 Hz 1920x1080 screen.
Their stylesheet SHA-256 was
`49d1b8aa7fb9a0798ce8500596cb2b7ecd084cfce46b85d4ab23a7b6a6fa8572`.
The deterministic repeating input generator used an in-image cursor, alternating wheel,
alternating pan with periodic Fit, and alternating contrast; every callback changed state.
Both requested an 8 ms **PreciseTimer** cadence, distinct from the earlier 13 ms traces.
The only intended configuration change between corrected windows was normal `app.exec()`
versus `app.processEvents()` with a 1 ms sleep. A 10 ms PreciseTimer heartbeat ran in both;
machine CPU and memory were sampled outside timed GUI callbacks.

| Corrected 7.5 s window | Native `app.exec()` | Manual pump |
| --- | ---: | ---: |
| Callbacks / requested generations / no-op callbacks | 578 / 587 / 0 | 688 / 699 / 0 |
| Detector paints / matched compositions | 562 / 562 | 563 / 563 |
| Actual callback and request interval p50 / p95 / p99 ms | 13.250 / 15.357 / 15.880 | 12.517 / 14.698 / 15.592 |
| Matched composed-frame interval p50 / p95 / p99 ms | 13.354 / 15.157 / 15.664 | 13.342 / 15.245 / 15.864 |
| Detector `paintGL` p95 / p99 ms | 0.736 / 0.865 | 0.698 / 0.790 |
| Completed paint to matching swap p95 / p99 ms | 4.629 / 4.958 | 4.697 / 5.131 |
| Wait to a later request after previous swap p95 / max ms | 0.606 / 1.169 | 2.037 / 2.854 |
| Duplicate/unmatched swaps; superseded completed paints | 9; 0 | 8; 0 |
| Unpresented requested generations | 25 | 136 |

The wait-to-later-request statistic did not account for a generation already pending at the
preceding swap, so it is not a true idle/backlog measure. These pair summaries also included
post-active drain swaps in their interval statistics; they are short-loop diagnostics, not
qualified active-only frame distributions. The manual loop delivered more callbacks and
coalesced more intermediate generations into essentially the same number of composed frames.
Both final generations composed, each task-owned autosave revision receipt reached the final
revision, both windows closed through the approved shell close path, and neither uploaded the
detector texture during the active window. The pair did not read the saved JSON back from disk.
The final modes were linear; cursor coordinates differed because callback counts differed.
The compact per-callback/paint/swap streams were 86,848/95,969 bytes with SHA-256
`320309502b9137e12bc2656dc76d704b9e81e15b4b426db91ebe764ec53675c1` and
`363798c62410e67fc0c8109810dc70280a13787faf360f0800b29e23979a1aa3`, respectively.
The external script and streams were removed after summary extraction. Native and manual
windows each had 563 horizontal and 563 vertical profile paints. Horizontal/vertical p95 paint
durations were 1.088/0.698 and 1.053/0.704 ms, respectively; native input-to-swap p95 was
15.021 ms and its heartbeat maximum 16.506 ms. Presented-profile latency was not measured.
Approximate process RSS was 245.5 to 241.5 MiB native and 244.1 to 241.6 MiB manual;
final whole-machine CPU samples were 11.5%/12.0%. The native console output was available
for this correction; the removed raw streams were not independently recalculated for the pair.

The short pair **did not reveal a composed-cadence difference between loop drivers** under its
changed-state 8 ms requested supply: both reported p95 intervals were below 16.7 ms and
both p99 intervals below 33.3 ms, with the active/drain boundary limitation above. It does
not prove sustained U02 acceptance, isolate the earlier 13 ms condition, or establish
physical monitor scanout latency. The sustained
native-loop packet below follows this diagnostic; no renderer change is supported by the pair.
U02 remains open, and the original measured misses and target remain unchanged.

#### U02 sustained native-loop qualification packet (2026-09-28; accepted)

The reviewed pair authorized a new 100 s aggregate active allowance. A tiny 0.01503 s
preflight checked empty/nonempty percentile inputs, a pre-construction OpenGL timing hook on
an 8x8 plane and approved window shutdown. Three native `QApplication.exec()` windows then ran
for 30.006, 30.003 and 30.000 s, totaling 90.025 s including preflight. No failed full-image
attempt or retry occurred. The external command was
`python -u -B <task-scratch>/u02_sustained.py selfcheck|run <worktree> <task-scratch>
[0.01503]` with `QT_QPA_PLATFORM=windows` and `PYTHONPATH=interactive;src`; the bracketed
preflight duration was passed only to `run` for the cumulative cap. The reporter SHA-256 was
`2d4c1130237bf58afd142b57a88f508f63fe98d9045f26995dc837978bbf8060`.

Production source `a5d645f41089f70be7f5be7ddbe1081dca373454` was unchanged. The shell
used the same Fusion/Segoe UI 10 stylesheet (SHA-256
`49d1b8aa7fb9a0798ce8500596cb2b7ecd084cfce46b85d4ab23a7b6a6fa8572`),
616x255 detector viewport, DPR 1 and Sceptre C27 (2) 1920x1080 screen at 74.99 Hz. The
same named 3000x3000 hBN OSC had decoded SHA-256
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`.
All 4,000 markers and both exact marginals remained enabled. Before each window the viewport
was reset to Fit, centered crosshair, linear automatic levels and visible layers, then settled
for 750 ms. A repeating four-way in-image cursor, alternating ±120 wheel, alternating ±(4,3)
pan with Fit every 64 actions, and alternating linear/signed contrast supplied changed state.
The input timer requested 8 ms and the heartbeat timer 10 ms, both `PreciseTimer`; actual
delivery is reported below. Machine load was sampled only at active boundaries. The earlier
13 ms traces remain separate conditions.

The external reporter recorded each callback's timestamp, action index, generation before/after
and desired state; `paintGL` start/end/generation; profile paint durations; and every swap's
timestamp, latest completed generation and current requested generation. Its swap match used
the newest completed paint, counted superseded paints and labeled swaps without a new paint
as repeated or unmatched. At runtime, all-swap intervals and fresh-generation intervals
included only pairs whose endpoints fell inside the same active window; final paint/save/close
drain was separate. Input-to-composition used the request timestamp for the matching
generation. For a *matched* preceding swap, a newer requested generation already pending
makes no-request gap zero; otherwise the gap ends at the next request. Two initial unmatched
swaps per window have unknown preceding state and were excluded from the corrected supply-gap
summary. This is request-supply timing, not a claim that GPU composition itself was idle.

The following values are p50 / p95 / p99 / max in ms. "All swaps" preserves the two initial
unmatched signals in each active window; "fresh" uses new, requested paint generations.

| Active-window measure | Window 1 | Window 2 | Window 3 |
| --- | --- | --- | --- |
| All-swap interval | 13.387 / 15.127 / 15.650 / 94.445 | 13.338 / 15.101 / 15.646 / 20.565 | 13.335 / 13.418 / 13.450 / 13.543 |
| Fresh requested composition interval | 13.387 / 15.118 / 15.633 / 16.484 | 13.338 / 15.101 / 15.646 / 20.565 | 13.335 / 13.418 / 13.450 / 13.543 |
| Input to matching composition | 13.027 / 14.938 / 15.495 / 16.344 | 12.917 / 14.879 / 15.424 / 20.037 | 12.947 / 13.200 / 13.268 / 13.337 |
| Actual callback/request interval | 13.266 / 15.239 / 15.802 / 16.669 | 13.292 / 15.211 / 15.781 / 20.504 | 13.333 / 13.682 / 13.887 / 14.840 |
| 10 ms requested heartbeat interval | 13.342 / 15.217 / 15.774 / 25.948 | 13.347 / 15.204 / 15.814 / 20.783 | 13.336 / 13.792 / 13.985 / 14.544 |
| Detector `paintGL` duration | 0.524 / 0.705 / 0.861 / 1.356 | 0.601 / 0.813 / 0.901 / 1.085 | 0.620 / 0.887 / 1.001 / 1.174 |
| Horizontal profile paint duration | 0.721 / 0.933 / 1.109 / 1.290 | 0.776 / 1.108 / 1.217 / 1.320 | 0.745 / 0.899 / 1.024 / 1.218 |
| Vertical profile paint duration | 0.554 / 0.685 / 0.782 / 0.949 | 0.565 / 0.726 / 0.812 / 0.962 | 0.539 / 0.652 / 0.713 / 0.845 |

| Active-window count | Window 1 | Window 2 | Window 3 |
| --- | ---: | ---: | ---: |
| Callbacks / requested generations / no-op callbacks | 2275 / 2311 / 0 | 2307 / 2344 / 0 | 2326 / 2363 / 0 |
| Active swap signals / fresh requested compositions | 2246 / 2244 | 2251 / 2249 | 2251 / 2249 |
| Detector paints / horizontal paints / vertical paints | 2244 / 2244 / 2244 | 2249 / 2249 / 2249 | 2249 / 2249 / 2249 |
| Coalesced or unpresented requests | 67 | 95 | 114 |
| Initial unmatched active swaps / same-generation repaints / superseded paints | 2 / 0 / 0 | 2 / 0 / 0 | 2 / 0 / 0 |
| Repeated swaps during post-active drain | 7 | 8 | 7 |
| Corrected known no-request gap p95 / max ms | 0.555 / 1.129 | 0.645 / 1.706 | 0.666 / 1.795 |
| Texture uploads during active window | 0 | 0 | 0 |

All fresh matched paint generations were requested and distinct, including the final
generations 2316, 4663 and 7029; none was a same-generation repaint or unrequested setup
paint. A read-only review recalculated the fresh-frame, response and heartbeat tuples from
the streams and found they matched the reporter's output. The 94.445 ms maximum between *all*
swap signals in window 1 is retained; its relation to the two unmatched initial signals is
unresolved, while the fresh-frame maximum was 16.484 ms and observed heartbeat maximum
25.948 ms. Among intervals with a matched preceding swap, no corrected no-request gap exceeded
4 ms. The reporter's original pending count of two per window came from treating preceding
unmatched swaps as if their prior generation were known; the corrected matched-predecessor
pending count was zero. All final requested generations composed.

After each active interval, the reporter waited for the final composition and matching
autosave revision, drained the owner/queue, then read the task-owned recovery JSON from disk.
Each document exactly equaled the final accepted project/view snapshot, including camera,
contrast, crosshair, layers, DPR, project/acquisition IDs and decoded source hash. The common
project/acquisition UUIDs were `cd4ee4fe-d35d-40a0-9a43-3d5038c99e5a` and
`b690edf3-7f36-4abc-a69d-4fbeb181ab71`. The native image stayed read-only with the same
native/display array identities and no texture reupload. Process RSS started/ended at
241.5/238.8, 240.0/239.7 and 240.5/240.6 MiB; final whole-machine CPU samples were
9.2%, 14.6% and 12.3%. Drains took 0.815/0.835/0.818 s outside active timing, and the
owned shell closed without a modal dialog. No fit or physical simulation ran.

The external result JSON SHA-256 was
`0f9d74ce30ba9dffc4aecd349f472ebfe57d7272beeafe7b4b95c1f0dc28c813`.
The three compact callback/paint/swap stream SHA-256 values were
`460146510f4049e70fd6f8bd54f66adf011ab67a8a23795b38c46502e17b1016`,
`53dee862263143fb6e0d08c5e3cb72ffe61ef20e394f97aab0e7bc3dd68cc8b5` and
`f0d94e1417c066984313b231b255df9466703adaee6c840b258988f616fb436c`.
The raw streams omitted exact active start/end timestamps, so their active boundary cannot
be reconstructed independently after cleanup; the live reporter filtered endpoints before
emitting summaries. The stream's active-swap prefix counts permit the reviewed all-fresh
recalculation above. Temporary reporter, streams, results and recovery root were removed
after extracting this evidence.

For this frozen 8 ms requested-supply scenario, all three active windows met the 16.7 ms p95
and 33.3 ms p99 frame targets using both all-swap and fresh-request distributions, the
50 ms p95 input-to-composition target, and no GUI heartbeat gap exceeded 100 ms. The
reported Qt composition boundary is not physical scanout. Profile paints were counted and
timed, but this packet did not bind each profile-state paint to a matching composed frame;
it does not newly qualify presented-profile latency. Earlier exact profile-value and
unchanged core evidence remain separate. The historical 13 ms request-supply misses are
preserved and their cause is still unknown. Independent review accepted U02 at `4c921c1`
for the declared changed-state scenario without promoting the older misses or claiming new
presented-profile latency.

#### U03 implementation checkpoint (2026-09-28; qualification pending)

The detector reader now follows or pins a native-coordinate profile center, supports independent
shaded row/column bands with numeric widths and pinned edge dragging, and offers sum, mean per
valid pixel, full-detector and drawn-ROI projections. The two intensity plots auto-scale or accept
independent finite pinned limits. Effective half-open bounds and center-bin support are visible.
Project and recovery documents save these choices with strict legacy/U02 defaults. Native int32
OSC import prepares both center bands and full-detector marginals in the worker; full projections
are cached by data identity and equivalent clipped band support. No prefix table was introduced.

Small independent enumeration covered non-square band/full/ROI sum, support and mean with and
without masks; separate checks covered signed/nonfinite and zero-support values, edge clipping,
integer overflow rejection, JSON roundtrip/older-view defaults, offscreen Qt click/pan/edge
gestures, rendered profile crossings and re-entry gaps at pinned intensity bounds, and an
external atomic project writer/reader roundtrip with the new state. The clipping check also
covered nearby large finite values and an overflowing opposite-sign difference. A direct
3000x3000 int32 broad ROI reduction cost about 39-43 ms before the one measured repair and
about 8-9 ms after using native
int64 sums without temporary validity arrays. The named hBN OSC worker prepared the 3000x3000
native image, center bands and full marginals in 0.157 s with 68.85 MiB retained result bytes.

An initial standalone-panel timing attempt ran three 30 s windows with a 13 ms requested timer.
Its 13.77-13.98 ms matched-profile p95 values are component observations only: it lacked the
production shell and markers, and its swap attribution did not prove presented profiles. A
corrected single native-shell window used the same 3000x3000 hBN source, 4,000 markers, the
production Fusion style, an 8 ms PreciseTimer input request and 10 ms heartbeat on the Sceptre
C27 (2) 1920x1080, DPR 1 display. The source decoded SHA-256 was
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`. The external
reporter was run with `QT_QPA_PLATFORM=windows`, `PYTHONPATH=interactive;src` and `python -u -B`;
it was removed after the packet. Its 30.009 s active interval had 2,650 mixed profile-query
callbacks and 2,231 requests with matching profile paints before the attributed composition.
All values below are p50/p95/p99/max in milliseconds.

| Corrected native-shell window | p50 / p95 / p99 / max |
| --- | --- |
| Profile input to matching composition | 12.567 / 14.215 / 17.051 / 24.278 |
| Fresh composition interval | 13.341 / 18.305 / 22.624 / 27.716 |
| All swap interval | 13.328 / 18.229 / 19.455 / 26.678 |
| Input callback duration | 0.361 / 9.288 / 9.817 / 14.158 |
| Actual input interval | 12.622 / 14.557 / 17.073 / 24.792 |
| Heartbeat interval | 13.025 / 22.322 / 23.788 / 35.540 |

There were 2,231 profile requests presented within the active window, 34 repeated swaps,
zero same-generation or superseded paints, zero unmatched swaps and one texture upload. Repeated
swap counts include setup and drain. The final generation `3316` and profile key matched after
the 4.021 s drain. The detector viewport was 616x180 logical pixels; process RSS was 312.8 MiB
at active start and 306.2 MiB at report, with no sampled process peak. The 18.305 ms fresh-frame
p95 misses the 16.7 ms target. Recovery save revision remained `-1` against view revision `2651`:
the external temporary recovery directory denied writes, so saved-document equality was not
verified. The corrected reporter emitted summaries without per-request raw streams or exact active
timestamps, preventing independent recomputation. The first three windows plus the corrected
window used 120.015 s of the 180 s active allowance; no further large run followed. This one
corrected window is partial evidence; sustained three-window qualification and shell recovery
readback remain open. No renderer or GUI bottleneck cause was established. Timing ends at Qt
composition, not physical scanout.

#### U03 native correctness and cadence attribution (2026-09-28; evidence only)

Production source remained `5dd2561`. An owned file directly under the writable external
visualization root passed create/write/read/delete before Qt; the previously denied temporary
child directory was not reused. The external native probe/reporter SHA-256 values were
`f081f0495c853eadb9a6cd9b93e843038d66e90c5293bfb0d4fdfac394821a71` and
`144c0c3e1263e896657510a9b951dabfeab41c776d87f08457b30c310a5c648d`.
Commands used `QT_QPA_PLATFORM=windows`, `PYTHONPATH=interactive;src`, `python -u -B`, and
`QT_SCALE_FACTOR=1` or `1.25` for tiny native runs. The tracked little-endian non-square OSC
decoded as 11 rows by 7 columns, SHA-256
`da0a7d85e8be6461ae3985b9514e5cbaf75d7f3d499d342c80b6841f7e118a63`; the
paired big-endian OSC produced the same native values.

At actual DPR 1.0 and 1.25, shown Qt windows mapped corner/interior native centers back to
the same coordinates. Framebuffer samples at every native row/column center matched the
independently calculated shaded row band `[4,8)`, column band `[2,5)`, and drawn ROI
`[1,6)` columns by `[2,9)` rows. Hover at `(2,3)` read 2,222 counts; click pinned it;
numeric placement and edge drags changed four rows to five and three columns to four. The
DPR 1 shell made a newer profile edit during an externally delayed save; normal autosave
published the exact final `ProjectDocument`. Reopening the recovery draft restored project
and acquisition IDs, source hash, center, follow state, widths, measure, scope, ROI and both
pinned scales. Both windows closed through normal owner drain. These are static native
geometry and recovery checks, not a new pan/zoom or monitor-migration campaign.

One 9.902 s attribution window then used the unchanged Shell/Fusion/Segoe UI 10 style,
3000x3000 hBN OSC (decoded SHA-256
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`), 4,000
markers, 616x180 viewport, DPR 1 and Sceptre C27 (2) 1920x1080 at 74.99 Hz. Its 8 ms
PreciseTimer repeated pointer-in-band, row width, column width, measure, full, the same broad
ROI `(100,2900,100,2900)`, band, pointer; the heartbeat requested 10 ms. There were no
no-op callbacks. Active performance-clock bounds were `378267.712927`–`378277.6148138`.
The three tiny commands and this active window together used under 20 s even counting the
tiny commands' full wall times. The independently recomputed active trace contained 5,456
action/reduction/publication/paint/swap records, 521,506 bytes, SHA-256
`c40dd7c6d644c330c764b34c3d1683081801497b042cf611598c30ce728555ef`.
The reporter and trace were external, with no repository harness retained. Values below are
p50/p95/p99/max ms; microsecond serialization changes a recomputed percentile by at most
0.001 ms.

| One U03 attribution window | p50 / p95 / p99 / max |
| --- | --- |
| Fresh composition interval | 13.356 / 18.206 / 22.699 / 28.075 |
| All swap interval | 13.342 / 18.116 / 18.628 / 19.205 |
| Profile input to matching composition | 12.573 / 14.062 / 16.990 / 18.910 |
| Actual input interval | 12.764 / 14.457 / 16.853 / 19.083 |
| Callback duration | 0.382 / 9.275 / 9.652 / 9.815 |
| Heartbeat interval | 13.198 / 22.261 / 23.667 / 25.165 |

Of 868 changed queries/publications, 732 had matching detector and marginal content at
composition; 136 were coalesced or unpresented within active timing. The trace has 732 fresh
detector paints, 743 paints for each marginal, 742 swaps, ten repeated swaps, two initial
unmatched swaps, and zero same-generation or superseded paints. Every recorded swap used the
newest completed detector paint; all 732 fresh presentations matched both profile generation
and buffer identity before composition. Final generation `1088` finished all paints by
`378277.610954` and swapped at `378277.614733`, before active end. The reporter stops paint
tracking at active end, so a hypothetical later final request would not be proven; this
observed final is proven inside the trace. The final normal autosave equaled the captured
document at revision 869, the owner drained on normal close, native/display arrays retained
identity and read-only flags, and the texture uploaded once.

There were 652 narrow-band reductions (p95 0.249 ms), 108 broad ROI reductions (p95 9.389 ms),
and zero full reductions because full marginals were cached. ROI callback p95 was 9.683 ms;
full-mode callback p95 was 0.216 ms. Independent trace review found 119 of 731 fresh intervals
over 16.7 ms: 108 included ROI work, ten band work, one no reduction. ROI-containing intervals
had median/p95 17.783/18.552 ms; band-containing intervals 13.171/14.470 ms, but the eight
longest intervals (22.699–28.075 ms) followed narrow-band actions. Known no-request gaps
after matched prior swaps had median/p95/max 0.383/0.666/9.164 ms, with no known pending
request at those prior swaps. These associations do not establish a sole cause or measure GPU
execution or physical scanout. Fresh-frame p95 still misses 16.7 ms. Reusing the unchanged ROI
projection across intervening full/band views, keyed by data and measure, is a small candidate
for a separately authorized measured repair; no optimization was made in this packet.

Source-derived resource accounting uses `N = rows * columns` and `S = rows + columns`.
One sum/support profile pack is `16S` bytes; imported center and full packs total `32S`.
Full mean shares support and adds `8S`; retained full, full mean and distinct current packs
use at most `40S`, with up to `56S` during a new-pack overlap. The two plots alias these
buffers; the imported center pack may be transiently retained while the worker result is
published. At 3000x3000, `N = 9,000,000`, `S = 6,000`, native int32 and display float32
planes each use 36,000,000 bytes, and retained profile caches are at most 240,000 bytes.
The reported OSC result was `8N + 32S + 6000` header bytes, or 68.85 MiB, below its 96 MiB
result cap. Admission also caps source bytes at 64 MiB, decoded bytes at 32 MiB, pixels at
12 million, and axes at 16,384 plus the OpenGL limit.

Unmasked int32 direct sums need output-sized scratch. For area `A`, a masked int32 reduction
can allocate approximately `5A` bytes (validity plus safe values) and a float64 reduction
approximately `9A` bytes; horizontal and vertical passes are sequential. Compact ROI outputs
add `16(h+w)` bytes alongside `16S` native-length aligned outputs; mean adds up to
`9 * max(h,w)` intermediate bytes. A 2800x2800 masked int32 ROI therefore has roughly
39.2 MB of broad scratch, and float64 roughly 70.56 MB; a full 3000x3000 float64 pass can
reach about 81 MB. An existing input mask is separate. The result cap is not a process peak
cap: preparation can hold raw/native/display arrays near `12N` plus an `N`-byte positive mask,
alongside a prior image's `8N_old` native/display arrays and prior caches during replacement,
before decoder, Python, Qt and driver overhead. Generic `set_image` copies its input and does
not have the OSC admission caps. These are conservative formulas, not measured RSS peaks or
allocator guarantees. Process RSS was 307.4/307.1 MiB at active start/report, Windows peak
working set was 327.5 MiB, and process CPU time rose 7.797 s from active start through
report/drain, not solely within active timing. New broad
ROI queries remain synchronous in the GUI; no worker cancellation or prefix-table behavior is
claimed. U03 remains unaccepted pending a reviewed correction and sustained qualification;
U02a had not started at this checkpoint.

#### U03 single-entry ROI reuse repair (2026-09-28; one-window comparison)

The detector panel now retains one ROI sum/support pack keyed by acquisition identity, native
image identity, data revision and effective native ROI bounds. It survives band/full views,
cursor/camera/contrast changes and measure toggles. Mean derives from the same readonly sums
and shares support; a new ROI, image or acquisition replaces the entry. The panel does not have
a mask/correction input or revision. The canonical reducer still owns masked, signed and
nonfinite semantics. First/new ROI queries remain synchronous in the GUI. No prefix table,
worker, dependency or backend was added.

Tiny external checks on non-square int32 and masked signed float64 arrays compared band/full/ROI
sum/support/mean to independent direct enumeration, including nonfinite and missing bins.
Instrumented canonical-reducer calls showed one ROI reduction through
ROI→band→full→same ROI and no extra reduction for sum/mean reuse. Different bounds, data
revision, image replacement and A–B–A acquisition resets caused misses. Invalid ROI bounds
and an invalid image were rejected without changing the prior image/cache; arrays stayed
readonly and the offscreen check made zero display uploads. Previously reviewed native DPR,
framebuffer, clipping, autosave/reopen and navigation evidence was not replayed.

One native comparison window used the same Shell/Fusion/Segoe UI 10 style, 3000x3000 hBN OSC
(decoded SHA-256 `137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`),
4,000 markers, 616x180 viewport, DPR 1 and Sceptre C27 (2) 1920x1080 at 74.99 Hz. The
same eight actions and formulas from the prior attribution packet supplied pointerA,
row width, column width, measure, full, fixed ROI `(100,2900,100,2900)`, band, pointerB
using an 8 ms PreciseTimer; heartbeat requested 10 ms. The ROI cache was empty before active
timing. External script SHA-256 was
`3871992fd1cae8d2dc229792963ddac98c027cdad13d37ebc4c265e2ad3c8c23`;
`python -u -B` ran with `QT_QPA_PLATFORM=windows`, `QT_SCALE_FACTOR=1` and
`PYTHONPATH=interactive;src`. A tiny hook/statistics/shutdown preflight and a direct writable
file preflight preceded the large image. Active bounds `379027.2025452`–`379057.1599226`
on the process clock span 29.957 s. The two tiny numerical attempts, the preflight command
and this one native window conservatively used under 40 s even counting the tiny commands'
full wall times. There was no large-image retry. The externally buffered trace contained
15,444 records, 1,596,632 bytes, SHA-256
`4c9465cf7e286e4d7cde7648d75f6ea2c8b4a70459a87b92ef31fc5084c50428`.
It was independently recomputed before cleanup. Values are p50/p95/p99/max milliseconds.

| U03 ROI reuse comparison | p50 / p95 / p99 / max |
| --- | --- |
| Fresh composition interval | 13.339 / 14.164 / 14.552 / 25.332 |
| All swap interval | 13.338 / 14.164 / 14.552 / 18.884 |
| Profile input to matching composition | 12.907 / 13.787 / 14.201 / 17.145 |
| Actual input interval | 13.276 / 14.218 / 14.653 / 17.424 |
| Callback duration | 0.367 / 0.522 / 0.582 / 9.244 |
| Heartbeat interval | 13.279 / 14.413 / 14.999 / 22.094 |

Of 2,340 changed requests/publications, ROI was selected 292 times: one cold reduction at
8.941 ms and 291 cache hits. Warm ROI callback p50/p95/p99/max was
0.147/0.190/0.239/0.399 ms; its one cold callback took 9.244 ms. The canonical reducer
handled 1,756 bands (p95 0.255 ms), zero full projections and one ROI. All 2,245 fresh
swaps matched completed detector and horizontal/vertical profile generations and buffer
identities in the worker reporter; its matching could reuse historical marginal paints, so
this window alone did not exclude a newer mismatched marginal paint at swap. There were
2,247 paints per marginal, 2,248 active swap events, one repeated swap, two initial unmatched
swaps, zero same-generation/superseded paints and no no-op callbacks. Final generation `2928`
composed at `379057.1598216`, before active end, and the final query key matched. Actual
saved JSON equaled the captured final `ProjectDocument` at revision 2341; the owner drained
and the window closed normally. Native/display identities and readonly flags persisted; the
texture uploaded once in total. The preparation snapshot showed zero uploads before its settle
period, so an active-only upload delta was not recorded.

The prior one-window baseline on unchanged `5dd2561` had 108 repeated ROI reductions at
9.389 ms p95 and fresh-frame/profile-response p95 of 18.206/14.062 ms. Under this repair,
fresh-frame/profile-response p95 were 14.164/13.787 ms. The first cold ROI was retained in
the new distribution: its one associated fresh interval was 17.371 ms. Only two of 2,244
fresh intervals exceeded 16.7 ms: that cold ROI interval and a 25.332 ms band-containing
interval. This one window meets the p95 16.7 ms, p99 33.3 ms, profile 50 ms and observed
heartbeat 100 ms thresholds for its declared supply, but does not establish sustained U03
acceptance or prove a sole frame-delay cause. The baseline and comparison have different
durations and incidental machine load; no second repair or run followed. Timing stops at Qt
composition, not monitor scanout or GPU execution.

The one-entry ROI sum/support pack adds `16S` bytes for `S = rows + columns`; optional mean
adds `8S` and shares support. At 3000x3000 these are 96,000 and 48,000 bytes. A new ROI
query can briefly overlap the old pack and its optional mean before the cache is replaced;
the current band/full view may also retain a distinct profile. No history of ROI planes or
additional native image is retained. First/new broad ROI work remains synchronous and was
not qualified for arbitrary masks, float64 data or larger admitted shapes. Active start/end
RSS was 305.6/311.9 MiB, Windows peak working set 327.5 MiB and precisely active process
CPU time 20.938 s; memory figures include Qt/driver/runtime and are not a GPU allocation
measure. U03 remains unaccepted pending supervisor review and later sustained evidence;
U02a had not started at this checkpoint.

#### U03 sustained native-shell qualification (2026-09-28; reviewed evidence, acceptance pending)

The repaired production source stayed at `ba80bf9`; `interactive/detector_panel.py` SHA-256 was
`372677217dafbaf3f28770563e7ef4c993daf940c8f0679ae11a6f3499b3f8bb`. One external
reporter (SHA-256 `a9d2cd35cc49ee421cbd125e912695c38594fa540bae668dba9abeb9a0c05c76`)
ran `python -u -B <reporter> --selfcheck`, then `python -u -B <reporter> 1`, `2` and `3` in
separate fresh processes, with `QT_QPA_PLATFORM=windows`, `QT_SCALE_FACTOR=1` and
`PYTHONPATH=interactive;src`. Here `<reporter>` was
`C:/Users/Kenpo/.codex/visualizations/2026/09/28/01a0e91c-3efa-74a3-a0fc-6a8354d0ed72/u03_final_reporter.py`;
each command ran from this worktree root. The self-check rejected a historical H/V match after a newer
marginal paint and checked missing/future marginal paints, repeated swaps and empty/single
quantiles; a tiny native GL-hook/normal-shutdown and external file-write preflight passed.
No failed large active window or retry followed. Three active windows totaled 89.9553962 s;
even adding the entire 1.301 s preflight command yields 91.2563962 s under the closed 100 s
allowance. The reporter and raw telemetry were removed after independent review.

Each process used the tracked 3000x3000 hBN OSC (decoded SHA-256
`137cd964f156d66144aea7b1ae2905aa383aeca5c8bebc35a0a6b5ae2724474d`), 4,000
markers, a 616x180 viewport at DPR 1 on Sceptre C27 (2), 1920x1080 at 74.99 Hz, and the
production Shell/Fusion/Segoe UI 10 style. Its worker prepared the native image and full
marginals before timing; the fixed ROI `[100,2900)` columns by `[100,2900)` rows was selected
with an empty ROI cache at active start. An 8 ms PreciseTimer cycled pointer A
`(30+37n mod 2940, 30+53n mod 2940)`, row width `1+n mod 11`, column width `1+n mod 13`,
sum/mean toggle, full, fixed ROI, band, then pointer B
`(30+19n mod 2940, 30+29n mod 2940)` for zero-based callback index `n`. A separate
10 ms PreciseTimer supplied heartbeat observations. All callbacks changed the profile key.
Quantiles below are p50/p95/p99/max in ms; cold ROI samples remain in their active intervals.

| Window, active bounds on process clock | All-swap interval | Fresh interval | Profile input to matching composition |
| --- | --- | --- | --- |
| 1: `379940.8703575`–`379970.8545207` (29.984 s) | 13.322 / 14.089 / 14.568 / 19.279 | 13.323 / 14.089 / 14.568 / 19.279 | 12.947 / 13.727 / 14.223 / 18.784 |
| 2: `379981.5690493`–`380011.551757` (29.983 s) | 13.337 / 13.998 / 14.412 / 33.155 | 13.337 / 13.995 / 14.409 / 33.155 | 12.938 / 13.641 / 14.103 / 32.381 |
| 3: `380022.3541054`–`380052.3426307` (29.989 s) | 13.327 / 14.097 / 14.416 / 28.081 | 13.327 / 14.097 / 14.416 / 28.081 | 12.956 / 13.728 / 14.066 / 27.832 |

| Window | Actual supply interval | Callback duration | Heartbeat interval | Detector / horizontal / vertical paint p95 |
| --- | --- | --- | --- | --- |
| 1 | 13.288 / 14.088 / 14.569 / 19.235 | 0.344 / 0.492 / 0.554 / 9.473 | 13.297 / 14.347 / 14.755 / 20.111 | 0.811 / 0.916 / 0.529 |
| 2 | 13.280 / 14.024 / 14.465 / 32.922 | 0.333 / 0.471 / 0.539 / 8.843 | 13.323 / 14.196 / 14.699 / 33.706 | 0.735 / 0.926 / 0.530 |
| 3 | 13.290 / 14.092 / 14.417 / 28.146 | 0.321 / 0.459 / 0.508 / 8.622 | 13.281 / 14.208 / 14.578 / 28.145 | 0.771 / 0.949 / 0.570 |

Windows 1/2/3 had 2,324/2,372/2,331 changed callbacks and requests; 2,250/2,249/2,249
active swaps and 2,248/2,247/2,248 fresh compositions with current H/V content. The
unpresented or coalesced request counts were 76/125/83. Repeated swaps were 0/0/1;
initial unmatched swaps 2/2/0. Obsolete-profile swaps, same-generation paints and
superseded paints were zero in every window. Each window made exactly one cold broad ROI
reduction (9.163/8.527/8.351 ms) and 289/295/290 ROI cache hits; band reductions were
1,744/1,780/1,749 and full reductions zero. Warm ROI callback p95 was
0.178/0.168/0.156 ms. The intervals containing the cold ROI were
18.5549/16.9358/16.9637 ms, retained in the distributions. Each window had three
fresh intervals above 16.7 ms; outliers including the 33.155 ms window-2 maximum remain
visible rather than trimmed. All p95 fresh intervals were below 16.7 ms, p99 below
33.3 ms, profile response p95 below 50 ms and observed heartbeat gap below 100 ms for
this declared input and display.

The serialized traces contained 17,664/17,795/17,743 records, with SHA-256
`f393a472fae0c92807f59d227c55be9c3ecbe28423142ab10b8006a3fbd96d7f`,
`270b1a25f7ed466adfdd1e4b780659f38096b5ffc5983b551126f43dfbef1bbb`, and
`b2aafe5202103b9f1f73c2c2817bcad2d90b21a5a17d5924401503c0c334e625`.
Unlike the earlier worker reporter, the new matcher used the **latest completed** G/H/V
paint before each swap, matched generation and actual H/V buffer IDs to the published
profile pack, and tracked paints through normal drain. Independent root and guardrail
reviews chronologically sorted the serialized completion timestamps and checked all
2,258/2,257/2,257 active-plus-drain swaps: zero false matches, phase-boundary errors or
frozen-action mismatches. Final requests 2908/2968/2917 matched the current key and
composed at `379970.8541936`/`380011.5513933`/`380052.3423131`, before their respective
active ends; later drain swaps also matched. Actual saved JSON equaled the final
`ProjectDocument` at revisions 2325/2373/2332. All owners drained and windows closed
normally. Native/display array identities and read-only flags persisted; active texture
upload delta was zero in each window.

Active start/end RSS was 306.5/311.5, 301.8/307.2 and 303.9/307.2 MiB; Windows process
peak working set was 328.6/327.7/327.8 MiB and active process CPU time
20.328/20.344/19.906 s. These process figures include Qt/runtime costs and do not measure
driver GPU allocation. This qualifies the frozen hBN mixed workload through Qt composition;
it does not measure physical scanout or qualify arbitrary masked/float64/max-shape broad
queries. First/new ROI reduction remains synchronous. Historical failed U03 windows and
their limits above remain evidence. U03 qualification is reviewed complete, pending final
supervisor acceptance; no scientific fitting inference or U02a work is implied.

### M2 — independent simulator

Likely ownership: optional forms/controller and the current detector worker. Configured simulations
reuse `pipeline/configured_simulation.py`; native drafts reuse `fitting/native_input.py`, supported
material bindings and `pipeline/conditional_detector.py`. Keep these actual contracts distinct.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [x] U09a: numeric parameter state | U01b/U08a | Provide explicit parameter descriptions, unit conversion, provenance, validation, undo/redo for U02a metadata mutations as well as physical and initial-value edits, and immutable launch snapshots. Reuse core constructors; simulation drafts and fit seed packs retain distinct types. Verify round trips for an admitted configuration without inventing fitted coordinates. |
| [ ] U12: configured simulator | U02/U03/U01c/U09a | Load an existing supported configuration, run the current preview/quantitative owner and save/reopen the independent draft. Show measure, backend, prefix/progress and failure state; no experimental image, fitted observations or 3D editor is required. |
| [ ] U12a: complete supported parameter forms | U12 | Expose all supported selected-model configuration fields in searchable grouped forms, including source, instrument, structure, mosaic, optics and execution. Compare field coverage with the capability inventory; unsupported combinations explain why. Keep configuration import/export working. |
| [ ] U12b: quantitative inspection | U12 | Bind exact cursor/profile/export values to immutable float64 snapshots with their draw prefix or route-specific numerical settings. Preview progression cannot mutate or relabel them; check direct reductions, lease consumption and bounded snapshot/upload memory. |
| [ ] U12c: independent native simulator | U12a/U12b | Map each admitted native model's physical/numerical parameters into an independent draft and the existing detector/integration owners. Load/edit/run/save with no observation pack or fit record; verify field coverage, canonical parameter binding and bounded like-observable output equivalence. Reuse shared views/jobs; do not wrap the fit-required render CLI, duplicate physics or invent a universal schema. |

### M3 — geometry fitting, one route at a time

Likely ownership: optional observation/result panels and the existing `selection/`,
`fitting/hbn.py`, `indexed_series.py` and `joint_geometry.py` owners. Preparation, fitting and
qualification remain explicit stages. Share an extracted CLI routine only when both CLI and GUI
actually need it; do not copy the orchestration into widgets.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U05a: direct/manual beam-center proposal | U03/U04/U09a | Add native click/numeric center and a worker-owned Gaussian ROI proposal with background, residuals and reliability. Check off-center/non-square coordinates, saturation/occlusion and stale rejection; adoption changes a seed only and survives save/reopen. |
| [ ] U08b: hBN preparation/fit boundary | U08a/U01c | Expose the narrow preparation and frozen-observation fitting inputs needed for admitted hBN seeds/bounds and cooperative cancellation. Preserve existing automatic defaults; make wavelength/ring/dark assumptions explicit. Verify residual/coordinate equivalence and actual optimizer inputs before enabling new controls. |
| [ ] U05b: hBN observation review | U04/U08b/U09a | Trace candidate rings with the canonical owner, show coverage/assignments and review only admitted edits. Persist the accepted ring pack, mask/config provenance and its revision; fitting consumes it unchanged and does not silently retrace. |
| [ ] U05: result binding and overlays | U02/U04/U01b | Bind contract-valid result records to acquisition/observation IDs; draw predicted peaks/rings, observed points, residuals and feature inspection. Distinguish no-fit, draft, candidate, selected and stale states; reject mismatched records and retain unavailable uncertainty. |
| [ ] U10: first hBN calibration workflow | U05a/U05b/U05/U08b/U09a | Numeric seeds -> reviewed ring pack -> explicit fit -> statistics/overlays -> named result/save/export works. Check seed/revision handoff, safe stop and late-result rejection; center/tilt and calibrant-private distance retain their meanings and actual qualification evidence. |
| [ ] U05c: sample discovery and freeze | U02a/U04/U09a | Bind a supported OSC series, geometry/material and reviewed mask to canonical discovery/indexing. Add narrow explicit mask/data input boundaries where necessary. Persist candidates, admissible review decisions and one frozen fit-ready pack; the fit entry point must not reindex internally. |
| [ ] U08c: indexed-series execution boundary | U08a/U01c/U05c | Use the existing explicit seeds/bounds and add safe cancellation/progress boundaries only where absent. Verify fixed references, incidence delta/trim gauges and unchanged frozen IDs with a small supported case. |
| [ ] U10a: sample-only geometry workflow | U05c/U08c/U05/U09a | Fit one admitted indexed series using its exact reviewed pack; save/export per-image/shared diagnostics and seed/fitted comparison. Insufficient tracks or unsupported single-image cases remain clearly unavailable; no manufactured observations or qualification. |
| [ ] U08d: joint geometry execution boundary | U08a/U01c | Add explicit admitted starting-state and safe-stop inputs while preserving the reduced gauge and existing defaults. Verify core seed/bounds handoff and required hBN/Bi2Se3/Bi2Te3 roster; do not generalize to arbitrary material groups through GUI assumptions. |
| [ ] U10b: joint geometry workflow | U10/U10a/U08d | Combine the admitted frozen calibrant/sample packs, review shared/local/fixed scopes and run the current joint owner. Persist result and supported hash-bound handoff; verify IDs, private calibrant distance, rank/qualification and stale downstream state. |
| [ ] U05d: geometry-derived center proposals | U05/U09a | Offer ring or sample estimates when the matching admitted result is available, including interpretation and coupling. An hBN result need not wait for a sample fit. Applying a shared estimate lists affected acquisitions; derived evidence is not counted twice. |

### M4 — reciprocal and experiment views

Likely ownership: narrow optional scene/reciprocal helpers and the existing parameter/result
controller. These are new presentations of the same model, not new geometry implementations.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U07: reciprocal geometry preview | U02/U09a | Show each image's on-demand draft/saved coverage and cursor Q in the declared frame; link available features. Compare nondefault wavelength/direction and off-panel cases with canonical APIs; reject stale jobs and fixed illustrative Ewald data. |
| [ ] U08: textured experiment scene | U02/U09a | Show canonical beam, sample, goniometer axes/pivots and detector with actual image/available overlays. Click-to-zoom, context return and camera presets work; verify texture corners, compound transforms, context recreation and no orbit-driven image upload. |
| [ ] U09: synchronized physical handles | U07/U08/U09a | Connect callouts, numeric fields and constrained arcs/arrows to existing parameter state without requiring a fit. One gesture is one undo; fixed/derived/unsupported coordinates stay explained. Simulator edits match canonical configuration. Enable each fit-specific mapping only after its execution boundary verifies the launched vector; handles cannot enable an unsupported scope. |
| [ ] U09b: experiment/simulator transfer | U09a/U12/U12b | Copy a compatible snapshot from numeric state into an independent draft without requiring the handle editor. Show exactly what transfers; verify units, provenance and unchanged source projects. Extend to the native route after U12c through its explicit admitted mapping. |

### M5 — existing prepared native fitting

Likely ownership: application stage/result bindings, `native_input.py`, `native_observations.py`,
`native_workflow.py`, `native_search.py`, `native_accuracy.py` and current material bindings.
Limit each delivery to one admitted model/stage. Existing physics and execution owners are reused.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U11: open a prepared native experiment | U01b/U01c/U05/U09a | Load one supported prepared physics/observations/plan set, verify hashes and show measured profiles, background/covariance policy and model capabilities. Save/reopen references without altering frozen arrays; do not describe this as preparation of new raw images. |
| [ ] U11a: mosaic stage | U11 | Run the existing search with the admitted mosaic coordinates/controls; persist initial/candidate/selected values and qualification. Verify frozen observable identity and explicit numerical settings on a bounded case. |
| [ ] U11b: ordered intensity stage | U11a | Bind a supported ordered model and declared upstream state, nuisance scale/background and active coordinates. Save/export matched-observable results; controls do not change membership or silently substitute an empirical baseline. |
| [ ] U11c: supported disorder stage | U11b | Expose disorder only for an actual supporting material binding with the ordered control and weak-direction diagnostics. Verify identical observable/measure and preserve unsupported native Bi states. |
| [ ] U11d: indexed geometry adoption | U11/U10a | Use the existing indexed OSC fit-record adoption path through `fixed_position_from_fit_record`; preserve its root/outer audits, provenance, qualification/model-limited distinction and downstream admission rules. Verify image/file/geometry identities and projector consistency; never reinterpret a joint handoff as an indexed result. |

Joint-handoff adoption is a distinct conditional extension after U10b and requires an explicitly
supported native binding. If one is unavailable, show that limitation and add a scoped task before
implementing it; it does not block the existing indexed-fit adoption route or become an implicit
artifact conversion.

### M6 — native observations from new acquisitions

This is scientific integration work, not a stage-panel widget. Likely ownership:
`measurement/continuous_regions.py`, current native measurement projectors,
`fitting/native_observations.py`, `radial_background.py` and a narrow preparation entry point.
Name the exact existing owner during U11e before editing; never duplicate the projection equation.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U11e: declare one preparation recipe | U04/U10a/U11 | Specify admitted acquisition/material/source/geometry, native membership, observable units, dark/background, full covariance, output identities and independent comparison/tolerances. Identify missing production boundaries before coding; unsupported recipes remain explicit. |
| [ ] U11f: implement numerical preparation | U11e | Produce a valid frozen native observation/background/covariance set for that recipe using authoritative projectors. Verify support, masks, signed counts, shared-dark covariance and hashes against the declared independent calculation; save external reproducible outputs. Split by numerical owner if more than five files are needed. |
| [ ] U11g: preparation UI and stage handoff | U11f/U11a | Guide selection/review/preparation of new imported acquisitions, show progress and persist the frozen result. A changed mask/calibration invalidates dependent preparations; a supported new input reaches each delivered admitted fitting stage without a handwritten manifest. Completion applies only to qualified recipes; it need not wait for unrelated disorder support. |

### M7 — complete daily use and release

Likely ownership: existing application inspector, project I/O and runtime result writers.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U13: sensitivity and supported uncertainty | U05/U09/U10 | Request bounded canonical parameter perturbations with declared held-fixed assumptions. Show covariance-based bands only with defensible evidence; retain rank/branch/invalid states and distinguish sensitivity from posterior confidence. |
| [ ] U02b: reusable import/setup | U02a/U09a | Add templates, source/reference/copy storage review and guided relink. Defaults carry provenance; template edits cannot mutate projects and matching filenames cannot substitute different bytes. |
| [ ] U14: complete exports and portable archive | U06/U10b/U11g/U12b/U14b | Extend earlier per-slice exports to selected cross-workflow products and a self-contained archive with size/identity review. Verify reopened inputs, parameters, observations/results and qualified/nominal statuses; retain exact underlying values for figures. |
| [ ] U14a: repeat-use and recovery integration | U02b/U14 | Integrate named attempts, duplicate experiments, selected result history and bounded recovery across all delivered stages. Verify interrupted saves, missing files and independence; no executable checkpoint or false solver resume. |
| [ ] U15: integrated release check | All prior tasks | Complete section 10's simultaneous inspection/3D/job and twenty-image journeys on declared hardware. Review optional imports, launch, keyboard/DPI behavior, cold/warm latency distributions, memory, cancel/close and failure recovery. Record actual limitations and remove temporary checks. |

### How each task is delivered

Keep each implementation commit to a small coherent change, normally one to three files and at
most five. If a row exceeds that, split it here before coding while retaining its acceptance
conditions; do not hide a second subsystem in the same commit. Share paths sequentially.

Each slice includes persistence for the state/results it introduces and export for usable outputs.
U14 integrates archive/export coverage; it is not permission to postpone earlier save or export.
Use the configured Ruff formatting/lint rules on touched Python; run package/import/affected CLI
checks when those boundaries change. Perform only focused external temporary numerical/interaction
checks for concrete risks, record inputs/commands/outcomes/limits, then remove the check code.
No retained test, proof or benchmark suite is introduced. A runtime diagnostic, when explicitly
requested, still follows the single external `.ra_diag.npz` contract.

For every checked task, record its coherent commit, delivered flow and measured evidence in this
section. A document review or an unchecked feature is not completion. At each milestone, perform
its end-to-end user journey; future capability limits must remain visible. New scientific support
needs its own evidence before enabling it, while independent UI/simulator work can continue.

Usability completion must be observed through complete journeys:

- A first-time user imports one image and sees the detector/profiles without prior configuration.
- A researcher imports hBN plus a supported angle series, assigns metadata in bulk, chooses the
  center, reviews frozen features, sets initial geometry and sees exactly what will be fitted.
- A simulator user loads/runs/saves a configuration without preparing an experiment.
- A user can follow an actionable blocked-stage explanation and preserve work after a corrupt
  file, bad estimate, unqualified fit or unsupported model.
- A returning user reopens/recovers a project, compares named attempts and exports a figure plus
  numerical values without overwriting the selected result.
- An expert reaches every supported parameter with keyboard alternatives; inspection/camera
  actions never edit geometry, masks or observation membership.

Use tracked example inputs for bounded walkthroughs. Obtain real researcher feedback when
available and record friction. This document makes no claim of implemented or measured success.

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

## 13. Independent audit disposition

A fresh read-only reviewer audited `bc47d7b` against the original objectives and the repository's
lightweight-design rules. Two required corrections and two optional simplifications were accepted
below. Earlier rendering/scientific-boundary and implementation-readiness repairs remain in their
own sections; their detailed audit history is preserved in `caaf9e6` and `bc47d7b`.

| Finding | Disposition |
| --- | --- |
| Required: native independent simulation had no delivery boundary | U12c now binds the existing native physics/detector/integration owners without observations or saved fit records. M2 completes both admitted simulator contracts, with explicit parameter coverage and no universal schema. |
| Required: independent geometry tools waited for fitting routes | Milestones are completion groupings. U07/U08 and simulator handles arrive after viewport/numeric state; only actual fit mappings wait for their execution boundary. Snapshot transfer needs numeric state, and hBN center proposals need only a matching hBN result. |
| Optional: exhaustive route inventory blocked the first plot | U08a starts with reader/presentation and one configured simulation route, then grows before each route is enabled. U00 no longer depends on an all-route inventory. |
| Optional: spatial picking index was mandatory before measurement | Begin with vectorized visible-feature picking; add the index only for a measured miss. Prefix tables, asynchronous uploads and extra processes remain conditional too. |

The reviewer found the other objectives covered: native local import, calibrated overlays, exact
horizontal/vertical bands, center proposals, initial values and parameter scopes/statistics,
per-image reciprocal coverage, physical scene controls, staged qualification, save/recovery/export
and responsive work. The early-delivery walkthrough now requires these geometry views together;
an image reader alone cannot be presented as the completed experiment interface.

Use current owners: [geometry-only construction](../src/rasim_next/pipeline/configured_simulation.py),
[native physical inputs](../src/rasim_next/fitting/native_input.py),
[native integration](../src/rasim_next/pipeline/conditional_detector.py) and the
[fit-bound native renderer](../scripts/render_native.py). Existing
[OSC selection](../src/rasim_next/selection/osc_series.py),
[geometry qualification orchestration](../scripts/fit_osc_geometry.py) and
[prepared-native adoption](../scripts/prepare_native.py) retain the previously identified frozen
observation, qualification and prepared-versus-new-input boundaries.

No evidence justified another optimizer, backend framework, general serializer or broader numerical
rewrite. U12c fills a missing application binding; new-acquisition native preparation remains its
own explicitly qualified scientific integration task. No runtime work or numerical/performance
validation was performed by these document audits; targets still require implementation evidence.
