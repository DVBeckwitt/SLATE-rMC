# Native desktop UI grand plan

Status: accepted feature scope; planned implementation, not delivered functionality.
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

This is the implementation plan and sole task checklist. All tasks remain unimplemented. Existing
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
to continue already authorized work. This update delivers planning only; implementation has not started.

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
| [ ] U02: detector viewport | U01a | Add pan/zoom, native pixels, signed/linear/log contrast and retained layers. Corner/interior fiducials, pointer and marginal axes remain aligned through resize/DPI changes; cursor/camera/contrast cause zero image uploads. |
| [ ] U03: exact marginal profiles | U02 | Add follow/pin crosshair, independent bands, sum/mean/full-image/ROI modes and support labels. Compare small direct reductions at edges, gaps and signed/nonfinite values; measure preparation and warm latency. Add prefix caching only for a measured need and verify its subtraction error. |
| [ ] U02a: multi-file import and metadata | U01b/U02 | Add lazy filmstrip, folder candidate review, roles, angle/exposure/material/CIF inputs and bulk metadata mapping. Mixed valid/corrupt files preserve successes; duplicates differ from repeated exposures; incomplete metadata does not block inspection. |
| [ ] U04: masks and regions | U03/U01b | Add rectangle/polygon masks with reasons, a persistent revision and bounded undo. Publish mask/profile generations atomically; mask display visibility and fit inclusion stay separate. |
| [ ] U04a: brush and imported masks | U04 | Add brush gestures and supported mask import with native orientation/shape validation. One gesture is one compact undo item; repeated strokes and rebuilds stay within the resource budget. |
| [ ] U06: comparison and cuts | U03/U02a | Display two images with compatible linked pan/limits, pinning, magnifier and explicit straight-line sampling. Keep exposure/units visible and masks/support matched; later result bindings reuse this view. |
| [ ] U14b: inspection export | U03/U01b | Export the current detector figure and exact profile values/support/units to an external destination. Reopen exported values and compare to the named data revision; unrelated fitting stages are not prerequisites. |

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

#### U02 sustained native-loop qualification packet (2026-09-28; supervisor review pending)

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
preserved and their cause is still unknown. U02 completion remains for supervisor review;
the plan checkbox is not changed here.

### M2 — independent simulator

Likely ownership: optional forms/controller and the current detector worker. Configured simulations
reuse `pipeline/configured_simulation.py`; native drafts reuse `fitting/native_input.py`, supported
material bindings and `pipeline/conditional_detector.py`. Keep these actual contracts distinct.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U09a: numeric parameter state | U01b/U08a | Provide explicit parameter descriptions, unit conversion, provenance, validation, undo and immutable launch snapshots. Reuse core constructors; simulation drafts and fit seed packs retain distinct types. Verify round trips for an admitted configuration without inventing fitted coordinates. |
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
