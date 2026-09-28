# Native desktop UI grand plan

Status: accepted feature scope; planned implementation, not delivered functionality.
Updated: 2026-09-28. Implementation-readiness audit baseline: `caaf9e6`.

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
- Batch peak markers and ring polylines; keep label count bounded by visibility/selection. Use a
  native-coordinate spatial index for pointer picking rather than scanning every feature each
  mouse event. Rebuild that index on geometry changes, not camera movement. Drawing culls never
  alter fit membership. Screen-pixel min/max envelopes may reduce dense 1D drawing while preserving
  extrema, missing-data gaps and exact underlying values/exports.
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

## 11. Build sequence and acceptance checklist

This is the implementation plan and sole task checklist. All tasks remain unimplemented. Existing
U identifiers are retained, with smaller lettered slices where the previous task was too broad.
Execution follows the order and dependencies below, not numeric ID order. The main agent is the
only writer; independent reviewers can inspect interfaces and completed changes.

### Releases a researcher can use

| Milestone | User-visible deliverable | Exit condition |
| --- | --- | --- |
| M0: choose the implementation | A measured rendering choice and a capability map grounded in current APIs | Alignment, buffer ownership and resource limits established before broad UI work |
| M1: inspect real data | Import one/many OSCs, exact marginal profiles, masks, comparison, save/reopen and basic export | A researcher can inspect and export an image without configuring a fit |
| M2: run the simulator | Independent configuration loading, supported parameter forms and explicit preview/quantitative output | Load, change, run, inspect and save a simulation without an experiment |
| M3: fit geometry | Reviewed observations, numeric starting values, beam-center tools, fitted overlays and statistics | hBN first, then supported sample-only and joint routes each work end to end |
| M4: edit the experiment visually | Reciprocal coverage and textured beam/sample/goniometer/detector scene with synchronized controls | Click-to-zoom and physical edits preserve canonical geometry and parameter ownership |
| M5: fit prepared native experiments | Supported mosaic, ordered and disorder workflows using existing prepared recipes | States, observations, measure, covariance and qualification remain intact |
| M6: prepare new native experiments | One explicitly admitted new-acquisition preparation path, then additional supported recipes | Imported counts become reproducible native observations under a qualified measurement contract |
| M7: complete daily use | Sensitivity, supported uncertainty, templates, archives and integrated release checks | Complete user journeys and declared performance/resource criteria pass |

M1 and M2 are useful deliveries on their own. A first geometry fit uses numeric inputs; it does not
wait for every 3D handle. M5 can use an existing valid prepared experiment without a new geometry
fit. M6 is required for the promised later-stage workflow on newly imported data; opening an
existing recipe alone does not complete that requirement.

### M0 — settle expensive decisions first

Likely ownership: this document, existing optional viewer components and visualization dependencies.
Keep the rendering comparison finite: one baseline, at most one alternative, and one focused repair
pass before recording the remaining bottleneck and a scoped next step. Preserve chosen live
components; remove discarded alternatives and all temporary checking code.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U08a: capability inventory | None | List each admitted simulation/fit route, acquisition requirements, parameter names/units/scopes, seed/bounds support, observation inputs, output schema and safe-stop limits. Include hBN wavelength/ring assumptions, calibrant-private distance, one-axis indexing, joint-series requirements and prepared-native limits. Every promised enabled control maps to an actual input; missing boundaries get an explicit task below. |
| [ ] U00: rendering decision | U08a | Use two existing native images, exact band profiles, batched overlays and a textured plane with bounded background work. Declare reference hardware/versions and numeric CPU/GPU budgets; compare alignment, cold/warm interaction and peak memory under section 10. Choose one compositor/plot path before expanding features; no full application or optimizer is needed here. |

### M1 — a useful detector reader

Likely ownership: `interactive/slate_app.py`, narrow optional project/I/O/presenter/profile helpers,
`src/rasim_next/io/osc.py` only if a measured boundary change is necessary, and
`interactive/README.md`. Do not allocate one module per row by default.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U01: shell and project identity | U00 | Provide a documented repository launch, the two workspaces, stable acquisition IDs and explicit empty/loading/error states. Widgets contain no scientific model state; normal numerical imports remain GUI-independent. |
| [ ] U01c: shared job lifecycle | U01 | Introduce bounded worker ownership, queued/running/cancel-requested/terminal states, generation rejection and responsive close. Exercise a small loading/preparation operation and late completion; acknowledgment and safe stop remain distinct. Numerical owners gain only their own later cancellation hooks. |
| [ ] U01a: first file import | U01c | File picker/drop imports one OSC/OSC.GZ asynchronously through the existing orientation boundary. Show native counts before scientific metadata is complete; verify tracked non-square inputs, corrupt-file handling and no second rotation. |
| [ ] U01b: save, reopen and draft recovery | U01a | Own one versioned numeric project schema with file identities and atomic save/autosave. Verify an interrupted save, moved/missing source, relink identity and close/reopen; originals remain unchanged and solver resume is never implied. |
| [ ] U02: detector viewport | U01a | Add pan/zoom, native pixels, signed/linear/log contrast and retained layers. Corner/interior fiducials, pointer and marginal axes remain aligned through resize/DPI changes; cursor/camera/contrast cause zero image uploads. |
| [ ] U03: exact marginal profiles | U02 | Add follow/pin crosshair, independent bands, sum/mean/full-image/ROI modes and support labels. Compare small direct reductions at edges, gaps and signed/nonfinite values; measure preparation and warm latency. Add prefix caching only for a measured need and verify its subtraction error. |
| [ ] U02a: multi-file import and metadata | U01b/U02 | Add lazy filmstrip, folder candidate review, roles, angle/exposure/material/CIF inputs and bulk metadata mapping. Mixed valid/corrupt files preserve successes; duplicates differ from repeated exposures; incomplete metadata does not block inspection. |
| [ ] U04: masks and regions | U03/U01b | Add rectangle/polygon masks with reasons, a persistent revision and bounded undo. Publish mask/profile generations atomically; mask display visibility and fit inclusion stay separate. |
| [ ] U04a: brush and imported masks | U04 | Add brush gestures and supported mask import with native orientation/shape validation. One gesture is one compact undo item; repeated strokes and rebuilds stay within the resource budget. |
| [ ] U06: comparison and cuts | U03/U02a | Display two images with compatible linked pan/limits, pinning, magnifier and explicit straight-line sampling. Keep exposure/units visible and masks/support matched; later result bindings reuse this view. |
| [ ] U14b: inspection export | U03/U01b | Export the current detector figure and exact profile values/support/units to an external destination. Reopen exported values and compare to the named data revision; unrelated fitting stages are not prerequisites. |

### M2 — independent simulator

Likely ownership: the optional application forms/controller and the current detector worker;
`pipeline/configured_simulation.py` remains the configuration/scientific authority.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U09a: numeric parameter state | U01b/U08a | Provide explicit parameter descriptions, unit conversion, provenance, validation, undo and immutable launch snapshots. Reuse core constructors; simulation drafts and fit seed packs retain distinct types. Verify round trips for an admitted configuration without inventing fitted coordinates. |
| [ ] U12: configured simulator | U02/U03/U01c/U09a | Load an existing supported configuration, run the current preview/quantitative owner and save/reopen the independent draft. Show measure, backend, prefix/progress and failure state; no experimental image, fitted observations or 3D editor is required. |
| [ ] U12a: complete supported parameter forms | U12 | Expose all supported selected-model configuration fields in searchable grouped forms, including source, instrument, structure, mosaic, optics and execution. Compare field coverage with the capability inventory; unsupported combinations explain why. Keep configuration import/export working. |
| [ ] U12b: quantitative inspection | U12 | Bind exact cursor/profile/export values to immutable float64 snapshots with their draw prefix. Preview progression cannot mutate or relabel them; check direct reductions, lease consumption and bounded snapshot/upload memory. |

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
| [ ] U05d: geometry-derived center proposals | U10/U10a | Offer supported ring/sample estimates beside direct/manual proposals, including their physical interpretation and coupling. Applying a shared estimate lists affected acquisitions; derived evidence is not counted twice. |

### M4 — reciprocal and experiment views

Likely ownership: narrow optional scene/reciprocal helpers and the existing parameter/result
controller. These are new presentations of the same model, not new geometry implementations.

| Task | Dependencies | Deliverable and focused verification |
| --- | --- | --- |
| [ ] U07: reciprocal geometry preview | U02/U09a | Show each image's on-demand draft/saved coverage and cursor Q in the declared frame; link available features. Compare nondefault wavelength/direction and off-panel cases with canonical APIs; reject stale jobs and fixed illustrative Ewald data. |
| [ ] U08: textured experiment scene | U02/U09a | Show canonical beam, sample, goniometer axes/pivots and detector with actual image/available overlays. Click-to-zoom, context return and camera presets work; verify texture corners, compound transforms, context recreation and no orbit-driven image upload. |
| [ ] U09: synchronized physical handles | U05a/U07/U08/U09a/U10 | Connect callouts, numeric fields and constrained arcs/arrows to the existing parameter state. One gesture is one undo; fixed/derived/unsupported coordinates stay explained. For each admitted route, compare handle edits with the actual launched vector; the scene cannot enable an unsupported fit scope. |
| [ ] U09b: experiment/simulator transfer | U09/U12a/U12b | Copy a compatible experiment snapshot into an independent simulation draft, or adopt a compatible simulation into a new experiment draft. Show exactly what transfers; verify units, provenance and unchanged source projects. |

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

The first implementation packet is U08a followed by U00: complete the precise capability/control
map, choose a reference machine/resource budget, then build and measure the small rendering slice.
Stop expanding that slice once the choice is justified. The next packet is U01/U01c/U01a so a user
can launch the app and open a real detector image. No full fit, parameter sweep or full-image
scientific campaign is required to make the rendering decision.

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

A separate read-only reviewer audited the usability draft against the current code on 2026-09-28.
The main writer checked the findings and revised this plan. The following are planning repairs;
they do not establish that runtime performance or missing implementation is already resolved.
The independent reviewer checked the revision and confirmed the findings were incorporated;
follow-up clarifications limit U00's scope and apply frame cadence only under active redraw demand.

| Finding | Plan correction and implementation evidence still required |
| --- | --- |
| Initial-value controls can exceed current fitting APIs | Sections 2/6, the U08a inventory, route-specific U08b/U08c/U08d and U09 require explicit seed/bounds handoff. Every enabled field must reach the optimizer; existing gauges/defaults remain authoritative. |
| Current GL presenter lacks viewport/layer alignment | Sections 9/10 and U00/U02 require one native transform, signed display, compatible overlay composition and DPI/alignment checks. Reuse narrow texture helpers, not the complete opaque widget unchanged. |
| Progressive float32 images cannot provide promised exact profiles | Section 4 and U12 bind inspection to named immutable quantitative snapshots, with explicit pending/historical states and no per-cursor full snapshot allocation. |
| Cancel/close promises exceed current safe interruption points | Section 10, U01c and the route-specific execution tasks require job lifecycle states, narrow execution hooks, measured noninterruptible phases and responsive safe shutdown. |
| Ewald illustration is not general acquisition geometry | Section 9 and U07 exclude its fixed formulas/textures from scientific data paths; verify canonical geometry at nondefault wavelength/direction. |
| Cache-only limits miss peak copies and history | Section 10 and U00/U04/U15 add a complete resource ledger, globally bounded publication, compact undo and reconstructible-buffer eviction. |
| 60 fps claim and 33 ms criterion disagree; cold/tail latency unbounded | Section 10 separates cadence, event latency, tails, preparation and stop time, with declared workloads/hardware and early measurement. |
| Packaging and early save/recovery ownership unclear | Section 9 and U01/U01b establish repository launch, schema and basic recovery before editing workflows; every slice saves its outputs and U14 completes archive/export coverage. |

Source boundaries checked: [detector presentation](../interactive/detector_viewer.py),
[illustrative Ewald viewer](../interactive/ewald_sphere_viewer.py),
[Monte Carlo presentation/quantitative snapshots](../src/rasim_next/pipeline/source_averaged_detector.py),
[joint geometry](../src/rasim_next/fitting/joint_geometry.py),
[hBN calibration](../src/rasim_next/fitting/hbn.py),
[indexed series](../src/rasim_next/fitting/indexed_series.py),
[OSC I/O](../src/rasim_next/io/osc.py) and [packaging](../pyproject.toml).
Recorded experiment history supplied no combined-UI timing evidence. No discovery/replay campaign
is justified for this documentation audit; future bounded measured comparisons follow the current
assessment and experiment-memory policies.

### Implementation-readiness audit and repair

A second read-only audit at `caaf9e6` checked whether the plan could be executed in small usable
deliveries. It found five sequencing/integration gaps; section 11 now addresses each one.

| Gap | Concrete repair |
| --- | --- |
| No task creates reviewed geometry observations from new images | U05b/U05c own canonical discovery, admissible review, frozen-pack persistence and exact fit consumption. Preserve qualification orchestration; no fit-time rediscovery. |
| Prepared native inputs were mistaken for preparation of new acquisitions | M5 opens existing supported recipes; M6 separately declares and implements a scientifically qualified observation/background/covariance preparation path. |
| Simulator waited for the entire geometry editor and fit APIs | U09a and U12/U12a/U12b deliver independent configuration, parameter forms and quantitative inspection in M2. |
| First fit required every physical handle and several fitters at once | Numeric state is early; hBN, indexed-series and joint routes have separate input/execution/result slices. M4 attaches scene controls to that existing state. |
| Export/persistence waited for unrelated advanced features | U01b and U14b deliver early saving/export; observations, simulator drafts and fit results persist in their own slices. U14 integrates complete archives. |

Additional source checks confirmed the fixed hBN wavelength/ring assumptions and the current
single-axis OSC-indexing admission rule; the capability inventory must expose those limits.
Relevant live boundaries include [OSC selection](../src/rasim_next/selection/osc_series.py),
[geometry CLI orchestration](../scripts/fit_osc_geometry.py),
[prepared native adoption](../scripts/prepare_native.py),
[native observation records](../src/rasim_next/fitting/native_observations.py),
[native physics inputs](../src/rasim_next/fitting/native_input.py) and
[measurement regions](../src/rasim_next/measurement/continuous_regions.py).
No application implementation or numerical/performance validation was performed by this audit.
The independent reviewer checked the revised sequence and identified one final route correction:
native geometry adoption currently consumes an indexed OSC fit record, not the distinct joint
handoff artifact. U11d now names that boundary and depends on U10a; joint adoption remains conditional
on its own supported binding. No remaining material sequence or qualification issue was reported.
