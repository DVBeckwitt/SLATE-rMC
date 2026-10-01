# Interactive tools

This directory contains every active, user-driven visualization entry point in the repository.
Batch simulations and proof utilities remain in `scripts/`; the immutable original-RA-SIM snapshot
under `reference/` is reference material rather than a supported application.

Install the visualization dependencies from the repository root:

```powershell
uv sync --frozen --group dev --extra visualization
```

## SLATE desktop shell

**Task04 comparison is implemented but not accepted:** its ordinary-opening responsiveness
gate exceeded the 100 ms limit. See the current Task04 disposition in `docs/DESKTOP_UI_PLAN.md`.

Launch the native application from the repository root:

```powershell
uv run --extra visualization python interactive/slate_app.py
```

If the visualization dependencies are already installed in the active Python environment, the
equivalent direct command is `python interactive/slate_app.py`.

The desktop requires the declared visualization extra, including both Matplotlib plotting backends.
If a backend is unavailable, startup prints the dependency error and installation command and exits
with status 2 before opening a window. Numerical package imports remain independent of the desktop.
The prepared-input editor has scrollable pages and two-column actions for smaller logical screens;
Tab traverses controls, and Escape closes the dialog. Detector shortcuts are Ctrl+0 (fit), Ctrl+1
(native pixels) and Ctrl+B (rectangle zoom); Alt+Left/Right changes the selected acquisition.

The shell has **Fit experiments** and **Simulator** workspaces. It starts with a local, unsaved
project and an empty acquisition browser. Use **Import files**, drop local `.osc` or `.osc.gz`
files, or **Review folder** to scroll the complete bounded list of direct files before confirming;
unsupported types are labeled. Each file has its own review
status and acquisition UUID. Failed and canceled rows can be retried; removing a row leaves its
source file untouched. Exact decoded-content duplicates are labeled but admitted separately, so
repeated exposures keep separate identities. The first admitted image opens automatically. Select
another from the browser, review table or filmstrip; only two prepared images are cached, and a
cold selection reloads and verifies its decoded hash before display. The detector shows native
int32 counts with exact linked profiles. No image is loaded or calculation started at launch.
The `interactive/` directory is not part of the installed numerical wheel, so run this entry
point from the checkout.

### Comparison and display cuts

Open **Compare images** and choose acquisitions A and B. Each keeps its source identity,
exposure, raw-count units, native profiles and mask revision. **Check detector frames** uses
the existing canonical geometry preparation; linked navigation is available only when both
verified detector lattices match in LAB. Shape alone is insufficient. **Lock raw-count limits**
shares the active limits; unlocking restores the two independent settings. No exposure or
solid-angle normalization is performed.

**Pin / replace active profiles** freezes only the exact vectors and their support. Selection,
query changes and mask edits cannot mutate that reference. A stale or missing binding is
identified explicitly; reopening rebinds only an unchanged acquisition/source/mask/query.
The magnifier retains native values and exclusion reasons in at most a 41-by-41 pixel region.

Choose **Draw line**, drag its endpoints, or enter native `(column,row)` endpoints and spacing
then **Apply line**. Escape cancels unfinished drawing; **Clear line** removes the defined cut.
Sampling selects nearest native pixel centers, with ties toward the larger index. Both exact
endpoints are retained and actual uniform spacing is no larger than requested. There is no
averaging or interpolation; repeated pixels are display samples. Zero support identifies
excluded, nonfinite or outside samples, and curves do not join across gaps. The 8192-sample
limit is explicit. **Cancel preparation** uses the application's existing job owner.

**Export comparison + cuts** writes a new external PNG and matching exact CSV with both
profiles, cuts, available reference values, support, units and complete identities/settings.
Both destinations must be new and cannot overwrite project or source files. Schema 8 saves
comparison state through the ordinary project writer and reads previous project versions.
Measured/model/residual intensity comparison is unavailable until a compatible intensity
result is admitted; these controls do not start a fit.

### Reusable setup and source storage

**Setup / sources** opens a modeless review for the selected acquisitions. **Capture setup**
uses declared metadata and a currently validated numeric draft. Name and edit the captured
values in their stored units; blank values omit a reusable default. Fixed/derived fields are
listed with their existing editor restrictions. Material and mount defaults are metadata labels;
unexposed material, coupled rotations, source laws and other configuration fields stay with the
target's canonical configuration. Capture does not confirm filename/header angle proposals.

Save/load a named external `.slate-template.json` (versioned, UUID-bound, at most 64 KiB).
**Review / apply to selected** validates complete target configurations for numeric defaults and
shows each acquisition, previous/proposed value, unit and origin before Apply. Templates contain
no acquisition IDs, masks, source-specific reference paths or fit qualification. Applied values
and a canonical template-snapshot hash are copied into each project acquisition. Editing or saving
a template later cannot mutate those copies. **Load draft** uses its acquisition's saved initial
values. The existing bounded Undo/Redo owns each committed setup or storage action; no-op,
invalid and canceled reviews add no action. Numeric/reference receipts are invalidated as needed.

Inputs are referenced in place by default. **Review copy to data folder** checks current identities
and shows all resolved sources/destinations, per-file/total byte sizes, hashes and derived paths.
Copying is limited to 512 files / 512 MiB per action, with 1 MiB streaming chunks and cooperative
cancellation on the existing global worker. Destinations must be new, external to Git checkouts,
and cannot alias any input or project. Verified repeated exposures retain independent UUIDs even
when they share immutable copied bytes. Originals are never changed; removing references never
deletes any file. Failed/canceled/stale work publishes no partial project binding. Already completed
byte-verified files may remain in the reviewed folder; the message states this explicitly.

Configuration copying includes its recorded dependent CIF. Relative layouts are preserved within
a deterministic bundle. An absolute CIF path is rewritten only in a derived configuration whose
new bytes/hash and provenance are recorded and canonically revalidated. Destination readback and
the exact resolved dependent CIF are checked before any binding is committed. The storage record
keeps the original path and separately identifies the path whose raw bytes were observed; a
matched decoded OSC relocation does not assert that historical compressed bytes matched.

**Locate / review relocation** shows missing/moved/unreadable references and affected acquisitions.
OSC relocation must match the saved decoded header/payload SHA-256; CIF/configuration relocation
must match actual saved reference bytes. Same filename/shape and a valid different file do not
match. To relocate a dependent CIF, choose the matching configuration/CIF bundle so its declared
path remains valid. **Replace CIF / configuration** is the separately labeled existing new-input
action. Project schema 8 saves applied defaults/receipts, original-storage provenance and current
bindings atomically, and reads schemas 1-7. A restored missing OSC is loaded through the original
bounded image owner. A selection/edit/open/close change rejects obsolete worker publication.

U02b is implemented with focused and native functional checks, pending consolidated root review.
The inherited ordinary Open delay B002 remains open; this is not a full responsiveness acceptance.
See `docs/DESKTOP_UI_PLAN.md` and the linked open-bug list there.

### Project and detector controls

Use **Save As** to name a `.slate.json` project, **Save** to write its current state, and **Open**
to reopen one. The document records project and acquisition UUIDs, acquisition order and names,
source paths and decoded OSC SHA-256 identities, reviewed metadata, selected workspace and
detector view. It does not embed image pixels or claim a resumable solver. Edits autosave atomically
to the named project;
before a project is named, they autosave to one UUID-named draft in the local SLATE-rMC recovery
location. **Recover Draft** opens the recovery chooser. Save status is shown beside the project
controls. Closing or opening another project offers Save, Discard and Cancel for unsaved edits.
Drafts are bounded to 32 project files and 32 MiB in that location.

Reopening checks each referenced OSC source independently. A missing, unreadable or changed source
remains in the browser with its original acquisition identity. Select it and use **Relink OSC**;
the replacement must have the same decoded OSC hash. A moved source can therefore be restored
without changing its acquisition UUID. Project save never modifies its OSC sources. Portable
project archives arrive in a later workflow; the independent Simulator is described below.

Reopening also checks the saved byte identities of standalone CIFs, configurations and recorded
configuration-dependent CIFs in the background. The review and inspector show current verified,
missing, changed, unreadable or unverified status while retaining the historical path and hash.
Older projects without a recorded dependent CIF identity show **unverified**; choose the
configuration again to bind its current dependent identity. Raw OSC browsing remains available.
Rechecking a reference updates the displayed status of every acquisition bound to the same file;
an unsuccessful recheck leaves those statuses unverified until the bytes can be checked again.

Select one or more admitted rows and use **Apply to selected** to enter role, specimen/mount,
commanded incidence in degrees, exposure in seconds, detector setup and material label. Unknown
values remain explicit. The review table is display only; changes go through these validated
controls. A filename such as `sample_5d.osc` offers an unconfirmed angle proposal;
**Confirm suggestion** accepts it. **Map pasted table** and **Map CSV file** preview explicit
column mappings before applying rows by acquisition UUID or selected-row order. The supported
hBN calibrant preset is a declared identifier; it does not infer calibration from image pixels.
**Replace CIF** and **Replace configuration** validate references through the package readers and retain
their path and SHA-256. A configuration also retains the bounded hash and resolved path of its
dependent CIF; configuration loading does not structurally parse that CIF. **Export metadata CSV**
writes an explicitly chosen file outside Git checkouts with
units and provenance. These metadata inputs do not start fitting or qualify a calibration.

The detector panel supports pointer-anchored wheel zoom, drag pan, **Box zoom**, **Fit** (Ctrl+0)
and **1:1 px** (Ctrl+1). The latter means one detector pixel per physical display pixel at the
current device pixel ratio. Arrow keys pan when the image has focus; Ctrl+B toggles box zoom.
Scroll the center pane to reach the filmstrip, review table and acquisition actions when they
extend below the detector.
The pointer readout names native `column_px`, `row_px` and the exact original count, and reports
when the pointer is outside the image. Horizontal and vertical marginal position labels use the
same native viewport transform. Numeric Low/High entries accept scientific notation; **Apply**
validates them and **Auto** uses extrema cached at image admission. Linear and signed modes change
only color; positive log marks nonpositive and nonfinite texture values with a checkerboard.
The native counts and exact marginal values remain unchanged. The profile center follows the
pointer until **Pinned center** is chosen or its native column/row is entered. The two shaded
integration bands have independent widths, adjustable by numeric entry or dragging their edges.
Choose sum or mean per valid pixel and integrate over bands, the full detector, or a drawn
inspection ROI. The status line shows effective native bounds and valid support at the selected
center. Missing bins are omitted from the plots. Horizontal and vertical intensity scales can be
pinned separately with finite low/high values; otherwise each plot scales to its current data.
Project save and recovery preserve these inspection settings. Full-detector projections are
prepared with the OSC import and cached per selected native image.
**Export figure + profiles** saves a PNG of the current detector viewport, overlays and visible
horizontal/vertical plots, plus a separate `.profiles.csv` with every native profile bin. Choose a
new external PNG filename; the paired CSV uses the same stem, and existing files are never replaced.
The CSV begins with `metadata` records for the project/acquisition UUIDs, decoded OSC hash, named
data revision, native shape, effective integration bounds, measure, units, display settings and PNG
hash. Its `profile` records give native column/row index, exact value, valid-pixel support and an
explicit missing flag. Integer sums remain decimal integers; float64 means use round-trip decimal
text. A zero-support sum is marked missing even though its value is zero; a zero-support mean is
`nan`. The PNG shows the selected viewport and display contrast; the CSV values come from the
native detector data, not from plotted pixels or display scaling. A selection or profile change
during destination choice rejects the export, while an already captured pair finishes from its
immutable snapshot if the selection changes during writing.
Image, crosshair and available
marker layers can be shown independently; fitted-result overlays are unavailable without a fit.
Contrast clipping is a display choice, while detector saturation remains unknown without a
supported source threshold. Q and scattering angles remain unavailable until geometry is bound.

Interactive admission limits are 64 MiB source bytes, 32 MiB decoded bytes, 12 million pixels and
16,384 pixels per axis, further capped by the active OpenGL context's texture-size limit.
The 160 MiB shared worker result limit includes the native int32 plane, float32 display plane, thumbnail,
exact center profiles and full-detector marginals. A review holds at most 128 candidates and
512 KiB of candidate paths; metadata table input is limited to 64 KiB and 128 rows. Imports
outside those limits fail with a visible message. The acquisition SHA-256 is over the decoded OSC
byte stream; it identifies exactly the header and payload consumed by the reader, whether the file
was plain or gzip compressed.

### Independent configured simulator

**Task06 is implemented; responsiveness acceptance remains blocked.** A canonical-run garbage
collection pause produced a 112.397 ms heartbeat gap against the 100 ms limit. Three passing
30-second windows have not been established. See the Task06 disposition in `docs/DESKTOP_UI_PLAN.md`.

Open **Simulator**, then **Load configuration** with a supported `rasim-simulation-v2` YAML.
No acquisition, observation pack, fit or scene edit is required. Search and grouped forms expose
all strict-loader fields, including optional values, with units, domains, defaults and applicability.
Values are preserved across unopened groups. Edit a field, then **Validate complete draft**;
validation uses the complete canonical configuration and geometry/source builder. Invalid values
remain visible for correction. Undo/redo restores draft content with fresh revisions. Field edits
supersede active work and never start a simulation automatically.

Source count/seed define the configured source ensemble. **MC draws per source state** and
**Detector seed** are separate. Select sampled source position or conditional-position integration
explicitly. The latter applies only to MC and retains the canonical model's support requirements.
Choose native MC pixel mass, pixel-center display density, macrobin display quadrature, reciprocal
density or nominal Ewald coating. Detector execution requires the supported finite Bi2X3 stack;
generic-CIF admission alone does not provide that renderer. Output enablement, CPU/CUDA selection
and route compatibility are enforced with visible reasons. Backend failures do not select another
backend. Auxiliary-only routes require detector output to be deselected explicitly.

**Run selected outputs** uses the existing numerical owners and publishes progressive prefixes.
Float32 presentation copies are owned separately from sampler leases. Exact cursor and linked
bands use an immutable float64 quantitative snapshot; previews say **Awaiting quantitative
snapshot**. **Inspect this snapshot** holds matching image/profiles at the next available boundary;
**Follow progression** resumes adoption. Large profile queries wait for the global worker after
active MC drains. Pixel-center density and macrobin quadrature remain display approximations;
macrobin indices are labeled, with native center arrays retained in exports. Reciprocal/Ewald
panels draw at most 512 sampled Qx/Qz points with linear canonical-density colors, while full
numeric arrays remain retained. No convergence or fit qualification is implied by completion. Auxiliary-only snapshots clear and
disable the detector and exact detector profiles; their exact exports contain only selected
auxiliary arrays. Reopening a detector snapshot restores that snapshot's own values and identity.

**Export exact snapshot** writes a new external NPZ with arrays and a UTF-8 numeric manifest,
including draft/run identities, configuration/CIF hashes, seeds, prefix, route, measure, units,
backend and qualification limitations. It reopens and compares every written array before
publication. **Reopen saved snapshot** verifies and parses the same hashed bytes. Optional configured
PNGs use the selected names and output directory and encode reciprocal/Ewald density colors.
Detector PNGs label their display sampling stride; exact full arrays are in the NPZ. Existing
files and source/project files cannot be overwritten. Cancellation can retain completed figures;
choose a new output directory before retrying. Missing Matplotlib disables configured figures
with a reason; exact numeric export remains available.

**Export configuration YAML** writes a validated derived configuration with absolute CIF binding
and verifies its physics revision on readback. Schema 10 projects save both independent drafts,
result references, transfer provenance and detector inspection settings; schemas 1-9 remain readable. Reopening keeps
saved results historical and requires explicit canonical validation before execution. Input
changes cannot mutate launched drafts, held snapshots or exported results.

Desktop execution caps are 12 million detector pixels, 16,384 per axis, 256 source states, 2,048
complete rods, 32 MiB auxiliary arrays and 160 MiB per publication. Declarations above the source
cap load for editing without constructing their ensemble; validation/execution require an explicit
supported count. Physical support is never pruned. CPU reservation is capped at 2 GiB and GPU
reservation at 512 MiB, including existing retained views/history and simulation buffers; these
are conservative reservations, not GPU allocator measurements. CPU workers are explicit in [1,4],
and `threadpoolctl` bounds nested BLAS to one thread while the sole worker runs. This visualization
extra avoids process-wide environment configuration and leaves core dependencies unchanged.
Canonical setup/JIT, auxiliary sampling, snapshot reductions and file publication have declared
noninterruptible phases. Cancel rejects obsolete publication immediately; safe stop and full drain
can take longer. Explicit Cancel discards queued simulation replacements and pending profile requests as well as
canceling active work. A later Run remains an explicit action. Draft load/edit, undo/redo and result
reference admission preflight the complete prospective project through the existing 1 MiB writer
boundary and view reserve. Rejected size changes keep the prior draft, history, result reference
and selection savable; declared input is never trimmed. Ordinary close waits asynchronously for
ownership to drain.

### Independent native simulator and reviewed experiment transfer

In **Simulator**, choose **Independent native recipe**, then **Load native physics**. Load an
expanded `rasim-native-fit-physics-v1` JSON for Bi2Se3, Bi2Te3, GD1, SiD1, Clean1 or B4, or a
previously exported `slate.native-draft.v1` envelope. No experiment, observation pack or fit is
required. The configured draft stays separate. Search the grouped native fields by name, unit or
owner; scalar, vector and matrix controls edit the declared input. Specimen coordinates come
from the selected canonical model and override its baseline sites/lattice. Unsupported native
Bi disorder and Pb reflectivity fail with a reason.

Set positive integer coherent repeats (Bi conventional cells / Pb c-axis layers), integrated
rectangle width (1 is full native resolution) and optional numerical proposal mosaic. Physical
thickness is `N*c + extra` angstroms. Explicit closed surface/phase fractions preserve tiny
probabilities, must agree with canonical shares and are cleared when their shares change.
**Validate complete draft** checks physical domains and the complete source/rod/panel declaration;
**Run selected outputs** starts CPU deterministic detector integration on the shared numerical
worker with one nested BLAS thread. Existing memory/panel/source/rod caps reject unsupported work
without pruning it. Setup and individual batches are noninterruptible; Cancel acknowledges the
request and safe drain follows asynchronously. Native CUDA and auxiliary routes are unavailable.

**Inspect this snapshot** holds an immutable whole-panel float64 sum of completed additive event
batches. Every rectangle has been evaluated for those batches; the integral remains incomplete
until all independent parts finish. **Follow progression** resumes presentation. Macrobin display
indices differ from exported native-pixel centers/edges. Exact profiles/export use float64 arrays;
completion remains nominal and does not establish convergence or a qualified fit. **Export native
draft JSON** saves the complete independent declaration. **Export exact snapshot** and **Reopen
saved snapshot** use the shared identity-checked numeric writer/reader. Reopened project drafts
require explicit validation before a new run.

For transfer, prepare the selected experiment's numeric state, choose **Experiment -> configured
draft** or one of the explicit **Experiment -> native recipe** targets, then **Review experiment
transfer**. Load the matching native recipe first. Review included values and units, destination
values that are retained, omitted acquisition/fit state and incompatible mappings. A configured
Gaussian spectrum retains the native line spectrum explicitly. **Apply** rechecks the same source
and destination snapshot plus fresh configuration/CIF hashes, then creates an independent draft;
the experiment stays unchanged and Run remains explicit. Changes while a review or confirmation
is queued reject that stale transfer. Undo/redo and ordinary project save/reopen preserve draft
values, provenance and result references.

Implementation checks follow the user's [nominal-proof policy](../docs/VALIDATION.md#desktop-implementation-verification).
Task07 implements both workflows; root acceptance, full sustained timing and locked-environment
compatibility remain separate. Existing fitting validation and qualification machinery is preserved.

### Native exclusion editing

Choose **Rectangle**, **Polygon** or **Brush** above the profiles, then choose user exclusion,
beamstop, detector gap or saturation. Rectangle and brush commit on release; polygon commits
with Enter. Escape cancels the unfinished gesture and returns to Inspect. **Reinclude** clears
the selected pixels. Masking preserves the integration center; middle drag, wheel and keyboard
navigation remain available. These reasons are explicit user annotations, with no automatic
threshold or residual filtering.

**Import .npy** accepts a C-order Boolean detector-native `[row,column]` array of exactly the
admitted image shape, at most 16 MiB. True retains existing membership; False adds exclusions
with the selected reason. Import never rotates, transposes, resizes or reinterprets OSC raw
indices. Use an exclusion reason for import. File name and SHA-256 are retained in provenance.
**Exclusions visible** controls the colored overlay independently of profile inclusion.

Each completed gesture or import is atomic. Undo/redo uses fresh persistent mask revisions;
invalid, canceled and no-op edits consume no action. Session history is limited to 32 actions
and 512 KiB per acquisition, with an 8 MiB total cap. New edits clear redo; oldest history is
evicted when a cap is reached. At most 64 completed gestures per image and 2 MiB of gesture
metadata can wait for preparation. A gesture has at most 2,048 vertices, brush radius at most
256 native pixels, and the committed mask at most 8,192 canonical row spans. Exceeding a limit
is reported instead of silently simplifying membership. The project retains its 1 MiB cap.

Rasterization and expensive masked profiles use the existing cancellable job owner. During
preparation, the previous committed mask/profiles remain labeled with their revision; narrow
bands may update against that prior mask. Export waits for current profiles. Mask visibility,
contrast and navigation do not change inclusion or upload a new reason plane. A real edit
uploads its changed reason rectangle; acquisition/context changes restore the complete texture.

Schema 6 saves the source-bound native shape, reasons, revision, provenance and overlay visibility
through the existing atomic project writer. Schemas 1–5 remain readable with no exclusions.
Undo history is session-only. Inspection export includes the mask revision, visibility and
provenance together with exact masked values and support. This slice introduces no observation
preparation, fit result or calibration qualification.

## Monte Carlo detector viewer

Render the complete native detector while changing source, mosaic, sample, and detector parameters:

```powershell
uv run --extra visualization python interactive/detector_viewer.py `
  --execution-backend cuda `
  --presentation-backend opengl
```

The viewer coalesces slider input at 5 ms, preserves the full detector grid, and progressively
refines draw prefixes before publishing the requested settled result. CUDA/OpenGL are explicit;
use `--execution-backend cpu --presentation-backend matplotlib` for the software path. Press `R`
to render, `0` to reset, or `Q` to close. See `docs/EXAMPLES.md` for the scientific measure and the
complete control semantics.

Beam positions now use a smooth Gaussian profile by default (`--beam-position smooth`). Each
scattering contribution projects and integrates its conditional beam profile into native pixels;
Monte Carlo samples divergence, wavelength and mosaic. Position/divergence correlations are
preserved. The qualitative starting budget is 128 source states and 8 mosaic draws per state;
increase `--source-samples` and `--draws-per-ki` or their sliders for refinement.

Smooth profiles require an unbounded planar sample and no external-path absorption. For other
configurations, or for the previous point-sampling method, use `--beam-position sampled`.
The former sampling budget is `--source-samples 1000 --draws-per-ki 49`. Smooth mode retains
off-panel contributions and lets beam intensity leave the panel without renormalization. A
six-standard-deviation integration cutoff omits at most approximately 6e-9 of each Gaussian's
mass, in addition to numerical quadrature error. This is a qualitative preview, not a fitted or
converged result.

## Bi2Se3/Bi2Te3 Ewald viewer

Open synchronized Ewald spheres and detector planes for Bi2Se3 and Bi2Te3 at 5, 10, and 15 degrees:

```powershell
uv run --extra visualization python interactive/ewald_sphere_viewer.py `
  --mode intensity `
  --stride 2
```

Dragging focuses a lightweight preview of the selected panel; releasing synchronizes all six
panels. Use the radio buttons or `I`/`C` to switch between intensity and cylinder sections, and
`X`, `Y`, or `Z` for axis-aligned views. Press `R` to reset or `Q` to close.


## hBN calibration and center proposals

Open one native OSC acquisition, then choose **hBN calibration** in the inspector or
**Choose beam center** above the detector/experiment views. This workflow works with hBN alone.
It uses the application-wide worker; preparation and optimizer termination are separate from
scientific qualification.

1. Choose the raw dark OSC and base detector YAML, and admit the selected image. The dialog shows
   exact source/dark/configuration/CIF hashes, metadata/mask revision, native shape, separate
   pitches, base rotation, beam direction, wavelength and hBN lattice/ring declarations. Dark
   counts are subtracted without exposure normalization. Identify the image as hBN powder before
   preparing rings or fitting; center proposals also work without that powder declaration.
2. Propose a center with a detector click or native `(column_px,row_px)` values, then explicitly
   adopt it as an initial estimate. Inspection crosshairs have no geometry authority. For a spot,
   drag or enter an ROI anywhere, optionally declare a raw saturation threshold, and fit a
   constant-background elliptical Gaussian. Review its named parameters/units, valid support,
   original-data/model/residual maps and profiles, model-dependent uncertainty and limitations.
   Gaussian adoption requires a current, reliable proposal. The observed spot center is not a
   determination of the geometric beam intercept.
3. Edit the actual five seeds and narrower admitted bounds, then **Apply seeds and bounds**.
   Display tilts are degrees; stored/optimizer tilts are radians. Centers are native pixels and
   distance is calibrant-private metres. Unchanged fields retain their exact stored values.
   `soft_l1`, positive `f_scale` and `max_nfev` (1–1000) are actual launch controls.
4. **Prepare ring observations for review** uses the canonical discovery/preliminary-refinement
   owner and active fitting mask. Coordinates, ring assignments and 10-degree sector IDs are
   immutable. Exclude a discovered point only with a reason, commit the review, and **Freeze
   reviewed observations**. At least 20 points in two rings are required. A changed input/review
   creates a new revision; an old result keeps its original relationship and is shown as stale.
5. Name the next result and explicitly **Fit the frozen observation pack**. Fit consumes those
   exact observations without retracing. Cancel immediately clears pending hBN work and requests
   cooperative stop; terminal drain is reported separately. Stale/canceled completions are rejected.
6. Inspect the result to show canonical predicted curves, frozen points and per-point residuals.
   Click an observation row for detector/profile inspection. Review launch/current/fitted values,
   errors or unavailable uncertainty, per-ring count/coverage/RMS, total/max residual, rank,
   conditioning, bound contacts and solver termination. Select a current candidate explicitly;
   its qualification label remains the existing owner's actual verdict. Private distance never
   becomes shared sample geometry. **Show draft review** returns to editable exclusions.
7. Save/autosave/Open retains named results, exact packs, proposals/adoption provenance, seeds,
   bounds, review decisions and selection. Reopened Gaussian proposals need a fresh explicit ROI
   fit before adoption. Export exact result or observation JSON, or a PNG with its exact
   `.png.values.json` companion. The values file is published first. Canceled exports may leave
   completed files; choose a new destination. Import requires matching acquisition, inputs and
   the current frozen pack. Existing files and input/result destinations are protected.

Desktop caps are eight acquisition drafts, four results per draft, 180 canonical observations,
16 export references per draft, a 65,536-pixel Gaussian ROI, existing bounded undo history and
prospective 1 MiB project admission. hBN source decoding uses the existing OSC admission limits;
launches account for retained views/history against 2 GiB CPU and 512 MiB GPU limits. There is
one CPU worker and one nested BLAS thread. Missing/changed references, insufficient observations,
unreliable proposals, mismatched imports and unsupported bounds remain unavailable with reasons.
Schema 11 adds hBN state; schemas 1–10 remain readable. These nominal workflow checks do not
establish sustained interaction timing, independent convergence or new engine qualification.


## Sample-only geometry series

Choose **Sample geometry series** in the inspector. Load a supported 2-8-image OSC manifest
(for example `configs/bi2te3_osc_geometry_fit.yaml`). No calibrant acquisition is required.

1. **Admit complete series / masks** freezes exact manifest/configuration/CIF/OSC hashes, image
   IDs/angles, decoded shapes, fixed geometry/material/source references and current project fitting
   masks. Duplicate acquisitions with conflicting masks require an explicit assignment resolution.
   Without matching acquisitions the native mask starts with no user exclusions; the canonical
   all-zero edge mask still applies. Display visibility has no fitting authority.
2. Edit initial/lower/upper values and fitted/fixed scopes, then **Commit displayed controls / review**. The nine shared mechanical corrections, three detector-calibration corrections, common
   incidence delta and complete Helmert contrast coordinates are actual owner arguments. Display
   angles are degrees, storage/solver angles are radians, physical offsets are metres and detector
   references are native pixels. Unchanged fields retain exact stored values. The configured reference
   geometry/material and solver defaults are read-only. Common incidence delta cannot be fitted with
   sample-normal x tilt; optional per-image trims use the owner's zero-sum basis and prior.
3. **Prepare discovery / indexing** runs canonical discovery/indexing and replicated-track
   admission on the entire roster. Inspect original candidates and decisions. Exclude a discovered
   observation only with a reason, commit, then **Commit review and freeze exact fit-ready pack**. Original
   coordinates/covariance/identities are preserved. Insufficient tracks/image admission and unsupported
   inputs remain unavailable. Pending edits block Fit; changed inputs/masks require re-admission and
   preparation, and changed review requires refreezing. Re-admission retains historical records.
4. Name a candidate and explicitly **Fit exact frozen observations**. This calls the existing indexed-series
   owner without rediscovery/reindexing. The global worker retains one active operation and the latest
   queued request. **Cancel** clears pending sample work and requests cooperative stop at boundaries
   and residual evaluations; terminal drain is separate. Stale/late results cannot publish.
5. Inspect one immutable result ID: launch/current/fitted values and units, native observed/predicted
   points and signed pixel residuals, per-image site/track metrics, scaled rank/conditioning/bounds,
   singular values/weak direction and solver termination. Selection is explicit inspection only.
   Dataset/precision qualification, calibrated covariance/standard errors and downstream adoption are
   unavailable without the existing additional audits. A successful solver does not supply these.
6. Ordinary Save/autosave/Open retains exact packs, controls/scopes, sources/masks, four named results,
   selection and 16 export references. Schema 12 adds sample state and schemas 1-11 remain readable.
   Exact observation/result JSON and PNG with `.png.values.json` use new external destinations; values
   publish first. Import/Open verifies canonical coordinates/metrics and the existing result predicate
   without fitting. Contradictory qualification/rank, changed sources and mismatched packs are rejected.
   Exported/source paths (including retained historical sources) are protected from project overwrite.

The existing prospective 1 MiB project cap applies before publication; bounded lossless packing
preserves exact original JSON text and memory reservations charge expanded state. Preparation reserves
one 12-Mpixel workspace plus up to eight admitted native planes against the existing combined 2 GiB
CPU/512 MiB GPU caps. Execution uses one CPU worker and one nested BLAS thread; this route uses no GPU.
Preparation can drain only after its current canonical phase completes. Missing historical sources
require resolution before record validation; automatic relinking/recovery is not provided here.
The latest implementation policy uses fitting mechanisms and saved-result checks without further
fits. Nominal wiring/persistence checks do not qualify physics or sustained interaction timing.


### hBN review and result consistency

Pending inclusion or exclusion-reason edits block Fit and worker requests. Commit and refreeze
before launch, or restore the exact committed checkboxes/reasons to restore frozen readiness.
Included candidates require an empty exclusion reason. Show draft/control/history actions cannot
silently discard pending review; committed undo/redo restores its exact pack/readiness. Save/Open
retains committed decisions and historical results, with preparation still separate from Fit.

Choosing another hBN result clears the previous point table, curves and overlays. Use **Inspect
result / show ring curves** to display the new record. Each readonly row and its crosshair inspection
uses that immutable result ID; explicit selection remains separate. Imported/project records must
match the shared existing qualification predicate and exact derived residual/support/contact
statistics, not just a recomputed hash or success label. Unknown uncertainty stays unavailable.
These mechanisms are checked with saved results and no new fitting/preparation; scientific and
sustained performance acceptance remain separate.


## Joint geometry

Open **Joint geometry** from the fit inspector. Review and freeze hBN, choose a genuine retained
hBN calibration candidate, then capture it explicitly. Review/freeze and capture Bi2Se3 and Bi2Te3
separately in the sample-series editor. Each capture stays retained when switching that editor.
Missing groups block Fit; a new successful standalone fit is not required. PbI2 capture is currently
unsupported.

Edit initial values and bounds in their displayed canonical units (rad, m, native px), then commit.
Pending visible edits are saved and block Fit until committed or explicitly reset. Fixed gauge and
unobserved coordinates are readonly zero references; hBN distance is a private nuisance coordinate.
Fit calls the real owner on the shared worker. Cancel clears queued work and cooperatively drains
active work. Source/mask/metadata changes revoke dependent selection; prior records remain historical.

Results show canonical parameter comparisons, qualification/rank/condition/uncertainty and hBN,
per-image and pooled diagnostics. Select a current-launch candidate explicitly. Save/Open/autosave
retain captures, controls, pending drafts and history. Import a previous exact result or v3 report
for inspection; absent starting values or frozen coordinates are shown as unavailable. Historical
inspection works without original files.

Export exact named results, or save/reload a qualified hash-bound joint geometry handoff with its
matching specimen manifest and detector base configuration. Reload verifies every predecessor.
These operations do not adopt experiment geometry or qualify mosaic/intensity. The original project
size and shared resource limits still apply; remove unselected history if a bounded limit is reached.


### Prepared inputs / draft plan

Open **Prepared inputs / draft plan**, choose existing native physics, observations and refinement
plan JSON, then **Load prepared set**. Profiles show frozen raw/background/signed corrected counts;
select a row to inspect exact values and its full covariance row. Marginal sigma is labeled separately.
The inputs tab shows original hashes and material/source/geometry provenance.

Edit the current start, declared search bounds or supported present stage method/budget fields,
then **Commit displayed draft**. Canonical units, physical bounds and declared fixed/active scopes
are retained. Pending text is autosaved and restored by Save/Open. Undo/redo restores exact
descriptions; reload original sources to restore arrays. Historical definitions are readonly.
**Reuse compatible historical starts** aligns by owner/name and explains incompatible changes.
Load an authoritative current plan to review new/removed definitions; unknown fields stay inert.
**Export committed plan** writes the exact committed plan outside the repository.
Run and indexed adoption are unavailable pending R4. Structural draft checks do not qualify a fit.


### Portable archive / exact exports

Open **Portable archive / exact exports**, select acquisitions/products, then **Review selection**.
Inspect exact identities, sizes, unavailable references and original qualification. Missing
selected predecessors block self-contained export; resolve them or explicitly change the selection
and review again. **Export reviewed archive** writes a new external `.slatezip`. State, selection
or source changes require a new review.

**Import archive elsewhere** creates a new `<archive>-reopened` directory under a chosen external
parent. Existing destinations are protected. Review its inventory, then **Open imported project**
through ordinary pending/dirty-state handling. Original snapshot/dependency bytes remain beside
a distinct project with explicit storage provenance. Keep that directory together or move it
through another archive. Windows relocation is established; cross-OS path interpretation is not.
Cancel and Close use the shared worker and clean unfinished owned staging. A completed directory
can remain after late GUI cancellation or stale admission.

Product export buttons expose existing owners. Prepared **Export exact measured data** writes
signed counts, frozen rows, marginal errors and full covariance to a new numeric NPZ. Configured
figure declarations still name their original explicit output directory; review new destinations
with that owner before figure export. Archiving computes no image or fit. Sample history stays
inspectable with live geometry revalidation marked unavailable; original qualification is retained.
Native fit Run/adoption remains disabled.


## Named attempts, independent copies and recovery (R7b)

Open **Attempts / independent copy / recovery** in the experiment inspector. Each row
is bound to its immutable owner and result/description identity. **Rename display** and
**Select for inspection** save separate metadata; scientific names, launch records, units,
definitions, hashes and qualification remain exact. Undo/Redo covers these metadata edits.
Inspection selection is separate from the current owner's scientific candidate selection.
**Open existing owner** locates the stable identity in the current route: prepared starts
use the existing explicit compatibility review; hBN requires its acquisition to be selected;
sample/joint retain their existing candidate-selection validators. Configured/native retained
outputs reopen their exact saved arrays, including historical snapshots, without running a
simulation. The current simulation draft and selected-output reference remain separate.

Simulation output history retains at most eight exact references per route. At the limit,
remove an old reference explicitly before admitting another output. Current selected references
cannot be removed from this catalog. Removal changes project references, not external files.
Names are limited to 128 entries of 256 characters; all attempt metadata is limited to 128 KiB
within the existing 1 MiB project admission. Removed owner history may leave a name with no
current row; it does not manufacture an attempt or restore a deleted scientific record.

**Review independent copy** shows a fresh project UUID, inherited owner/history identities,
the exact source snapshot identity and complete pending/current state. **Publish reviewed copy**
requires a new external destination and uses no-overwrite publication. **Open published copy**
uses ordinary dirty-state handling and rechecks the exact published bytes. Edits, selections,
pending text, masks, configurations and Undo after Open belong to the new project; recovery
uses its new UUID. Original scientific owners and result identities remain inherited inspection.
Active reuse must pass the existing route's checks. Exact referenced files remain explicitly
shared; use the portable archive when independent file storage is required.

**Recover Draft** opens bounded recovery review. Choose a valid candidate by project UUID,
name, source identities and age, then **Open reviewed recovery**. Invalid/partial candidates
are listed with reasons. Changed bytes require another review. Source checks and exact Relink
remain explicit; restoring a project never resumes a solver. Pending text, selected inspection
and original result qualification are retained.

Recovery is limited to 32 drafts of at most 1 MiB each. Autosave remains enabled. Existing
recovery bytes can be replaced/retired only when their exact owned or reviewed SHA256 matches;
unknown/changed drafts remain for explicit review. Saves flush a unique sibling temporary file
before one atomic publication, with cooperative cancellation before publication. A failure
before publication preserves prior saved bytes; cancellation after publication may leave a
complete external copy which is not adopted by stale work. Cleanup is limited to owned paths.
This is not a filesystem-wide transaction, concurrent-writer lock or power-loss guarantee.

R7b's focused desktop and publication checks establish implementation behavior only. No new
fit, prediction, observation preparation or scientific/performance campaign was run. R4 Run/
adoption, R5 execution and R6 preparation remain unavailable. B014's equivalent 64-source
long-reference trigger met the original 12-second autosave and 8-second drain limits; the
original timeout remains unexplained and root-owned, rather than a claimed deadlock/data-loss fix.


## Acquisition review and frozen-output handoff (R6)

Open **Acquisition preparation / frozen outputs**. Select a current imported OSC and one
of the explicitly listed preparation scopes. **Review current inputs** reads exact source
bytes and decoded OSC identity, native shape and recorded reference hashes on the shared
worker. The receipt retains the native mask, metadata revision, setup receipt, initial
proposals, current geometry-owner digests and original qualification labels. Pending notes
are proposals only. Dark/mask acquisition links are reviewed as identities; no subtraction,
background transfer or numerical membership is created by this panel.

**Numerical Prepare unavailable** shows the concrete missing recipe or execution boundary.
There is no currently admitted complete new raw-acquisition recipe in this workflow.
The existing frozen-adoption script verifies frozen measurements; it does not specify a
new background or signal/control layout. Its execution/publication integration remains
outside this delivery. Future parameter rosters are not prerequisites for input review.

Choose a retained prepared description and **Inspect frozen counts / full covariance**.
The existing typed owner reads original physics, observation, plan and array identities,
including explicitly relocated archive bytes. Original raw/background/signed corrected
counts, validity and frozen row IDs remain ordered; the full covariance is retained.
Marginal uncertainty is displayed separately and never replaces cross-row covariance.
The complete observation JSON, including extra provenance and background/control metadata,
is displayed without discarding unknown fields. Selecting another description clears
previous displayed arrays until that description is inspected. Inspection does not replace
the current description, original parameter definition, pending text or result history.

Frozen-output selection is independent of imported OSC selection. A saved decoded NPZ
cannot be silently identified with another OSC by matching sample names. The original raw
kind/hash, physical binding and frozen geometry remain authoritative; current geometry is
not adopted or reprojected. **Open prepared draft / stage editor** opens the existing editor
with the selected immutable description. Historical starts require explicit compatible
reuse through that owner. Current engine capabilities are displayed separately from
original definitions. Fitting Run and indexed adoption remain unavailable.

Schema 18 stores pending selections/notes, selected description SHA256 and at most eight
immutable input review receipts, within 128 KiB and the existing project admission.
Each receipt is at most 32 KiB; notes are at most 8 KiB. Remove an old review explicitly
at the limit. Scientific output/history stays with NativeFitSession and existing named
attempts. Save/Open, autosave, independent copies and recovery use existing project state.
Changes to source, mask, calibration, geometry or controls cancel stale review work;
historical receipts remain inspectable and never assert current scientific readiness.

Archive **views** includes acquisition reviews and their exact verified predecessor bytes.
Select all acquisition/geometry/prepared dependencies when retaining those reviews.
Import preserves receipt/owner identities and uses the existing explicit storage map;
original receipt paths remain historical provenance. Missing historical dependencies
block a self-contained export until that review is removed or views are deselected.
Review generates no external scientific files. Cancel/Open/Close use the one shared
worker and existing drain; previous reviews and drafts remain intact.


## Declared native stage review and editing (R5)

In the prepared draft editor, select a declared stage by its saved name/order. The
review shows canonical owner/name/unit identities, declared and canonical indices,
active/held/fixed roles, preceding stage, frozen input hashes, source/integration
choices and exact numerical declarations. Stage names do not establish scientific
capability. Derived thickness, phase/parent fractions and candidate-dependent inactive
directions remain with the existing material owner; no candidate values are inferred.

For supported definitions, edit an earlier stage's **Active canonical names** as a
JSON list and its declared historical-guard boolean. These are unvalidated draft
proposals. Existing method/budget, finite-difference and parameter controls remain.
**Commit displayed draft** uses existing non-solving validation; it does not admit
a launch. Global fixed/gauge definitions and the final stage's free-coordinate
ordering stay read-only. Unknown settings/coordinates retain their original data
and explicit incompatibility. No stage template, bound, prior or gauge is invented.

The native owner profiles nonnegative acquisition scale; guarded historical SLSQP
owns its literal scale coordinate. Frozen background/full covariance remain exact;
there is no fitted-background control or new transfer recipe. Bi native disorder
is unavailable. Pb coordinates are limited to the actual loaded phase/parent roster;
source support is not measured stage qualification.

Schema 19 retains schemas 1-18 and stores selected stage in NativeFitSession. Old
sessions default to no saved stage selection. Pending text, immutable descriptions
and history use existing Save/Open, autosave, Undo/Redo, independent copy, recovery
and portable archive owners. Invalid pending active-list text stays inert until
explicitly corrected and committed. Stage/upstream changes invalidate pending owner
work through the existing lifecycle; historical records are not upgraded.

Native stage-result import is unavailable: refinement writes its result in the
script; rendering that result constructs an evaluator. There is no independent
typed read-only native stage-result owner. Initial/candidate/selected outcomes,
rank/covariance, weak directions and qualification cannot be manufactured from
stage declarations. Existing genuine hBN/joint geometric diagnostics remain exact
in their inspectors, distinct from native mosaic/ordered/disorder results. Run and
indexed adoption remain unavailable pending R4. U11a/b/c execution is not accepted
by this independent editor delivery.


### Full simulation output history (B024)

Configured and native routes retain at most eight distinct exact output references.
If an export completes while that route is full, the panel refuses project admission
and explains that an old reference must be removed explicitly. The exported snapshot
remains at the reported external path; the current draft, selected output, histories
and prior savable project state stay intact. There is no silent eviction.

Open **Attempts / independent copy / recovery**, select an older simulation output
that is not the current selected output, and choose **Remove old simulation reference**.
This removes the project reference and leaves its external file untouched. A subsequent
export can then be admitted and saved through the ordinary workflow.
