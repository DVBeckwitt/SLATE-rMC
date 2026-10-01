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
and verifies its physics revision on readback. Schema 9 projects save the independent draft,
result reference and detector inspection settings; schemas 1-8 remain readable. Reopening keeps
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
