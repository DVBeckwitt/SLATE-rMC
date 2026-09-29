# Interactive tools

This directory contains every active, user-driven visualization entry point in the repository.
Batch simulations and proof utilities remain in `scripts/`; the immutable original-RA-SIM snapshot
under `reference/` is reference material rather than a supported application.

Install the visualization dependencies from the repository root:

```powershell
uv sync --frozen --group dev --extra visualization
```

## SLATE desktop shell

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
project archives and simulation controls arrive in later workflows.

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
**Bind CIF** and **Bind configuration** validate references through the package readers and retain
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
The 96 MiB worker result limit includes the native int32 plane, float32 display plane, thumbnail,
exact center profiles and full-detector marginals. A review holds at most 128 candidates and
512 KiB of candidate paths; metadata table input is limited to 64 KiB and 128 rows. Imports
outside those limits fail with a visible message. The acquisition SHA-256 is over the decoded OSC
byte stream; it identifies exactly the header and payload consumed by the reader, whether the file
was plain or gzip compressed.

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
