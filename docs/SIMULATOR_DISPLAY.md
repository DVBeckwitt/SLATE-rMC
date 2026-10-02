# Simulator display and contrast

The Simulator initially uses Linear with **Auto 99%**. The upper display level is
the 99th percentile of finite positive display cells; zeros and invalid cells do
not determine that percentile. This is a viewing heuristic, not a change to
intensity, sampling, normalization, physical integration or numerical qualification.
Linear cannot reveal every order of magnitude at once. Positive log remains an
explicit choice. Full range uses finite extrema, including strong peaks.

## Controls and stability

- **Upper / Apply** sets a manual display limit. Low is also editable.
- **Exposure** changes display limits only; increasing it lowers Upper and can
  reveal weaker signal while saturating bright peaks.
- **Auto 99%** returns to automatic contrast. Once positive signal arrives, levels
  are held separately for each display bin size and mode while that run refines.
  A new run gets fresh automatic levels. Empty initial frames do not lock a range.
- Manual levels survive progressive frames and new runs until Auto is selected.
  Saved project display levels reopen as explicit restored/manual levels; Auto
  can then be restored. Exposure is represented by the saved resulting levels.
- The visible caption identifies Auto/manual status, display bin size and an
  approximate percentage of **positive display cells** above Upper. The clipping
  estimate uses a bounded CDF with 0.1-percentile steps. Signed Low may also clip
  negative values; the tooltip identifies this additional possibility.

Automatic Linear on signed data uses its full finite range. Signed mode uses
opposite finite limits around zero. Positive log hides nonpositive values and
requires positive levels in the normal float32 shader range. Tiny signals below
that range remain identified as such; no epsilon is added to scientific pixels.
Entirely invalid bins remain checkerboard. User choices are not silently replaced
by Positive log on publication.

## Fit-to-window presentation

Simulator results prepare a bounded sequence of 2 x 2, 4 x 4, and larger disjoint
pixel-sum display arrays on the existing worker. Summation uses float64; textures
are float32 presentation arrays. Edge bins may contain fewer input cells. Finite
included values conserve their sum before float32 texture rounding. Excluded and
nonfinite cells contribute no signal; a bin with no valid cells remains invalid.
The nearest texture grid is selected finely enough to avoid minifying its bins at
the viewport's device-pixel scale. This retains isolated hits that point sampling
of a full detector texture can miss. There is no blur or invented fine detail.

The shader maps original detector coordinates directly to their sum-bin index,
including partial edge bins. Original image shape controls zoom/pan, cursor, masks
and overlays. Native 1:1 zoom uses the original full-resolution texture. A saved
macrobin result uses its existing input-cell coordinates; these display reductions
do not create new physical detector coordinates or a new integration rule.

Displayed limits apply to **sums per display bin**, with the original observable's
units. For pixel-center density displays, this is a sum of density samples, not
an independently integrated mass. Auto adapts once when a different bin size is
selected. Manual limits retain the same sum-unit values across zoom; clipping and
the bin-size caption show the changed presentation. Cursor readouts, quantitative
profiles and exact result export always use the original arrays. Display levels
and contrast statistics are never included in exact scientific result export.

The experiment/OSC viewer retains its existing presentation and contrast behavior.
This repair uses the shared texture owner only when Simulator sum levels are
explicitly supplied. Configured and native resource ledgers reserve the additional
display/preparation memory, and the 160 MiB publication bound remains enforced.

## Delivery scope and evidence limits

This display repair does not change source/draw counts, numerical methods, fitting
hooks or physics. The separately paused higher-sampling, Qz-guide and held-drag
work remains uncommitted in the UI checkout; it is not accepted by this fix.
Existing numerical/runtime qualification limits and acceptance rows remain open.

Focused checks reused genuine saved native results and the retained method study.
Six study arrays supported the 99th-percentile tradeoff; GUI before/after Linear
views use identical input values. Explicit non-scientific fixtures covered sparse
hits/outliers, odd edges, masks/nonfinite data, empty/tiny/signed cases and progress
stability. Actual Save/Open/reopen and exact-array export were checked. These are
presentation/I/O checks, not new physical intensity calculations, accuracy or
refresh-speed qualification. Restart the running UI with the existing launcher
to load this Python change, then reopen the project/result. Saved levels remain
manual until Auto 99% is selected.
