# SLATE OSC + hBN for Windows XP

A portable native Win32 viewer and hBN detector calibrator. Run `SLATE-OSC-XP.exe`.
No Python, Qt, installation, network connection or GPU is required. Target: 32-bit
Windows XP on an SSE2-capable processor (Pentium 4 or later). Use a 1024 by 768 or
larger display. Allow memory for 36 MB per 3000 by 3000 source/dark image, display
buffers and a new image during replacement; 256 MB or more system RAM is advisable.
The supplied build has been checked on modern Windows, not executed on an actual XP
installation. XP compatibility and workstation performance require that final check.

Extract the entire ZIP to a local folder on the XP computer, then double-click the
EXE. The distribution includes `demo/hBN_calibrant_5m.osc` and `demo/darkImg.osc`.
With these files and the built-in preset, Calculate hBN should report QUALIFIED,
column/row tilts about `-0.383836 / -1.344322 deg`, beam center
`(1452.694443,1596.706498) px` and calibrant distance `75.966066 mm`.

## Viewing

1. Choose **Open OSC**. The input must be an uncompressed R-AXIS `.osc`; decompress
   repository `.osc.gz` files first. A dark is optional for general viewing.
2. **Fit image**, drag to pan, and use the wheel or Zoom buttons to zoom. Click to
   select a full-resolution row/column profile. Plots use a min/max envelope to keep
   narrow peaks visible when many samples share one screen column.
3. Shift-drag selects a rectangular ROI. The status reports inclusive column/row
   limits, pixel count, signed sum and mean. Interrupted selections are discarded.
4. Auto contrast chooses the 1st and 99.5th percentiles of a small mean-binned preview.
   Black/white levels and Log display affect presentation only. At low zoom the
   display uses that preview; cursor values, profiles and ROI statistics always use
   original integer pixels. Log display compresses `max(value-black,0)`.
5. **Open dark** requires exactly matching dimensions. **Show raw - dark** subtracts
   it at scale one for display, measurements and numeric exports; there is no exposure
   normalization. Negative values remain signed. Clear dark is in the File menu.

Coordinates are SLATE detector-native: clockwise from raw OSC exactly once on import,
column increases right, row down, and integer coordinates are zero-based pixel centers.
This differs from the old OSC_Reader's raw-order display. All tools and overlays agree
on this orientation. High-range 16-bit OSC values are decoded into signed 32-bit storage.
Inputs are limited to 25 million pixels and 16384 pixels per axis. File length,
signature and endian-specific dimensions are checked before publication. A failed or
canceled load preserves the previous image. ANSI XP file paths must fit MAX_PATH.

## hBN calibration

Open the hBN image and its matching dark. Verify the visible geometry inputs. The
provided preset matches `configs/joint_hbn_crystal_geometry_fit.yaml`: 0.1 mm pitches,
1.5406 A wavelength, center `(1453.12,1596.422)`, distance 74 mm and zero initial tilts.
The hBN lattice is `a=2.504 A, c=6.661 A`; all ten settings can be read explicitly
through **File > Load instrument preset**. The built-in defaults match `instrument.ini`;
the file is not silently loaded at startup. Do not transfer the preset to another
instrument without verifying its geometry and wavelength.

**Pick initial center** uses the next image click. **Calculate hBN** traces rings
`002,100,101,102,004`, then fits two detector tilts, two beam-center coordinates and
the calibrant-private distance along the beam. The initial center/distance must be
close enough to place the rings in their search windows. Tilts use active local-x
(column) followed by current-local-y (row): `base @ Rx @ Ry`. This tool explicitly
fixes the SLATE base to `[[1,0,0],[0,0,1],[0,-1,0]]` and the LAB beam to `(0,1,0)`.
It does not determine absolute detector roll. The calibrant distance is not a fitted
crystal/sample distance.

The narrow C port follows `src/rasim_next/fitting/hbn.py`, the scientific authority:
360 spokes, full-profile Gaussian contrast, robust peak selection, 36 sectors per
ring, a two-ring initial fit and four normal-to-curve refinement rounds. Bilinear
sampling uses full-resolution `log1p(max(raw-dark,0))` values only for detection;
original signed measurements are not changed. This is an inherited detection
recipe, not a validated background model or intensity fit. The supplied dark is
used unchanged at scale one; matching exposure remains the operator's obligation.

The standalone solver is bounded damped Gauss-Newton for the same soft-L1 objective
(scale one), not SciPy's TRF implementation. Coordinate scaling, fitted domains,
robust-Jacobian SVD, conditional covariance and qualification gates are preserved.
No other numerical model is introduced into the Python package. This compatibility
port is required by XP; GUI and batch share its one C implementation.

Qualified means solver convergence, rank five, scaled condition below 1e8, no active
bounds, at least eight points and 0.15 angular coverage on each of five rings, and
every ring RMS at most 2.5 px. Bounds are +/-0.15 radians, an on-panel center, and
distance 40-120 mm. Solver convergence alone is insufficient. Green curves are
qualified, orange unqualified, and red points are the retained measured selections.
Inspect the overlay. No user exclusion-mask editor is included in this version.
Reports preserve unqualified status, bounds and conditional standard errors; those
errors are not calibrated physical confidence intervals.

Loading a new source/dark or changing geometry invalidates the old calibration.
Long reads, fitting and full-matrix exports use one cancelable worker. Controls are
locked during those operations to prevent stale results; Close requests cancellation.
Preview preparation and small profile/ROI operations are synchronous.

## Export and batch

- **Native CSV/ASC matrix:** original-resolution values in native row order, with
  comment headers recording shape, coordinate convention, raw/dark state and CRC32
  source fingerprints. ASC is plain whitespace-delimited native ASCII, not the old
  Rigaku `(DAS)^2` exchange format. Neither export applies display contrast/log.
- **Profiles and ROI:** full-resolution selected row and column, with axis labels,
  pixel-center coordinates, source fingerprints and optional signed ROI statistics.
- **Visible view BMP:** the currently rendered viewport, including its crop/contrast,
  without curve, cursor, ROI or profile overlays. This is a display image, not counts.
- **Save hBN calibration:** parameters, units, seeds, qualification, per-ring metrics,
  conditional errors, source CRC32, retained observations and residuals in one CSV.

Outputs stage to a temporary file and replace their destination only after a complete
write. The GUI asks before overwriting. OSC and INI extensions cannot be export targets.
CRC32 detects accidental changes; it is not a cryptographic provenance guarantee.

The same executable supports unattended calibration without opening a window:

```text
SLATE-OSC-XP.exe --fit hBN.osc dark.osc result.csv [instrument.ini]
```

Exit codes: 0 qualified result saved, 3 unqualified result saved, 2 input/calculation/
output failure. Batch explicitly replaces the requested CSV. No file is written on
calculation failure. Passing a single OSC filename instead opens it in the viewer.

Angle/q-space rebinning and the old viewer's additional analysis tools are outside
this version. They need separate geometry, measure and conservation checks.

## Build on a modern development computer

Use the independently published LLVM-MinGW-XP 22.1.7 MSVCRT x86_64-host ZIP:
https://github.com/mon/llvm-mingw-xp/releases/tag/llvm-mingw-xp-22.1.7

Archive SHA256:
`042b2df678eab60f42bf47d0024bec3572c25d327b19b0aa14cda2b622ef3e1b`.
The host architecture is separate from the i686 application target. Do not substitute
an ordinary UCRT compiler and assume the result runs on XP. This build has no runtime
dependencies beyond XP system DLLs; the accompanying runtime license notices belong
beside the executable.

```powershell
./build.ps1 -CompilerRoot C:/tools/llvm-mingw-xp-22.1.7-msvcrt-x86_64 -OutputDirectory C:/output/SLATE-OSC-XP
```

Build outputs must be external to the repository. The build uses strict warnings,
SSE2 double arithmetic, no fast-math, static support libraries and PE subsystem 5.1.
Inspect PE imports and exercise the GUI and batch executable before distribution.
No persistent test/proof runner or generated binary is stored in this repository.
