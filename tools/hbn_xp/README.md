# SLATE OSC + hBN for Windows XP

A portable native Win32 viewer, hBN detector calibrator, angular integrator and CIF reference viewer.
Run `SLATE-OSC-XP.exe`.
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
The packaged `demo/hBN_analysis_geometry.csv` is the applied geometry from that
example. Load it through File for a quick angular-analysis demonstration; it is
specific to the supplied calibrant at its measured position.
The demo folder also contains PbI2, Bi2Se3 and Bi2Te3 CIFs from the repository.
These are separate material references, not models of the supplied hBN image.

The application is already native C11/C++17. Version 6 reduces repeated work without
changing precision: strip-based OSC decoding, block-local previews, Gaussian weights
reused within each calibration, and fewer polygon copies/clips during integration.
Additional OSC staging memory is 186 kB for a 3000-column file (at most 1 MiB total).
No additional runtime or hardware requirement is introduced.

On the development computer, median times from three runs of 32-bit builds with
identical compiler settings and supplied hBN/dark files changed from 142 to 97 ms
for both OSC loads, 21 to 7 ms
for a dark-subtracted preview, 233 to 200 ms for calibration, and 4.80 to 3.41 seconds
for an 800 by 360 angular integration. These are component timings with warm file
cache on modern Windows, not predicted XP times. Complete decoded pixels, previews,
fit observations/results, angular signal/area/panel arrays and count-accounting
totals matched version 5 bit for bit. Focused checks also cover endian/strip edges,
signed partial preview blocks, reflected profile filters, beam-center and phi-wrap
boundaries, cropping, masking, cancellation and malformed file rejection. They
establish implementation equivalence, not a new physical calibration validation.

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

The pressed Analysis/hBN/CIF button identifies the current page. The default view
is a single detector image. Switching pages or views and resizing the window
preserve zoom and the native point at the viewport center; **Fit image** explicitly
resets the view. Black/white, Auto contrast, Apply levels, Fit image and Zoom affect
the detector only and are disabled in angular-only view. Log display applies to both
images; the angular map automatically scales to its current finite bin values.

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
Inspect the overlay. The analysis mask does not change hBN ring detection.
Reports preserve unqualified status, bounds and conditional standard errors; those
errors are not calibrated physical confidence intervals.

Loading a new source/dark or changing geometry invalidates the old calibration.
Long reads, fitting, angular integration and numeric exports use one cancelable worker. Controls are
locked during those operations to prevent stale results; Close requests cancellation.
Preview preparation and small profile/ROI operations are synchronous.

## Angular analysis

1. Calculate hBN, then choose **Analysis > Use hBN fit**. This explicitly applies
   the qualified detector pose and calibrant distance. If the sample is elsewhere,
   enter its beam-direction distance in **Sample mm** and click **Apply**. The hBN
   distance alone does not establish the sample position. Alternatively, enter
   manual geometry on the hBN page and choose **Analysis > Apply inputs**.
2. Choose **Detector**, **Phi vs 2theta**, or **Side by side**. Hover on the detector
   for native column/row, signed counts, 2theta and phi. The angular-map cursor shows
   angles and the corresponding continuous detector position. Phi is undefined at
   the direct beam. All angle controls use degrees.
3. Set 2theta and phi limits and their requested bin widths, then **Integrate**.
   The map and both profiles share these limits: I(2theta) integrates over phi;
   I(phi) integrates over 2theta. Bin counts round upward, with equal bin widths
   adjusted to fit the exact limits. **Full range** resets to 0..80 degrees and
   -180..180 degrees; click Integrate to recompute. At most 4096 radial bins, 1440
   azimuthal bins and 1,048,576 total bins are admitted. Actual widths must be at
   least 1e-7 radians (about 0.00000573 degrees).
4. Select **Angular sector** as the mouse tool. Drag in detector space between two
   angular corners to select an annular sector, or drag a rectangle on the angular
   map. Releasing recomputes the map and profiles. Detector dragging chooses the
   shorter phi arc (at most 180 degrees); use the fields or angular map for wider
   sectors. A phi end below the start crosses the seam: 170 to -170 means 20 degrees.
5. **Mask rectangle** and **Unmask rectangle** operate on full detector pixels.
   Masked pixels are shaded red. Click Integrate after changing the mask. **Clear
   mask** removes all exclusions. Masks affect angular analysis only; native matrix,
   row/column profiles, ROI measurements and hBN ring detection retain their original
   behavior. A new source clears the mask; masks are not saved in geometry files.
6. **Map / profiles** selects total signed counts, mean counts, or valid pixel area.
   Missing bins have no valid mean; gray marks missing support, red marks bins whose
   available panel support is entirely masked. Profiles reduce signal and area first,
   then divide for means. Coverage means effective detector pixels, not a percentage
   of a complete ring. The map uses nearest-bin display sampling; profiles use a
   min/max envelope. Display settings never change integration values.
7. **File > Save analysis geometry** and **Load analysis geometry** preserve the
   applied pose, sample distance, dimensions, pitches, wavelength and calibration
   provenance. A geometry remains available when opening another same-size image;
   verify that the instrument configuration and sample position still match.
   Editing hBN seeds does not silently modify applied analysis geometry.

Phi is scattering azimuth, not a sample motor angle. The frame uses the incident
beam and the detector column axis projected perpendicular to it. In an untilted
native view phi is zero upward, positive toward the left, and wrapped to [-180,180).
Tilted-detector readout and inverse overlays use this same frame and the full pose.

The C projector follows `measurement/angle_space.py:compile_detector_angle_projector`
and the detector-oriented AngleFrame in `selection/osc_series.py`. It maps physical
pixel corners, divides ordinary pixels along the top-left/bottom-right diagonal,
and clips those angular triangles into bins. Beam-pole pixels use physical-edge
fans; azimuthal seams are unwrapped before clipping. Weights use the full pixel's
angular polygon area. Cropped support is not renormalized. This is the accepted
corner-polygon approximation, not exact integration of a continuously curved pixel.
For each bin, S=sum(weight*count), N=sum(weight for unmasked pixels), and P=sum(weight
before masking). Means are S/N; N/P is the valid fraction of the available panel
support. No solid-angle, polarization, exposure or background correction is applied.
Signed raw-dark values are preserved, including negative measurements.

The integrator streams two rows of corner coordinates and keeps three double arrays
for bins. The default 800 by 360 grid adds about 6.6 MiB, plus about 1.1 MiB for a
3000 by 3000 mask when used. Moving the cursor computes only its coordinate transform.
Integration runs on demand; slower XP hardware remains responsive to Cancel.
Angular profile display values are cached until the integration or output mode
changes, adding 9,280 bytes for the default grid. Selection and guide redraws reuse
them without repeating the grid reductions. Numeric exports retain their original
signed reductions and precision.

## CIF reference beside the image

1. Choose **CIF** and **Load CIF**. Enter the measurement wavelength in angstroms
   and the maximum 2theta in degrees. **Use geometry wavelength** copies the applied analysis
   wavelength; click **Calculate** after changing inputs. The CIF's own wavelength
   is not substituted for the measurement wavelength. A CIF can be calculated
   before opening an OSC.
2. **Guides > Powder 2theta arcs** draws every distinct Bragg angle along phi.
   **Arcs + Qz rods / L ticks** adds orange constant-Qr rods and yellow HKL ticks.
   Set **Incidence deg** (sample incidence) and **Normal phi deg** (transverse sample-normal
   direction), then **Apply a1/a2 fiber**. This explicitly declares that direct
   a1/a2 lie in the sample surface, b3 is the texture normal, and crystallites may
   rotate through the full fiber azimuth. Positive incidence means the beam enters
   that surface; normal phi 0 points up and +90 points left in the angle frame.
   These are measurement inputs, not values inferred from the CIF or hBN fit.
   Incidence must be strictly within +/-89 degrees; normal phi within +/-180.
   Mounting is session-only; enter/apply it again after restarting the program.
3. **View** selects detector, angular or split view. Both images use the same
   physical outgoing rays and applied detector pose. Angular views need an existing
   integration. I(2theta) shows only the powder-angle lines. Guides need matching
   applied image geometry and wavelength (relative agreement 1e-10). Edited CIF
   inputs hide guides until recalculated; edited mounting hides oriented guides
   until applied. **Guides > Off** and **Labels** control display independently.
4. The upper side table switches between **P: 2theta groups**, **R: Qz rod groups**,
   and **T: HKL tick groups**. Each coincident contour is drawn once. A label such
   as `T12 (1 0 3) +5` means five additional HKLs in that group. Select a group with
   the mouse or arrow keys: its guide turns cyan and the lower table lists every
   signed HKL, 2theta and individual raw |F|^2 in electrons squared. The upper
   table's **max |F|^2** is the largest individual member, not a sum. **By strength**
   sorts these maxima; **By position** restores angle order (Qr order for rods).
   Sorting preserves the selected group and scrolls it into view. **More rows**
   collapses the setup fields to enlarge both tables; **Show setup** restores them.
   View choice and calculation status remain visible. Hover over a table row for
   the full label and more precise values. The CIF inspector uses a wider sidebar
   so signed HKLs and intensities fit without horizontal scrolling at normal XP
   font size. For the largest image, use a single view rather than Side by side.
   R labels use literal `L` because that line spans continuous L; R membership
   includes different L values along the same rod. P and T groups differ: equal
   powder angle does not imply equal Qr,Qz. Systematic extinctions and Friedel
   mates remain in the tables and CSV; no strength cutoff is applied.
5. Labels avoid each other; if the view is crowded, a visible count reports labels
   hidden for overlap. The lines and complete member tables remain. In oriented
   mode labels prioritize rods and HKL ticks; select a P group to label its arc.
   Zoom or use the group table to inspect crowded reflections.
6. **Export peaks** saves every signed HKL, d, 2theta, complex F, raw |F|^2, Qr/Qz,
   P/R/T group IDs, wavelength, source/data CRC32, atom counts and factor mappings.
   Its header records the applied mounting and whether mounting inputs were edited.
   Failed/canceled loads preserve the previous table. Recalculate edited CIF inputs
   before export. Group IDs are local to that CIF and reflection range.

The oriented guides use external-air kinematic geometry, without refraction,
mosaic spread or an oriented single-crystal azimuth model. A 00L tick is physically
accessible only at its specular Ewald condition; an arbitrary 00L powder ring does
not imply an oriented 00L spot at the current incidence. Negative Qz and both
physical Ewald roots are retained. The axial (0,0) rod intersects the elastic
sphere at one nonzero point, shown with an R label at `2theta=2*|incidence|`;
it is not an integer-L tick unless that Bragg condition is met. The direct beam
is excluded. No changes are made to the measured image.
Raw |F|^2 alone does not predict measured counts. No multiplicity,
relative normalization, Lorentz/polarization, absorption, electron-radius factor,
detector solid-angle or intensity correction is applied.

For a general cell, `B=2*pi*A^-T`, `n_c=b3/|b3|`, `Qz=(B*hkl).n_c`, and
`Qr=|B*(h,k,0) - n_c*(B*(h,k,0)).n_c|`. Qz therefore includes h/k offsets;
`2*pi*L/c` is not substituted for an oblique cell. P groups compare `1/d^2`,
R groups compare `Qr^2`, and T groups compare Qz within one R group. Coincidence
uses a fixed anchor and `512*DBL_EPSILON*max(1,abs(a),abs(b))` in those quantities,
not screen resolution or a chained tolerance. The exact axial rod remains separate
from small nonzero-Qr rods. Grouping never combines amplitudes or intensities.

With `k=2*pi/lambda`, incidence alpha, beam b, sample normal n, tangent
`t=(b+sin(alpha)*n)/cos(alpha)` and `v=n cross t`, a rod has
`q=x*t +/- sqrt(Qr^2-x^2)*v + Qz*n`, where
`x=(2*k*sin(alpha)*Qz-Qr^2-Qz^2)/(2*k*cos(alpha))`.
Only real elastic roots and forward detector intersections are drawn. A shared
cache projects these rays into both image spaces, splitting invalid paths and phi
seams. It refines the curve for the current zoom/range (midpoint targets 0.3 detector
screen pixels and 0.2 pixels per angular axis); changing view scale rebuilds it.
Limits are 250000 points, 20000 paths and 18 subdivision levels. Exceeding a limit
hides the image guides with an explicit message; the full peak tables remain.

The compatibility port follows `materials/crystal.py`, `materials/optics.py` and
`ordered/amplitudes.py`. Gemmi 0.7.5 parses and expands the CIF once. For each signed
reflection, d comes from the general-cell reciprocal metric and
`2theta = 2 asin(wavelength/(2d))`. With `s=1/(2d)`:

```text
F = sum_sites occupancy * (f0(s) + f'(E) + i*f''(E))
              * exp(-8*pi^2*Uiso*s^2) * exp(+2*pi*i*(h*x+k*y+l*z))
raw SF intensity = |F|^2; E[eV] = 12398.419843320026 / wavelength[A]
```

The offline `cif_scattering.bin` contains Waasmaier elastic factors and Chantler
anomalous factors from XrayDB 4.5.8/database 9.2. The elastic ion is used when
available; otherwise the neutral factor is used and the mapping is shown explicitly
in the status and CSV. Anomalous factors use the element. CIF dispersion values
are not used. Each f' interval preserves the main program's local seven-knot cubic
interpolation; f'' uses log-log linear interpolation. Keep this file beside the EXE.
Its payload CRC32 is checked when loaded. It adds about 3.8 MiB on disk; transient
factor storage is released when the calculation finishes. Grouping and the curve
cache require no recalculation during ordinary cursor movement.

Known CIF Uiso or Biso values are honored. Missing/null displacement requires the
visible **Unknown Uiso = 0 A^2** choice (enabled initially); the count of assumed
source sites is reported. Anisotropic displacement metadata is rejected. Occupancy,
fractional coordinates, unique labels and explicit element/ionic species must be
present for every site. Supported symbol forms include C, Fe, Cu+, Cu2+ and O2-;
indirect atom-type identifiers require conversion to explicit species. The parser
requires one named data block, finite cell parameters and a consistent resolvable
space group. Conflicting symmetry aliases are rejected.

Gemmi's canonical per-source-site symmetry orbit uses a 0.4 A coincidence tolerance.
This tool additionally verifies that every generated mate lies within 0.0001 A of
a retained same-label mate. Ambiguous near-special positions are rejected; no
coordinates are snapped or occupancy changed. Distinct source labels stay distinct.
Low-precision coordinates may need correction or an explicitly expanded P1 CIF.

Limits keep the XP calculation bounded: 2 MiB CIF, 2000 source sites, 192 symmetry
operations, 10000 expanded atoms, 2 million candidate hkl, 100000 retained reflections
and 50 million atom/reflection terms. Oversized requests fail with advice to reduce
maximum 2theta; results are never truncated. Full factors support H through U,
energy from 250 eV through the selected elements' Chantler table maximum, and
`sin(theta)/wavelength <= 6 / A`. Maximum 2theta must be strictly between 0 and
180 degrees. The 000 direct beam is excluded.

## Export and batch

- **Native CSV/ASC matrix:** original-resolution values in native row order, with
  comment headers recording shape, coordinate convention, raw/dark state and CRC32
  source fingerprints. ASC is plain whitespace-delimited native ASCII, not the old
  Rigaku `(DAS)^2` exchange format. Neither export applies display contrast/log.
- **Profiles and ROI:** full-resolution selected row and column, with axis labels,
  pixel-center coordinates, source fingerprints and optional signed ROI statistics.
- **Visible view BMP:** the currently rendered viewport, including its crop/contrast,
  and mask shading, without curve, cursor, ROI or profile overlays. This is a display
  image, not counts.
- **Save hBN calibration:** parameters, units, seeds, qualification, per-ring metrics,
  conditional errors, source CRC32, retained observations and residuals in one CSV.
- **Export angles:** one CSV containing geometry/provenance, the complete angular
  map, I(2theta), and I(phi), identified by the `kind` column. Each row records angle
  bounds, S, N, P, mean and valid fraction. Blank means/fractions indicate missing
  support. Full input, masked and outside-window totals make count accounting
  explicit. The current mask has a CRC32 fingerprint; its bitmap is not embedded.

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

Reciprocal-space rebinning, intensity corrections and background fitting are outside
this version.

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
./build.ps1 -CompilerRoot C:/tools/llvm-mingw-xp-22.1.7-msvcrt-x86_64 -GemmiRoot C:/tools/gemmi-0.7.5 -Python C:/project/.venv/Scripts/python.exe -OutputDirectory C:/output/SLATE-OSC-XP
```

Build outputs must be external to the repository. The build uses strict warnings,
SSE2 double arithmetic, no fast-math, static support libraries and PE subsystem 5.1.
Inspect PE imports and exercise the GUI and batch executable before distribution.
No persistent test/proof runner or generated binary is stored in this repository.

Build-time additions: the unmodified Gemmi 0.7.5 source ZIP (SHA256
`16c7d5dc414e4a1ca884688d0146cb5a39a571812a930f8a9f8436a6dd3ff525`), and a modern
Python environment with NumPy, SciPy and XrayDB 4.5.8/database 9.2. The data exporter
checks the database SHA256 before generating the offline table. Gemmi supplies the
existing CIF/symmetry parser instead of a new partial CIF parser; its C++ runtime is
statically linked. These dependencies are not installed on XP. See CIF-NOTICES.txt,
Gemmi-LICENSE.txt and PEGTL-LICENSE.txt for source availability and licenses.
