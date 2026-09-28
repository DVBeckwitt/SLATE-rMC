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
project and an empty acquisition browser. Use **Import OSC** or drop exactly one local `.osc` or
`.osc.gz` file onto the window. Import runs in a bounded worker; the detector shows native int32
counts with exact linked profiles. A failed or canceled import leaves the previously selected image
usable. Selecting an older acquisition reloads its source and checks its stored decoded hash before
showing it. The OSC header supplies image dimensions and byte order, but angles and calibration remain
unknown. The project is not saved yet; project opening/saving, multi-file import and simulation
controls arrive in later workflows. No image is loaded or calculation started at launch. Close the
window normally to exit. The `interactive/` directory is not part of the installed numerical wheel,
so run this entry point from the checkout.

Interactive admission limits are 64 MiB source bytes, 32 MiB decoded bytes, 12 million pixels and
16,384 pixels per axis, further capped by the active OpenGL context's texture-size limit.
The 96 MiB worker result limit includes the native int32 plane, float32 display plane and exact
center profiles. Imports outside those limits fail with a visible message. The acquisition SHA-256
is over the decoded OSC byte stream; it identifies exactly the header and payload consumed by the
reader, whether the file was plain or gzip compressed.

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
