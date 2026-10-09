# Simulator scene controls

The Simulator workspace shows detector output beside the current experiment
geometry. Open the existing **Launch SLATE UI.cmd**, choose **Simulator**, and
load or edit a draft through **Menu > Advanced parameters**. That action also
opens the retained full parameter editor. Geometry appears before an intensity
result; opening the scene or changing its camera does not calculate intensity.

## Everyday edits

Click a device label, its leader arrow, or its rendered surface. The device
selector provides the same targets, including every declared goniometer axis
in composition order. Select one parameter, enter its exact value, and set
**Step** for keyboard adjustment. Translations have a straight constrained
handle; supported rotations have an arc. Drag the selected handle, or use the
arrow keys while the scene has focus. Parameters retain their declared units
and frames: LAB positions are metres, angles are displayed in degrees, and
detector reference coordinates are native `(column, row)` pixels.

One drag makes one Undo action. **Escape** cancels the gesture. **Stop** retains
the latest draft and ends live dispatch and pointer ownership. Inline edits,
Advanced parameters, Undo/Redo, and Save/Open share the same immutable draft.
Geometry follows edits through the existing worker. **Live** uses the existing
latest-only intensity dispatcher during a held gesture; no latency guarantee
is implied.

Drag empty scene space with the left button to orbit, use the right button to
pan, and use the wheel to zoom. Camera presets and **Back to experiment** change
only the view. They do not dirty the draft or reupload an unchanged texture.

## Geometry and route limits

Configured drafts expose source position, declared base translation, mount/sample
offset, actual declared motor angles, detector translation/reference/tilts, and
their applicable nonspatial parameters. Mount/sample offset handles follow the
canonical moving goniometer frame. Incidence is derived from the current mean
ray and sample normal; a general motor angle is not renamed incidence.
Coupled beam direction/basis, complete rigid matrices, axis rosters and pivots
remain in Advanced. No independent hardware or motor chain is invented.

Native recipes expose their supported LAB sample and detector pose controls.
A separately declared base, mount, or motor chain is unavailable in that route.
Crystal/material, mosaic, sampling, display, external path, and identity belong
to compact selected sections. A mosaic label does not assert a probability cone.
Unbounded sample patches and holder glyphs are schematic. The detector reference
crosshair and direct-beam intersection are distinct.

## Current geometry and retained intensity

Scene landmarks use canonical compiled rigid geometry. The scene uses the
detector's immutable sum-display array and contrast settings only when the result
draft exactly matches the current geometry draft. Otherwise the scene shows a
schematic detector and identifies the mismatch; retained detector output remains
historical. Undo may restore identical declarations with a newer revision, which
still requires a matching result before attaching its texture. Camera changes
reuse the same array. Raw numeric image cells, profiles, and exports are unchanged.

Focused integration evidence covers independent rigid-transform identities,
actual Qt picking and gestures, shared values and history, Save/Open, canceled
Live dispatch, stale-worker rejection, native controls, and 150% scaling.
These geometry and software checks do not qualify physical intensity, fitting
accuracy, or frame rate. No new physical prediction was run for this integration.

## Compact layout

**Profile controls** expands the detector position/width, follow/pin, measure and
ROI controls in a scrolling panel. Fit, contrast and exposure remain in the main
detector toolbar. Profile buttons keep their full caption height at desktop
scaling; the detector image and geometry scene remain the primary workspace.

The LAB triad uses a fixed screen size and separate colored +X/+Y/+Z legend.
Sample normal and detector reference/row/column meanings appear in the caption
below the scene when their device is selected. Only relevant spatial arrows are
prominent; their true canonical positions are unchanged. Schematic annotations
use a short complete caption, rather than a truncated scene footer.
