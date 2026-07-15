# OSC and detector coordinates

Original RASIM stores the OSC array and the simulation image in different orientations. The new repository fixes this once at the I/O boundary.

## Coordinate domains

```text
OscRawIndex
    row and column in the decoded file payload

DetectorIndex
    row and column in the canonical detector-native array

DetectorCoordinate
    continuous (column_px, row_px) in the canonical detector-native frame
```

These types are not interchangeable tuples.

## Canonical orientation

The expected conversion, to be confirmed by the characterization fixture, is

```python
detector_native = np.rot90(osc_raw, -1)
```

For an OSC raw array of shape `(H_raw, W_raw)`, the detector-native array has shape `(W_raw, H_raw)` and an OSC raw index maps as

\[
r_{\mathrm{det}}=c_{\mathrm{raw}},
\qquad
c_{\mathrm{det}}=H_{\mathrm{raw}}-1-r_{\mathrm{raw}}.
\]

The inverse is

\[
r_{\mathrm{raw}}=H_{\mathrm{raw}}-1-c_{\mathrm{det}},
\qquad
c_{\mathrm{raw}}=r_{\mathrm{det}}.
\]

## Required bootstrap fixture

Use a non-square synthetic OSC payload, for example `7 x 11`, containing:

- four distinct corners
- one off-axis interior marker
- one marker at the declared beam-center metadata coordinate
- one high-range encoded pixel

Prove:

- signature and header parsing
- endian choice
- dimensions and payload size
- high-range decoding
- raw and detector-native shapes
- every marker mapping
- exact inverse mapping
- pixel-center mapping
- beam-center metadata mapping

## Prohibitions

- No physics or rendering module may rotate, flip, or transpose detector data.
- No general `display_rotation` setting exists in the numerical core.
- Display orientation, if later needed by a GUI, is outside the scientific coordinate contract.
- `2theta/phi` and caking must consume detector-native coordinates and cannot redefine the OSC conversion.
