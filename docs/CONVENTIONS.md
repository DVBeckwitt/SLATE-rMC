# Mathematical conventions

## Frames

Use right-handed frames:

```text
lab
    fixed beamline frame

goniometer
    rigid frame defined by declared axes and pivots

sample
    film surface and sample-holder frame

crystal
    direct and reciprocal lattice frame

detector
    origin, column axis, row axis, and outward normal
```

Transforms are named `target_from_source`. Points and vectors are different operations. Translation applies only to points.

## Rotation convention

- Column vectors.
- Active rotations.
- Right-hand rule.
- Radians internally.
- Composition is explicit: `lab_from_sample @ sample_from_crystal`.
- Every rotation states its pivot.
- Rotation matrices must remain orthogonal with determinant `+1`.

## Units

```text
instrument positions and pixel pitch    metre
wavelength and crystal lengths          angstrom
wavevectors and reciprocal vectors      inverse angstrom
angles                                  radian
continuous detector coordinates         pixel units
```

Public data fields include unit suffixes or typed unit metadata.

## Wavevectors

Define

\[
\mathbf Q_{\mathrm{external}}=\mathbf k_{f,\mathrm{air}}-\mathbf k_{i,\mathrm{air}},
\qquad
\mathbf Q_{\mathrm{internal}}=\mathbf k_{f,\mathrm{film}}-\mathbf k_{i,\mathrm{film}}.
\]

They are different quantities. Every API states which one it uses.

At a planar interface, conserve the tangential component and compute the normal mode from

\[
k_{z,2}=\sqrt{(n_2 k_0)^2-|\mathbf k_{\parallel}|^2}.
\]

The shared branch selector enforces the requested propagation direction and non-growing evanescent behavior. The time convention and branch sign are recorded in code and proof traces.

Real phase wavevectors define elastic event geometry. Imaginary normal components define field decay and attenuation.

## Scalar interface coefficient

The first validated off-specular model uses

\[
T_{12}=\frac{2k_{1z}}{k_{1z}+k_{2z}}.
\]

Entrance and exit intensity weighting uses

\[
W_T=|T_{\mathrm{in}}|^2|T_{\mathrm{out}}|^2.
\]

Do not average s and p power-transmission coefficients in this scalar model.

## Detector coordinates

- Array indexing: `[row, column]`.
- Continuous coordinates: `(column_px, row_px)`.
- Integer index `(r,c)` refers to the pixel centered at `(c,r)` unless a file format explicitly defines edge coordinates.
- Beam center is stored as `(column_px,row_px)` in detector-native coordinates.

## Rod and family conventions

Every physical `(h,k)` rod has a unique `rod_id`. A `family_id` records shared in-plane radius without collapsing rods.

For a hexagonal cell,

\[
m=h^2+hk+k^2,
\qquad
Q_r=\frac{2\pi}{a}\sqrt{\frac{4m}{3}}.
\]

For a general cell, `Qr` comes from the in-plane reciprocal metric. Floating `Qr` alone is not a stable identity. The phase, reciprocal-cell revision, exact family key, and rod IDs are part of selection provenance.

## Branch convention

Branch identity is assigned only after integrated event geometry exists. For a non-specular pair, compute a wrapped in-plane reciprocal azimuth from the event `Q` in the declared sample or crystal basis. The sign-to-label mapping defines branch `0` and `1` and is versioned with the basis, wrapping convention, and deadband.

Raw OSC row/column sign and display orientation are never branch identity. Detector-native side is derived by projecting the already identified event and is used only for measured association and presentation.

`00L` cases that collapse at the branch boundary use `branch_id=None` with an explicit `COLLAPSED_00L` status. Optimizers never switch identity dynamically.
