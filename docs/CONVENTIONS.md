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

An intrinsic Ewald direction is the unit vector of the real internal-film
`kf_film_sample` in the sample frame. Its solid angle `dOmega_film` is distinct from external-air
ray solid angle and detector pixel solid angle. With `k = |ki_film_sample|`, the intrinsic sphere
uses `Q_sample = k * kf_hat_film_sample - ki_film_sample` and `dA_Q = k^2 dOmega_film`.
An incident-`ki`-aligned coordinate triad may be used only as a display basis; it does not change
the sample-frame `Q` contract. A LAB-oriented sphere in an exploded schematic is a translated and
scaled direction glyph, not reciprocal coordinates sharing a metre origin with detector planes.

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
- Detector-local `+z` is the outward normal. The active face is approached from detector-local
  `z < 0`, so a valid outgoing ray satisfies `n_D dot kf_hat > 1e-14`. Back-side and tangent
  approaches are invalid and carry zero geometry and intensity.

## Rod and family conventions

Every physical `(h,k)` rod has a unique `rod_id`. A `family_id` records shared in-plane radius without collapsing rods.

For a hexagonal cell,

\[
m=h^2+hk+k^2,
\qquad
Q_r=\frac{2\pi}{a}\sqrt{\frac{4m}{3}}.
\]

For a general cell, `Qr` comes from the in-plane reciprocal metric. Floating `Qr` alone is not a stable identity. The phase, reciprocal-cell revision, exact family key, and rod IDs are part of selection provenance. `family_m = h^2+hk+k^2` is hexagonal metadata only; general-cell integer-L indexing groups candidates by reciprocal-metric transverse radius and preserves representative signed `(h,k)` in marker identity. The accepted layered-CIF mounting uses direct vectors `a1,a2` as the surface lattice and reciprocal `b3` as continuous `L`.

## Branch convention

The live branch identifier belongs to the analytic Ewald root, not to a sampled event or detector
side. For a nonzero rod, branch `1` is the lower-`u` quadratic root and branch `2` is the upper-`u`
root. A retained regular nonzero `m=0` solution uses branch `0`; the algebraic direct `Q=0` root is
suppressed. Tangent and no-root cases emit no ordinary-density branch.

Raw OSC row/column sign, display orientation, reciprocal azimuth, and detector-native side are never
Ewald-root identity. A future measured-selection side or azimuth label must be a separately named,
versioned contract and must not overload branches `0`, `1`, or `2`.
