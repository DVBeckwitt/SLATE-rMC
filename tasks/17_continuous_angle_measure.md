# T17 — Continuous detector-to-angle measure

Status: `READY`

## Activation

This narrow task is activated by the user request to retain the detector field as a continuous
function of canonical `(phi, 2theta)` and to render matched 3,000 × 3,000 detector/angle views. It
splits the continuous coordinate pullback from historical T15. A continuous-`S/N` finite-bin
reducer, reciprocal remapping, experimental masks/background, and angle-space fitting remain
deferred; the accepted finite-pixel polygon projector remains available unchanged.

## Owned result

For the fixed `AngleFrame`, invert angles to detector coordinates and apply the detector-area
coordinate Jacobian exactly once:

```text
(c,r) = detector_coordinates(2theta, phi)
J = |d(c,r)/d(2theta,phi)| = sin(2theta) / pixel_solid_angle_sr
S = d(c,r) J
N = J
I = S/N
```

The detector solid angle is used only to express this coordinate Jacobian; it is not an added
acceptance correction. Invalid/off-panel directions and the exact polar coordinate have zero
`S`, `N`, and `I`. A later finite bin must integrate `S` and `N` separately before division.

## Owned paths

- `src/rasim_next/geometry/angles.py`
- `src/rasim_next/measurement/continuous_angle.py`
- `src/rasim_next/measurement/__init__.py`
- the pose-bound instrument exposure in `src/rasim_next/fitting/geometry.py`
- compact permanent tests and the contract/proof documentation for this slice

The standalone comparison renderer was retired during repository cleanup. Its numerical
detector/angle APIs and all coordinate-measure proofs remain supported; the external figure is
historical acceptance evidence, not a required runtime tool.

## Acceptance

- A tilted, rectangular detector Jacobian agrees with an independent finite-difference oracle.
- Seam periodicity, support validity, and the zero-Jacobian pole are explicit.
- A real one-state all-root detector function satisfies pointwise `S=dJ`, `N=J`, and `I=S/N`.
- The angle wrapper owns the corrected detector pose and cannot accept a mismatched instrument.
- A finite-bin consumer is documented to integrate `S` and `N` separately; direct continuous bins
  converge to but do not claim bitwise identity with the finite-pixel polygon projector.
- An external side-by-side figure contains two 3,000 × 3,000 numeric fields and labels every
  positive-strength nominal exact-integer-L marker.
- Ruff, the compact permanent suite, registered proofs, seed verification, benchmark, and clean
  worktree gates pass before handoff.
