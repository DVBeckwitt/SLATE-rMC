# Continuous Ewald and detector strategy

Status: **implemented and authoritative**.

## 1. Latent Bragg space

For each physical rod `r=(h,k)`, use curvilinear coordinates `(alpha,beta,u)`:

```text
Q_s(alpha,beta,u)
  = C R_beta(beta) R_alpha(alpha) [h*b1 + k*b2 + u*d].
```

`alpha` and `beta` are rotations, not Cartesian reciprocal coordinates. `u` is the inverse-angstrom
coordinate along the rotated rod. At zero mosaic, the mapping reduces to the unmosaicked rod; at
nonzero mosaic, different orientations generally meet the elastic shell at different `u` values.

The per-rod latent density is

```text
b_r = p(alpha,beta) * P_r * S_r(u/|b3|; |ki_air|).
```

The wrapped mosaic law uses folded alpha and full periodic beta. Every `(h,k)` structure strength,
population, and geometry is evaluated before any exact-family sum.

## 2. Analytic Ewald roots

Elastic scattering inside the film satisfies

```text
|ki_film + Q_s(alpha,beta,u)|^2 = |ki_film|^2.
```

For fixed `(r,alpha,beta)` this is a quadratic in `u`. The stable line/sphere solver returns the two
regular roots, a typed tangent, or no root. It verifies

```text
|Q + ki_film| = |ki_film|
```

within the declared residual tolerance. The intrinsic coating applies the coarea factor once and
then tests exit and active-panel visibility. No sphere mesh, texture, triangulation, or global sphere
sampling is created.

`m=0` is geometrically well defined, but its zero-transverse-width direct root at `Q=0` is singular
and excluded. A top-exit detector can see a separate nonzero `m=0` solution. The all-roots runtime
includes it only after proving the positive gap `|Q| > -ki_z > 0`, which keeps the visible support
away from the direct-root singularity; the gap is retained in result provenance.

## 3. Exit wave and attenuation

Each retained internal `kf_film = ki_film + Q` is refracted independently at the exit interface.
Tangential wavevector is conserved and the shared complex-normal branch chooses the outgoing air
mode. The intensity multiplier is

```text
|t_in * t_out|^2 * uniform_depth_attenuation.
```

Real phase wavevectors carry geometry; imaginary normal components determine decay. Backward,
parallel, evanescent-without-outgoing, and nonintersecting rays retain explicit status and zero
detector intensity.

## 4. Continuous detector pullback

The efficient production direction begins at an arbitrary detector coordinate `(c,r)`:

```text
detector point
  -> unit outgoing air direction
  -> kf_air
  -> inverse exit refraction to kf_film
  -> Q_sample = kf_film - ki_film
  -> all inverse (rod,alpha,beta,u,root) branches
  -> summed detector-coordinate density.
```

The exact Q-surface and inverse-latent determinants perform the surface-to-plane change of variables.
This is equivalent to first forming the coarea-weighted shell and then mapping it to the detector,
but avoids constructing or projecting that intermediate surface. No solid-angle or coarea factor is
applied twice.

The zero-width rod pushforward is finite almost everywhere and has integrable caustic curves where
the latent map folds. Exact caustic coordinates are flagged. They do not prevent mapping the measure
to the detector: a finite pixel box has a finite integral even when the pointwise density is
unbounded on a measure-zero curve.

## 5. Source averaging and wavelength variation

Every Monte Carlo source state first produces its own canonical `ki`, elastic roots, exit optics,
and detector-coordinate field. States and wavelengths are independent and sum as intensities:

```text
d_total(c,r) = sum_s d_s(c,r).
```

This summed callable is created before pixel integration. It does not collapse different wavelengths
onto a fictitious common `kf`; each state is evaluated at the same detector coordinate with its own
wavevector magnitude and refraction.

## 6. Pixel integration and rendering

The native pixel value is the deterministic box integral of `d_total(c,r)`. Fixed Gauss rules give a
fast preview; adaptive refinement supplies a reported error/tolerance path. A pixel center, display
interpolation, Gaussian blur, histogram, or bilinear point deposit is not a pixel integral.

Reciprocal and Ewald 3D figures may sample the callable functions for display. Those arrays are
ephemeral views and are never cached as the scientific model.

## 7. Proof obligations

Permanent evidence covers mosaic normalization and signed directions, analytic roots versus an
independent quadratic oracle, elastic closure, per-rod/family reduction, latent-to-detector
round-trips, scalar/compiled/CUDA agreement, caustic finite-box behavior, quadrature refinement,
source/wavelength summation, detector tilts, invalid support, and once-only factor ownership.
