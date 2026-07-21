# Result measure

This document defines the factors and units of every current forward result. No downstream caller
may silently renormalize one of these measures.

## Source measure

`IncidentSampleBatch.source_weight` is an empirical probability mass over complete source rows.
Valid source weights sum to one. Entrance transport multiplies it by the declared sample-footprint
acceptance. Independent source rows, wavelengths, phases, and polarization states add as
intensities.

## Continuous Bragg measure

For physical rod `r`, folded tilt `alpha in [0, pi]`, full azimuth `beta in [0, 2*pi)`, and rod
coordinate `u` in inverse angstroms,

```text
Q_r(alpha,beta,u) = C R_beta R_alpha (h*b1 + k*b2 + u*d)

b_r(alpha,beta,u)
  = p(alpha,beta) * population_r * S_r(L=u/|b3|; K)
```

`p` is the normalized wrapped mosaic density with respect to `dalpha dbeta`. The folded-alpha
factor is included exactly once, full beta already represents both signed tilt directions, and the
total continuous probability is one. `S_r` is a nonnegative per-rod finite-stack strength in
angstrom squared. `b_r` therefore has measure ID
`latent_bragg_density_A2_rad2_inv.v1`; integrating over `du` also contributes inverse-angstrom
measure.

Each `(h,k)` rod is evaluated before summing intensities within exact family `m`. Independent rods
never sum as amplitudes, and no summed family structure factor is attached to a representative rod.

## Analytic Ewald restriction

For incident film-phase wavevector `ki`, the elastic constraint is

```text
|ki + Q_r(alpha,beta,u)| = |ki|.
```

The line/sphere equation is solved analytically for every retained root `u_j(alpha,beta)`. The
intrinsic coating diagnostic evaluates

```text
e_r,j(alpha,beta) = b_r(alpha,beta,u_j) * J_Ewald,j,
```

where `J_Ewald` is the one required coarea factor. Its measure ID is
`detector_visible_intrinsic_ewald_latent_density_A2_rad2_inv.v1` after restricting validity to the
active panel. Tangencies and no-root regions have explicit status. The zero-width `m=0` direct root
at `Q=0` is excluded. For the top-exit detector geometry, the retained nonzero `m=0` solution is
separated from that root by the proven gap `|Q| > -ki_z > 0`; the all-roots detector path includes
only this regular detector-visible support and reports the gap.

## Detector-coordinate measure

A floating-point detector coordinate `(c,r)` determines one outgoing air ray. Exit refraction is
solved using the shared complex-normal branch to obtain the film-phase `kf`; then

```text
Q_sample(c,r) = kf_film_sample(c,r) - ki_film_sample
|Q_sample + ki_film_sample| = |ki_film_sample|.
```

For every physical rod, the inverse latent branches mapping to this `Q` are enumerated. The
pre-binned density is

```text
d_r(c,r) = sum_inverse_branches [
    b_r(alpha,beta,u)
    * J_Qsurface(c,r) / |J_latent(alpha,beta,u)|
    * |t_in * t_out|^2 * attenuation
    * source_weight * footprint * phase_population * polarization
]
```

This direct change of variables is equivalent to the intrinsic Ewald coarea restriction followed
by the surface-to-detector map; it does not apply the coarea factor twice. The result is
`raw_detector_coordinate_density_A2_per_px2.v1` and is callable at arbitrary detector coordinates.
All inverse branches and rods sum as intensities.

The zero-transverse-width rod model has integrable caustic curves. The coordinate density is an
almost-everywhere representative: exact positive caustic points are marked and may be infinite,
while finite pixel box integrals remain the authoritative detector values. Display interpolation or
blur is not a physical regularization.

`pixel_solid_angle_sr` is geometry metadata. The raw detector-coordinate density already contains
the required coordinate-change Jacobian, so solid angle is not multiplied a second time as an
acceptance or efficiency factor. Any later area-normalized or caked observable must be separately
named and apply its correction once.

## Pixel measure

For native pixel box `P_ij`,

```text
I_ij = integral_Pij d(c,r) dc dr.
```

`DetectorPixelMass.measure_id` is `raw_detector_pixel_mass_A2.v1`. Deterministic fixed or adaptive
quadrature evaluates the continuous function inside the box. A display macrobin uses the same box
integral over a larger declared rectangle and is labelled
`raw_detector_macrobin_fixed_quadrature_estimate_A2.v1`.

No point deposition, histogram, pixel supersampling claim, per-reflection normalization, or image
maximum normalization belongs to the physical result. Masks, background, saturation, detector
efficiency, and detector PSF remain separate future operators.
