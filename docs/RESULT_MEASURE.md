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

The coordinate contributes only when its ray approaches the active detector face with
`n_D dot kf_hat > 1e-14`. Back-side, tangent, off-panel, or otherwise invalid coordinates carry
zero density.

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

## Continuous angle-coordinate measure

Let `t = 2theta` and let the fixed `AngleFrame` define canonical `phi`. The inverse geometry gives
`(c,r) = T^-1(t,phi)` on the active detector. For one detector pixel area viewed from that same
angle-frame origin,

```text
J(t,phi) = |d(c,r) / d(t,phi)|
         = sin(t) / pixel_solid_angle_sr(c,r).
```

`pixel_solid_angle_sr` appears here only as an exact geometry identity for the coordinate measure;
it is not multiplied into the raw detector physics as an acceptance factor. The continuous
normalized angle function returns

```text
S(t,phi) = d(c,r) J(t,phi)
N(t,phi) = J(t,phi)
I(t,phi) = S(t,phi) / N(t,phi) = d(c,r).
```

The measure IDs are `raw_detector_angle_signal_density_A2_per_rad2.v1`,
`detector_area_density_px2_per_rad2.v1`, and
`raw_detector_area_normalized_intensity_A2_per_px2.v1`. Invalid or off-panel directions have zero
`S`, `N`, and `I`. At `t=0` the azimuth is undefined and `J=0`, so the duplicated point is invalid
for pointwise normalization even when the direct-beam detector coordinate exists.

An ordered-intensity selected-center observation first restricts the already source-summed detector
function to the rods named by one frozen reflection group and then samples `S(t,phi)` at its frozen
center. Its measure ID is `selected_group_angular_signal_density_A2_per_rad2.v1`. Source states and
retained roots are reduced before this selection; no per-source normalization, recentering, scale,
or residual is defined. This point density is neither the finite-bin integral below nor a native
pixel count. It can be compared only with an observation carrying the same selected-component
extraction contract.

For a finite angular bin `B`, the only admitted reduction is

```text
I_B = integral_B S dt dphi / integral_B N dt dphi.
```

Dividing or averaging pointwise ratios before this reduction is not equivalent. Direct integration
of the continuous function converges to the existing finite-pixel corner-polygon projector as the
detector pixels are refined. It does not claim bitwise identity at finite pixel size because that
projector treats each already-integrated detector pixel as uniform over its angular polygon.

## Pixel measure

For native pixel box `P_ij`,

```text
I_ij = integral_Pij d(c,r) dc dr.
```

`DetectorPixelMass.measure_id` is `raw_detector_pixel_mass_A2.v1`. Deterministic fixed or adaptive
quadrature evaluates the continuous function inside the box. A display macrobin uses the same box
integral over a larger declared rectangle and is labelled
`raw_detector_macrobin_fixed_quadrature_estimate_A2.v1`.

The optional stochastic terminal estimates the same finite pixel mass by drawing
`omega_sm=(alpha,beta)` from the declared mosaic probability for every canonical source stratum
`s`. Counter lanes are reserved for known-zero strata to preserve source-index identity, but those
states execute no root or physics work:

```text
Ihat_P = sum_s (1/M) sum_m sum_(r,j)
         W_srj(omega_sm) 1[T_srj(omega_sm) in P].
```

`r` is a physical signed rod, `j` is every retained analytic root, `T` is the canonical forward
exit/refraction and detector map, and `W` contains source/footprint/phase/polarization mass, rod
population, finite-stack strength, the Ewald coarea factor once, and entrance/exit optical and
attenuation factors. Because the proposal is exactly the mosaic law, its density cancels and is not
multiplied into `W`. No detector-coordinate Jacobian or detector solid-angle factor belongs to a
forward deposit. Invalid, no-root, and off-panel draws have zero weight and remain in the divisor.

`MonteCarloDetectorPixelMass.measure_id` is
`raw_detector_pixel_mass_monte_carlo_estimate_A2.v1`. `image_A2` contains weighted raw pixel-mass
estimates, not integer photon counts. `attempted_root_count` and `visible_hit_count` are work
ledgers, the detector-visible m=0 support gap remains explicit provenance, and the retained
per-draw total masses are seed-stability diagnostics. Near a regular
Ewald fold, the coarea weight can scale as the inverse square root of distance to tangency; its mean
is integrable while its second moment may diverge. The result therefore makes no ordinary
central-limit or standard-error claim. Refined deterministic latent integration is the acceptance
oracle.

`MonteCarloDetectorPresentation` is not a second scientific measure. It is a transient full-native
float32 rendering lease obtained from the same unnormalized accumulator, divided by the completed
draw count. It performs no spatial resampling, detector Jacobian, solid-angle correction,
normalization, or change of pixel ownership. The authoritative retained/exported result remains the
validated float64 `MonteCarloDetectorPixelMass` above.

`DetectorCoordinateDensityImage` is different: `sample_detector_pixel_center_density(...)` samples
the completed all-source, all-rod, all-root function once at each native center and retains
`raw_detector_coordinate_density_A2_per_px2.v1`. It is a display-only coordinate-density sample,
not a pixel-box mass, macrobin mass, calibrated count expectation, or raw OSC count image.

No undeclared or unweighted histogram, general point-deposition API, pixel supersampling claim,
per-reflection normalization, or image maximum normalization belongs to the physical result. Masks,
background, saturation, detector efficiency, and detector PSF remain separate future operators.
