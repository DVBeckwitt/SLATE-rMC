# Result measure

## Explicit shared spherical mosaic and raw-count refits

`painted_ewald.normal_density.SphericalMosaicDensity` declares the opt-in
`spherical_wrapped_tied_orientation.v1` model. It is a restricted orientation law,
not a complete SO(3) distribution. For each existing wrapped component `g_i(alpha)`,
`Z_i = 2 integral_0^pi g_i(alpha) sin(alpha) d(alpha)`. The directed density per
spherical area is `F(alpha) = sum_i p_i g_i(alpha)/(pi Z_i)`. Each component has
unit mass, so the Lorentzian probability remains its integrated mass fraction.
The nonzero-rod inverse uses `F(alpha) sin(alpha)` per `d(alpha) d(beta)` and
retains the established tied crystal rotation and every inverse preimage.

The local-lamella stitched `m=0` calculation uses that same directed law. Its
two plane-normal preimages have separate signed structure strengths:
`I = C_plus F(alpha) + C_minus F(pi-alpha)`. Do not replace both strengths with
`S(+L)`: even an ordered CIF can have unequal signed intensities with complex
atomic factors. In this spherical measure the `sin(alpha)` Jacobian cancels
analytically; the remaining normal-transfer denominator is `|Q_air|^2`.
`CompiledDetectorEvaluator.compile_local_m0_transfer` exposes the two physical
coefficients before probability multiplication, bound to its immutable source,
geometry, structure and optical state. Recompile after changing those inputs.

This named model is available through the CPU compiled evaluator and explicit
sparse-term reduction. It does not silently redefine the older flat-latent
mosaic APIs, CUDA mosaic model, or immutable reference results. The permanent
proofs check spherical mass, signed two-atom interference, sparse/compiled
nonzero agreement, and positive caustics independently.

For the six-sample physical-intensity refit, the declared observable is raw
counts averaged over each fixed detector-area region:
`prediction = exposure_scale * full_physical_simulation + nonnegative_background`.
Every nonzero rod and local stitched `m=0` contributes inside every region.
One common mosaic and physical phase population describe the entire exposure.
Only instrument-invalid data and predeclared geometric support are excluded.
No simulation-side sideband subtraction, extra empirical sample-Q damping,
peak suppression, positive-L selection, intensity-dependent mask, or per-region
rescaling is part of this observable. CIF displacement factors remain physical
structure inputs. Log plots do not floor nonpositive data; the raw values remain
in the objective and archive.

Earlier sideband-conditioned fitting commands implement a different observable;
they are not an implementation or acceptance oracle for this raw-count refit.
Historical outputs remain provenance only. Numerical qualification must compare
region shapes, not just their summed mass: the six-sample rerun found material
q2-to-q4 changes in nonzero peaks even when specular profiles were stable;
selected Bi2Te3 `m=3` peaks still changed by about 27-30% at q8-to-q16.
Nominal-family fold coordinates are not sufficient for source-shifted support
edges. Source-specific finite-region integration must be qualified before these
candidate fits are promoted.
Neither an optimizer success flag nor a lower objective certifies convergence
or physical identifiability. A bound-active or zero-probability mosaic component
must be reported as unresolved rather than as a measured width.

Compiled Parratt evaluators use zero reflection for an equal-zero interface
normal pair. Other undefined denominators and nonfinite recursion outputs fail
explicitly, including at the CUDA host boundary; they never substitute an
unrelated kinematic strength. Canonical exit round-trip checks retain tangential
conservation and outward sign, but test the normal dispersion residual in
squared-wavevector units to avoid amplifying rounding error near the horizon.

This document defines the factors and units of every current forward result. No downstream caller
may silently renormalize one of these measures.

## Source measure

`IncidentSampleBatch.source_weight` is a probability/quadrature mass over complete source rows.
Valid source weights are finite, nonnegative, and sum to one; they need not equal `1/N`. In the
discrete-line model each declared line probability is divided among that line's geometry rows so
the realized line masses are exact. Entrance transport retains source mass and declared
sample-footprint acceptance as separate fields. Detector source-phase construction multiplies
those fields once with the flat-film illuminated-path factor
`w_illum = 1/|direction_sample,z|`, phase population, and configured polarization. Independent
source rows, wavelengths, phases, and polarization states add as intensities.

Physical discrete-source sampling requires at least one row per declared line. A separate nominal
geometry API returns one mean-wavelength companion row and rejects it at every source-weighted
detector-measure intensity boundary. Source-free intrinsic coating/locus densities may use that
row only to draw geometry or reference landmarks; they are not an empirical source intensity or a
likelihood observable.
Equal per-line row counts reuse the same geometry grid. When a nonzero position/divergence
correlation is declared and a line has at least eight rows, that line is moment-matched to the full
declared four-dimensional covariance. Zero-correlation antithetic LHS grids and smaller correlated
grids remain finite quadrature approximations; non-divisible line counts additionally carry an
explicit unequal-grid model ID.

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

## Intrinsic Ewald solid-angle measure

For internal-film incident magnitude `k = |ki_film_sample|`, an outgoing unit direction
`n_film_sample` defines

```text
kf_film_sample = k * n_film_sample
Q_sample = kf_film_sample - ki_film_sample.
```

For every non-specular physical rod, all regular inverse latent preimages of this `Q` are summed as
intensities. The exact almost-everywhere density is

```text
rho_Omega(n_film_sample) = sum_(rod,preimage) [
    p(alpha,beta) * population_r * S_r(L) * k^2 / |J_latent(alpha,beta,u)|
].
```

For the tied-rotation map, `|J_latent| = |w*x|`. The mosaic probability is already defined with
respect to folded-alpha/full-beta `dalpha dbeta`, so no `sin(alpha)` factor is introduced. The
factor `k^2` converts internal Ewald-sphere area to internal-film outgoing solid angle,
`dA_Q = k^2 dOmega_film`. The Ewald coarea factor cancels against the latent-root surface Jacobian
and is not multiplied again.

The result measure is `intrinsic_ewald_direction_density_A2_per_sr.v1`. Its steradian is the
internal-film outgoing-`kf` direction measure, not external-air solid angle and not detector pixel
solid angle. Integrating this density over `dOmega_film` gives intensity in `A2`; a sampled display
texture is not itself an integrated observable.

The full-sphere selection contains every direction on the `Q` sphere centered at `-ki_film_sample`
with radius `k` and applies no sample-frame `Qz`-sign mask. Both `Qz` signs occur in the validated
grazing-incidence Bi2Se3 case; this is not a universal normal-incidence claim. It applies no
exit-refraction, front-face, active-panel, or detector-visibility mask. The present intrinsic
measure excludes the entire `m=0` rod; detector-visible regular nonzero `m=0` support remains a
separately declared detector measure. At `|J_latent|=0`, an integrable fold is marked as a caustic.
A strictly positive numerator carries `+inf`; a zero numerator uses the zero Radon--Nikodym
representative at that measure-zero point. Display clipping, blur, and interpolation are not
physical regularization.

### Detector-visible intrinsic Ewald direction measure

The separately named
`detector_visible_intrinsic_ewald_direction_density_A2_per_sr.v1` uses native coordinates on one
declared active detector panel to parameterize only the corresponding internal-film direction
patch. For canonical top-exit/front-face/panel validity `V_D(n)`, its density is

```text
rho_visible(n) = V_D(n) * sum_(rod,regular preimage) [
    p(alpha,beta) * population_r * S_r(L) * k^2 / |J_latent(alpha,beta,u)|
].
```

This remains an intrinsic `A2/sr` measure: detector visibility is only a selection mask. No source,
optical, attenuation, detector-coordinate Jacobian, detector solid angle, or Ewald coarea factor is
introduced. Unlike the complete-sphere diagnostic, it includes the regular nonzero `m=0`
preimages. With incident `ki_z < 0` and visible top-exit `kf_z > 0`, every selected point obeys
`|Q| >= Qz > -ki_z > 0`; the collapsed direct `Q=0` root therefore cannot enter. The returned gap
is part of the result contract. The direct/specular contribution remains owned by the separately
named Parratt, kinematic, and composite specular models.

An explicitly configured empirical Parratt stitch may replace only the regular `(h,k)=(0,0)`
strength in this same continuous `A2` detector measure. Its low branch converts dimensionless
Parratt reflectivity into finite-stack strength units; its high branch is assigned exactly from the
existing internal-phase kinematic strength. It neither creates the excluded direct `Q=0` root nor
changes a nonzero-`m` rod. The external normal transfer is taken from the detector event, not
reconstructed from internal phase.

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
    * |t_in * t_out|^2 * uniform_depth_attenuation
    * exp[-mu_external(wavelength) * ray_distance]
    * source_weight * footprint * w_illum * phase_population * polarization
]
```

This direct change of variables is equivalent to the intrinsic Ewald coarea restriction followed
by the surface-to-detector map; it does not apply the coarea factor twice. The result is
`raw_detector_coordinate_density_A2_per_px2.v1` and is callable at arbitrary detector coordinates.
All inverse branches and rods sum as intensities. External attenuation uses an intensity
coefficient in inverse metres; table mode requires an exact sampled-wavelength key and never
interpolates, clamps, or falls back to a scalar.

On the common non-specular subset, the same identity can be written

```text
d(c,r) = rho_Omega(n_film(c,r))
         * J_Qsurface(c,r) / k^2
         * optical_weight(c,r) * source_phase_weight.
```

Exit refraction determines `n_film` from the external detector ray. The detector result may also
contain its separately supported regular nonzero `m=0` contribution. Consequently detector
textures are independently evaluated pullbacks, not geometric warps or projections of displayed
sphere colors.

The zero-transverse-width rod model has integrable caustic curves. The coordinate density is an
almost-everywhere representative: exact positive caustic points are marked and may be infinite,
while finite pixel box integrals remain the authoritative detector values. Display interpolation or
blur is not a physical regularization.

`pixel_solid_angle_sr` is geometry metadata. The raw detector-coordinate density already contains
the required coordinate-change Jacobian, so solid angle is not multiplied a second time as an
acceptance or efficiency factor. Any later area-normalized or caked observable must be separately
named and apply its correction once.

## Unified local-lamella `m=0` detector measure

When a `ParrattStitchStack` declares `local_lamella_follows_mosaic.v1`, its inverse-reflection map owns the complete
nonzero `(0,0)` detector field. For each sampled incident direction and detector ray, their external
momentum-transfer direction defines the unoriented local normal. The internal film normal mode gives
`L = 2 Re(kz_film) / |b3|`. The roughened Parratt--kinematic crossover is evaluated as one strength
`S_stitch(L)` in `A2`; it is not drawn or scaled as a separate curve. The per-source density is

```text
d_m0(c,r) = p_plane(alpha) * population_00 * S_stitch(L)
            * k0^2 * pixel_solid_angle_sr(c,r)
            * source_phase_weight * Thomson * sample_Q_envelope
            / (|Q_air|^2 * sin(alpha)).
```

The source phase weight already contains source probability, footprint acceptance, illuminated-path
weight, and phase population. External detector-path attenuation is applied separately once using
the actual sample-intersection-to-detector ray length. Parratt contains the applicable interface
optics, so entrance/exit transmission is not multiplied again. Every actual sampled direction and
wavelength is evaluated before incoherent reduction. Exact direct-beam `Q=0` is rejected; positive
mirror-map caustics remain explicit and are resolved by later finite-bin integration. The output stays in the same
`raw_detector_coordinate_density_A2_per_px2.v1` measure as every other rod. There is no second
count scale, horizontal shift, detector raster, smoothing, or resolution convolution.

The separate `fixed_external_qz_m0_strength.v1` convention retains the ordinary regular inverse
map, its positive-Q support gap, mosaic measure, and detector transfer. It replaces only the
`(0,0)` kinematic strength with the empirical Parratt--kinematic crossover, using
`|Q_air dot Q_film_hat|` for the external normal transfer. Nonzero rods are unchanged. This path
supports compiled CPU/CUDA and forward sampling; the local-lamella path rejects forward sampling.
The two interface conventions express different assumptions, not equivalent implementations.

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

### Intrinsic exact-rational PbI2 landmark strength

For one admitted structural landmark `j` with exact layer coordinate `L_j` and complete unique
signed-rod set `H_j`, the fixed-parent response is

```text
R[j,c] = sum_(h,k in H_j) S_c(h,k,L_j; wavelength,N,epsilon).
```

Each `S_c` is the existing nonnegative finite-per-layer PbI2 parent strength in `A2`. Independent
signed rods and parent populations add as intensities. Detector root-side identities are collapsed
because they do not change this intrinsic structural query; their keys remain sampling provenance.
The response measure ID is
`pointwise-intrinsic-summed-signed-rods-layer-L-strength-A2.v1`.

This is neither detector density nor a finite peak integral. It contains no source probability,
rod population beyond the explicit sum, mosaic probability, optical/attenuation/polarization
factor, detector-coordinate Jacobian, detector solid angle, exposure, background, or pixel/ROI
integration. Therefore measured detector counts or mosaic nuisance scales cannot be supplied as
observations of this measure without a separately declared transport and covariance boundary.

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
exit/refraction and detector map, and `W` contains source/footprint/illuminated-path/
phase/polarization mass, rod population, finite-stack strength, the Ewald coarea factor once,
entrance/exit optical and film attenuation, and external detector-path attenuation. Because the
proposal is exactly the mosaic law, its density cancels and is not
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

The optional conditional-position terminal replaces `1[T in P]` by the probability that the
projected Gaussian beam position lies in pixel box `P`, conditional on the sampled direction.
Its measure is `raw_detector_pixel_mass_conditional_position_estimate_A2.v1` and its position model
is `conditional_gaussian_pixel_integral.v1`. Source rows contain conditional position means;
the residual covariance is integrated, preserving position/divergence correlation. Divergence,
wavelength and mosaic remain sampled. This requires an unbounded planar sample, zero external-path
absorption and forward-flight margins exceeding eight spatial standard deviations when nonzero.
Six-sigma marginal/conditional cutoffs omit at most `6*Phi(-6)` (approximately `5.92e-9`) of each
Gaussian's mass, in addition to numerical box-quadrature error. Off-panel mass is lost, never
renormalized, and profiles centered outside the panel can contribute. A zero-width profile uses
the existing exact pixel owner. This is a separately declared approximation, not image blur.

`MonteCarloDetectorPresentation` is not a second scientific measure. It is a transient full-native
float32 rendering lease obtained from the same unnormalized accumulator, divided by the completed
draw count. It performs no spatial resampling, detector Jacobian, solid-angle correction,
normalization, or change of pixel ownership. The authoritative retained/exported result remains the
validated float64 `MonteCarloDetectorPixelMass` above.

`DetectorCoordinateDensityImage` is different: `sample_detector_pixel_center_density(...)` samples
the completed all-source, all-rod, all-root function once at each native center and retains
`raw_detector_coordinate_density_A2_per_px2.v1`. It is a display-only coordinate-density sample,
not a pixel-box mass, macrobin mass, calibrated count expectation, or raw OSC count image.

## Continuous incidence-exposure estimate

For a declared normalized motor exposure density `q(theta)` over one incidence interval, the
detector-native scan observable is

```text
d_scan(c,r) = integral q(theta) d_theta(c,r) dtheta,
integral q(theta) dtheta = 1.
```

`IncidenceAngleQuadrature` represents this probability measure with fixed positive dimensionless
node masses that already sum to one and an explicit calibration revision; it never silently
normalizes either quantity or supplies a default angle-axis identity. Uniform exposure uses
`q=1/(theta_upper-theta_lower)`. The estimate retains
`raw_detector_coordinate_density_A2_per_px2.v1`; total fluence or dwell is a separate dataset
scale. Its reduction ID is
`fixed_quadrature_estimate_of_incoherent_incidence_angle_probability_average.v1`.

Each node carries the complete corrected sample pose and incident transport, including footprint,
refraction, attenuation, detector mapping, inverse roots, and finite-stack strength. The optimized
path may compile angle-invariant detector physics once. The calibrated scan builder computes each
node's incident transport upstream; its exact engine rebind then recomputes transport-derived
evaluator fields, detector projection, and m=0 support gap. All
nodes reuse the exact same source rows and empirical source weights. Incidence angle is not added to the
source Monte Carlo coordinates. Invalid source/node contributions remain zero under the original
source and exposure divisors; surviving contributions are never renormalized. The v1 calibrated
configured-series builder is stricter and fails closed unless every source state is valid at every
angle node; wrappers assembled through another declared calibration path may retain partial
validity with the original full-source divisor. Physical rods,
retained roots, sources, and incidence nodes add incoherently as intensities.

The detector-coordinate point-density wrapper uses one immutable composite Gauss--Legendre rule;
its `h`/`p` convergence remains an external audit. The separate finite-region
`evaluate_adaptive_scan_oracle(...)` performs deterministic physical-panel refinement using
covariance-whitened absolute coarse/fine error and caller-declared topology, fold, and window
events. It preserves normalized exposure mass and signed ROI contrast while holding source and
evaluator revisions fixed. Neither path claims native-pixel mass. A node
landing exactly on an inverse-density caustic fails closed; it is never skipped, shifted, clipped,
or assigned zero. Intervals that need finite-region fold correction require node-specific fold
plans or an incidence-fold substitution. A single fixed-pose fold plan cannot be applied to a
multi-pose scan. The initial API is therefore authoritative for detector-native point-density and
display sampling, not native-pixel mass or the current corrected-region fitting path.

For the local-lamella stitched `m=0` term, the moving specular caustic can cross a fixed continuous
detector coordinate. Its `1/sin(alpha)` factor then makes the incidence-averaged point density
logarithmically singular on the swept trace. More angle nodes cannot turn that exact pointwise
observable into a finite value. A finite scan image at that trace requires the distinct joint
incidence-angle by detector-pixel-box mass integral; it must not be obtained by clipping or shifting
the point singularity.

No undeclared or unweighted histogram, general point-deposition API, pixel supersampling claim,
per-reflection normalization, or image maximum normalization belongs to the physical result. Masks,
saturation, detector efficiency, detector PSF, and general acquisition-matched backgrounds remain
separate future operators. The matched-region comparison below hash-verifies the declared dark;
the current Bi2Se3 basis sets its scale to zero because it is not acquisition-matched, then applies
one frozen radial halo.

## Matched-region count comparison

For row `r`, verified native OSC pixels and the declared dark OSC define the signed field
`D = raw - s_dark dark`. Raw and dark use the same sparse continuous-region projector `W`; no
negative value is clipped. Independent raw/dark counting covariance propagates as
`W diag(max(raw,1) + s_dark^2 max(dark,1)) W^T`, including the cross-OSC covariance induced by
reusing the same dark exposure when `s_dark` is nonzero. When no acquisition-matched dark exists,
the explicit `s_dark=0` basis keeps the dark bytes hash-verified while its subtraction, covariance,
and cross-OSC contribution are exactly zero. The corrected native pixels define a piecewise-constant
measured count-density field.
A sparse data-only operator integrates that field over the declared phi/two-theta or signed-Qr/L
rectangle. Its weights are detector areas of pixel/region overlap under independently refined
continuous cubature, so `C_r` and its support use the same rectangle as the model without smoothing.
Fractional sharing propagates the full regularized plug-in count covariance
`W diag(max(c,1)) W^T`; the declared one-count variance floor is revision-bound. Model mass `M_r` is
the continuous detector density integrated directly over that rectangle with the detector-area
Jacobian. It is never sampled or aggregated as a model raster.
Dataset scale `s_d` multiplies every family in dataset `d` once.
The frozen radial calibration supplies background mass `B_r` and covariance. Adjacent-anchor
conditioning is applied first. If peak aggregation is declared, a fixed matrix `G` sums every
conditioned signal row exactly once into its trusted peak: `y_peak=G y`, `mu_peak=G mu`, and
`Sigma_peak=G Sigma G.T`. Otherwise the conditioned signal rows remain separate. Both forms
use one nonnegative `s_d` per dataset through the shared native search. The identical
two-anchor projection transforms the full measured-count and radial-background covariance before
whitening. Division by support occurs only for displayed density profiles. Where a
continuous region crosses the inverse-map fold, fixed-rule mass from the complete same-`|b|` rod
group is removed and replaced by exact-`x` cubature with `x^2 = Qr^2 - b^2`.

## Sparse structure-response factorization

At selected continuous detector coordinates, contract v13 stores
`d_i = sum_t R_it S_t`. `R_it` has units `px^-2` (the detector-density to structure-strength
ratio) and contains every
fixed source, wavelength, signed rod, inverse root, mosaic, optical, polarization, attenuation,
sample-Q-envelope, and detector-Jacobian factor. `S_t` is the candidate provider's nonnegative
`A2` strength at its exact `(h,k,L,k_norm)` query. Applying a candidate introduces no detector
solid-angle efficiency correction, normalization, per-profile scale, or hidden pruning; source,
wavelength, rod, root, and parent contributions remain incoherent intensities.

`ParameterizedStructureRegionModel` integrates the applied raw detector density through each
block's exact `ContinuousRegionQuadrature`. Candidate site or population coordinates change only
`S`; transfer, quadrature, measured background and covariance stay frozen. Separate dataset
scales are profiled jointly with the full declared covariance at each candidate.
Regular kinematic `00L` uses the same factorization after its positive-Q support gate. The optional
local-lamella Parratt composite is a separate declared observable and is not part of this response.


## Conditional native strength-weighted measure

Regular conditional integration uses W(u) du, W=S+(u)+S-(u), including canonical
physical rod populations and incoherent mixture weights. Interior Gaussian nodes
carry physical mass and S+/W, S-/W fractions. The shared compiler applies dL/du
once; signed contraction retains every original source, cone, optical, attenuation,
polarization, phase and deposition factor. No probability/image renormalization
is introduced. Analytic zero populations/occupancies contribute zero; unresolved
sampled-zero measures fail.

Native images/regions accumulate integrated native pixel mass through literal
fractional memberships. Continuous `density_at` is a distinct A2/px2 observable
through the same prepared event kernels. Native parent/child indicators and discrete
moments do not qualify continuous regions, axial convergence or a full physical
image. Earlier continuous matched-region and structure-response contracts above
remain separate declared observables.
