# Result and factor measure

## Declared final output

The integrated core returns a nonnegative detector-native array of sampled scattering mass in `angstrom^2` per pixel. Its normalized ensemble mean is the raw detector observable; it is not differential per solid angle.

The default source is fixed-seed randomized Latin-hypercube Gaussian sampling with independent dimensions, antithetic pairs, and an odd central ray. Every row has exact empirical mass `1/N`; the generating Gaussian PDF is never a weight. Deterministic Gauss–Hermite is an oracle only.

The raw result does not apply incident flux, exposure, gain, detector quantum efficiency, background, pixel solid angle, maximum normalization, or display rescaling.

## Complete-pool selection and deposition

For each incident ray and independent phase/parent, valid-support construction forms one candidate pool spanning every individual `(h,k)` rod and valid mosaic/`Q` solution. Each candidate retains its own rod, orientation, `Q`, elastic `kf`, detector hit, scattering strength, mosaic mass, and other physical factors. `Qr` is never candidate identity.

For candidate `i`,

\[
m_i=w^{\mathrm{src}}w^{\mathrm{recip}}_i w^{\mathrm{pop}}
S_i W^{\mathrm{opt}}_i W^{\mathrm{foot}} W^{\mathrm{pol}}_i,
\qquad T=\sum_i m_i,
\qquad P(i)=m_i/T.
\]

`S_i` is polarization-neutral `r_e^2` times raw electron² in `angstrom^2`. T04 or T05 applies the single `core.scattering` conversion exactly once. `w_recip` is the candidate mosaic/Jacobian mass, and source and population masses are independent incoherent factors.

T07 selects a configurable `N` outgoing events from the complete pool by seeded cumulative inverse CDF (legacy default `50`). Every selected event receives exactly `T/N` and uses its selected candidate's own geometry and hit. For detector pixel `p`,

\[
M_p=\sum_s \frac{T}{N}D_{sp}.
\]

Bilinear deposition splits that mass once and reports edge clipping explicitly. There is no per-reflection normalization and no post-selection source PDF, structure factor, mosaic mass, selection probability, or solid-angle multiplier. Deterministic/adaptive support construction precedes statistical selection; two-pass or streaming enumeration is preferred so the incident×rod×mosaic product is not retained.

## Optical model for the first reference core

Use

\[
W_T=|T_{\mathrm{in}}|^2|T_{\mathrm{out}}|^2
\]

and, for a uniformly scattering film of thickness `t`,

\[
\overline W_{\mathrm{prop}}=
\frac{1-\exp[-2(\kappa_i+\kappa_f)t]}
{2(\kappa_i+\kappa_f)t}.
\]

Then

\[
W^{\mathrm{opt}}=W_T\overline W_{\mathrm{prop}}.
\]

This is the current off-specular scalar reference model described by manuscript equations `eq:si_scalar_transmission`, `eq:si_entry_exit_transmission`, `eq:si_transmission_intensity`, `eq:si_abs_complex_kz`, and `eq:si_full_optical_weight_lambda`.

A path-length attenuation result may also be exposed when an event has an explicitly defined scattering depth. Full multilayer distorted fields are deferred to a separately named later model.

## Coherence

Sum amplitudes before squaring for:

- atoms in one unit cell or layer motif
- coherent layers in an ordered finite stack
- correlated layer pairs in the transition model

Sum intensities for:

- independent source samples and wavelengths
- incoherent mosaic orientations
- distinct crystalline phases
- distinct parent-rich stacking populations
- distinct rods after their own event geometry is evaluated


## Polarization declaration

Polarization is separate from the scalar Fresnel coefficient. Every dataset/model pairing declares one of:

```text
MODEL
    compute the Thomson scattering polarization factor from a declared
    incident polarization state and outgoing direction

DATA_CORRECTED
    measured data were consistently polarization-corrected and W_pol = 1

UNITY_APPROXIMATION
    W_pol = 1 is an explicit approximation recorded in provenance
```

There is no silent unity default. A full vector polarization-resolved interface-field calculation remains deferred.

## Event Jacobian and Lorentz terminology

`w_recip` contains the candidate measure/Jacobian required by reciprocal support construction. It enters `m_i` once and is not multiplied after selection. Do not add a second empirical or powder Lorentz factor.

## Solid angle

For a flat detector pixel of area `A_pixel`, ray distance `R`, detector normal `n`, and unit ray direction `r_hat`,

\[
\Delta\Omega=A_{\mathrm{pixel}}
\frac{|\hat{\mathbf n}\cdot\hat{\mathbf r}|}{R^2}.
\]

`pixel_solid_angle_sr` is immutable geometry metadata for optional later caking or analysis. It cannot change the raw detector image. Bilinear deposition is numerical support allocation, not a physical point-spread function.

## Reflectivity

Pure Parratt, pure kinematic specular intensity, and the named smooth composite are separate outputs. They are not silently added to off-specular scattering strength. Integration declares how a specular-family output enters the detector image.
