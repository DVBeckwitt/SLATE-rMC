# Result and factor measure

## Declared final output

The integrated core returns a nonnegative detector-native array of sampled scattering mass in `angstrom^2` per pixel. Its normalized ensemble mean is the raw detector observable; it is not differential per solid angle.

The default source uses `independent_gaussian_antithetic_lhs.v2` with an explicit
`numpy_pcg64.v1` generator. For `N=2p`, each of the five independent dimensions permutes the
lower strata `0..p-1`, draws one coordinate strictly inside each selected `N`-stratum, and places
its exact complement in the paired upper stratum on the adjacent row. For `N=2p+1`, the final row
is exactly the middle coordinate `0.5`. Every row has exact empirical mass `1/N`; the generating
Gaussian PDF is never a weight. The batch retains canonical, unit/frame-labelled text for only the
declared source parameters plus its independently validated SHA-256 revision. Sampling-model ID,
RNG-model ID, and nonnegative seed are separate fields; together with the parameter revision and
realized rows they determine the complete source revision. Packet or worker layout is not source
provenance. Deterministic Gauss–Hermite is an oracle only.

The complete canonical source realization is generated and hashed once. Worker count, packet size,
backend, and completion order neither enter the RNG key nor create slice revisions. The
self-contained incident ledger preserves every row and its `1/N` mass through geometry or optical
failure; rejected rows are not renormalized away.

The raw result does not apply incident flux, exposure, gain, detector quantum efficiency, background, pixel solid angle, maximum normalization, or display rescaling.

## Current finite-pool selection and deposition

For each incident ray and independent phase/parent, the current T07 comparison path forms one
candidate pool spanning every individual `(h,k)` rod and valid mosaic/`Q` solution that also has a
valid detector hit. `pipeline.selection.CandidatePool` requires exactly those detector-valid event
rows. Each candidate retains its own rod, orientation, `Q`, elastic `kf`, detector hit, scattering
strength, mosaic mass, and other physical factors. `Qr` is never candidate identity.

For candidate `i`,

\[
m_i=w^{\mathrm{src}}w^{\mathrm{recip}}_i w^{\mathrm{pop}}
S_i W^{\mathrm{opt}}_i W^{\mathrm{foot}} W^{\mathrm{pol}}_i,
\qquad T=\sum_i m_i,
\qquad P(i)=m_i/T.
\]

`S_i` is polarization-neutral `r_e^2` times raw electron² in `angstrom^2`. T04 or T05 applies the single `core.scattering` conversion exactly once. `w_recip` is the candidate mosaic/Jacobian mass, and source and population masses are independent incoherent factors.

T07 selects a configurable `N` outgoing events from this finite detector-valid pool by seeded
cumulative inverse CDF (legacy default `50`). Its current `T` and CDF are therefore
detector-conditioned. Every selected event receives exactly `T/N` and uses its selected
candidate's own geometry and hit. For detector pixel `p`,

\[
M_p=\sum_s \frac{T}{N}D_{sp}.
\]

Bilinear deposition splits that mass once and reports edge clipping explicitly. There is no per-reflection normalization and no post-selection source PDF, structure factor, mosaic mass, selection probability, or solid-angle multiplier. Deterministic/adaptive support construction precedes statistical selection; two-pass or streaming enumeration is preferred so the incident×rod×mosaic product is not retained.

This finite-pool behavior is temporary comparison evidence for parallel Tasks 1.1--1.2; it is not
the accepted continuous-coating measure. The future continuous-coating replacement uses
detector-unconditioned component masses and CDFs. Under that replacement, detector projection and
clipping occur after sampling; a detector miss is rejected mass and cannot change source/incident
revisions, component masses, or coating CDFs.

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
