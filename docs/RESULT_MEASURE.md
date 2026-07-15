# Result and factor measure

## Declared final output

The first integrated core returns a nonnegative detector-native array of relative expected photon mass per pixel for explicit normalized source distributions, explicit phase/population weights, and the selected scalar optical model.

It does not silently apply incident flux, exposure, gain, detector quantum efficiency, background, maximum normalization, or display rescaling.

## Event-to-pixel equation

For detector pixel `p`,

\[
M_p=
\sum_e
w^{\mathrm{src}}_e
w^{\mathrm{recip}}_e
w^{\mathrm{pop}}_e
I^{\mathrm{model}}_e
W^{\mathrm{opt}}_e
W^{\mathrm{foot}}_e
W^{\mathrm{pol}}_e
\Delta\Omega_e
D_{ep}.
\]

Definitions:

```text
w_src
    integrated source and wavelength probability mass

w_recip
    integrated mosaic/Ewald support mass, including the declared event Jacobian

w_pop
    declared incoherent crystalline-phase or parent-population mass

I_model
    ordered or stacking differential scattering intensity for the event

W_opt
    scalar entrance/exit field intensity factor times attenuation

W_foot
    footprint acceptance or partial source-cell mass

W_pol
    scattering polarization factor, distinct from Fresnel interface fields

DeltaOmega
    detector-pixel solid angle when I_model is per solid angle

D_ep
    mass-conserving deposition fraction from event e to pixel p
```

Every factor has one owner and is multiplied exactly once in integration.

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

`w_recip` already contains the measure/Jacobian required when the reciprocal support is integrated. Do not multiply a second empirical or powder Lorentz factor unless an independent derivation shows it is absent from the event measure. The trace and factor ledger record this decision.

## Solid angle

For a flat detector pixel of area `A_pixel`, ray distance `R`, detector normal `n`, and unit ray direction `r_hat`,

\[
\Delta\Omega=A_{\mathrm{pixel}}
\frac{|\hat{\mathbf n}\cdot\hat{\mathbf r}|}{R^2}.
\]

Apply it only when converting differential intensity per solid angle to detector pixel mass. Bilinear deposition is numerical support allocation, not a physical point-spread function.

## Reflectivity

Pure Parratt, pure kinematic specular intensity, and the named smooth composite are separate outputs. They are not silently added to off-specular event intensity. Integration declares how a specular-family output enters the detector image.
