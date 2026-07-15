# Forward dovetail matrix

This document is the authoritative producer-consumer map for the four forward worktrees. A branch may use synthetic contract fixtures during isolated development. Concrete cross-branch calls are assembled only by T07 integration.

## Runtime data flow

```text
T03 source samples
    -> T02 incident geometry and entrance optics
    -> T03 mosaic and Ewald event support

T04 rod catalog ----------------------^
T04 material optics -> T02 optics

T03 scattering events
    -> T04 ordered event intensities
    -> T05 stacking event intensities using T04 layer amplitudes
    -> T02 exit transport and continuous detector hits

T02 hits + T04 or T05 intensities
    -> T07 factor assembly, solid angle, deposition, detector-native image
```

## Interface ownership

| Contract or factor | Producer | Consumer | Acceptance rule |
|---|---|---|---|
| `IncidentSampleBatch` | T03 sampling | T02 geometry | weights are integrated source/wavelength mass and sum under the declared normalization |
| `MaterialOptics` | T04 materials | T02 optics and T04 Parratt | wavelength grid, complex index convention, `delta`, `beta`, and absorption are identical |
| `IncidentStateBatch` | T02 geometry/optics | T03 Ewald | frames, real phase wavevector, complex normal mode, source ID, and validity are explicit |
| `RodCatalog` | T04 ordered | T03 Ewald, T08 selection | every `(h,k)` rod remains distinct; exact family metadata is preserved |
| `ScatteringEventBatch` | T03 Ewald | T02 exit transport, T04 ordered, T05 stacking, T07 integration | event IDs, internal `Q`, `Qz`, `L`, wavelength, outgoing film phase wavevector, and reciprocal mass agree |
| `LayerAmplitudeResult` | T04 ordered motifs | T05 stacking | event/rod alignment, phase convention, motif gauge, and normalization are explicit |
| `EventIntensityResult` | T04 ordered or T05 stacking | T07 integration | same event IDs and declared differential-intensity measure; one model selected per event contribution |
| `OutgoingWaveBatch` | T02 optics | T02 detector and T07 integration | exit amplitude, attenuation, phase direction, and validity are separate fields |
| `DetectorHitBatch` | T02 geometry | T07 measurement/render | continuous `(column_px,row_px)`, solid angle, and event IDs only; no deposition |
| `PixelContributionBatch` | T07 render | T07 image reduction | deposition weights conserve event mass under the declared clipping rule |

## Factor ownership

```text
source and wavelength mass       T03
footprint acceptance             T02
mosaic/Ewald mass and Jacobian   T03
ordered or stacking intensity    T04 or T05
phase/parent population mass     T07 from T04/T05 metadata
Fresnel and attenuation          T02
scattering polarization          T07 from source state and event direction
detector solid angle             T07
deposition support               T07
```

No factor may be hidden inside a contract owned by another branch. T06 review checks both omissions and duplicates.

## Import rule

During isolated proof work:

- T02, T03, T04, and T05 import only bootstrap-owned core contracts and shared primitives.
- They do not import another physics branch's concrete modules.
- Synthetic fixtures stand in for upstream and downstream contracts.

After integration, `pipeline/` is the only layer that calls concrete modules across domains. Shared numerical primitives remain in `core/` and are never copied into branch packages.

## Mandatory boundary proofs

Before a branch is accepted for integration, prove:

1. schema version, field names, shapes, dtypes, units, frames, and measure metadata
2. stable source, rod, event, and model IDs
3. no implicit interpolation or family collapse at an interface
4. valid/invalid status propagation without sentinel overloading
5. round-trip serialization through the common proof format
6. exact factor ownership under `docs/RESULT_MEASURE.md`

## Vertical integration order

```text
1. T04 RodCatalog -> T03 with synthetic IncidentStateBatch
2. T03 IncidentSampleBatch -> T02 -> T03 events
3. T03 RodQueryBatch -> T04 ordered intensity
4. T04 LayerAmplitudeResult + T03 query -> T05 stacking intensity
5. T03 outgoing film wavevector -> T02 exit transport and hits
6. T02 hits + T04 intensity -> T07 detector mass
7. substitute T05 intensity under the same interface
8. run the non-square tiny end-to-end case
```

Each slice must pass before the next branch is merged.

## Reuse by later fitting

```text
instrument/source fit
    rebuild source and/or instrument states

sample geometry fit
    rebuild incident states, events, hits, and response

mosaic fit
    reuse instrument and rods; rebuild reciprocal weights/support as required

ordered or stacking intensity fit
    reuse event geometry, hits, and detector response whenever model parameters do not change them
```

Selection and fitting consume the integrated identities and compiled states. They do not create alternate forward paths.
