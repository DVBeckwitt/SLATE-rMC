# Forward dovetail matrix

This document is the authoritative producer-consumer map for the four forward worktrees. A branch may use synthetic contract fixtures during isolated development. Concrete cross-branch calls are assembled only by T07 integration.

## Runtime data flow

```text
one complete canonical T03 `IncidentSampleBatch`
    -> T02 incident geometry and entrance optics
    -> self-contained `IncidentStateBatch`
    -> T03 mosaic and Ewald event support (no raw-source rejoin)

T04 rod catalog ----------------------^
T04 material optics -> T02 optics

T03 scattering events
    -> T04 ordered scattering strengths
    -> T05 stacking scattering strengths using T04 layer amplitudes
    -> T02 exit transport and continuous detector hits

T02 hits + T04 or T05 strengths
    -> T07 complete-pool selection, conservative deposition, raw detector image
```

## Interface ownership

| Contract or factor | Producer | Consumer | Acceptance rule |
|---|---|---|---|
| `IncidentSampleBatch` | T03 sampling | T02 geometry | one complete PCG64/v2 realization; every row has exact empirical mass `1/N`; no generating PDF weight or worker-local generation |
| `CompiledInstrument` | geometry compilation | T02 geometry, T03 reciprocal orientation, T07 measurement | retained causal transforms/calibration/support plus owner-computed `sample_geometry_revision` under schema `sample_entrance_revision.v2`; local goniometer composition and dead transform derivatives are not stored |
| `MaterialOptics` | T04 materials | T02 optics and T04 Parratt | one sorted unique exact wavelength grid; `n_complex` is the sole optical-array authority and the owner supplies `material_revision` under schema `material_optics_revision.v2` |
| `IncidentStateBatch` | T02 geometry/optics | T03 Ewald | sole incident authority with exact state/sample identity, wavelength, polarization, status, mass, `ki`, optics, source-parameter/source/sample/material revisions, and incident-model ID; no companion source join or downstream rehash |
| `RodCatalog` | T04 ordered | T03 Ewald, T08 selection | every `(h,k)` rod remains distinct; exact family metadata is preserved |
| `ScatteringEventBatch` | T03 Ewald | T02 exit transport, T04 ordered, T05 stacking, T07 integration | pre-selection candidates retain exact rod/orientation/`Q`/`kf` identity and mosaic/Jacobian mass for one all-rod pool; no per-reflection normalization or post-selection reweighting |
| `LayerAmplitudeResult` | T04 ordered motifs | T05 stacking | event/rod alignment, phase convention, motif gauge, and normalization are explicit |
| `LayerNormalQBatch` | future T07 projection | T05 stacking | exact event/rod/phase/gauge alignment; projection comes from event `Q` and T04 metadata with no sample-`Qz` fallback |
| `EventIntensityResult` | T04 ordered or T05 stacking | T07 integration | unweighted polarization-neutral `r_e² × electron²` in `angstrom²`; typed unit-cell, finite-total, or finite-per-layer normalization |
| `OutgoingWaveBatch` | T02 optics | T02 detector and T07 integration | exit amplitude, attenuation, phase direction, and validity are separate fields |
| `DetectorHitBatch` | T02 geometry | T07 measurement/render | continuous `(column_px,row_px)`, event IDs, and solid-angle metadata; solid angle never enters raw rendering |
| `PixelContributionBatch` | T07 render | T07 image reduction | deposition weights conserve event mass under the declared clipping rule |

## Factor ownership

```text
uniform source empirical mass    T03
footprint acceptance             T02
mosaic/Jacobian candidate mass   T03
ordered or stacking strength     T04 or T05, including r_e^2 exactly once
phase/parent population mass     T07 from T04/T05 metadata
Fresnel and attenuation          T02
scattering polarization          T07 from source state and event direction
pool selection and equal T/N     T07
deposition and clipped mass      T07
detector solid-angle metadata    T02, excluded from raw rendering
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
2. generate one complete T03 `IncidentSampleBatch` -> T02 `IncidentStateBatch` -> T03 events;
   reciprocal construction consumes only the state batch
3. T03 RodQueryBatch -> T04 ordered strength
4. T04 LayerAmplitudeResult + T07 LayerNormalQBatch -> T05 stacking strength
5. T03 outgoing film wavevector -> T02 exit transport and hits
6. T02 hits + T04 strength -> T07 raw detector image
7. substitute T05 strength under the same interface
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
Parallel Task 1.1 introduces no runtime packet layout. Any private packet layouts introduced by
future staged execution belong to parallel Task 1.4, after the Task 1.3 numeric boundary, and are
row-index views of the canonical incident ledger. They inherit the complete parent revision/model
envelope and restore canonical order before comparison or reduction; worker count, tile size, and
completion order cannot change source generation, incident status, or numeric payload.
