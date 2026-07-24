# T13: ordered intensity fit

Status: `COMPLETE_FIXED_POSITION_250_STATE_SYNTHETIC_SELECTED_CENTER_SLICE`.

Branch: `codex/bi2se3-250ki-refit`

This contract-v10 slice preserves the finite-ROI mass API but uses the distinct source-averaged
selected-center response for the active Bi2Se3 proof. It does not restore retired event/hit APIs.

## Goal

With geometry, detector center, lattice, mosaic, atom positions, optics, and stacking frozen, fit
the three unique site occupancies and directional `Ur/Uz` factors to three incidence datasets at
once. Every incident state must contribute to one detector function per incidence before any
normalization, scale projection, comparison, or residual.

The accepted slice proves synthetic selected-component identifiability and acceleration. It does
not claim ordered-intensity extraction or structure recovery from unresolved raw OSC counts.

## Owned paths

```text
src/rasim_next/fitting/ordered_intensity.py
src/rasim_next/fitting/__init__.py
src/rasim_next/pipeline/_continuous_detector_kernel.py
src/rasim_next/pipeline/configured_simulation.py
src/rasim_next/pipeline/source_averaged_detector.py
scripts/recover_bi2se3_mosaic.py
scripts/recover_bi2se3_ordered_intensity.py
tests/test_fitting.py
tests/test_integration.py
the live contracts, architecture, validation, examples, ledger, roadmap, and task records
```

## Required behavior

- use one shared 250-row source realization across 5, 10, and 15 degrees;
- bind every response to exact source count and source revision;
- reduce source intensities incoherently into one detector function per incidence;
- retain all `88/78/72` frozen selected centers, including every weak nonzero identity and all six
  admitted branchless `m=0` anchors in unfiltered synthetic mode;
- when a measured mosaic is supplied, compile exactly its fit-eligible identities and reject stale,
  missing, or unknown-dataset selection records;
- require positive finite baseline support from the complete source-averaged detector for every
  admitted `m=0` anchor;
- reject mixed point-density and finite-ROI-mass observations;
- fit any enabled subset of `oBi`, `oSe1`, `oSe2`, `Ur`, and `Uz`, freezing the complement exactly;
- reject attempts to move the two Wyckoff coordinates in this phase;
- reject the exact common-occupancy gauge when analytic image scales are enabled unless one positive
  occupancy is fixed as ratio reference; and
- keep structural parameters global and dataset scales dataset-specific, with no per-source or
  per-peak fit amplitudes.

## Implementation

For each frozen selected center, the compiler evaluates six occupancy probes through the selected-
group view of the complete source-averaged, all-root detector. Thirteen Chebyshev-Lobatto `Uz`
nodes compile the noncommuting source/root `Qz` response; 12 interlaced full-detector nodes must pass
the declared `2e-3` maximum occupancy-contracted signal-error gate. The conservative profile-local
generalized-eigenvalue certificate covers every occupancy direction with a declared `1e-12`
extinct-mode floor; no weak profile borrows a scale from a stronger profile.
Accepted rod-family `Qr^2` damping is exact. Optimizer
iterations contract only the occupancy quadratic and directional damping; they do not project the
detector or recalculate the full structure factor.

Recovery schema v3 records compiler contract
`source-averaged-selected-center-occ-quadratic-chebyshev-qz-spectral.v2` and its `1e-12` spectral
floor. Rendering rejects stale schema/compiler provenance before evaluating the detector.

Synthetic truth is generated independently through a fresh authoritative combined-detector
evaluation at the same frozen selected centers. The selected-center measure is
`selected_group_angular_signal_density_A2_per_rad2.v1`, not integrated peak mass or raw OSC counts.

## Proof commands

```powershell
python -m compileall -q src
ruff check src scripts/recover_bi2se3_ordered_intensity.py tests
pytest -q
python scripts/recover_bi2se3_ordered_intensity.py `
  --source-sample-count 250 `
  --mosaic-result C:\path\outside\the\repository\bi2se3-real-mosaic\bi2se3_real_mosaic_fit.json `
  --execution-backend cuda `
  --output C:\path\outside\the\repository\bi2se3-sf\ordered_intensity_result.json `
  --json
git diff --check
```

## Handoff

Commit SHA: recorded in the final branch handoff because a commit cannot contain its own SHA.

Accepted ordered-model revision: `bi2se3_fixed_position_occ_directional_u.v1`.

The unfiltered simultaneous 5/10/15-degree proof retains 238 centers (`88/78/72`), including six
`m=0` and every frozen weak nonzero identity. The measured-mosaic command above instead retains its
exact `2/5/8 = 15` fit-eligible profiles, including five `m=0`. All three responses require source revision
`42e8108b3ba9c66cb7752bc7e9d2f9fdb8401987a8f9a0eb272d3f4edeaa39a7` and 250 states.

For the current measured-selection handoff, the maximum 13-node/12-interlaced-node `Uz` certificate
is `1.19096e-11`; the compact response agrees with a fresh combined-detector prediction to
`8.09542e-16`. Absolute recovery identifies all five coordinates with rank 5, condition `31.1434`,
maximum residual `1.9984e-15`, and maximum parameter error `1.22e-15`. Relative recovery fixes
`oBi=1`, identifies the two occupancy ratios and `Ur/Uz` with rank 4, condition `12.8583`, maximum
residual `1.1102e-15`, and maximum parameter error `1.17e-16`. No coordinate contacts a bound.

The measured-selection responses contain `156/390/624` coefficients. Compilation took `171.912 s`;
fresh versus cached equivalent truth prediction took `3.83954/0.0007503 s` (`5117.35x`); absolute
and relative fits took `0.04147/0.02670 s`; total proof time was `249.477 s`. Traced peak memory was
`62,674,047` bytes. The current ordered JSON SHA-256 is
`432a402bf675c0411c399fd7f0327175be36b54576a611dfc431cdaef5a048ea`; it binds mosaic SHA-256
`69c782f67216ea595b72abced82a9660833d1acbee7b38ffaedb2de7d81df829` and is external at
`C:\Users\Kenpo\.codex\visualizations\2026\07\22\019f8a1c-5dba-7ce2-afae-11e6953c18f4\bi2se3_250ki_sf_gate_v2\ordered_intensity_result.json`.

The separate unfiltered full-catalog proof has `6864/6084/5616` coefficients, a
`2.084780066e-10` certificate, and `8.976202640e-15` direct parity. Its `88/78/72` catalog is not
the measured `--mosaic-result` handoff.

The separate display-only 3,000-by-3,000 render used CUDA, all 250 states, all 85 rods, all roots,
and `m=0`. It took `3839.456 s` for the three incidences and evaluated
`4,577,845/4,377,290/4,170,019` detector centers after the conservative top-exit cull. This raster
was generated after fitting and is not a fit input or a count-calibrated comparison.

Legacy classification is `CORRECTED` for raw, unrounded CIF-derived strength with explicit
occupancy and directional damping, and `NO_ORACLE` for raw-OSC component extraction, background,
noise, PSF, count calibration, or real-data parameter recovery.

Permanent tests retain the combined source-sum contract, m=0 and nonzero direct-oracle parity,
occupancy-quadratic/direct-strength parity, directional damping, source count/revision binding,
mixed-measure rejection, freeze/gauge/rank/bound contracts, and native-center sampling. Each
protects a distinct public or scientific invariant.

Limitations: the current structure basis is Bi2Se3-specific; atom motion, per-site `Uij`, calibrated
background/noise/PSF, uncertainty, raw-OSC deblending, and material-neutral structure bases remain
unproven. New materials require their own structure basis, identity catalog, and response proof.
