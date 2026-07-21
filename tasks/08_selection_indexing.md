# T08: post-integration selection and indexing

Status: `COMPLETE`.

This task was replanned against the detector-native integer-L and OSC-orientation contracts. It
does not restore deleted event or hit APIs.

Branch: `codex/real-osc-indexing`

## Goal

Create stable rod-family, reflection-group, and physical branch identities and associate measured detector-native observations without embedding identity changes inside an optimizer.

## Owned paths

```text
src/rasim_next/selection/
tests/test_selection.py
scripts/index_bi2se3_osc.py
this task's execution-plan and handoff sections
```

## Reference map

```text
original RASIM
    ra_sim/utils/calculations.py:48-62,90-117
    ra_sim/gui/geometry_q_group_manager.py:1197-1329
    ra_sim/fitting/caked_geometry_objective.py:188-228

manuscript
    sections/refinement_workflow.tex:29-59
    2D_Supplemental/SI_failure_modes.tex:691-748
```

## Required work

1. Discover detector features globally in a tiled inverse-resampled `(2theta, phi)` search cake.
   This API accepts no marker catalogue or predicted marker coordinates.
2. Refine each proposal once on the detector-native raster and retain separate ridge-support and
   localization weights. These are conservative operational weights, not calibrated statistical
   coverage.
3. Convert only the discovered detector points to `q_sample` with the canonical detector Ewald
   geometry.
4. Infer family `m`, integer `L`, analytic Ewald branch, and beta-root sign from reciprocal
   geometry. Use `solve_integer_l_ewald_roots` as the branch authority; never infer branch from
   left/right display placement.
5. Apply quarter-separation family, integer-L, and beta-root gates with a frozen localization-weight
   multiplier. Reject tangencies, noncoincident symmetry-rod aliases, conflicting labels, and
   unresolved ownership.
6. Generate exact alpha-zero integer-L anchor coordinates only after the discrete label is frozen.
7. Keep weak or absent sites absent; never infer extinction from non-detection.
8. Accept a root-side branch track only after three ordered local sites and replication at two
   distinct incidences with at least two common L anchors. Detector-data hashes, indexing contexts,
   and predicted track geometry must also be distinct. Export no site from a rejected track, and
   export an individual exact-L site to geometry only when that `(m,L,root)` identity itself appears
   at two distinct incidences.
9. Hash exact image, mask, detector/cake context, reciprocal context, policies, decisions, and
   branch-track outcomes.
10. Apply OSC clockwise orientation once at `read_osc`; detect in cake coordinates, refine and fit
    in native `(column_px, row_px)` coordinates.
11. Require the discovery calibration hash to match the q-indexing calibration exactly. Bound the
    post-label alpha-zero anchor distance at 32 pixels; this preserves the real peak tracks while
    rejecting a reproducible 58--81 pixel parallel-ridge alias.

## Proof

- non-square rotated-detector cake/native round trip
- global discovery without marker-position input and invariance under image-unit scaling
- exact q-space label-algebra recovery of configured Bi2Se3 labels and root signs
- withheld-center noisy raster discovery and q indexing under a deliberately shifted coarse panel
- conservative anisotropic ridge-support covariance plus separate label covariance
- global assignment ambiguity rejection and hash stability
- distinct-incidence/shared-L branch replication and duplicated-data spoof rejection
- once-clockwise OSC orientation and direct legacy-native coordinate audit
- position-free 3000x3000 native simulation qualification after indexing is frozen

## Commands

```bash
PYTHONPATH=src python -m compileall -q src
ruff check src/rasim_next/selection scripts/index_bi2se3_osc.py tests/test_selection.py
PYTHONPATH=src pytest -q -p no:cacheprovider tests/test_selection.py
PYTHONPATH=src python scripts/index_bi2se3_osc.py
PYTHONPATH=src python scripts/index_bi2se3_osc.py --simulation-diagnostic EXTERNAL.ra_diag.npz
git diff --check
```

## Stop conditions

Stop `BLOCKED` if the continuous rod/root and detector contracts do not expose enough frame,
identity, or projection metadata. Request the smallest shared-contract change rather than inferring
identity from display pixels.

## Execution plan

State: COMPLETE

1. Freeze coordinate, mask, identity, and confidence contracts. `COMPLETE`
2. Implement local predicted-window indexing for comparison and geometry handoff. `COMPLETE`
3. Implement position-prior-free global cake discovery and q-space indexing. `COMPLETE`
4. Validate the three real OSC images and external native simulation qualification. `COMPLETE`
5. Run repository gates, remove temporary audits, and create one coherent commit. `COMPLETE`

## Handoff

Status: READY

Commit SHA: reported by the final branch handoff because a commit cannot contain its own SHA

Selection schema version: `measured-position-free-peak-discovery.v1`,
`position-free-reciprocal-indexing-context.v2`, and
`measured-indexing-manifest.v2`, plus `bi2se3-position-free-simulation-qualification.v2`

Public APIs: `discover_measured_cake_peaks`, `index_discovered_integer_l_peaks`,
`index_measured_integer_l_branches`, and `select_confident_branch_tracks`, with their immutable
policy, discovery, image-result, manifest, decision, and branch-track records.

Legacy classifications:

- `MATCH`: one clockwise OSC rotation at the I/O boundary and direct legacy displayed
  `(legacy_raw_x, legacy_raw_y)` detector coordinates.
- `CORRECTED`: tracked CSV `observed_column_px` applies an extra horizontal reflection and is not
  used as a numerical oracle. The named first divergence is the CSV canonicalization
  `observed_column_px = 2999 - legacy_raw_x`; the accepted once-clockwise native coordinate is
  `(legacy_raw_x, legacy_raw_y)`. Legacy side index 0 maps to tag branch 2/root `+1`, while side
  index 1 maps to tag branch 1/root `-1`.
- `NO_ORACLE`: non-detection is not classified as physical extinction, and exact integer-L
  alpha-zero anchors are not asserted to be raster maxima.

Proof summary:

- Five compact selection tests pass. The end-to-end test withholds all six raster centers from the
  indexer, recovers all six `(m,L,branch,root)` labels within 0.5 pixels, retains the same labels
  under a 3.606-pixel coarse-calibration shift, and returns no labels for noise plus masked-edge
  artifacts. The blind q-label test also rejects authoritative tangencies, conflicting rod labels,
  and noncoincident symmetry-rod anchors.
- The full repository suite passes 75 tests. Exact detector/cake round trips, image-unit scaling,
  mixed-frame rejection, noncoincident rod rejection, assignment ambiguity, missing-site handling,
  duplicate-data spoof rejection, and exact hash mutation are covered.
- The real 5/10/15-degree run globally discovers the detector features without marker positions.
  With the 32-pixel anchor gate it accepts both m=1 root-side tracks at all three incidences and
  exports 10, 8, and 8 replicated exact-L sites respectively; the manifest is
  `sha256-f333ecbc8b1e62a5d2dcf45ab7e78b6bcaaa699fe65afc5af458b6d6e319a365`.
  The site-level incidence gate removes the lone 10-degree L=13/tag-2 point: it was locally
  detectable but appeared at only one incidence and lay 29.707 pixels from the corresponding
  direct legacy pick. Of the remaining exported sites, 24 have legacy rows and agree at 1.741
  pixels RMS and 5.103 pixels maximum; the two 10-degree L=4 sites have no legacy-table rows.
- The direct-coordinate audit was performed only after discovery. Across all 82 legacy selections,
  the direct native coordinates have 49.5-count median local contrast and 76 sites above 10 counts;
  the reflected CSV coordinates have 9-count median contrast and none above 25 counts. The 5-degree
  003 site contains 83,328 counts at direct `(1455, 1473)` but 46 counts at the reflected location.
  Direct coordinates reproduce the stored cake angles with median absolute errors of 0.000242
  degrees in `2theta` and 0.000062 degrees in `phi`; the reflected coordinates miss by 2.26 and
  90.99 degrees respectively.
- The external 3000x3000 depth-6 raster discovers 90 features and recovers 11 locally confident m=1
  sites. Observed-to-alpha-zero-anchor error is 2.709 pixels RMS and 5.130 pixels maximum. Depth-4
  versus depth-6 indexing has identical label keys, 0.000080-pixel RMS position shift, and
  0.000193-pixel maximum shift. A measured run took 34.090 seconds and peaked at 716,222,464 bytes
  RSS with the proof-resolution one-pixel search cake.
- Real three-image indexing took 101.725 seconds. A two-pixel search reduced the 3000x3000 run to
  13.045 seconds and 469,995,520 bytes, but selected a parallel real-data ridge; the production
  runner therefore retains the proof-resolution one-pixel grid.

Known limitations:

- The available 3000x3000 artifact is `ADAPTIVE_UNRESOLVED_DIAGNOSTIC` with 6,836 unresolved pixels.
  The runner rejects it by default and permits it only with an explicit non-accepted diagnostic
  override. It covers one 5-degree, m=1, analytic-branch-2 image, not three simulated incidences or
  the full all-family source-averaged detector.
- The same-context integer-L catalogue checks label algebra and anchor round trips; it is not an
  independent physics oracle and supplies no detectable-site recall denominator.
- Ridge-support and localization matrices are deterministic conservative weights, not calibrated
  covariance coverage. Geometry fitting must not interpret the multiplier as a probabilistic
  three-sigma statement.
- Absolute labels still require the configured lattice, incident state, and a calibration close
  enough for q-space separation. This stage fits neither mosaic nor simulated intensity.

Minimum integration request:

- Correct the protected shared provenance in `examples/DATA_PROVENANCE.md` and
  `src/rasim_next/proof/reference.py` so the accepted detector-native coordinate is direct
  `(legacy_raw_x, legacy_raw_y)`, and retain an OSC-intensity/cake-angle regression for that mapping.
