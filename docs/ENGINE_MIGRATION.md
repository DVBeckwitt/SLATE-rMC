# Native engine migration

The sole regular conditional engine now uses positive strength-weighted Gaussian
axial quadrature and native-pixel-error angular acceptance. The local-m0 physical
endpoint chart remains. Unstitched Pb `(0,0)` rods use the same regular interior-node
engine; no fabricated singular q=0 sample is introduced. Physical factors and observation memberships are preserved;
quadrature changes invalidate earlier numerical qualifications. Scoped interval
evidence supports method selection, not whole-image superiority or a qualified fit.

Write a new physics input explicitly:

```text
python scripts/migrate_native_physics.py OLD_PHYSICS.json NEW_PHYSICS.json
```

The command accepts `rasim-native-fit-physics-v1`, rejects unknown old controls and
creates a new `rasim-native-fit-physics-v2` file exclusively. Every physical field
and original file is preserved. The v2 loader rejects v1 with this migration
instruction. Never overwrite archived input/result evidence. Reprepare predictions,
checkpoints and qualifications under the new engine/source identity.

Observation descriptors and experiment catalogs also bind physical bytes. A new
physics file alone does not migrate those consumers. Create new external copies;
keep archived catalog/descriptors and all baseline/results unchanged:

1. Place the new v2 physics in a new external input directory. Copy the observation
   JSON to that directory under a new name. For every existing file reference with
   `path` and `sha256`, resolve its path against the original descriptor directory
   and retain that exact SHA. These references may be absolute in the descriptor.
2. Replace only the copied descriptor's `physical_input` with the new v2 path and
   SHA256. Add `engine_migration_provenance` containing the original descriptor
   path/SHA, original physical-input path/SHA, and `numerically_qualified: false`.
   Keep arrays, support, backgrounds, covariance, guards and raw acquisition intact.
3. Copy the catalog externally. Change that experiment's `observations` and
   `physics` entries to paths relative to the new `--input-root`, with their new
   exact SHAs. Remove its `archived_baseline` from the new catalog; retain the old
   entry in provenance. Do not use `--with-baseline` for the migrated experiment.
4. Run `prepare_native.py --sample SAMPLE --input-root NEW_INPUT_ROOT
   --catalog NEW_CATALOG --output-directory NEW_PREPARED_DIRECTORY`, or prepare the new
   descriptor directly using `--observations NEW_DESCRIPTOR`. Preparation verifies
   all bound bytes and relocates them without changing measurement membership.
   Use the resulting descriptor and its bound v2 physical input for fresh execution.

5. Copy the intended fit plan externally; do not edit the historical recipe.
   Preserve its original `experiment_binding` as migration provenance. Set the new
   `experiment_binding.physics_sha256` to SHA256 of the prepared descriptor's bound
   physical file. Set `experiment_binding.observation_sha256` to the prepared
   descriptor's `preparation.source_observation_sha256` (the copied source descriptor
   SHA, **not** the prepared descriptor SHA). Without preparation use the exact
   source observation descriptor SHA. Keep acquisition IDs, measurements, guards,
   parameter ownership and physical constraints unchanged. Apply the numerical
   control changes below. Use fresh output and qualification state; retain no old
   selected/optimizer status. The tracked recipe hashes deliberately remain historical
   and require this explicit external rebinding before current execution.

The scoped migration check exercises this copy/rebind and hash-verification path;
it does not qualify numerical predictions. Original source-observation provenance
remains explicit, while the changed descriptor obtains a new input revision.
Saved fits made with the old engine cannot be rendered as current fits: input/source
identities differ and rendering rejects them. Old parameter values may only seed a
new explicitly unqualified run; old selection, optimizer or numerical status cannot
be inherited. Current saved results must replay under their own current identities.

Plans retain `rasim-native-refinement-plan-v1`; update numerical declarations:

| Previous control | Current declaration |
| --- | --- |
| `axial_power`, `seed` | `local_m0_axial_power`, `local_m0_seed` for the active endpoint chart |
| `angular_power` | `angular_initial_power`; explicit `local_m0_angular_power` for local m0 |
| `axial_peak_spacing_L`, `axial_peak_half_width_L` | `local_m0_peak_spacing_L`, `local_m0_peak_half_width_L` |
| `angular_resolution_fraction` | `local_m0_angular_resolution_fraction` |
| `local_m0_angular_power=None` | Explicit former inherited angular power |
| Ordinary selectors, manual edges/meshes and width caps | Remove; no alternate regular engine |
| `workers` greater than one | Set to 1; existing `prediction_workers` parallelizes candidates |

Review staged `numerical_checks` and `reference_correction.numerical_override`
as well as `integration_override`. Old regular refinement metadata cannot qualify
new axial accuracy. Choose current scalar/Gauss/angular controls and independently
assess predictions and objective contrasts. Work/memory guards are not accuracy
refinements. Inactive local-m0 refinements are rejected. Both tracked live native
plans now use current controls; external plans need the same explicit review.

Default regular controls are Gaussian order 4, cheap scalar order 16 and scalar
phase step pi/4. Angular native L1 indicators use rtol 5e-5 and atol 1e-15; spatial
order remains 16. Changed strength, source, geometry, material, thickness or actual
mosaic rebuilds preparation. Complete G/L columns prepare both pure laws independently.

Native regions preserve literal fractional pixel memberships. Rendering streams
compact `(bounds, mass)` patches into one caller image. For bin size above one,
bounds index binned cells. Resume regenerates the identical candidate stream and
skips completed accepted windows. Continuous `density_at` remains A2/px2 through
the same prepared kernels; native-pixel acceptance does not qualify that observable.

Covariance/rank, physical constraints, guard checks, fit qualification and selection
states remain mandatory. Failed/timed-out streams and any accumulated prefix are
incomplete. Migration and constructor invariants do not promote fits.
