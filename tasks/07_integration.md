# T07: native-detector integration

Branch: `feat/integration`

Begin only after T06 approves corrected T02–T05 SHAs and the follow-up shared-measure commit.

## Goal

Compose the four proven subsystems into the seeded statistical raw native-detector model without duplicating physics. Preserve reusable fixed-geometry states for later fitting.

## Owned paths

```text
src/rasim_next/measurement/
src/rasim_next/render/
src/rasim_next/pipeline/
tests/test_integration.py
this task's execution-plan and handoff sections
```

## Production and recycling boundary

- Integrate exactly one source sampler from T03, implement one weighted selector, and implement one conservative bilinear depositor.
- For each statistically sampled incident ray and independent phase/parent, reuse that ray across all eligible individual `(h,k)` rods and every valid mosaic/Q solution; build one complete candidate-mass pool for that ray+phase, never one pool per reflection.
- Each candidate retains its rod/reflection ID, orientation/Q solution, full `Q`, elastic `kf`, detector hit, model/structure-factor strength, mosaic mass, and every other once-only physical factor.
- Select a configurable outgoing-event count per source+phase (legacy default 50) from the complete pool by seeded cumulative inverse CDF.
- Each selection uses that candidate's own `Q`, `kf`, rod, and hit; never reuse one `kf` across reflections.
- Assign every selection `total_pool_mass / selected_count`; do not reapply structure factor, mosaic mass, or selection probability.
- Prefer two-pass/streaming enumeration so the full incident×rod×mosaic product is not retained. Preserve individual rods; never collapse a `Qr` family.
- Deterministic/adaptive candidate support and all-candidate raster paths are oracle/diagnostic only, never the production rasterizer.

## Mandatory integration order

1. Correct the shared result measure, contracts, names, and factor ledger after superseded base `b4c10fa`.
2. Merge locally approved T04 rods, T02 incident/exit geometry, corrected T03 source/candidates, then T05 stacking support.
3. Prove one sampled ray reaches all eligible individual rods and candidate-specific mosaic/Q solutions.
4. Join each candidate to its own ordered or stacking strength, optical factors, polarization, phase/parent population, and detector hit exactly once.
5. Build one complete mass pool per source+phase, normalize it, and select by seeded cumulative inverse CDF.
6. Assign uniform selected-event mass, deposit bilinearly once, and report explicit edge clipping.
7. Substitute stacking under the same candidate/selection boundary only after single-phase conservation passes.
8. Run the tiny case, authoritative high-sample comparison, and fitting-relevant performance workloads.

## Raw observable

The raw native image is the accumulated selected-event mass. Source and mosaic distributions act only through sampling/selection frequency. Pixel solid angle is metadata or optional later analysis and cannot multiply raw events or pixels; exposure and overall normalization remain separate.

## Tiny end-to-end case

Use a non-square detector, fixed seed, one ray reused across at least two reflections, nonzero sample/detector rotations, refraction and attenuation, multiple mosaic solutions, candidate-specific `kf`/hits, one stacking substitution, and declared clipping. Record pool mass, selection counts, assigned mass, deposited/clipped mass, peak pixel, and selected pixels.

## Required proof — exactly eight obligations

1. Fixed seed is exactly repeatable, including complete-pool selection.
2. Source histograms recover declared distributions and correlations.
3. Mosaic selection frequencies converge to normalized candidate intensities across the complete ray+phase pool; one-ray/two-reflection candidates retain distinct `kf`/hits and frequency responds to structure-factor and mosaic-mass ratios.
4. Increasing event count preserves the normalized ensemble mean and noise falls approximately as `1/sqrt(N)`.
5. One selected event deposits its assigned mass exactly once; bilinear deposition conserves it except for declared edge clipping, and batch deposit sums to total pool mass.
6. A high-sample ensemble matches the authoritative legacy raw-detector observable without solid-angle correction through a named comparison stage.
7. A mutation detects double-weighting sampled source or mosaic probabilities, including structure-factor/mosaic/probability reapplication after selection.
8. Changing or enabling solid-angle metadata cannot change the raw detector image.

## Commands

```bash
python -m compileall -q src
ruff check src tests
pytest -q
python -m rasim_next.proof all --json
python -m rasim_next.proof tiny-end-to-end --json
git diff --check
```

## Performance and boundaries

Profile high-sample full rendering and repeated ordered/stacking evaluation with geometry fixed; report time, peak memory, candidate count, selected count, and speed relevant to later fitting. Broad sweeps remain external artifacts.

Do not add a framework, PSF, caking, fitting, backend, generic RNG layer, proof infrastructure, or reference mutation. Keep pure specular composition explicit and separate.

## Handoff

Record integrated SHAs, eight-proof results, authoritative comparison stage, mass/clipping evidence, performance, and remaining limitations.
