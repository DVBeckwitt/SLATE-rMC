# Performance strategy

## Result-first rule

Do not commit to CPU, GPU, CUDA, Numba, JAX, C++, or another backend before the integrated reference path is profiled. The accepted production method is the fastest implementation that preserves the declared observable within a measured error bound and memory limit.

## Algorithmic priorities

1. Compile instrument transforms once.
2. Use fixed-seed empirical source samples and deterministic/adaptive candidate support.
3. Construct localized Ewald support instead of scanning a full circle when possible.
4. Stream or use two passes instead of materializing the full incident-by-rod-by-mosaic product.
5. Keep candidate geometry separate from scattering strength so later fits can reuse it.
6. Evaluate ordered and stacking models only at event-required `Qz` or `L` coordinates.
7. Cache rod grids only when profiling proves reuse outweighs interpolation error and memory.
8. Separate continuous detector hits from deposition.
9. Use fixed seeds and reproducible reduction in proof and fitting modes.
10. Render only selected detector regions during fitting when the objective does not require the full image.

## Repeated fitting workloads

Later optimization should keep these states resident or cached:

```text
CompiledInstrument
CompiledSourceSamples
CompiledIncidentStates
CompiledRodCatalog
CompiledScatteringEvents
CompiledDetectorHits
CompiledDetectorResponse
```

Parameter dependency:

```text
scale or background
    final combination only

ordered or stacking parameters
    scattering strength and detector reduction

mosaic parameters
    reciprocal weights and possibly event support

optical constants or thickness
    refraction, optical weight, and downstream reduction

sample or detector geometry
    full incident states, events, hits, and response
```

Batch parameter evaluation is desirable for multi-start, finite differences, profile likelihoods, and population methods.

## Benchmark set

Record equivalent work for:

```text
small proof
    individual rays, few rods, direct oracles

medium forward
    representative detector simulation and rod catalog

large forward
    maximum intended source and reciprocal sampling

fit-structure
    repeated ordered or stacking intensity evaluations with fixed event geometry

fit-geometry
    repeated full invalidating evaluations on selected peak observations
```

Record wall time, peak memory, transfer time if applicable, setup/compile time, reuse time, hardware, precision, and error versus the reference path.

## Selection rule

Choose the production path after integration. One subsystem may use a different internal method if it preserves the same public contracts and does not create a general backend abstraction.
