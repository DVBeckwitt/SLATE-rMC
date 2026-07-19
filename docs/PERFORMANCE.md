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
IncidentSampleBatch
IncidentStateBatch
RodCatalog
ScatteringEventBatch
DetectorHitBatch
```

Parameter dependency:

```text
scale or background
    final combination only

ordered or stacking parameters
    scattering strength and detector reduction

mosaic parameters
    reciprocal weights and possibly event support

optical constants
    incident refraction, optical weight, and downstream reduction

film thickness
    attenuation, optical weight, and downstream reduction; not incident-state revisions

sample entrance pose or support
    full incident states, events, hits, and response

sample_from_crystal
    reciprocal events, hits, and response; not incident states

detector geometry
    detector hits and response only; not incident states or coating masses/CDFs
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

## Beam-to-`ki` authority-cutover evidence

The clean `f106c45` geometry/optics proof ran 512 equivalent float64/complex128 work items on
Windows 11, Python 3.13.13, NumPy 2.2.6, SciPy 1.15.1, and Intel64 Family 6 Model 183. The vector
path took `0.9387 ms`, the scalar oracle took `16.8892 ms` (ratio `17.99`), and the untimed vector
call had an incremental `tracemalloc` peak of `451,425` bytes for `36,864` input and `86,528`
retained-output numeric bytes. Maximum point and complex-normal-wavevector errors were zero;
maximum amplitude error was `2.22e-16`.

The contract cleanup does not claim that timing as a new optimization. It removes three stored
float64 material arrays (`24*M` numeric bytes for `M` wavelengths), two consumer-zero compiled
transform payloads, repeated transport revision hashing, and noncausal angle-cache invalidation.
The guarded reuse check observed zero revision-helper calls during incident transport. PERF-01
remains future and must record `NO_CHANGE` unless representative all-valid, mixed, all-invalid, and
repeated-geometry profiling justifies the optional allocation cleanup.

## Selection rule

Choose the production path after integration. One subsystem may use a different internal method if it preserves the same public contracts and does not create a general backend abstraction.
