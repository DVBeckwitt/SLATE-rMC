# Performance policy

Correctness, measure identity, and deterministic proof precede acceleration. Optimize the number of
physical evaluations and reusable state before changing processors or precision.

## Current work model

The runtime never materializes a detector × rod × orientation × source Cartesian product. For each
requested detector-coordinate block it:

1. builds detector rays and refracted film `kf` in vectorized arrays;
2. prunes rods whose reciprocal lines cannot enter that source state's elastic-reach ball;
3. evaluates analytic inverse branches and finite-stack strength in bounded blocks;
4. sums rods and independent source states into one coordinate-density result; and
5. consumes that field by selected-center evaluation, display-only native-center sampling, or one
   requested detector box quadrature.

The source-averaged evaluator therefore presents one callable detector field even when source rows
have different directions and wavelengths. Each state retains its own elastic and optical geometry;
only intensities are reduced.

## Reusable immutable state

- Compiled instrument transforms and detector axes.
- Canonical incident state table.
- CIF expansion, material optics, reciprocal basis, and physical rod catalog.
- Finite-2H strength parameters.
- Mosaic constants and elastic-reach bounds.
- Packed detector evaluator arrays and reachable-rod indices.
- Pixel quadrature nodes and weights.
- Certified selected-center occupancy/`Uz` response coefficients.

A source, sample, material, wavelength, mosaic, structure, or detector revision invalidates only its
declared downstream state. No hidden module-global cache or import-time device setup is permitted.

## CPU and CUDA

The NumPy implementation is the transparent proof path. The compiled CPU path removes Python
dispatch from repeated coordinate evaluation. The CUDA path batches source states, detector
coordinates, and reachable rods on device and must agree with the scalar/NumPy oracle within the
frozen observable tolerance.

Backend choice is explicit in YAML. Kernel compilation is normally paid once per process and kernel
signature, not once per incident state. Changing detector tilt or distance changes packed data, not
the kernel program. A Python, NumPy, Numba, device, dtype, or kernel-code change may require a new
compilation.

The response compiler and display renderer may choose different explicit backends because they
perform different work shapes. The ordered artifact records the compiler backend, and the image
manifest separately records the renderer backend for every incidence. This is permitted only across
the permanent CPU/CUDA all-root detector parity boundary; it does not permit a hidden backend or a
change in source, instrument, rods, mosaic, structure, or observable.

## Quality controls

- Use float64/complex128 for proof.
- Display macrobins and low-order fixed quadrature are declared preview estimates.
- Native-center images are display-only density samples, not pixel masses or count-calibrated data.
- Native-pixel adaptive quadrature reports unresolved pixels and cannot silently claim convergence.
- Approximation quality is assessed on final mass, normalized L1 shape, centroid, per-rod mass, and
  invalid-support behavior, not visual similarity alone.
- Many low-quality source-state contributions may reduce source Monte Carlo noise, but they do not
  repair a biased detector quadrature. Detector quadrature must still meet its own error target.
- A selected-center accelerator records source count/revision, retained anchor count, interlaced
  response-certificate error, and fresh combined-detector direct-oracle parity.

## Benchmark protocol

Record warm and cold wall time separately, coordinate-evaluation count, active source/rod count,
backend, dtype, detector region, quadrature rule, total detector mass, normalized image error,
centroid shift, and peak resident memory. Benchmark outputs are external diagnostics, not committed
fixtures or permanent tests.

For selected-center fitting, replace mass/image metrics with response coefficient count, source
count/revision, anchor count (including `m=0`), compile time, fresh-versus-cached equivalent
prediction time, certificate error, direct-oracle error, and peak memory.

## Continuous-incidence scan reuse

The optimized Bi2X3 scan path compiles fitted structure, mosaic, rod, envelope, and Parratt state
once, then exactly rebinds the calibrated incident transport and rigid detector/sample projection
for every incidence node. For a fault-free finite stack it also evaluates the coherent layer sum
with the stable constant-work Dirichlet identity instead of a loop over layers. The local-lamella
`m=0` evaluator passes its outgoing direction as three scalars; this avoids one Numba-managed
three-element array allocation per source and detector coordinate.

On the RTX 3060 historical pre-acquisition-correction Bi2Se3 view (uniform commanded 5--20 degree
exposure, composite
Gauss--Legendre order 16 on four equal subintervals, 64 retained angle nodes, 250 shared source
states, 19 rods, fitted 50-layer 3R structure, local-lamella `m=0`, and a 200 x 375 stride-8
detector-coordinate grid), a clean CPython 3.12 process evaluated 4.8 million
angle--detector-coordinate pairs in `float64` and measured:

- `0.729 s` to build all calibrated scan inputs;
- `16.069 s` for one fully fitted and stitched detector template;
- `0.179 s` to derive all 64 calibrated geometry views;
- `29.639 s` for the full-grid CUDA/CPU hybrid JIT warmup;
- `1317.646 s` (`21.961 min`) for the public 64-node averaged evaluation; and
- `1364.373 s` (`22.740 min`) whole-process elapsed after imports.

The prior equivalent retained fine-function run used CPython 3.13.13 and took `3466.78 s`
(`57.78 min`), including `1656.93 s` spent constructing 64 independent engines. The observed
same-hardware, cross-runtime wall-time ratio is `2.54x` overall and about `102x` for engine
construction; those ratios therefore include any CPython/runtime difference. The reusable path
reproduced the retained 64-node detector field
to relative L1 `2.91e-15`, relative L2 `2.69e-15`, and maximum absolute `2.54e-20 A2/px2`; all 250
sources remained valid and no sampled coordinate was caustic. The sampled point-density grid sum
was `4.119949946223436e-05 A2/px2`; it is not a detector mass because no pixel boxes were
integrated, and a centroid shift was not retained. A bounded repeat with the same 64-view wrapper
and one complete 75,000-coordinate node measured peak host RSS `564,613,120 B` (`538.457 MiB`).
Per-process GPU memory was unavailable under WDDM and is not reported as zero. This is a timing and
retained-field parity check for the same fixed quadrature, not an angular-convergence claim: the
retained point-density scan remains model-limited until an independent angle-rule refinement meets
its declared tolerance. These measurements are not timing or convergence evidence for the
authoritative 5--25-degree acquisition.

## Progressive full-native Monte Carlo

Task [T19](../tasks/19_fluid_detector_viewer.md) separates immutable detector physics from one
explicit mutable, thread-confined `CompiledMonteCarloDetectorSampler`. Draw-count changes generate
only the missing fixed-width Philox prefix. Detector-pose changes compile one batched projection,
swap only four projection arrays, and reset sampling while retaining source/rod/transport/physics
state; sample/goniometer changes rebuild incident transport, and a validity-topology or source-count
change replaces the sampler.
CPU uses at most four private full-native float64 blocks with stable reduction. CUDA keeps packed
state plus raw float64 and presentation float32 images resident, streams roots directly with no
event table, and polls cancellation between adaptive source-state chunks. One long-lived viewer
worker owns the sampler and CUDA context.

Every preview retains the complete native detector shape. The CUDA presentation kernel normalizes
the accumulator into a full-native float32 buffer and copies it through one reused pinned host
frame. OpenGL uploads all pixels to a persistent R32F texture; only masking, logarithmic color
mapping, and nearest viewport presentation occur in the shader. The float32 lease is not a
scientific result. Matplotlib is an explicit full-native fallback that consumes the authoritative
float64 snapshot, and neither execution nor presentation changes backend implicitly.

On the RTX 3060 acceptance workload (1,000 source states, 85 rods, 3,000 x 3,000 pixels), the clean
baseline took `5.687 s` for bundle construction, `0.189/0.870 s` for warm CPU 1/49 draws, and roughly
`0.9--1.7 s` for Matplotlib presentation. The retained CUDA path measured `4.398 s` configured
construction, `1.057 s` sampler allocation, and `2.434 s` for the first preview including JIT.
Median warm incremental stages were `16.494 ms` at draw 1, `16.823 ms` for 1-to-4, `28.666 ms` for
4-to-8, and `244.746 ms` for 8-to-49. Materializing an already-computed 49-draw authoritative
float64 snapshot added `59.351 ms`.

Before the projection-only specialization, repeated detector-pitch and detector-translation probes
needed roughly `69--81 ms` through display preparation because they rebuilt 1,000 immutable
evaluator views and repacked unchanged transport. With the specialization, 10 interleaved warmed
revisions of each kind measured `28.074 ms` median (`26.474--30.986 ms`) for pitch and `28.342 ms`
(`25.798--30.850 ms`) for column translation through full-native texture preparation. The pitch
breakdown was `0.485/6.403/17.701/3.083 ms` for instrument construction, projection rebind,
one-draw preview, and texture preparation; translation was `0.529/6.188/17.885/2.854 ms`. All 20
revisions stayed below `31 ms`, and the four resident transport device-array identities never
changed. Persistent OpenGL texture upload plus shader draw separately measured `11.691 ms` median
(`14.936 ms` p95) with `glFinish`. No source states, rods, roots, selected-stage draws, or pixels
were reduced.
Adaptive long-refinement chunks preserved single launches for the 1/3/4-draw increments and reduced
post-signal stale-cancellation latency to `11.594 ms` median, `28.935 ms` maximum.

The workspace owned `105.464 MiB` of device memory, including the `68.665 MiB` raw float64 image,
`34.332 MiB` float32 image, and two small failure-atomic geometry-buffer sets, plus one `34.332 MiB`
pinned host frame. Process RSS was about `353 MiB` in the memory probe. Cold/JIT figures are not
interaction latency, and none of these timings is a scientific tolerance.
