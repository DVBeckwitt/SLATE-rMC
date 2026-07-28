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
