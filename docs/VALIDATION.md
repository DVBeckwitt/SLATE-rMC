# Assessing changes and scientific results

## Current policy

At the user's direction, this repository retains no test suites, proof runners or
error-injection/benchmark harnesses. Do not add test-only fixtures or rebuild that infrastructure
inside or outside the repository. Historical task prompts and personal skills do not override
this policy. The former suite and detailed results remain in Git at
`358362e952289f5c4d563d46eb02a522b237561c`; those commands describe that revision only.

Runtime validation is part of the product. Preserve input/measure checks, physical domains,
rank/covariance diagnostics, numerical-qualification gates, candidate/selection distinction,
and atomic result/recovery integrity. Immutable reference evidence and measured inputs remain.

## Match verification to the change

- For deletions and literal moves, inspect callers and compare surviving code with the parent.
- Check formatting/linting, package construction, imports and affected CLI/I/O boundaries.
- Use small external temporary checks for concrete changed behavior. Record the commands,
  inputs, outcomes and limits, then remove the temporary check code.
- Stop when the actual risk is addressed. A cleanup is not permission for fits, full images,
  parameter sweeps or a replacement permanent suite.
- Report production and development-infrastructure line changes separately.

## Scientific evidence

Software checks do not establish a physical fit or numerical convergence. A scientific change
must declare its observable, units, probability measure, normalization, source/rod support,
geometry, coherence and factor ownership. Compare like work and like observables. Preserve
the existing acceptance tolerances and independently justified references. A correction needs
a named first divergent stage and evidence supporting the corrected result.

For a declared numerical comparison, use
`abs(candidate - reference) <= atol + rtol * declared_scale` with a scale from physical inputs,
analytic bounds or fixed reference data, never chosen to conceal disagreement. IDs, ordering,
shape, status, rod keys and index maps require exact equality. Near-zero handling must retain
units and a justified absolute bound. Compare complex amplitudes as complex values and raw
intensities before clipping or display transforms. Convergence needs a real refinement variable
and evidence at the requested observable; nominal output remains explicitly nominal.

The selected fitting workflow and current empirical Gaussian/Lorentzian baseline are described
in [ARCHITECTURE.md](ARCHITECTURE.md#selected-fitting-workflow). Historical success does not
waive current data/physics limitations. Report unsupported parameters and unqualified results.

## Historical full-image comparison gates

The removed T44 task record used a frozen 3000x3000 Bi2Se3 image, all 85 signed rods,
32 original source rows, both signed strength sheets and one declared local-lamella
composite. Its full-image gate was relative L1 `E1=sum(abs(I-R))/sum(R)` at most 1%,
with independently assessed reference allowance `U_R<=0.002` and conservative comparison
`(E1+U_R)/(1-U_R)<=0.01`. Keep E1 and U_R separate; empirical refinement is an allowance,
not a rigorous certificate, and global L1 does not bound every weak pixel.

No cropping, fitted renormalization, threshold pruning or survivor normalization may
improve that comparison. Every source/group must finish. Axial, angular, cone, spatial
and source sensitivity are separate obligations; coupled axial/angular refinement must
include its mixed difference. Cache reuse must cover actual candidate dependencies.
Fitting additionally requires the whitened prediction, parameter-contrast and objective
checks in [NATIVE_REFINEMENT.md](NATIVE_REFINEMENT.md). Partial images, isolated intervals
and fixed-node component checks cannot qualify a whole image or fit.

These preserve the historical acceptance contract, not an instruction to restart the
stopped campaign. Any future comparison needs current authorization, frozen inputs,
complete declared support and one finite deadline including closure. A different nominal
study must state its own limits without relabelling nominal evidence as this qualification.
