# Architecture

## Package layout

```text
src/rasim_next/
    core/
        units.py
        frames.py
        transforms.py
        wave_modes.py
        interfaces.py
        contracts.py
        validity.py

    io/
        osc.py
        orientation.py

    geometry/
        instrument.py
        sample.py
        detector.py
        transport.py

    optics/
        refraction.py
        attenuation.py

    materials/
        cif.py
        atomic_data.py
        optical_constants.py

    sampling/
        source.py
        wavelength.py
        mosaic.py
        quadrature.py

    reciprocal/
        lattice.py
        rods.py
        ewald.py
        events.py

    ordered/
        structure_factor.py
        layer_motif.py
        finite_stack.py
        rod_model.py

    reflectivity/
        parratt.py
        kinematic.py
        composite.py

    stacking/
        transition.py
        enumeration.py
        finite_intensity.py
        parent_models.py

    measurement/
        factors.py
        solid_angle.py

    render/
        deposition.py
        detector_image.py

    pipeline/
        compile.py
        simulate.py

    selection/          # post-integration
        rod_families.py
        branches.py
        observations.py
        roi.py

    fitting/            # post-integration
        contracts.py
        context.py
        invalidation.py
        source.py
        detector.py
        geometry.py
        mosaic.py
        ordered_intensity.py
        stacking_intensity.py
        result.py

    proof/
        traces.py
        compare.py
        cli.py
        diagnostics.py
```

A different internal arrangement is acceptable if ownership and dependency rules remain equivalent.

## Dependency direction

```text
core
  -> io, geometry, optics, materials, sampling
  -> reciprocal, ordered, reflectivity, stacking
  -> measurement, render
  -> pipeline
  -> selection
  -> fitting
```

Lower layers never import higher layers. Only `pipeline` assembles the forward model. Only `fitting` owns optimizers and objectives. Selection owns rod/branch association but no physics equations.

## Bootstrap-owned shared mathematics

Bootstrap implements and proves:

- units and frame tags
- `RigidTransform` inverse and composition
- OSC index mapping
- complex square-root branch selection
- normal-wavevector calculation
- scalar interface amplitude
- common validity/status codes
- trace records and event IDs
- no-physics synthetic plumbing

Geometry and reflectivity consume the same normal-wavevector and interface primitives. They do not reimplement them.

## Parallel ownership

```text
geometry-optics
    io/osc.py, geometry/, optics/

mosaic-ewald
    sampling/, reciprocal/ewald.py, reciprocal/events.py

ordered-reflectivity
    materials/, reciprocal/lattice.py, reciprocal/rods.py,
    ordered/, reflectivity/

stacking-transition
    stacking/

integration
    measurement/, render/, pipeline/

post-integration selection
    selection/

post-integration fitting
    fitting/ contracts and one stage-specific module per future worktree
```

Shared files change only through a small separately reviewed contract commit.

## Reusable compiled states

The forward model exposes immutable state boundaries:

```text
CompiledInstrument
CompiledSourceSamples
CompiledIncidentStates
CompiledRodCatalog
CompiledScatteringEvents
CompiledDetectorHits
CompiledDetectorResponse
```

They are not optimizer objects. They are dependency and reuse boundaries.

Invalidation rules:

```text
geometry parameter change
    invalidates incident states, events, hits, and response

mosaic parameter change
    invalidates reciprocal weights and possibly event support, but not instrument compilation

ordered or stacking intensity parameter change
    invalidates model intensities, but may reuse event geometry, hits, and response

scale/background nuisance change
    invalidates only final combination and objective
```

This is the central performance requirement for later fitting.

## Production acceleration

The architecture does not require CPU, GPU, CUDA, Numba, JAX, C++, or another technology. After the integrated reference path exists, profile representative full simulation and repeated-fit workloads. Choose the implementation that gives the fastest correct final observable under bounded error and memory.
