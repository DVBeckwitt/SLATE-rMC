# General-CIF pipeline research

The geometry predictor, mosaic profile fitter, and covariance-aware matched-region optimizer are
already material-neutral. The duplicated material path begins at detector transfer: the
source-averaged compiled evaluator packs `Bi2X3FiniteStackStrength`, and the ordered fitter embeds a
three-site Bi2X3 quadratic.

The smallest shared boundary is therefore the existing structure-free `DetectorStructureResponse`:
it contains the source/optics/mosaic/detector coefficient for every physical rod, inverse root,
exact `L`, and wavelength. A source-averaged sparse form can be compiled once and applied to any
reciprocal-basis-bound strength model. This makes the current Bi2X3 evaluator an accelerator behind
the same contract rather than a separate scientific pipeline.

For a general layered-film CIF, the default ordered model is the complete periodic unit-cell
amplitude multiplied by an explicitly declared finite coherent repeat along the third lattice
translation. Unknown displacement values require an explicit calculation value. Disorder and
polytype mixtures are not inferred from the CIF; they remain alternative strength models.

The current nine-coordinate geometry correction intentionally excludes detector center and
distance. Those calibration coordinates must be a separate optional pack so the default numerical
path and serialized legacy results remain unchanged. Simultaneously active translation-like sample
or pivot coordinates must pass the existing scaled-rank gate rather than being silently reduced.

Ordered refinement needs data, not element-specific code. A compact affine basis over expanded CIF
site fractional coordinates, occupancies, and isotropic displacements is sufficient for tied or
fixed parameters while preserving the existing `unit_cell_amplitude` equation. The basis must be
validated for topology, bounds, gauges, and exact site ordering before fitting.
