# Bi2Te3 example

`structures/Bi2Te3_cod_9011962.cif` is the full-occupancy COD 9011962 structure normalized with
explicit Bi and Te type symbols for the strict CIF boundary. The three OSC files and the R-centered
geometry, region, and structure-fit recipes use the same public staged contracts as Bi2Se3.

The material-specific data are declarative: CIF/configuration, OSC manifest, accepted detector
regions, fixed site ADPs, parameter bounds, and priors. Position, optional lattice, the tracked
`experiment/fixed_mosaic.json`,
background, A/B/C/joint fitting, continuous profiles, and rendering all use the shared layered-
quintuple implementation. This is the current generalization boundary; arbitrary crystal families
require a separate material-basis adapter rather than a branch inside the numerical core.
