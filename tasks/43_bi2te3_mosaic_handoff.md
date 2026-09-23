# T43: Bi2Te3 joint geometry to mosaic handoff

Status: `GEOMETRY_VALIDATED_NUMERICAL_BOUNDARY_MEASURED_MOSAIC_UNQUALIFIED`

Branch: `codex/bi2te3-mosaic-handoff-repair`

## Repair

The earlier experimental Bi2Te3 mosaic adapter added fitted absolute detector tilts to the
legacy tilted Bi2Te3 detector and applied the fitted beam-center offset a second time at the
detector reference. The joint fit used the zero-tilt Bi2Se3 detector base. The reusable handoff now
rebases only detector rotation, derives and checks the fitted beam line, retains the detector
translation, pitch, shape and reference, and applies the shared and sample-local corrections once.
`zS` is sample-local; conditional `zB` is already represented by the beam line.

`save_joint_geometry_handoff` and `load_joint_geometry_handoff` bind the reduced joint report,
detector-base config, specimen config, both CIF files, OSC geometry manifest, image roster,
commanded angles and OSC bytes. Reload reconstructs the rebased source and detector, and rejects
changed inputs or a changed position record. `scripts/replay_joint_geometry_handoff.py` creates
and checks the external geometry checkpoint. `scripts/replay_bi2te3_mosaic_response.py` consumes
that checkpoint through `load_joint_geometry_handoff` and `build_fixed_experiment_series`, so both
its derived source and its fixed position enter the actual detector mosaic response.

## Proof and status

The qualified report is `joint_geometry_report_reduced_v3_final.json`, SHA256
`8c19068bdc47e29a09588b0a2c95e3c2dcf92a01291dfa8cc9a6504cf9c4bff5`.
The reloaded handoff reproduces all 35 retained Bi2Te3 sites at 5, 10 and 15 degrees with zero
detector-coordinate difference from the joint fitter. Measured RMS/max site residuals are
`0.630007/1.230024`, `0.893597/2.190828`, and `0.849124/1.954111` px. The permanent synthetic
regression independently rebuilds the downstream source and detects a wrong beam origin, second
detector-reference shift, and second tilt application.

The earlier corrected-geometry preflight retained seven profiles under unchanged signal and pair
gates. Its 64-state mixture reached the Lorentzian lower bound (`G=0.759225` deg, `L=0.35` deg,
`eta=0.778238`, objective `0.354150`). This is diagnostic only. A corrected, fixed-layout
41-bin pilot for both 10-degree `m=1,L=10` root sides compared 250 and 500 source states. The
gate was fixed before measuring: raw and amplitude-profiled relative L2 must both be at most 0.02,
all 41 bins must remain valid, aggregate evaluation must be under 30 minutes, and peak process RSS
under 1 GB. At Gaussian sigma 0.909 degree, raw discrepancies are 0.01345 and 0.00762; at
Lorentzian HWHM 0.585 degree, 0.01802 and 0.01061. At Lorentzian HWHM 0.35 degree, the negative
root fails at 0.02624 raw and 0.02411 amplitude-profiled. Aggregate evaluation took 97.673 seconds
and peaked at 319,602,688 bytes of whole-process RSS. The direct profile evaluator avoids the
historical 9.3 GB
prebuilt support/layout path for this bounded pilot; a full shared response-bank fit has not been
run. This failed narrow-component gate prevents promoting the three-parameter mixture. The
corrected seven-profile family split, background sensitivity and pitch-gauge propagation are also
unproved. Earlier `geo2` 250/500 responses are invalid because they used the double-corrected
geometry. No Bi2Te3 mosaic is promoted; the tracked `fixed_mosaic.json` is unchanged.

The preserved root-cause and preflight evidence is external under the September 18
`bi2te3_mosaic_qualification` visualization directory. This branch establishes the reusable
geometry input and a corrected downstream numerical response with a measured convergence limit.
A physical mosaic claim still requires the listed numerical and family checks. Legacy
classification for the corrected handoff is `CORRECTED`: it agrees with
the prior adapter through input loading and first diverges at detector-base rotation and beam-center
application, then follows the qualified joint fit's independent detector predictions.
