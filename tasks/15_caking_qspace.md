# T15: later caking and reciprocal-space measurement transform

Status: DEFERRED. Begin only after T14 and native-detector fit validation.

Branch: `feat/caking-qspace`

## Goal

Add `2theta/phi`, caking, and selected fixed-`Qr` profiles as measurement transforms that reuse accepted geometry and selection identities without changing upstream physics or fit results.

## Required work

- consume canonical detector-native arrays
- use accepted instrument geometry
- generate signal and normalization fields
- prove non-square mapping and mass conservation
- sum signal and normalization over selected masks before division
- reuse immutable rod-family, reflection-group, and branch identities
- keep this layer optional for stacking-profile analysis

This task cannot redefine OSC orientation, detector coordinates, rod identity, branch identity, or upstream model normalization.
