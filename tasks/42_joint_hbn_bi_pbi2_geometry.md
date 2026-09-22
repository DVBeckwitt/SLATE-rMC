# T42: joint hBN, BiX, and 2H-PbI2 geometry

Status: PASS_REDUCED_REFERENCE_GEOMETRY
Branch: `codex/joint-hbn-bi-pbi2-geometry`
Base: `9cbc1be8d4194148f64d17ca13e3f982a567314b`
Decision: [D043](../docs/DECISIONS.md#d043-joint-geometry-fixes-the-unsupported-pitch-gauge-before-intensity-fitting)

## Goal

Add PbI2 Y1 and Y2 to the automatic hBN/Bi2Se3/Bi2Te3 geometry fit. Use only confidently
identified 2H peaks, give Y1 and Y2 independent sample tilt and `zS` coordinates, and share all
detector, beam-center, goniometer-axis, pivot, `zB`, and incidence-zero coordinates across every
specimen.

## Selection and ownership

- The existing strong multi-L branch-track gate remains unchanged for Bi2Se3 and Bi2Te3.
- Sparse PbI2 evidence is admitted only when blind indexing recovers the same full 2H integer-L
  key at two distinct incidences, from distinct detector and geometry contexts, and the observed
  incidence motion agrees with the predicted motion within 8 px RMS.
- Y1 and Y2 each retain the two `m=1`, `L=1` roots at both 5 and 10 degrees. The motion-incoherent
  `m=3` key and every singleton key are excluded before optimization.
- hBN owns only detector intrinsic tilts and beam center; its plane distance remains private.
- Detector distance, pitch, shape, roll, wavelength, source declaration, and nominal transforms
  remain static. Bi2Se3 sample-x tilt remains the incidence-zero gauge.

## Measured result

The observable fit passes. hBN is 0.8251 px RMS. Per-image crystalline RMS ranges are 0.873-2.481
px for Bi2Se3, 0.604-0.860 px for Bi2Te3, 1.143-1.596 px for Y1, and 0.318-0.395 px for Y2.
The pooled crystalline RMS is 1.2189 px and the maximum site error is 6.3030 px.

The unrestricted 21-coordinate decomposition remains unqualified. Its global pivot-pitch offset
reaches the +1.0 mm bound and dominates the weakest scaled direction at 0.9829. A measured profile
from -3 to +3 mm changes the pooled detector RMS only from 1.301 to 1.210 px, while fixing pivot
pitch alone transfers the weak direction to goniometer-axis pitch with 0.9828 loading. These two
pitch-direction mechanical coordinates therefore describe an unsupported gauge for this dataset.

The accepted reduced parameterization fixes goniometer-axis pitch and pivot pitch displacement to
their nominal references and fits the remaining 19 coordinates. It is full-rank at 19/19, has
scaled condition 978.86, has no active fitted bounds or uncertain fitted parameters, and passes the
unchanged detector residual gates. hBN is 0.8251 px RMS; pooled crystalline RMS is 1.2314 px and the
maximum site error is 6.3043 px. The change from the unrestricted fit is 0.0124 px pooled RMS.

This qualifies detector-predictive geometry over the observed angle range. The fixed mechanical
references are declared rather than reported as measurements. Conditional `zB=-0.03091 +/-
0.01807 mm` is the fitted beam line relative to the nominal pivot plane; the physical beam offset
from the unknown true rotation center remains unqualified.

Measured artifacts:

- unrestricted fit: `C:\Users\Kenpo\.codex\visualizations\2026\09\18\01a0b67f-592e-76b3-b9b4-5d6d86215329\joint_hbn_bi_pbi2_geometry\joint_geometry_report_final_v2.json`
- fixed-coordinate profile: `C:\Users\Kenpo\.codex\visualizations\2026\09\18\01a0b67f-592e-76b3-b9b4-5d6d86215329\joint_hbn_bi_pbi2_geometry\joint_geometry_pivot_profile.json`
- accepted reduced fit: `C:\Users\Kenpo\.codex\visualizations\2026\09\18\01a0b67f-592e-76b3-b9b4-5d6d86215329\joint_hbn_bi_pbi2_geometry\joint_geometry_report_reduced_v3_final.json`

## Verification

```powershell
python -m compileall -q src scripts
ruff check src/rasim_next/fitting src/rasim_next/selection tests scripts
pytest -q tests/test_joint_geometry.py tests/test_hbn_calibration.py tests/test_selection.py
python scripts/verify_seed.py
git diff --check
```

An independent mechanical pivot datum or wider incidence range is still required before assigning
physical meaning to the two fixed pitch references or to physical `zB`. More samples at only 5 and
10 degrees with free local `zS` coordinates will not resolve that mechanical decomposition.
