# T42: joint hBN, BiX, and 2H-PbI2 geometry

Status: BLOCKED_GONIOMETER_PIVOT_UNIDENTIFIABLE
Branch: `codex/joint-hbn-bi-pbi2-geometry`
Base: `9cbc1be8d4194148f64d17ca13e3f982a567314b`

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

The complete decomposition remains unqualified. The data Jacobian is full-rank at 21/21 with
scaled condition 11517.9, but the global pivot-pitch offset reaches the +1.0 mm bound. The weakest
scaled direction has 0.9829 loading on that coordinate. Derived `zB` is -1.0309 +/- 3.1544 mm.
Adding the two-angle PbI2 data therefore verifies that the 2H peaks can participate in the common
fit and improves observable coverage, but it does not supply the missing pivot lever arm.

Measured artifact:
`C:\Users\Kenpo\.codex\visualizations\2026\09\18\01a0b67f-592e-76b3-b9b4-5d6d86215329\joint_hbn_bi_pbi2_geometry\joint_geometry_report_final_v2.json`

## Verification

```powershell
python -m compileall -q src scripts
ruff check src/rasim_next/fitting src/rasim_next/selection tests scripts
pytest -q tests/test_joint_geometry.py tests/test_hbn_calibration.py tests/test_selection.py
python scripts/verify_seed.py
git diff --check
```

The next qualifying measurement must add independent mechanical pivot information or a wider
incidence range that materially changes the pivot lever arm. More samples at only 5 and 10 degrees
with free local `zS` coordinates will not resolve the measured weak direction.
