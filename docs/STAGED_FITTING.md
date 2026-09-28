# Supported staged fitting

Use `prepare_native.py`, `refine_native.py` and `render_native.py` for native Bi/Pb work.
Geometry is adopted from a hash-bound experiment. Changed calibration requires a separate
prepared experiment and recipe binding; it is never a historical replay. The reduced
joint hBN/crystal geometry calibrator remains the geometric owner.

## Historical Bi2Te3 starting recipe

`configs/bi2te3_historical_native.json` binds the delivered September 11 result,
original geometry/source/observation roster, N15, 500 A film, G/L law and historical
objective. It starts a new conditional search of mosaic and termination fractions;
it does not reproduce the old optimizer trajectory. Source files and exact settings
are recorded in that recipe and `NATIVE_REFINEMENT.md`.

To admit cell, atomic, ADP and morphology coordinates, derive a separate plan:

```python
import json
from pathlib import Path
plan = json.loads(Path("configs/bi2te3_historical_native.json").read_text())
names = [p["name"] for p in plan["parameters"]]
plan.pop("fixed_parameters")
plan["recipe_id"] = "bi2te3-joint-ordered-native.v1"
plan["stages"] = [
    dict(name=label, active_parameters=active, method="slsqp",
         maximum_iterations=250, enforce_historical_guards=True)
    for label, active in [("mosaic", names[13:18]),
                          ("ordered_structure", names[:13]), ("joint", names)]
]
plan["sensitivity"] = True
plan["notes"] = ["New coupled search; archived geometry remains fixed.",
    "N15 is conditional. Film thickness is N*c + extra and now changes.",
    "All21 admitted coordinates released finally; weak directions are reported.",
    "No native Bi stacking-disorder coordinates are available."]
output_path = Path(r"C:\external\bi2te3_joint_plan.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(plan, indent=2) + "\n")
```

Replace the `C:\external` placeholder with a writable directory outside the repository.
Write generated plans and results there. Bounds are declarations,
not uncertainty intervals. Do not freeze a weak occupancy or ADP to manufacture
identification. N, geometry and source remain explicitly conditional here.

## Pb ordered control and disorder

`configs/gd1_staged_native.json` uses the current Pb finite-stack recurrence,
N72/FINITE_TOTAL and the archived first21 GD1 seed/bounds. It adopts geometry and
source, initializes G/L mosaic, then ordered structure/surface, releases disorder,
and finally releases all21 coordinates. The first start is pure2H; the second is
the archived mixed-law seed. No fitted result is implied by these starting values.

The independent ordered control uses the same inputs, objective, numerical rule
and nuisance treatment. Derive it from the same recipe:

```python
import json
from pathlib import Path
plan = json.loads(Path("configs/gd1_staged_native.json").read_text())
names = [p["name"] for p in plan["parameters"]]
fixed = dict(zip(names[16:], [1.0, 0.0, 0.0, 1/3, 0.5], strict=True))
plan["recipe_id"] = "gd1-pure2h-control-native.v1"
plan["fixed_parameters"] = fixed
for start in plan["starts"]:
    for i, name in enumerate(names):
        if name in fixed:
            start[i] = fixed[name]
plan["stages"] = [s for s in plan["stages"] if s["name"] != "disorder"]
for stage in plan["stages"]:
    stage["active_parameters"] = [n for n in stage["active_parameters"] if n not in fixed]
plan["sensitivity"], plan["profiles"] = False, []
plan["notes"] = ["Conditional pure2H nested control; initial orientation remains free.",
    "Final joint stage releases the same nonstacking coordinates as the full model.",
    "Inactive phase1 parent shares are explicit. Identification belongs to the released model."]
output_path = Path(r"C:\external\gd1_ordered_control.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(plan, indent=2) + "\n")
```

Zero epsilon in an average of different transition laws need not be ordered.
The control sets the phase0 fraction to one and both epsilons to zero. Initial
orientation is an independent intensity mixture of deterministic ordered stacks.
Neither a better objective nor an optimizer success establishes numerical convergence,
physical adequacy or unique parameters. Recipes explicitly retain nominal status;
qualification uses the existing observable gates when separately requested.

## Other materials and acquisitions

All six catalog inputs remain supported: Bi2Se3/Bi2Te3 ordered native models;
GD1/SID1 Pb21; CLEAN1 Pb27; B4 Pb25. Their phase rosters and normalization belong
to the physical input. A generic CIF alone does not supply a source, geometry,
measurement operator, background or disorder law.

For programmatic generic CIF or supported fixed-parent responses, use
`AffineCifFiniteStackParameterization` / `ParameterizedStructureRegionModel` and
`fit_structure_regions` in `fitting.matched_regions`. It calls the same
`fit_native_parameters` search. Declare `FitParameter` names, units, bounds and
one specimen owner, provide explicit starts and frozen matched observations/background.
One exposure scale per acquisition is profiled jointly with full covariance.
Anchor conditioning acts on model and data; optional peak aggregation acts on both.
Data-only local rank and conditioning must pass; priors cannot supply missing data rank.
The result is a shared search result with acquisition IDs, fitted native counts,
strength revision and sensitivity, not a second result/optimizer API.
Only `GaussianCalibration` blocks are supported; arbitrary prior callbacks are retired.
Objectives are full sums of squared residuals (the retired region fitter reported half).
Generic sparse models still require declared quadrature qualification; no generic
full-image/automatic region-discovery or arbitrary disorder binding is claimed.

## Retired alternatives

The selected-center ordered-intensity optimizer, mosaic component-bank/shape-only
optimizer and intrinsic stacking-profile optimizer are archived in Git at `1f65a09`.
Their different objectives are withdrawn, not described as equivalent native fits.
Continuous profile evaluation, canonical G/L density, atomic/finite-stack equations,
fixed-parent Pb providers and geometry fitting remain. No test/proof harness is retained.
