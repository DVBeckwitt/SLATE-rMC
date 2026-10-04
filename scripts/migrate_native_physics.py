"""Explicitly migrate native physics v1 inputs to the sole strength-Gauss engine.

Writes a new v2 file; never changes source physics or saved scientific evidence.
Old regular-engine controls are retired. Local-m0 proposal controls are retained
where they have the same meaning. All previous numerical qualifications expire.
"""

import argparse
import json
from pathlib import Path

from rasim_next.pipeline.fiber_detector import FiberIntegrationRule


def migrate_record(record):
    if record.get("schema") != "rasim-native-fit-physics-v1":
        raise ValueError("migration requires a native physics v1 record")
    result = dict(record)
    old = record["integration_rule"]
    unknown = set(old) - {
        "axial_meshes",
        "local_m0_maximum_axial_panel_width_Ainv",
        "batch_size",
        "angular_integration",
        "source_latent_radius",
        "axial_peak_spacing_L",
        "maximum_axial_panel_width_Ainv",
        "maximum_angular_panel_nodes",
        "local_m0_axial_peak_coordinate",
        "seed",
        "angular_power",
        "local_m0_angular_power",
        "angular_resolution_fraction",
        "stitch_grid_size",
        "quadrature_kind",
        "regular_q_bounds_Ainv",
        "local_m0_q_bounds_Ainv",
        "angular_support",
        "cone_quadrature_order",
        "axial_power",
        "angular_panel_edges_rad",
        "axial_peak_half_width_L",
        "maximum_backward_probability",
        "frozen_ewald_bounds_Ainv_rad",
    }
    if unknown:
        raise ValueError(f"unknown v1 integration controls: {sorted(unknown)}")
    retained = (
        "source_latent_radius",
        "maximum_backward_probability",
        "maximum_angular_panel_nodes",
        "batch_size",
        "cone_quadrature_order",
        "stitch_grid_size",
        "regular_q_bounds_Ainv",
        "local_m0_q_bounds_Ainv",
        "frozen_ewald_bounds_Ainv_rad",
        "local_m0_axial_peak_coordinate",
    )
    controls = {name: old[name] for name in retained if name in old}
    for before, after in (
        ("axial_power", "local_m0_axial_power"),
        ("seed", "local_m0_seed"),
        ("axial_peak_spacing_L", "local_m0_peak_spacing_L"),
        ("axial_peak_half_width_L", "local_m0_peak_half_width_L"),
        ("angular_resolution_fraction", "local_m0_angular_resolution_fraction"),
    ):
        if before in old:
            controls[after] = old[before]
    local_power = old.get("local_m0_angular_power")
    controls["local_m0_angular_power"] = (
        old.get("angular_power", 5) if local_power is None else local_power
    )
    controls["angular_initial_power"] = old.get("angular_power", 5)
    FiberIntegrationRule(**controls)
    result["schema"] = "rasim-native-fit-physics-v2"
    result["integration_rule"] = controls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    migrated = migrate_record(json.loads(args.input.read_bytes()))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(migrated, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Migrated numerical engine only. Reprepare predictions and qualify new results.")


if __name__ == "__main__":
    main()
