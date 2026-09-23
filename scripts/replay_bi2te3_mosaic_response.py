"""Evaluate one Bi2Te3 mosaic profile from the qualified joint geometry handoff."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rasim_next.fitting.fixed_experiment import build_fixed_experiment_series  # noqa: E402
from rasim_next.fitting.fixed_lattice import FixedLatticeState  # noqa: E402
from rasim_next.fitting.joint_geometry_handoff import load_joint_geometry_handoff  # noqa: E402
from rasim_next.fitting.mosaic import (  # noqa: E402
    MosaicProfileDefinition,
    MosaicProfileIdentity,
    MosaicReflectionGroupKey,
    evaluate_continuous_mosaic_profiles,
)
from rasim_next.geometry import detector_coordinates_to_angles  # noqa: E402
from rasim_next.materials import read_crystal  # noqa: E402
from rasim_next.pipeline.configured_simulation import (  # noqa: E402
    build_nominal_ewald_context,
    build_source_averaged_detector,
    evaluate_nominal_integer_l_markers,
)
from rasim_next.selection import build_osc_angle_frame  # noqa: E402


def _peak_process_rss_bytes() -> int:
    if os.name != "nt":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage if sys.platform == "darwin" else usage * 1024)

    import ctypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise OSError("unable to read peak process working set")
    return int(counters.PeakWorkingSetSize)


def replay(
    checkpoint: Path,
    *,
    source_count: int,
    component: str,
    width_deg: float,
    root_sign: int,
) -> dict[str, object]:
    handoff = load_joint_geometry_handoff(checkpoint)
    if handoff.config.material.phase_id != "bi2te3" or root_sign not in {-1, 1}:
        raise ValueError("the replay requires the Bi2Te3 handoff and one root sign")
    if (
        component not in {"gaussian", "lorentzian"}
        or not math.isfinite(width_deg)
        or width_deg <= 0
    ):
        raise ValueError("the component and its width must be physical")
    crystal = read_crystal(
        handoff.config.material.cif_path,
        phase_id="bi2te3",
        expected_sha256=handoff.config.cif_sha256,
    )
    lattice = FixedLatticeState.implicit_cif(crystal.direct_basis_A)

    def series(count: int):
        return build_fixed_experiment_series(
            handoff.config,
            position=handoff.position,
            fixed_lattice=lattice,
            source_sample_count=count,
            gaussian_sigma_rad=math.radians(width_deg if component == "gaussian" else 1.0),
            lorentzian_half_width_rad=math.radians(width_deg if component == "lorentzian" else 0.5),
            lorentzian_probability=0.0 if component == "gaussian" else 1.0,
        )

    nominal = series(1)[1]
    context = build_nominal_ewald_context(nominal)
    markers = evaluate_nominal_integer_l_markers(context)
    matches = np.flatnonzero(
        (markers.family_m == 1)
        & (markers.integer_L == 10)
        & (markers.branch == 2)
        & (markers.root_sign == root_sign)
    )
    if matches.size != 1:
        raise ValueError("the selected m=1,L=10 marker is not unique")
    row = next(
        item
        for item in json.loads(
            (ROOT / "examples/bi2te3/observations/indexed_catalog.json").read_text(encoding="utf-8")
        )["rows"]
        if item["image_id"] == "Bi2Te3_10d_5m"
        and item["family_m"] == 1
        and item["integer_L"] == 10
        and item["root_sign"] == root_sign
    )
    active = series(source_count)[1]
    detector = build_source_averaged_detector(active)
    frame = build_osc_angle_frame(
        mean_direction_lab=active.config.source.mean_direction_lab,
        instrument=active.instrument,
        sample_intersection_lab_m=context.incident.states.sample_intersection_lab_m[0],
        revision="bi2te3-joint-handoff-mosaic-10deg.v1",
    )
    angles = detector_coordinates_to_angles(
        np.asarray([row["column_px"]]),
        np.asarray([row["row_px"]]),
        instrument=active.instrument,
        angle_frame=frame,
    )
    if not bool(angles.valid[0] & angles.azimuth_valid[0]):
        raise ValueError("the tracked centroid is outside the corrected angle frame")
    with (ROOT / "examples/bi2te3/experiment/staged_fit_replay.toml").open("rb") as stream:
        profile = tomllib.load(stream)["mosaic"]
    definition = MosaicProfileDefinition(
        identity=MosaicProfileIdentity(
            dataset_id="Bi2Te3-10deg",
            incidence_angle_rad=math.radians(active.config.instrument.axis_rotations[0].angle_deg),
            group_key=MosaicReflectionGroupKey(
                group_id="bi2te3:m=1:L=10",
                rod_catalog_revision=detector.rod_catalog_revision,
                member_rod_hk=markers.contributing_rod_hk[int(matches[0])],
                branch_mode="EXPLICIT_NONZERO",
                layered_family_m=1,
                layered_integer_L=10,
            ),
            branch_id=1 if root_sign < 0 else 2,
            analytic_branch_id=2,
        ),
        center_two_theta_rad=float(angles.two_theta_rad[0]),
        center_phi_rad=float(angles.phi_rad[0]),
        two_theta_half_width_rad=math.radians(profile["two_theta_half_width_deg"]),
        phi_half_width_rad=math.radians(profile["nonzero_phi_half_width_deg"]),
        phi_bin_count=profile["phi_bin_count"],
        two_theta_gauss_order=profile["nonzero_two_theta_gauss_order"],
        phi_gauss_order=profile["nonzero_phi_gauss_order"],
    )
    print("Evaluating corrected 10-degree m=1,L=10 detector response", flush=True)
    result = evaluate_continuous_mosaic_profiles(
        detector,
        angle_frame=frame,
        definitions=(definition,),
        profile_revision="bi2te3-joint-handoff-source-convergence.v1",
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "source_count": source_count,
        "component": component,
        "width_deg": width_deg,
        "root_sign": root_sign,
        "rod_count": len(definition.identity.group_key.member_rod_hk),
        "valid_bin_count": int(np.count_nonzero(result.valid)),
        "intensity": result.intensity[0].tolist(),
        "valid": result.valid[0].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-count", type=int, choices=(250, 500), required=True)
    parser.add_argument("--component", choices=("gaussian", "lorentzian"), default="gaussian")
    parser.add_argument("--width-deg", type=float, default=0.909)
    parser.add_argument("--root-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args()
    started = perf_counter()
    result = replay(
        args.checkpoint,
        source_count=args.source_count,
        component=args.component,
        width_deg=args.width_deg,
        root_sign=args.root_sign,
    )
    result["wall_time_s"] = perf_counter() - started
    result["peak_process_rss_bytes"] = _peak_process_rss_bytes()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
