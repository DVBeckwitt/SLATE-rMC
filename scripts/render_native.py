"""Render the same declared native fit, with explicit candidate status and optional full pixels."""

import argparse
import hashlib
import importlib.metadata
import io
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_observations import load_native_fit_observations
from rasim_next.fitting.native_workflow import make_native_evaluator, native_physics_with
from rasim_next.io.diagnostics import validate_diagnostic_destination, write_diagnostic
from rasim_next.measurement.continuous_regions import NativePixelRegionProjection


def render(
    physics_path,
    observation_path,
    result_path,
    output,
    *,
    candidate=False,
    full_image=False,
    bin_size_px=1,
    profile_projection_path=None,
    checkpoint_seconds=60,
    resume=False,
):
    started = perf_counter()
    physics_path, observation_path, result_path, output = map(
        Path, (physics_path, observation_path, result_path, output)
    )
    validate_diagnostic_destination(output, repository_root=Path(__file__).resolve().parents[1])
    if not np.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ValueError("render checkpoint interval must be positive")
    if resume and not output.is_file():
        raise FileNotFoundError(output)
    if not resume and output.exists():
        raise ValueError("render output exists; use resume or another path")
    result_bytes = result_path.read_bytes()
    with np.load(io.BytesIO(result_bytes), allow_pickle=False) as saved:
        result = json.loads(saved["manifest_json"].tobytes())
        prediction_name = (
            "optimizer_candidate_prediction_count" if candidate else "selected_prediction_count"
        )
        expected_prediction = saved[prediction_name].copy() if prediction_name in saved else None
        synthetic_target = saved["synthetic_target"].copy() if "synthetic_target" in saved else None
    if result.get("schema") != "rasim-native-refinement-result-v2":
        raise ValueError("renderer requires the public native refinement result schema")
    point = result.get("optimizer_candidate") if candidate else result.get("selected")
    if point is None:
        raise ValueError("result has no requested selected fit or optimizer candidate")
    original = load_native_fit_physics(physics_path)
    observations = load_native_fit_observations(observation_path)
    if (
        original.input_revision != result["physics_input_revision"]
        or observations.input_revision != result["observation_input_revision"]
    ):
        raise ValueError("render inputs differ from the fitted experiment")
    plan = result["plan"]
    observations = replace(observations, objective_kind=plan.get("objective", "gls"))
    if "synthetic" in plan and (
        synthetic_target is None
        or synthetic_target.shape != observations.net_count.shape
        or np.iscomplexobj(synthetic_target)
        or np.any(~np.isfinite(synthetic_target))
    ):
        raise ValueError("synthetic rendering requires the saved finite aligned fitted target")
    physics = native_physics_with(original, plan, {})
    evaluator = make_native_evaluator(physics, observations, plan)
    values, n, scale = np.asarray(point["parameters"]), point["N"], point["scale"]
    bound, arguments, mosaic, stack = evaluator.bind(values, n)
    detectors = tuple(
        replace(
            part.detector(mosaic=mosaic, **arguments),
            specular_stitch_stack=stack,
            proposal_mosaic=evaluator.proposal_mosaic,
        )
        for part in bound.integration_parts()
    )
    rows, columns = detectors[0].detector_shape_rc
    if (
        type(bin_size_px) is not int
        or bin_size_px < 1
        or rows % bin_size_px
        or columns % bin_size_px
    ):
        raise ValueError("bin size must exactly divide the native detector")
    root = Path(__file__).resolve().parents[1]
    implementation = dict(
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src").rglob("*.py"))
        },
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependencies={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "gemmi", "xraydb")
        },
    )
    profile_projection = None
    projection_sha256 = None
    if profile_projection_path is not None:
        payload = Path(profile_projection_path).read_bytes()
        projection_sha256 = hashlib.sha256(payload).hexdigest()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            shape = archive["detector_shape_rc"]
            count = archive["observation_count"]
            if (
                shape.shape != (2,)
                or shape.dtype.kind not in "iu"
                or count.shape != ()
                or count.dtype.kind not in "iu"
            ):
                raise ValueError("projection shape and count must be integral metadata")
            profile_projection = NativePixelRegionProjection(
                detector_shape_rc=tuple(int(v) for v in shape),
                observation_count=int(count),
                quadrature_revision=projection_sha256,
                **{
                    name: archive[name]
                    for name in (
                        "flat_pixel_index",
                        "observation_row",
                        "pixel_column_index",
                        "detector_area_weight_px2",
                    )
                },
            )
        if profile_projection.detector_shape_rc != detectors[0].detector_shape_rc:
            raise ValueError("render projection must use the fitted detector-native frame")
    manifest = dict(
        schema="rasim-native-render-v1",
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        physics_input_revision=original.input_revision,
        observation_input_revision=observations.input_revision,
        status="unqualified_optimizer_candidate" if candidate else "selected_fit",
        candidate=point,
        objective_kind=observations.objective_kind,
        detector_revision=detectors[0].fixed_physics_revision,
        detector_partition_revisions=[d.fixed_physics_revision for d in detectors],
        full_image=full_image,
        bin_size_px=bin_size_px,
        profile_projection_sha256=projection_sha256,
        implementation=implementation,
        completed_batches=0,
        partition_completed_batches=[0] * len(detectors),
        partition_finished=[False] * len(detectors),
        fit_numerical_status=result["numerical_status"],
        image_numerical_status="not_qualified",
        target_kind="synthetic" if "synthetic" in plan else "measured",
        complete=False,
    )
    arrays = dict(
        measured_region_count=synthetic_target if "synthetic" in plan else observations.net_count,
        valid=observations.valid,
    )
    image_key = (
        "simulated_detector_native_count" if bin_size_px == 1 else "simulated_detector_cell_count"
    )
    if full_image:
        arrays[image_key] = np.zeros((rows // bin_size_px, columns // bin_size_px))
        arrays["cell_column_center_px"] = (
            np.arange(columns // bin_size_px) + 0.5
        ) * bin_size_px - 0.5
        arrays["cell_row_center_px"] = (np.arange(rows // bin_size_px) + 0.5) * bin_size_px - 0.5
    if resume:
        with np.load(io.BytesIO(output.read_bytes()), allow_pickle=False) as saved:
            old = json.loads(saved["manifest_json"].tobytes())
            if (
                old.get("schema") != manifest["schema"]
                or type(old.get("completed_batches")) is not int
                or old["completed_batches"] < 0
            ):
                raise ValueError("invalid native render checkpoint")
            for name in (
                "result_sha256",
                "detector_revision",
                "detector_partition_revisions",
                "status",
                "full_image",
                "bin_size_px",
                "profile_projection_sha256",
                "implementation",
            ):
                if old[name] != manifest[name]:
                    raise ValueError(
                        "render checkpoint belongs to a different result or pixel rule"
                    )
            if profile_projection is not None and "display_profile_count" in saved:
                profile = saved["display_profile_count"]
                if (
                    profile.shape != (profile_projection.observation_count,)
                    or np.iscomplexobj(profile)
                    or np.any(~np.isfinite(profile))
                    or np.any(profile < 0)
                ):
                    raise ValueError(
                        "render checkpoint profile must contain aligned finite nonnegative counts"
                    )
                arrays["display_profile_count"] = profile.copy()
            if full_image:
                for name in (image_key,):
                    if saved[name].shape != arrays[name].shape:
                        raise ValueError("render checkpoint shape mismatch")
                    arrays[name] = saved[name].copy()
                    if (
                        np.iscomplexobj(arrays[name])
                        or np.any(~np.isfinite(arrays[name]))
                        or np.any(arrays[name] < 0)
                    ):
                        raise ValueError(
                            "render checkpoint pixels must be finite nonnegative counts"
                        )
                manifest["completed_batches"] = old["completed_batches"]
                counts, finished = (
                    old.get("partition_completed_batches"),
                    old.get("partition_finished"),
                )
                if (
                    not isinstance(counts, list)
                    or not isinstance(finished, list)
                    or len(counts) != len(detectors)
                    or len(finished) != len(detectors)
                    or any(type(c) is not int or c < 0 for c in counts)
                    or any(type(done) is not bool for done in finished)
                    or sum(counts) != old["completed_batches"]
                ):
                    raise ValueError("invalid native render partition checkpoint")
                manifest["partition_completed_batches"] = counts
                manifest["partition_finished"] = finished
    arrays["prediction_region_count"] = scale * evaluator.predict(values, n)
    if expected_prediction is None or not np.allclose(
        arrays["prediction_region_count"], expected_prediction, rtol=5e-12, atol=1e-10
    ):
        raise ValueError(
            "current renderer does not reproduce the saved candidate region prediction"
        )
    root = Path(__file__).resolve().parents[1]
    last_checkpoint = perf_counter()

    def save():
        manifest["elapsed_seconds"] = perf_counter() - started
        write_diagnostic(output, arrays=arrays, manifest=manifest, repository_root=root)

    save()
    if profile_projection is not None and "display_profile_count" not in arrays:
        arrays["display_profile_count"] = sum(
            (
                scale * detector.compile_native_response(profile_projection).evaluate()
                for detector in detectors
            ),
            np.zeros(profile_projection.observation_count),
        )
        save()
    if full_image:
        for part, detector in enumerate(detectors):
            if manifest["partition_finished"][part]:
                continue
            first = manifest["partition_completed_batches"][part]
            for i, contribution in enumerate(
                detector.iter_native_pixel_batches(batch_offset=first, bin_size_px=bin_size_px),
                start=first,
            ):
                arrays[image_key] += scale * contribution
                manifest["completed_batches"] += 1
                manifest["partition_completed_batches"][part] = i + 1
                if perf_counter() - last_checkpoint >= checkpoint_seconds:
                    save()
                    print(
                        json.dumps(dict(completed_batches=manifest["completed_batches"])),
                        flush=True,
                    )
                    last_checkpoint = perf_counter()
            manifest["partition_finished"][part] = True
            save()
    # An interruption between array accumulation and its counter must never
    # publish partial state. Recovery uses the last consistent atomic checkpoint.
    manifest["complete"] = True
    save()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("physics", "observations", "result", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--full-image", action="store_true")
    parser.add_argument("--bin-size-px", type=int, default=1)
    parser.add_argument("--profile-projection", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-seconds", type=float, default=60)
    args = parser.parse_args()
    print(
        render(
            args.physics,
            args.observations,
            args.result,
            args.output,
            candidate=args.candidate,
            full_image=args.full_image,
            bin_size_px=args.bin_size_px,
            profile_projection_path=args.profile_projection,
            checkpoint_seconds=args.checkpoint_seconds,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
