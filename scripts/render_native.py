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
from rasim_next.proof.diagnostics import write_diagnostic


def render(
    physics_path,
    observation_path,
    result_path,
    output,
    *,
    candidate=False,
    full_image=False,
    checkpoint_seconds=60,
    resume=False,
):
    physics_path, observation_path, result_path, output = map(
        Path, (physics_path, observation_path, result_path, output)
    )
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
    detector = bound.detector(mosaic=mosaic, **arguments)
    detector = replace(
        detector, specular_stitch_stack=stack, proposal_mosaic=evaluator.proposal_mosaic
    )
    if not np.isfinite(checkpoint_seconds) or checkpoint_seconds <= 0:
        raise ValueError("render checkpoint interval must be positive")
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
    manifest = dict(
        schema="rasim-native-render-v1",
        result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        physics_input_revision=original.input_revision,
        observation_input_revision=observations.input_revision,
        status="unqualified_optimizer_candidate" if candidate else "selected_fit",
        candidate=point,
        detector_revision=detector.fixed_physics_revision,
        full_image=full_image,
        implementation=implementation,
        completed_batches=0,
        fit_numerical_status=result["numerical_status"],
        image_numerical_status="not_qualified",
        target_kind="synthetic" if "synthetic" in plan else "measured",
        complete=False,
    )
    arrays = dict(
        measured_region_count=synthetic_target if "synthetic" in plan else observations.net_count,
        valid=observations.valid,
        prediction_region_count=scale * evaluator.predict(values, n),
    )
    if expected_prediction is None or not np.allclose(
        arrays["prediction_region_count"], expected_prediction, rtol=5e-12, atol=1e-10
    ):
        raise ValueError(
            "current renderer does not reproduce the saved candidate region prediction"
        )
    rows, columns = detector.detector_shape_rc
    if full_image:
        arrays.update(simulated_detector_native_count=np.zeros((rows, columns)))
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
                "status",
                "full_image",
                "implementation",
            ):
                if old[name] != manifest[name]:
                    raise ValueError(
                        "render checkpoint belongs to a different result or pixel rule"
                    )
            if full_image:
                for name in ("simulated_detector_native_count",):
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
    elif output.exists():
        raise ValueError("render output exists; use resume or another path")
    root = Path(__file__).resolve().parents[1]
    started = last_checkpoint = perf_counter()

    def save():
        manifest["elapsed_seconds"] = perf_counter() - started
        write_diagnostic(output, arrays=arrays, manifest=manifest, repository_root=root)

    save()
    if full_image:
        first = manifest["completed_batches"]
        for i, contribution in enumerate(
            detector.iter_native_pixel_batches(batch_offset=first), start=first
        ):
            arrays["simulated_detector_native_count"] += scale * contribution
            manifest["completed_batches"] = i + 1
            if perf_counter() - last_checkpoint >= checkpoint_seconds:
                save()
                print(json.dumps(dict(completed_batches=i + 1)), flush=True)
                last_checkpoint = perf_counter()
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
            checkpoint_seconds=args.checkpoint_seconds,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
