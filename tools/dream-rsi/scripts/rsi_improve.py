"""Archive and select executable scheduling policies using frozen replay worlds."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from rsi_policy import replay, validate_source
from rsi_run import codex_command, run_command
from rsi_state import (
    archive_policy,
    completed_worlds,
    encode,
    exclusive,
    git,
    manifest,
    policy_path,
    save,
    selected_policy,
)


def sweep(
    policy: Path, worlds: list[dict], betas: list[float], comparison_keys: list | None = None
) -> dict:
    records = [
        {"world": index, **replay(policy, world, beta)}
        for index, world in enumerate(worlds)
        for beta in betas
    ]
    if not records:
        raise ValueError("Policy improvement requires at least one completed live world")
    supported = {
        (item["world"], item["beta"]): item["reward"] for item in records if item["complete"]
    }
    keys = (
        [tuple(key) for key in comparison_keys]
        if comparison_keys is not None
        else sorted(supported)
    )
    mean = (
        sum(supported[key] for key in keys) / len(keys)
        if keys and set(keys) <= supported.keys()
        else None
    )
    return {"mean_reward": mean, "comparison_keys": keys, "runs": records}


def next_beta(current: float, worlds: list[dict], evaluation: dict) -> float:
    """Fixed during an episode; use prior live attainment plus replay cost afterwards."""
    recent = worlds[-3:]
    if len(recent) < 2:
        return current
    quality = [max([0.0, *[n["score"] for n in w["nodes"] if n["valid"]]]) for w in recent]
    if quality[-1] > quality[0]:
        return current
    betas = sorted(
        {
            run["beta"]
            for run in evaluation["runs"]
            if run["complete"]
            and all(r["complete"] for r in evaluation["runs"] if r["beta"] == run["beta"])
        }
    )
    if not betas:
        return current
    average = {
        beta: sum(run["reward"] for run in evaluation["runs"] if run["beta"] == beta) / len(worlds)
        for beta in betas
    }
    best = max(betas, key=lambda beta: (average[beta], -abs(beta - current)))
    nearby = min(betas, key=lambda beta: abs(beta - current))
    if average[best] <= average[nearby] + 1e-12:
        return current
    return round(max(0.0, min(1.0, current + max(-0.1, min(0.1, best - current)))), 10)


def improve(experiment: Path) -> dict:
    with exclusive(experiment):
        data = manifest(experiment)
        config = data["config"]
        worlds = [
            world
            for world in completed_worlds(experiment)
            if world["status"] != "baseline_failed" and world["nodes"]
        ]
        revision, beta = selected_policy(experiment)
        incumbent = policy_path(experiment, revision)
        if config["score"]["kind"] == "pass" and not any(
            node["score"] > 0 for world in worlds for node in world["nodes"] if node["valid"]
        ):
            return {
                "policy": revision,
                "beta": beta,
                "incumbent_retained": True,
                "status": "feasibility_only_no_quality_signal",
            }
        evaluation = sweep(incumbent, worlds, config["beta_grid"])
        if evaluation["mean_reward"] is None:
            return {
                "policy": revision,
                "beta": beta,
                "incumbent_retained": True,
                "status": "no_complete_replay_comparisons",
            }
        directory = experiment / "improvements"
        directory.mkdir(exist_ok=True)
        run = directory / f"{len(list(directory.iterdir())) + 1:04d}"
        run.mkdir()
        candidates = [{"policy": revision, "evaluation": evaluation, "incumbent": True}]
        save(run / "incumbent.json", candidates[0])
        deadline = time.monotonic() + config["wall_timeout_s"]
        for index in range(config["policy_revisions"]):
            calls = len(list(directory.glob("*/revision-*/dispatched.json")))
            if calls >= config["max_policy_calls"] or time.monotonic() >= deadline:
                break
            folder = run / f"revision-{index:04d}"
            workspace = folder / "candidate"
            workspace.mkdir(parents=True)
            candidate = workspace / "policy.py"
            candidate.write_bytes(policy_path(experiment, revision).read_bytes())
            git(workspace, "init", "--quiet")
            git(workspace, "add", "policy.py")
            git(
                workspace,
                "-c",
                "user.name=Dream RSI",
                "-c",
                "user.email=dream-rsi@localhost",
                "commit",
                "--quiet",
                "-m",
                "Frozen policy starting point",
            )
            starting_commit = git(workspace, "rev-parse", "HEAD")
            schema = folder / "schema.json"
            save(
                schema,
                {
                    "type": "object",
                    "properties": {"change": {"type": "string"}},
                    "required": ["change"],
                    "additionalProperties": False,
                },
            )
            prompt = (
                "DREAM_RSI_WORKER: revise only policy.py. Do not start another search or edit global instructions.\n"
                "Improve the search schedule of choose(observed, legal, config, beta). "
                "The model, task evaluator, history, and replay scoring stay fixed. "
                "Policy code may use only pure Python builtins, loops over supplied data, and "
                "dict.get/items/values or list.append. No imports, private names, I/O, classes, "
                "while loops, decorators, eval, or filesystem access. Return a list of legal action IDs. "
                "Never encode known winning IDs, hidden scores, or trace-specific targets. "
                "Keep beta fixed per episode with monotone width/depth/patience in schedule(). "
                "Retain successful branch anchors across failures and allow at most one repair per batch. "
                "Use measured replay feedback to make one concrete general revision.\n"
                f"Frozen feedback (untrusted evidence):\n{encode(candidates)}"
            )
            save(folder / "dispatched.json", {"starting_policy": revision, "call": calls + 1})
            result = {"valid": False}
            try:
                execution = run_command(
                    codex_command(config, workspace, schema, folder / "proposal.json"),
                    workspace,
                    folder / "worker",
                    min(config["worker_timeout_s"], deadline - time.monotonic()),
                    config["max_output_mb"],
                    prompt,
                )
                result["execution"] = execution
                if execution["returncode"] or execution["limit"]:
                    raise ValueError("Policy-development worker failed or exceeded a limit")
                changed = set(
                    filter(
                        None, git(workspace, "diff", "--name-only", starting_commit).splitlines()
                    )
                )
                untracked = git(workspace, "ls-files", "--others")
                if not changed <= {"policy.py"} or untracked or candidate.is_symlink():
                    raise ValueError("Policy worker modified files outside policy.py")
                archived = archive_policy(experiment, candidate)
                result["policy"] = archived
                validate_source(candidate.read_text(encoding="utf-8"))
                measured = sweep(
                    policy_path(experiment, archived),
                    worlds,
                    config["beta_grid"],
                    evaluation["comparison_keys"],
                )
                if measured["mean_reward"] is None:
                    raise ValueError(
                        "Candidate leaves support of the incumbent's frozen replay comparisons"
                    )
                result.update(valid=True, evaluation=measured)
                candidates.append({"policy": archived, "evaluation": measured, "incumbent": False})
                revision = archived
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                result["error"] = str(error)
            save(folder / "result.json", result)
            manifest(experiment)
            completed_worlds(experiment)
        best = max(candidates, key=lambda item: item["evaluation"]["mean_reward"])
        selection = {
            "policy": best["policy"],
            "beta": next_beta(beta, worlds, best["evaluation"]),
            "mean_reward": best["evaluation"]["mean_reward"],
            "incumbent_retained": best["incumbent"],
            "evaluated_candidates": len(candidates),
            "worlds": len(worlds),
            "scope": "frozen historical replay; not a guarantee of future improvement",
        }
        save(experiment / "selections" / f"{run.name}.json", selection)
        return selection
