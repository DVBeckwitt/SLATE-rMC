"""Command line for persistent Codex discovery, replay, and policy improvement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from rsi_improve import improve, sweep
from rsi_run import initialize, recover, run_online
from rsi_state import (
    completed_worlds,
    encode,
    manifest,
    policy_path,
    project_key,
    read,
    save,
    selected_policy,
    state_home,
    utc,
)


def history(repo: Path, home: Path) -> dict:
    directory = home / "projects" / project_key(repo)
    experiments = []
    for path in sorted(directory.glob("*/experiment.json")):
        experiment = path.parent
        data = manifest(experiment)
        worlds = completed_worlds(experiment)
        attempts = [node for world in worlds for node in world["nodes"]]
        valid = [node for node in attempts if node["valid"]]
        best = max(valid, key=lambda node: node["score"], default=None)
        revision, beta = selected_policy(experiment)
        experiments.append(
            {
                "experiment": str(experiment),
                "task": data["task"],
                "base": data["base"],
                "cycles": len(worlds),
                "attempts": len(attempts),
                "selected_policy": str(policy_path(experiment, revision)),
                "beta": beta,
                "best": {key: best.get(key) for key in ("id", "score", "commit", "proposal")}
                if best
                else None,
            }
        )
    notes = [read(path) for path in sorted((directory / "notes").glob("*.json"))]
    return {"project": str(repo), "experiments": experiments, "notes": notes[-30:]}


def report(experiment: Path) -> dict:
    data = manifest(experiment)
    worlds = completed_worlds(experiment)
    policy, beta = selected_policy(experiment)
    attempts = [node for world in worlds for node in world["nodes"]]
    valid = [node for node in attempts if node["valid"]]
    best = max(
        (node for node in valid if node["score"] > 0), key=lambda node: node["score"], default=None
    )
    incomplete = [
        str(path)
        for path in (experiment / "cycles").glob("*")
        if not (path / "world.sha256").exists()
    ]
    return {
        "experiment": str(experiment),
        "task": data["task"],
        "base": data["base"],
        "policy": policy,
        "beta": beta,
        "completed_cycles": len(worlds),
        "coding_calls": len(list((experiment / "cycles").glob("*/attempts/*/dispatched.json"))),
        "evaluated_attempts": len(attempts),
        "feasible_candidates": [
            {key: node.get(key) for key in ("id", "commit", "workspace", "proposal")}
            for node in valid
        ],
        "policy_calls": len(
            list((experiment / "improvements").glob("*/revision-*/dispatched.json"))
        ),
        "best_improvement": best,
        "baseline_retained": best is None,
        "incomplete_cycles": incomplete,
        "locked": (experiment / "coordinator.lock").exists(),
        "cycles": [
            {
                "status": world["status"],
                "metrics": world.get("metrics"),
                "policy": world["policy"],
                "beta": world["beta"],
            }
            for world in worlds
        ],
    }


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Freeze a clean repository, task, evaluator and budget")
    init.add_argument("--repo", type=Path, default=Path.cwd())
    init.add_argument("--task", required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--policy", type=Path, help="Reuse an archived policy for a related task")
    init.add_argument(
        "--allow", action="append", help="Narrow the candidate writable paths for this task"
    )
    init.add_argument("--state-root", type=Path, default=state_home())
    for name in (
        "run-online",
        "replay",
        "improve-policy",
        "cycle",
        "status",
        "report",
        "dry-run",
        "recover",
    ):
        command = commands.add_parser(name)
        command.add_argument("experiment", type=Path)
        if name == "cycle":
            command.add_argument("--count", type=int, default=1)
        if name == "replay":
            command.add_argument("--policy", type=Path)
    for name in ("history", "remember"):
        command = commands.add_parser(name)
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--state-root", type=Path, default=state_home())
        if name == "history":
            command.add_argument("--all-projects", action="store_true")
        if name == "remember":
            command.add_argument("--mechanism", required=True)
            command.add_argument("--outcome", required=True)
            command.add_argument("--evidence", required=True)
            command.add_argument("--repair", default="")
    args = parser.parse_args(arguments)
    try:
        if args.command == "init":
            config = read(args.config)
            if args.allow:
                config["writable_paths"] = args.allow
            result = {
                "experiment": str(
                    initialize(
                        args.repo.resolve(),
                        args.task,
                        config,
                        args.state_root.resolve(),
                        args.policy,
                    )
                )
            }
        elif args.command == "history":
            if args.all_projects:
                result = {
                    "experiments": [
                        {
                            "experiment": str(path.parent),
                            "task": read(path)["task"],
                            "repo": read(path)["repo"],
                        }
                        for path in sorted(
                            (args.state_root / "projects").glob("*/*/experiment.json")
                        )
                    ]
                }
            else:
                result = history(args.repo.resolve(), args.state_root)
        elif args.command == "remember":
            note = {
                "created": utc(),
                "mechanism": args.mechanism,
                "outcome": args.outcome,
                "evidence": args.evidence,
                "repair": args.repair,
                "kind": "session_evidence_not_replay",
            }
            path = (
                args.state_root
                / "projects"
                / project_key(args.repo)
                / "notes"
                / (time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex() + ".json")
            )
            save(path, note)
            result = {"record": str(path)}
        elif args.command in {"status", "report"}:
            result = report(args.experiment.resolve())
        elif args.command == "dry-run":
            result = manifest(args.experiment.resolve()) | {"dispatch": False}
        elif args.command == "replay":
            experiment = args.experiment.resolve()
            data = manifest(experiment)
            revision, _ = selected_policy(experiment)
            result = sweep(
                args.policy or policy_path(experiment, revision),
                completed_worlds(experiment),
                data["config"]["beta_grid"],
            )
        elif args.command == "run-online":
            result = run_online(args.experiment.resolve())
        elif args.command == "improve-policy":
            result = improve(args.experiment.resolve())
        elif args.command == "recover":
            result = recover(args.experiment.resolve())
        else:
            if not 1 <= args.count <= 10:
                raise ValueError("cycle --count must be between 1 and 10")
            experiment = args.experiment.resolve()
            results = []
            for _ in range(args.count):
                world = run_online(experiment)
                if world["status"] == "baseline_failed":
                    results.append({"status": "baseline_failed"})
                    break
                selection = improve(experiment) if world["nodes"] else {"status": "no_attempts"}
                results.append({"online": world.get("metrics"), "selection": selection})
            result = {"cycles": results, "report": report(experiment)}
        print(encode(result))
        if result.get("status") == "baseline_failed" or any(
            item.get("status") == "baseline_failed" for item in result.get("cycles", [])
        ):
            return 2
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(encode({"status": "error", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
