"""One prefix-only policy interface shared by live search and historical replay."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from rsi_state import encode


def validate_source(source: str) -> None:
    """Policies are a deliberately small, pure subset of Python, not plugins."""
    if len(source.encode()) > 65536:
        raise ValueError("Policy exceeds 64 KiB")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Invalid policy syntax: {error}") from error
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.ClassDef,
        ast.With,
        ast.AsyncWith,
        ast.Await,
        ast.AsyncFunctionDef,
        ast.Global,
        ast.Nonlocal,
        ast.Delete,
        ast.Try,
        ast.TryStar,
        ast.Raise,
        ast.Yield,
        ast.YieldFrom,
        ast.While,
    )
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Policies cannot use {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Private names are not available to policies")
        if isinstance(node, ast.Attribute) and node.attr not in {
            "get",
            "append",
            "items",
            "values",
        }:
            raise ValueError(f"Policy attribute is not permitted: {node.attr}")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith("_")):
            raise ValueError("Decorators and private function names are not permitted")


def legal_actions(nodes: list[dict], config: dict) -> list[dict]:
    branches = {node["branch"] for node in nodes}
    parents = {node["parent"] for node in nodes}
    legal = []
    if len(branches) < config["max_width"]:
        legal.append({"id": "new", "parent": "root", "branch": len(branches)})
    for node in nodes:
        if node["id"] not in parents and node["depth"] < config["max_depth"]:
            legal.append({"id": node["id"], "parent": node["id"], "branch": node["branch"]})
    return legal


def prefix(nodes: list[dict]) -> list[dict]:
    fields = ("id", "parent", "branch", "depth", "score", "valid", "repairable", "failure")
    return [{key: node[key] for key in fields} for node in nodes]


def decide(
    policy: Path, nodes: list[dict], legal: list[dict], config: dict, beta: float
) -> list[dict]:
    if not legal:
        return []
    safe_config = {key: config[key] for key in ("workers", "max_calls", "max_width", "max_depth")}
    payload = {"observed": prefix(nodes), "legal": legal, "config": safe_config, "beta": beta}
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), str(policy.resolve())],
        input=encode(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=5,
    )
    if result.returncode:
        raise ValueError(f"Policy failed: {result.stderr[-2000:]}")
    chosen = json.loads(result.stdout)
    if not isinstance(chosen, list) or not all(isinstance(item, str) for item in chosen):
        raise ValueError("Policy must return a list of action IDs")
    if len(chosen) != len(set(chosen)) or len(chosen) > config["workers"]:
        raise ValueError("Policy returned duplicate actions or exceeded the worker cap")
    by_id = {item["id"]: item for item in legal}
    if not set(chosen) <= by_id.keys():
        raise ValueError("Policy selected an illegal or unsupported action")
    return [by_id[item] for item in chosen]


def objective(nodes: list[dict], rounds: int, config: dict) -> dict:
    best = max([0.0, *[node["score"] for node in nodes if node["valid"]]])
    count = len(nodes)
    parallelism = count / max(1, rounds)
    return {
        "quality": best,
        "probes": count,
        "rounds": rounds,
        "parallelism": parallelism,
        "utilization": parallelism / config["workers"],
        "reward": best - config["work_penalty"] * count + config["parallel_bonus"] * parallelism,
    }


def replay(policy: Path, world: dict, beta: float) -> dict:
    """Recorded outcomes only; never invokes the coding worker or task evaluator."""
    config = world["config"]
    recorded = world["nodes"]
    observed = []
    revealed = set()
    rounds = []
    unsupported = ""
    while len(observed) < config["max_calls"]:
        transitions = {}
        legal = legal_actions(observed, config)
        for action in legal:
            children = [
                node
                for node in recorded
                if node["parent"] == action["parent"] and node["id"] not in revealed
            ]
            if not children:
                continue
            child = children[0]
            transitions[action["id"]] = child
        actions = decide(policy, observed, legal, config, beta)
        actions = actions[: config["max_calls"] - len(observed)]
        if not actions:
            break
        if any(action["id"] not in transitions for action in actions):
            unsupported = "selected continuation was not recorded"
            break
        if any(not set(transitions[action["id"]]["context_ids"]) <= revealed for action in actions):
            unsupported = "selected continuation depends on unseen context"
            break
        rounds.append(
            {"visible": sorted(revealed), "actions": [action["id"] for action in actions]}
        )
        batch = [transitions[action["id"]] for action in actions]
        observed.extend(batch)
        revealed.update(node["id"] for node in batch)
    result = objective(observed, len(rounds), config) | {
        "beta": beta,
        "revealed": [node["id"] for node in observed],
        "trajectory": rounds,
        "recorded_probes": len(recorded),
        "support": "recorded dependency-respecting continuations only",
        "complete": not unsupported,
        "censored_reason": unsupported,
    }
    if unsupported:
        result["reward"] = None
    return result


def main() -> None:
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    validate_source(source)
    names = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace = {"__builtins__": names}
    exec(compile(source, "<scheduling-policy>", "exec"), namespace)
    payload = json.load(sys.stdin)
    result = namespace["choose"](
        payload["observed"],
        payload["legal"],
        payload["config"],
        payload["beta"],
    )
    print(encode(result))


if __name__ == "__main__":
    main()
