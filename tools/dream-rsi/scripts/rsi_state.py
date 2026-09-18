"""External immutable experiment records and frozen task configuration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def encode(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path: Path, payload: bytes) -> None:
    """Atomically publish new bytes; never replace existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".publishing-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save(path: Path, value: object) -> None:
    publish(path, encode(value).encode("utf-8"))


def utc() -> str:
    return datetime.now(UTC).isoformat()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def state_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "dream-rsi"


def project_key(repo: Path) -> str:
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return hashlib.sha256(str(common.resolve()).casefold().encode()).hexdigest()[:16]


def validate_config(config: dict) -> dict:
    defaults = {
        "workers": 1,
        "max_calls": 6,
        "max_width": 3,
        "max_depth": 3,
        "beta": 0.6,
        "beta_grid": [0.2, 0.6, 1.0],
        "policy_revisions": 1,
        "max_policy_calls": 2,
        "worker_timeout_s": 900,
        "evaluator_timeout_s": 900,
        "wall_timeout_s": 3600,
        "max_disk_mb": 4096,
        "max_output_mb": 16,
        "work_penalty": 0.01,
        "parallel_bonus": 0.0,
        "score": {"kind": "pass"},
        "protected_paths": [
            "AGENTS.md",
            "tests/**",
            "reference/**",
            ".codex/**",
            ".agents/**",
            ".github/**",
        ],
        "environment": {},
        "worker_context": "ancestry",
        "constraints": "",
        "require_valid_baseline": False,
        "directions": [],
    }
    unknown = set(config) - set(defaults) - {"writable_paths", "evaluator", "codex"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = defaults | config
    if type(config["require_valid_baseline"]) is not bool:
        raise ValueError("require_valid_baseline must be boolean")
    if not isinstance(config["directions"], list) or not all(
        isinstance(s, str) and s.strip() for s in config["directions"]
    ):
        raise ValueError("directions must be a list of nonempty mechanism descriptions")
    for key in (
        "workers",
        "max_calls",
        "max_width",
        "max_depth",
        "policy_revisions",
        "max_policy_calls",
        "max_disk_mb",
        "max_output_mb",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("worker_timeout_s", "evaluator_timeout_s", "wall_timeout_s"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for beta in [config["beta"], *config["beta_grid"]]:
        if not isinstance(beta, (int, float)) or not 0 <= beta <= 1:
            raise ValueError("beta and beta_grid values must lie in [0, 1]")
    if not config["beta_grid"]:
        raise ValueError("beta_grid must not be empty")
    for key in ("work_penalty", "parallel_bonus"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["worker_context"] not in {"ancestry", "observed"}:
        raise ValueError("worker_context must be ancestry or observed")
    for name in ("writable_paths", "protected_paths"):
        if not isinstance(config.get(name), list) or not config[name]:
            raise ValueError(f"{name} must be a nonempty list")
        for pattern in config[name]:
            if not isinstance(pattern, str) or pattern.startswith(("/", "\\")) or ":" in pattern:
                raise ValueError(f"Invalid repository-relative pattern: {pattern!r}")
            if ".." in pattern.split("/") or "\\" in pattern:
                raise ValueError("Path patterns use forward slashes and cannot traverse parents")
    commands = config.get("evaluator")
    if not isinstance(commands, list) or not commands:
        raise ValueError("Declare at least one evaluator command before starting a search")
    for command in commands:
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(s, str) for s in command)
        ):
            raise ValueError("Evaluator commands must be argv lists, not shell strings")
    score = config["score"]
    if score.get("kind") not in {"pass", "metric"}:
        raise ValueError("score.kind must be pass or metric")
    if score["kind"] == "metric":
        if not isinstance(score.get("key"), str) or not score["key"]:
            raise ValueError("A metric score requires a key in the last evaluator's JSON output")
        if score.get("direction") not in {"min", "max"}:
            raise ValueError("Metric direction must be min or max")
    if not isinstance(config["environment"], dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in config["environment"].items()
    ):
        raise ValueError("environment must map names to strings")
    return config


def manifest(experiment: Path) -> dict:
    data = read(experiment / "experiment.json")
    if (
        digest(experiment / "experiment.json")
        != (experiment / "experiment.sha256").read_text().strip()
    ):
        raise ValueError("Frozen experiment configuration changed")
    return data


@contextmanager
def exclusive(experiment: Path):
    """One coordinator per experiment; a stale lock is explicit, never auto-deleted."""
    lock = experiment / "coordinator.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(encode({"pid": os.getpid(), "created": utc()}))
    except FileExistsError as error:
        raise ValueError(
            f"Experiment locked: {lock}. Inspect the owning process before recovery."
        ) from error
    try:
        yield
    finally:
        lock.unlink()


def archive_policy(experiment: Path, source: Path) -> str:
    revision = digest(source)
    target = experiment / "policies" / f"{revision}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != revision:
            raise ValueError("Archived policy changed")
    else:
        publish(target, source.read_bytes())
    return revision


def policy_path(experiment: Path, revision: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", revision):
        raise ValueError("Invalid policy revision")
    path = experiment / "policies" / f"{revision}.py"
    if digest(path) != revision:
        raise ValueError("Archived policy changed")
    return path


def completed_worlds(experiment: Path) -> list[dict]:
    worlds = []
    for file in sorted((experiment / "cycles").glob("*/world.json")):
        if not file.with_suffix(".sha256").exists():
            continue
        if digest(file) != file.with_suffix(".sha256").read_text().strip():
            raise ValueError(f"Frozen history changed: {file}")
        worlds.append(read(file))
    return worlds


def selected_policy(experiment: Path) -> tuple[str, float]:
    data = manifest(experiment)
    selections = sorted((experiment / "selections").glob("*.json"))
    if selections:
        selection = read(selections[-1])
        policy_path(experiment, selection["policy"])
        return selection["policy"], selection["beta"]
    return data["initial_policy"], data["config"]["beta"]


def context_nodes(nodes: list[dict], parent: str, mode: str) -> list[dict]:
    if mode == "observed":
        return list(nodes)
    by_id = {node["id"]: node for node in nodes}
    ancestry = []
    while parent != "root":
        node = by_id[parent]
        ancestry.append(node)
        parent = node["parent"]
    return list(reversed(ancestry))
