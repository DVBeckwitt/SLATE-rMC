"""Public orchestration boundaries; no model calls or productivity experiments."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def rsi(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "tools" / "dream-rsi" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return SimpleNamespace(
        **{
            name: importlib.import_module("rsi_" + name)
            for name in ("state", "policy", "run", "improve")
        },
        cli=importlib.import_module("dream_rsi"),
        installer=importlib.import_module("install"),
        scripts=scripts,
    )


@pytest.fixture
def repository(tmp_path, rsi):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "core.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "evaluate.py").write_text(
        "import json\nfrom core import value\nassert 1 <= value <= 10\nprint(json.dumps({'quality': value}))\n",
        encoding="utf-8",
    )
    (repo / "AGENTS.md").write_text("Preserve the evaluator.\n", encoding="utf-8")
    rsi.state.git(repo, "init", "--quiet")
    rsi.state.git(repo, "add", ".")
    rsi.state.git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "--quiet",
        "-m",
        "baseline",
    )
    return repo


def config(rsi, **overrides):
    return rsi.state.validate_config(
        {
            "writable_paths": ["core.py"],
            "protected_paths": ["AGENTS.md", "evaluate.py"],
            "evaluator": [["{python}", "-B", "evaluate.py"]],
            "score": {"kind": "metric", "key": "quality", "direction": "max"},
            "max_calls": 3,
            "max_depth": 2,
            "beta": 1.0,
            "codex": "test-codex",
            **overrides,
        }
    )


def node(identity, parent, branch, depth, score, dependencies=(), repairable=False):
    return {
        "id": identity,
        "parent": parent,
        "branch": branch,
        "depth": depth,
        "score": score,
        "valid": score is not None,
        "repairable": repairable,
        "failure": "evaluation" if score is None else "",
        "context_ids": list(dependencies),
    }


def test_replay_is_prefix_only_dependency_respecting_and_deterministic(tmp_path, rsi, monkeypatch):
    policy = tmp_path / "policy.py"
    policy.write_text(
        "def choose(observed, legal, config, beta):\n    return [a['id'] for a in legal][:1]\n"
    )
    records = [
        node("a", "root", 0, 1, 1),
        node("b", "root", 1, 1, 2),
        node("c", "a", 0, 2, 999, ("a", "b")),
    ]
    world = {"nodes": records, "config": config(rsi, max_width=2)}
    payloads = []
    original = rsi.policy.decide

    def observe(policy, nodes, legal, config, beta):
        payloads.append(rsi.policy.prefix(nodes))
        return original(policy, nodes, legal, config, beta)

    monkeypatch.setattr(rsi.policy, "decide", observe)
    first = rsi.policy.replay(policy, world, 1.0)
    assert first == rsi.policy.replay(policy, world, 1.0)
    assert first["revealed"] == ["a", "b", "c"]
    assert payloads[0] == []
    assert "999" not in json.dumps(payloads[:3])
    assert first["trajectory"][2]["visible"] == ["a", "b"]
    exhausted = rsi.policy.replay(policy, {"nodes": records[:1], "config": config(rsi)}, 1.0)
    assert exhausted["probes"] == 1
    assert not exhausted["complete"] and exhausted["reward"] is None
    dependency = {"nodes": records, "config": config(rsi, max_width=2)}
    policy.write_text("def choose(observed, legal, config, beta):\n    return [legal[-1]['id']]\n")
    censored = rsi.policy.replay(policy, dependency, 1.0)
    assert censored["censored_reason"] == "selected continuation depends on unseen context"


def test_policy_rejects_illegal_batches_and_impure_source(tmp_path, rsi):
    policy = tmp_path / "policy.py"
    for expression in ("['new', 'new']", "['hidden']"):
        policy.write_text(f"def choose(observed, legal, config, beta):\n    return {expression}\n")
        with pytest.raises(ValueError):
            rsi.policy.decide(
                policy, [], rsi.policy.legal_actions([], config(rsi)), config(rsi), 0.6
            )
    for source in (
        "import os",
        "x = (1).__class__",
        "x = open('history.json')",
        "while True: pass",
        "def malformed(",
    ):
        policy.write_text(source + "\ndef choose(observed, legal, config, beta):\n    return []\n")
        with pytest.raises(ValueError):
            rsi.policy.decide(
                policy, [], rsi.policy.legal_actions([], config(rsi)), config(rsi), 0.6
            )


def test_valid_anchor_survives_failure_and_repair_is_bounded(rsi):
    cfg = config(rsi, max_depth=5, max_width=1)
    policy = rsi.scripts / "policy.py"
    nodes = [node("a", "root", 0, 1, 4), node("b", "a", 0, 2, None, ("a",), True)]
    actions = rsi.policy.decide(policy, nodes, rsi.policy.legal_actions(nodes, cfg), cfg, 1.0)
    assert actions[0]["parent"] == "b"
    assert rsi.policy.objective(nodes, 2, cfg)["quality"] == 4
    nodes += [
        node("c", "b", 0, 3, None, ("a", "b"), True),
        node("d", "c", 0, 4, None, ("a", "b", "c"), True),
    ]
    assert rsi.policy.decide(policy, nodes, rsi.policy.legal_actions(nodes, cfg), cfg, 1.0) == []
    namespace = {}
    exec(policy.read_text(), namespace)
    schedules = [namespace["schedule"](beta, cfg) for beta in (0.0, 0.6, 1.0)]
    for key in schedules[0]:
        assert [item[key] for item in schedules] == sorted(item[key] for item in schedules)


def fake_workers(rsi, monkeypatch, protected=False):
    real = rsi.run.run_command
    calls = []

    def command(
        argv, cwd, output, timeout, max_output_mb, prompt="", environment=None, cancel=None
    ):
        if argv[0] != "test-codex":
            return real(argv, cwd, output, timeout, max_output_mb, prompt, environment, cancel)
        calls.append(prompt)
        output.mkdir(parents=True)
        message = Path(argv[argv.index("--output-last-message") + 1])
        if (cwd / "policy.py").exists():
            with (cwd / "policy.py").open("a") as stream:
                stream.write("\n# Equivalent candidate; retain incumbent on a tie.\n")
            message.write_text(json.dumps({"change": "equivalent"}))
        else:
            core = cwd / "core.py"
            value = int(core.read_text().split("=")[1])
            core.write_text(f"value = {value + 1}\n")
            if protected:
                (cwd / "evaluate.py").write_text("raise SystemExit(0)\n")
            message.write_text(
                json.dumps(
                    {
                        key: "specific measured repair"
                        for key in ("mechanism", "evidence", "change", "benefit", "risk", "repair")
                    }
                )
            )
        return {"argv": argv, "returncode": 0, "limit": "", "elapsed_s": 0.01}

    monkeypatch.setattr(rsi.run, "run_command", command)
    monkeypatch.setattr(rsi.improve, "run_command", command)
    return calls


def test_online_replay_improvement_and_next_cycle_preserve_evidence(
    tmp_path, repository, rsi, monkeypatch
):
    calls = fake_workers(rsi, monkeypatch)
    experiment = rsi.run.initialize(repository, "Improve quality", config(rsi), tmp_path / "state")
    world = rsi.run.run_online(experiment)
    assert 1 <= len(world["nodes"]) <= 3
    assert all(item["valid"] for item in world["nodes"])
    assert (repository / "core.py").read_text() == "value = 1\n"
    snapshot = (experiment / "cycles/0001/world.json").read_bytes()
    count = len(calls)
    revision, _ = rsi.state.selected_policy(experiment)
    rsi.improve.sweep(rsi.state.policy_path(experiment, revision), [world], [0.2, 0.6, 1.0])
    assert len(calls) == count
    selection = rsi.improve.improve(experiment)
    assert selection["incumbent_retained"]
    assert selection["evaluated_candidates"] == 2
    second = rsi.run.run_online(experiment)
    assert second["policy"] == selection["policy"]
    assert snapshot == (experiment / "cycles/0001/world.json").read_bytes()
    with pytest.raises(FileExistsError):
        rsi.state.save(experiment / "cycles/0001/world.json", {})
    (experiment / "cycles/0001/world.json").write_text("{}")
    with pytest.raises(ValueError, match="history changed"):
        rsi.state.completed_worlds(experiment)


def test_protected_edits_and_dirty_input_are_rejected(tmp_path, repository, rsi, monkeypatch):
    fake_workers(rsi, monkeypatch, protected=True)
    experiment = rsi.run.initialize(repository, "Improve", config(rsi), tmp_path / "state")
    world = rsi.run.run_online(experiment)
    assert not any(item["valid"] for item in world["nodes"])
    assert all("protected" in item["failure"] for item in world["nodes"])
    assert "assert 1 <= value" in (experiment / "baseline/evaluate.py").read_text()
    (repository / "core.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="clean committed"):
        rsi.run.initialize(repository, "Preserve edits", config(rsi), tmp_path / "state")


def test_timeout_recovery_and_global_installation(tmp_path, repository, rsi):
    cfg = config(rsi)
    result = rsi.run.evaluate(
        repository,
        cfg
        | {
            "evaluator": [["{python}", "-c", "import time; time.sleep(30)"]],
            "evaluator_timeout_s": 0.1,
        },
        tmp_path / "timeout",
        1.0,
        float("inf"),
    )
    assert not result["valid"] and result["failure"] == "timeout"
    experiment = rsi.run.initialize(repository, "Recover", cfg, tmp_path / "state")
    cycle = experiment / "cycles/0001"
    revision, beta = rsi.state.selected_policy(experiment)
    rsi.state.save(cycle / "start.json", {"policy": revision, "beta": beta})
    rsi.state.save(cycle / "attempts/n0001/pending.json", {"id": "n0001"})
    assert rsi.run.recover(experiment)["recovered"]
    recovered = rsi.state.completed_worlds(experiment)[0]
    assert recovered["pending"] == ["n0001"] and recovered["nodes"] == []
    (cycle / "world.sha256").unlink()
    assert not rsi.state.completed_worlds(experiment)
    assert rsi.run.recover(experiment)["recovered"]
    assert rsi.state.completed_worlds(experiment)[0] == recovered
    home = tmp_path / "codex"
    home.mkdir()
    (home / "AGENTS.md").write_text("Keep existing preferences.\n")
    rsi.installer.install(rsi.scripts.parent, home)
    first = (home / "AGENTS.md").read_text()
    rsi.installer.install(rsi.scripts.parent, home)
    assert first == (home / "AGENTS.md").read_text()
    assert first.startswith("Keep existing preferences.")
    assert "automatically" in first and (home / "skills/dream-rsi/SKILL.md").exists()


def test_feasibility_search_does_not_learn_to_do_nothing(tmp_path, repository, rsi, monkeypatch):
    fake_workers(rsi, monkeypatch)
    cfg = config(rsi, score={"kind": "pass"}, max_calls=1)
    experiment = rsi.run.initialize(repository, "Validated change", cfg, tmp_path / "state")
    rsi.run.run_online(experiment)
    result = rsi.improve.improve(experiment)
    assert result["status"] == "feasibility_only_no_quality_signal"
    assert rsi.cli.report(experiment)["feasible_candidates"]
    (repository / "core.py").write_text("value = 0\n")
    rsi.state.git(repository, "add", "core.py")
    rsi.state.git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@localhost",
        "commit",
        "--quiet",
        "-m",
        "Reproduce failing baseline",
    )
    repair = rsi.run.initialize(repository, "Repair", cfg, tmp_path / "state")
    world = rsi.run.run_online(repair)
    assert not world["baseline"]["valid"]
    assert world["nodes"][0]["valid"] and world["nodes"][0]["score"] == 1.0
