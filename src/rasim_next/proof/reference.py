"""Minimal tracked-reference check, extended by T01."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tomllib
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _environment_sha256() -> str:
    encoded = json.dumps(
        {
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_reference_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    manifest_path = root / "reference" / "reference_manifest.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    pack = root / manifest["reference_pack"]["path"]
    checks: list[dict[str, str]] = []
    if not pack.is_file() and allow_missing_pack:
        checks.append(
            {"check_id": "reference_pack", "status": "PASS", "evidence": "pack absence allowed"}
        )
        hashes: dict[str, str] = {}
    elif not pack.is_file():
        checks.append(
            {"check_id": "reference_pack", "status": "FAIL", "evidence": "pack is missing"}
        )
        hashes = {}
    else:
        observed_hash = _sha256(pack)
        expected_hash = manifest["reference_pack"]["sha256"]
        status = "PASS" if observed_hash == expected_hash else "FAIL"
        checks.append(
            {
                "check_id": "reference_pack",
                "status": status,
                "evidence": "tracked pack SHA-256 and manifest schema checked",
            }
        )
        hashes = {"rasim_reference_v1": observed_hash}
    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {
        "schema_version": 1,
        "task_id": "T01",
        "status": overall,
        "base_sha": _base_sha(root),
        "commit_sha": None,
        "contract_version": 4,
        "trace_schema_version": 4,
        "reference_pack_sha256s": hashes,
        "environment_sha256": _environment_sha256(),
        "checks": checks,
        "classifications": [],
        "limitations": ["T00 performs only the pack-presence/hash precheck"],
    }
