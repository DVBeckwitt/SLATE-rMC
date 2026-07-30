"""Canonical scientific identity for staged-fit artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

STAGED_FIT_STAGE_SCHEMA_VERSION = "rasim-staged-fit-replay-stage-v2"
STAGED_FIT_STAGE_NAMES = ("geometry", "mosaic", "ordered_intensity", "render")

_SHA256_PREFIX = "sha256-"
_VOLATILE_SCIENTIFIC_FIELDS = frozenset(
    {
        "artifact",
        "artifact_sha256",
        "benchmark",
        "diagnostic",
        "elapsed_seconds",
        "execution_device",
        "geometry_setup_wall_time_seconds",
        "frozen_reindex_wall_time_seconds",
        "indexing_pass_wall_times_seconds",
        "indexing_wall_time_seconds",
        "initial_start_fit_wall_time_seconds",
        "manifest_path",
        "output_directory",
        "path",
        "peak_memory_bytes",
        "primary_fit_wall_time_seconds",
        "runtime",
        "timing_seconds",
        "wall_time_seconds",
    }
)


def _stable_scientific_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_scientific_value(item)
            for key, item in sorted(value.items())
            if key not in _VOLATILE_SCIENTIFIC_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_stable_scientific_value(item) for item in value]
    if isinstance(value, Path):
        raise TypeError("scientific revisions cannot contain paths")
    return value


def staged_fit_scientific_revision(stage: str, payload: Mapping[str, Any]) -> str:
    """Hash a path-, device-, and timing-free scientific stage payload."""

    if stage not in STAGED_FIT_STAGE_NAMES:
        raise ValueError(f"unsupported replay stage {stage!r}")
    encoded = json.dumps(
        {
            "schema": "rasim-staged-fit-scientific-revision-v2",
            "stage": stage,
            "payload": _stable_scientific_value(payload),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


def verify_staged_fit_stage_artifact(
    document: Mapping[str, Any],
    *,
    expected_stage: str,
) -> str:
    """Verify one v2 stage envelope and return its bound scientific revision."""

    if document.get("schema_version") != STAGED_FIT_STAGE_SCHEMA_VERSION:
        raise ValueError("unsupported staged-fit artifact schema")
    if document.get("stage") != expected_stage:
        raise ValueError(f"expected a {expected_stage} staged-fit artifact")
    revision = document.get("scientific_revision")
    if not isinstance(revision, str) or not revision.startswith(_SHA256_PREFIX):
        raise ValueError("staged-fit artifact lacks a scientific revision")
    payload = {name: value for name, value in document.items() if name != "scientific_revision"}
    if revision != staged_fit_scientific_revision(expected_stage, payload):
        raise ValueError("staged-fit artifact changed its scientific revision")
    return revision
