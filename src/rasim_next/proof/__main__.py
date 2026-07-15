"""Command-line dispatcher for compact proof commands."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from typing import Any

_COMMANDS = {
    "core": ("rasim_next.proof.core", "run_core_proof"),
    "references": ("rasim_next.proof.reference", "run_reference_proof"),
    "geometry-optics": ("rasim_next.geometry.proof", "run_proof"),
    "mosaic-ewald": ("rasim_next.reciprocal.proof", "run_proof"),
    "ordered-reflectivity": ("rasim_next.ordered.proof", "run_proof"),
    "stacking-transition": ("rasim_next.stacking.proof", "run_proof"),
    "integration": ("rasim_next.pipeline.proof", "run_proof"),
}


def _emit(result: dict[str, Any]) -> None:
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main(arguments: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if arguments is None else arguments)
    command = args.pop(0) if args else ""
    allowed_flags = {"--json", "--allow-missing-pack"}
    unknown_flags = sorted(set(args) - allowed_flags)
    if command not in _COMMANDS or unknown_flags:
        _emit(
            {
                "status": "FAIL",
                "error": "unknown proof command or option",
                "command": command,
                "unknown_options": unknown_flags,
            }
        )
        return 2
    module_name, function_name = _COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
        runner: Callable[..., dict[str, Any]] = getattr(module, function_name)
        result = runner(allow_missing_pack="--allow-missing-pack" in args)
    except Exception as error:
        _emit(
            {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "command": command,
            }
        )
        return 1
    _emit(result)
    return 0 if result.get("status") in {"PASS", "READY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
