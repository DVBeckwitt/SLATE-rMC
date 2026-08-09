"""Run or inspect one material-neutral geometry -> mosaic -> SF workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from rasim_next.fitting.workflow import (
    FIT_WORKFLOW_STAGE_NAMES,
    load_fit_workflow,
    planned_fit_workflow,
    run_fit_workflow,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--through", choices=FIT_WORKFLOW_STAGE_NAMES, default="sf")
    selection.add_argument("--only", choices=FIT_WORKFLOW_STAGE_NAMES)
    parser.add_argument("--plan", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        workflow = load_fit_workflow(
            arguments.case,
            output_directory=arguments.output_directory,
        )
        through = arguments.through if arguments.only is None else "sf"
        if arguments.plan:
            payload = planned_fit_workflow(
                workflow,
                through=through,
                only=arguments.only,
            )
        else:
            results = run_fit_workflow(
                workflow,
                through=through,
                only=arguments.only,
            )
            payload = {
                "material_id": workflow.material_id,
                "model_family": workflow.model_family,
                "stages": [
                    {
                        "stage": result.stage,
                        "status": (
                            "reused"
                            if result.reused
                            else (
                                "provided"
                                if not workflow.stages[result.stage].commands
                                else "completed"
                            )
                        ),
                        "manifest": str(result.manifest_path),
                    }
                    for result in results
                ],
            }
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"layered fit failed: {error}")
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
