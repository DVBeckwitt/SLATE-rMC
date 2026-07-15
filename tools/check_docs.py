from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
TASK = re.compile(r"^  - id: (T\d{2})\n((?:    [^\n]+\n?)+)", re.MULTILINE)


def repository_files(pattern: str) -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob(pattern)
        if ".git" not in path.parts and ".venv" not in path.parts
    )


def check_task_index(errors: list[str]) -> None:
    text = (ROOT / "tasks/index.yaml").read_text(encoding="utf-8")
    if "schema_version: rasim-task-index-v3" not in text or "\ntasks:\n" not in text:
        errors.append("tasks/index.yaml: invalid header")
    seen: set[str] = set()
    indexed_files: set[str] = set()
    for task_id, body in TASK.findall(text):
        fields = dict(
            match.groups()
            for match in re.finditer(r"^    ([a-z_]+): (.+)$", body, re.MULTILINE)
        )
        task_file = fields.get("file", "")
        dependencies = {
            item.strip()
            for item in fields.get("depends_on", "[]").strip("[]").split(",")
            if item.strip()
        }
        if task_id in seen or not task_file or not (ROOT / task_file).is_file():
            errors.append(f"tasks/index.yaml: invalid {task_id}")
        if not fields.get("branch") or not fields.get("status") or not dependencies <= seen:
            errors.append(f"tasks/index.yaml: incomplete {task_id}")
        seen.add(task_id)
        indexed_files.add(task_file)
    numbered = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tasks").glob("[0-9][0-9]_*.md")
    }
    if indexed_files != numbered:
        errors.append("tasks/index.yaml: numbered task coverage mismatch")


def check_links(errors: list[str]) -> None:
    for document in repository_files("*.md"):
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                errors.append(
                    f"{document.relative_to(ROOT).as_posix()}: broken local link {target!r}"
                )


def main() -> int:
    errors: list[str] = []
    for path in [*repository_files("*.toml"), ROOT / "uv.lock"]:
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            errors.append(f"{path.relative_to(ROOT).as_posix()}: {error}")
    if [path.relative_to(ROOT).as_posix() for path in repository_files("*.yaml")] != [
        "tasks/index.yaml"
    ]:
        errors.append("only tasks/index.yaml is supported")
    check_task_index(errors)
    check_links(errors)
    if errors:
        print("\n".join(sorted(errors)))
        return 1
    print("documentation links, task index, TOML, and YAML checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
