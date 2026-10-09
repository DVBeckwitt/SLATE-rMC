"""Explicit file relocation at read boundaries; content identities remain caller-owned."""

from collections.abc import Mapping
from pathlib import Path


def resolve_storage_path(path: Path, stored_paths: Mapping[Path, Path] | None = None) -> Path:
    original = Path(path).resolve()
    return original if stored_paths is None else stored_paths.get(original, original)
