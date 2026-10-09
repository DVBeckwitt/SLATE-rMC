"""Explicit hash-bound archive storage locations; scientific records retain original paths."""

import json
from pathlib import Path

from sample_state import encoded


def storage_files(text):
    if type(text) is not str or len(text.encode()) > 128 * 1024:
        raise ValueError("Archive storage map exceeds 128 KiB")
    value = json.loads(text)
    if type(value) is not dict:
        raise ValueError("Archive storage mapping must be an object")
    if not value:
        return ()
    if (
        set(value) != {"schema", "archive_sha256", "files"}
        or value["schema"] != "slate.archive-storage.v1"
    ):
        raise ValueError("Unsupported archive storage mapping")
    rows = value["files"]
    if type(rows) is not list or len(rows) > 256:
        raise ValueError("Archive storage exceeds 256 files")
    seen = set()
    for row in rows:
        if set(row) != {"original", "stored", "sha256", "size"}:
            raise ValueError("Malformed archive storage row")
        for key in ("original", "stored"):
            if (
                type(row[key]) is not str
                or not 0 < len(row[key]) <= 4096
                or not Path(row[key]).is_absolute()
            ):
                raise ValueError("Archive storage needs bounded absolute paths")
        key = str(Path(row["original"]).resolve()).casefold()
        if key in seen:
            raise ValueError("Archive storage has duplicate original names")
        seen.add(key)
        if type(row["size"]) is not int or not 0 <= row["size"] <= 512 * 1024**2:
            raise ValueError("Archive storage has invalid file size")
        _digest(row["sha256"])
    _digest(value["archive_sha256"])
    encoded(value)
    return tuple(rows)


def _digest(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Archive storage requires SHA256")


def stored_path(path, storage=(), expected=None):
    from rasim_next.io.storage import resolve_storage_path

    original = resolve_storage_path(Path(path))
    for row in storage:
        if original == Path(row["original"]).resolve():
            if expected is not None and expected != row["sha256"]:
                raise ValueError("Archive storage identity differs from scientific predecessor")
            return resolve_storage_path(original, {original: Path(row["stored"])})
    return original
