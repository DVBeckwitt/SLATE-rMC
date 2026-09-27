"""Atomic, external-only retained diagnostic output."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def write_diagnostic(
    destination: str | Path,
    *,
    arrays: Mapping[str, ArrayLike],
    manifest: Mapping[str, Any],
    repository_root: str | Path,
) -> Path:
    """Atomically write one external NPZ with one embedded JSON manifest."""

    path = Path(destination).resolve()
    root = Path(repository_root).resolve()
    if path == root or path.is_relative_to(root):
        raise ValueError("diagnostic destination must be outside the repository")
    if not path.name.endswith(".ra_diag.npz"):
        raise ValueError("diagnostic destination must end with .ra_diag.npz")
    if not path.parent.is_dir():
        raise ValueError("diagnostic destination parent must already exist")
    if "manifest_json" in arrays:
        raise ValueError("manifest_json is reserved for the embedded manifest")

    payload: dict[str, np.ndarray[Any, Any]] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("diagnostic array names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise ValueError(f"diagnostic array {name!r} may not use object dtype")
        payload[name] = array
    manifest_bytes = json.dumps(
        dict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["manifest_json"] = np.frombuffer(manifest_bytes, dtype=np.uint8)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path
