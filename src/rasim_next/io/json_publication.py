"""Atomic publication of caller-validated JSON documents."""

import json
import os
import tempfile
from pathlib import Path


def publish_json_document(destination: Path, document: object, *, allow_nan: bool = True) -> Path:
    """Write one JSON document with a unique temporary file and guaranteed cleanup."""
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
