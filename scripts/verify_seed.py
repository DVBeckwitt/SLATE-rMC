from __future__ import annotations

import gzip
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

errors: list[str] = []

file_manifest = json.loads((ROOT / "FILE_MANIFEST.json").read_text())
for item in file_manifest["files"]:
    path = ROOT / item["path"]
    if not path.is_file():
        errors.append(f"missing: {item['path']}")
        continue
    if path.stat().st_size != item["size_bytes"]:
        errors.append(f"size: {item['path']}")
    if sha256(path) != item["sha256"]:
        errors.append(f"sha256: {item['path']}")

ref_meta = tomllib.loads((ROOT / "reference/reference_manifest.toml").read_text())
ref = ROOT / ref_meta["reference_pack"]["path"]
if sha256(ref) != ref_meta["reference_pack"]["sha256"]:
    errors.append("reference pack hash")
with np.load(ref, allow_pickle=False) as data:
    manifest = json.loads(bytes(data["manifest_json"]).decode())
    if manifest["schema_version"] != ref_meta["reference_pack"]["schema_version"]:
        errors.append("reference schema")
    if not manifest.get("cases"):
        errors.append("reference cases")

examples = tomllib.loads((ROOT / "examples/MANIFEST.toml").read_text())
for item in examples["file"]:
    path = ROOT / item["path"]
    if sha256(path) != item["sha256"]:
        errors.append(f"example hash: {item['path']}")

for path in (ROOT / "examples").rglob("*.osc.gz"):
    with gzip.open(path, "rb") as handle:
        if handle.read(5) != b"RAXIS":
            errors.append(f"gzip OSC signature: {path.relative_to(ROOT)}")

forbidden = list(ROOT.rglob("*.ra_diag.npz"))
if forbidden:
    errors.append("diagnostic file under repository")

if errors:
    print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
    raise SystemExit(1)
print(json.dumps({"status": "PASS", "files": len(file_manifest["files"]), "reference_cases": len(manifest["cases"])}, indent=2))
