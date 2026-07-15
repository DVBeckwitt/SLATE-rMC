from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import rasim_next


def test_seed_import_and_reference_pack() -> None:
    assert rasim_next.__version__ == "0.0.0"
    pack = Path(__file__).parents[1] / "reference" / "rasim_reference_v1.npz"
    with np.load(pack, allow_pickle=False) as data:
        manifest = json.loads(bytes(data["manifest_json"]).decode())
    assert manifest["schema_version"] == "rasim-reference-pack-v1"
    assert manifest["cases"]
