"""Raw complex structure-factor helpers."""

from __future__ import annotations

import contextlib
from functools import lru_cache
import io
from pathlib import Path


@lru_cache(maxsize=16)
def _load_crystal_cached(cif_path: str):
    import Dans_Diffraction as dif

    with contextlib.redirect_stdout(io.StringIO()):
        xtl = dif.Crystal(cif_path)
        xtl.Symmetry.generate_matrices()
        xtl.generate_structure()
    return xtl


def _load_crystal(cif_path: str | Path):
    return _load_crystal_cached(str(Path(cif_path).resolve()))


