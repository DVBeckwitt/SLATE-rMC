"""Independent native physical drafts; no observations, fit or mutable model owner."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from simulation_state import _hash, _path

NATIVE_RECIPES = ("bi2se3", "bi2te3", "gd1", "sid1", "clean1", "b4")
MAX_NATIVE_PHYSICS_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class NativeSimulationDraft:
    draft_id: UUID
    revision: int
    recipe: str
    physics_path: Path
    imported_sha256: str
    physics_json: str
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameter_values: tuple[float, ...]
    coherent_repeats: int
    bin_size_px: int = 1
    proposal_mosaic: tuple[float, float, float] | None = None
    transfer_provenance: str = ""
    surface_fractions: tuple[float, ...] | None = None
    phase_fractions: tuple[float, ...] | None = None

    def __post_init__(self):
        _path(self.physics_path)
        _hash(self.imported_sha256)
        if (
            not isinstance(self.draft_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
            or self.recipe not in NATIVE_RECIPES
        ):
            raise ValueError("invalid native draft identity, revision or admitted recipe")
        if (
            type(self.physics_json) is not str
            or not 0 < len(self.physics_json.encode()) <= MAX_NATIVE_PHYSICS_BYTES
        ):
            raise ValueError("native physical declaration exceeds 128 KiB")
        if (
            type(self.parameter_names) is not tuple
            or type(self.parameter_units) is not tuple
            or type(self.parameter_values) is not tuple
            or not 0 < len(self.parameter_names) <= 64
            or len(self.parameter_names) != len(self.parameter_values)
            or len(self.parameter_units) != len(self.parameter_values)
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or any(
                type(v) is not str or not 0 < len(v) <= 128
                for v in (*self.parameter_names, *self.parameter_units)
            )
            or any(
                type(v) not in (float, int) or not math.isfinite(v) for v in self.parameter_values
            )
        ):
            raise ValueError("native parameter names, units and finite values must align")
        if (
            type(self.coherent_repeats) is not int
            or not 1 <= self.coherent_repeats <= 1000000
            or type(self.bin_size_px) is not int
            or not 1 <= self.bin_size_px <= 16384
        ):
            raise ValueError(
                "native repeats and rectangle bin size must be positive bounded integers"
            )
        if self.proposal_mosaic is not None and (
            type(self.proposal_mosaic) is not tuple
            or len(self.proposal_mosaic) != 3
            or any(
                type(v) not in (float, int) or not math.isfinite(v) for v in self.proposal_mosaic
            )
        ):
            raise ValueError("proposal mosaic needs sigma/gamma radians and probability")
        for fractions in (self.surface_fractions, self.phase_fractions):
            if fractions is not None and (
                type(fractions) is not tuple
                or not 1 <= len(fractions) <= 16
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in fractions
                )
                or not math.isclose(sum(fractions), 1, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError("exact declared fractions must be finite normalized probabilities")
        if (
            type(self.transfer_provenance) is not str
            or len(self.transfer_provenance.encode()) > 16384
        ):
            raise ValueError("native transfer provenance exceeds 16 KiB")

    @property
    def physics_sha256(self):
        return hashlib.sha256(self.physics_json.encode()).hexdigest()

    @property
    def parameter_sha256(self):
        return hashlib.sha256(
            json.dumps(
                {
                    "names": self.parameter_names,
                    "units": self.parameter_units,
                    "values": self.parameter_values,
                    "coherent_repeats": self.coherent_repeats,
                    "surface_fractions": self.surface_fractions,
                    "phase_fractions": self.phase_fractions,
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()


def native_draft_document(draft):
    if draft is None:
        return None
    return {
        **asdict(draft),
        "draft_id": str(draft.draft_id),
        "physics_path": str(draft.physics_path),
    }


def native_draft_from_document(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != set(NativeSimulationDraft.__dataclass_fields__):
        raise ValueError("invalid independent native draft document")
    row = value.copy()
    row["draft_id"] = UUID(row["draft_id"])
    row["physics_path"] = Path(row["physics_path"])
    for key in (
        "parameter_names",
        "parameter_units",
        "parameter_values",
        "proposal_mosaic",
        "surface_fractions",
        "phase_fractions",
    ):
        if row[key] is not None:
            row[key] = tuple(row[key])
    return NativeSimulationDraft(**row)


@dataclass(frozen=True, slots=True)
class NativeSimulationReference:
    path: Path
    sha256: str
    draft_id: UUID
    draft_revision: int
    recipe: str
    physics_sha256: str
    parameter_sha256: str
    numerical_sha256: str
    measure: str
    completed_batches: int

    def __post_init__(self):
        _path(self.path)
        for value in (
            self.sha256,
            self.physics_sha256,
            self.parameter_sha256,
            self.numerical_sha256,
        ):
            _hash(value)
        if (
            not isinstance(self.draft_id, UUID)
            or self.recipe not in NATIVE_RECIPES
            or type(self.draft_revision) is not int
            or not 0 <= self.draft_revision < 2**63
            or type(self.completed_batches) is not int
            or self.completed_batches < 0
            or type(self.measure) is not str
            or not 0 < len(self.measure) <= 256
        ):
            raise ValueError("invalid native result reference")


def native_reference_document(value):
    if value is None:
        return None
    return {**asdict(value), "path": str(value.path), "draft_id": str(value.draft_id)}


def native_reference_from_document(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != set(NativeSimulationReference.__dataclass_fields__):
        raise ValueError("invalid native result reference document")
    row = value.copy()
    row["path"] = Path(row["path"])
    row["draft_id"] = UUID(row["draft_id"])
    return NativeSimulationReference(**row)
