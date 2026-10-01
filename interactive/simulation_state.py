"""Independent configured drafts and exact external-result identities."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

import numpy as np

MAX_SIMULATION_YAML_BYTES = 64 * 1024
SIMULATION_ROUTES = (
    "monte_carlo",
    "pixel_centers",
    "macrobins",
    "reciprocal_space",
    "ewald_surface",
)


def _hash(value: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid simulation SHA-256")


def _path(value: Path) -> None:
    if not isinstance(value, Path) or not value.is_absolute() or not 0 < len(str(value)) <= 4096:
        raise ValueError("simulation paths must be bounded absolute paths")


@dataclass(frozen=True, slots=True)
class SimulationDraft:
    draft_id: UUID
    revision: int
    configuration_path: Path
    imported_sha256: str
    yaml_text: str
    cif_path: Path
    cif_sha256: str
    route: str = "monte_carlo"
    position_mode: str = "sampled"
    draw_count: int = 8
    detector_seed: int = 1729
    transfer_provenance: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.draft_id, UUID)
            or type(self.revision) is not int
            or not 0 <= self.revision < 2**63
        ):
            raise ValueError("invalid independent draft identity or revision")
        _path(self.configuration_path)
        _path(self.cif_path)
        _hash(self.imported_sha256)
        _hash(self.cif_sha256)
        if (
            type(self.yaml_text) is not str
            or not 0 < len(self.yaml_text.encode("utf-8")) <= MAX_SIMULATION_YAML_BYTES
        ):
            raise ValueError("simulation configuration exceeds 64 KiB")
        if self.route not in SIMULATION_ROUTES or self.position_mode not in (
            "sampled",
            "conditional_position",
        ):
            raise ValueError("unsupported simulation route or position mode")
        if type(self.draw_count) is not int or not 1 <= self.draw_count <= 1_000_000:
            raise ValueError("detector draws must be an integer in [1, 1000000]")
        if type(self.detector_seed) is not int or not 0 <= self.detector_seed < 2**64:
            raise ValueError("detector seed must be an unsigned 64-bit integer")

        if (
            type(self.transfer_provenance) is not str
            or len(self.transfer_provenance.encode()) > 16384
        ):
            raise ValueError("configured transfer provenance exceeds 16 KiB")

    @property
    def configuration_sha256(self) -> str:
        return hashlib.sha256(self.yaml_text.encode("utf-8")).hexdigest()


def simulation_draft_document(draft: SimulationDraft | None) -> dict | None:
    if draft is None:
        return None
    return {
        **asdict(draft),
        "draft_id": str(draft.draft_id),
        "configuration_path": str(draft.configuration_path),
        "cif_path": str(draft.cif_path),
    }


def simulation_draft_from_document(value: object) -> SimulationDraft | None:
    if value is None:
        return None
    fields = {
        "draft_id",
        "revision",
        "configuration_path",
        "imported_sha256",
        "yaml_text",
        "cif_path",
        "cif_sha256",
        "route",
        "position_mode",
        "draw_count",
        "detector_seed",
    }
    if type(value) is not dict or set(value) not in (fields, fields | {"transfer_provenance"}):
        raise ValueError("invalid simulation draft document")
    row = value.copy()
    row["draft_id"] = UUID(row["draft_id"])
    row["configuration_path"] = Path(row["configuration_path"])
    row["cif_path"] = Path(row["cif_path"])
    return SimulationDraft(**row)


@dataclass(frozen=True, slots=True)
class SimulationReference:
    path: Path
    sha256: str
    draft_id: UUID
    draft_revision: int
    configuration_sha256: str
    cif_sha256: str
    route: str
    measure: str
    draw_prefix: int
    detector_seed: int

    def __post_init__(self) -> None:
        _path(self.path)
        for digest in (self.sha256, self.configuration_sha256, self.cif_sha256):
            _hash(digest)
        if (
            not isinstance(self.draft_id, UUID)
            or type(self.draft_revision) is not int
            or not 0 <= self.draft_revision < 2**63
        ):
            raise ValueError("invalid simulation result draft identity")
        if (
            self.route not in SIMULATION_ROUTES
            or type(self.measure) is not str
            or not 0 < len(self.measure) <= 256
        ):
            raise ValueError("invalid simulation result measure")
        if (
            type(self.draw_prefix) is not int
            or not 0 <= self.draw_prefix <= 1_000_000
            or type(self.detector_seed) is not int
            or not 0 <= self.detector_seed < 2**64
        ):
            raise ValueError("invalid simulation result draw prefix or detector seed")


def simulation_reference_document(value: SimulationReference | None) -> dict | None:
    if value is None:
        return None
    return {**asdict(value), "path": str(value.path), "draft_id": str(value.draft_id)}


def simulation_reference_from_document(value: object) -> SimulationReference | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {
        "path",
        "sha256",
        "draft_id",
        "draft_revision",
        "configuration_sha256",
        "cif_sha256",
        "route",
        "measure",
        "draw_prefix",
        "detector_seed",
    }:
        raise ValueError("invalid simulation result reference")
    row = value.copy()
    row["path"] = Path(row["path"])
    row["draft_id"] = UUID(row["draft_id"])
    return SimulationReference(**row)


@dataclass(frozen=True, slots=True)
class SimulationExportWork:
    destination: Path
    manifest: bytes
    arrays: tuple[tuple[str, np.ndarray], ...]
    storage_json: str = "{}"

    @property
    def argument_bytes(self) -> int:
        return (
            len(self.manifest)
            + len(self.storage_json.encode())
            + len(str(self.destination).encode())
            + 512 * len(self.arrays)
        )

    def validate(self) -> None:
        from archive_storage import storage_files

        storage_files(self.storage_json)
        _path(self.destination)
        if (
            type(self.manifest) is not bytes
            or len(self.manifest) > 512 * 1024
            or type(self.arrays) is not tuple
            or not 0 < len(self.arrays) <= 32
        ):
            raise ValueError("simulation export exceeds its metadata/array limit")
        if len({k for k, _ in self.arrays}) != len(self.arrays):
            raise ValueError("duplicate simulation export array")
        total = 0
        for key, array in self.arrays:
            if (
                type(key) is not str
                or not key.isidentifier()
                or not isinstance(array, np.ndarray)
                or array.dtype.kind not in "iufb"
                or array.flags.writeable
                or not array.flags.c_contiguous
            ):
                raise ValueError("simulation export needs named immutable numeric arrays")
            total += array.nbytes
        if total > 160 * 1024 * 1024:
            raise ValueError("simulation export arrays exceed 160 MiB")
