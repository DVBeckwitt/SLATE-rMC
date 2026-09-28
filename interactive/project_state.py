"""Project and acquisition identity for the desktop application.

Persistence is added here when the project save/reopen slice is delivered. This
module has no Qt or numerical-package dependency.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

PROJECT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Acquisition:
    acquisition_id: UUID
    name: str
    source_path: Path
    source_sha256: str

    @classmethod
    def create(cls, name: str, source_path: Path, source_sha256: str) -> "Acquisition":
        name = name.strip()
        if not name:
            raise ValueError("An acquisition needs a name")
        if len(source_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in source_sha256
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        return cls(uuid4(), name, Path(source_path), source_sha256)


@dataclass(frozen=True, slots=True)
class Project:
    project_id: UUID
    name: str
    acquisitions: tuple[Acquisition, ...] = ()
    schema_version: int = PROJECT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported project schema version: {self.schema_version}")
        ids = [item.acquisition_id for item in self.acquisitions]
        if len(ids) != len(set(ids)):
            raise ValueError("Acquisition IDs must be unique within a project")

    @classmethod
    def create(cls, name: str = "Untitled project") -> "Project":
        name = name.strip()
        if not name:
            raise ValueError("A project needs a name")
        return cls(uuid4(), name)

    def add_acquisition(self, name: str, source_path: Path, source_sha256: str) -> "Project":
        acquisition = Acquisition.create(name, source_path, source_sha256)
        return replace(self, acquisitions=(*self.acquisitions, acquisition))

    def rename_acquisition(self, acquisition_id: UUID, name: str) -> "Project":
        name = name.strip()
        if not name:
            raise ValueError("An acquisition needs a name")
        if not any(item.acquisition_id == acquisition_id for item in self.acquisitions):
            raise KeyError(acquisition_id)
        return replace(
            self,
            acquisitions=tuple(
                replace(item, name=name) if item.acquisition_id == acquisition_id else item
                for item in self.acquisitions
            ),
        )

    def reorder_acquisitions(self, order: Sequence[UUID]) -> "Project":
        by_id = {item.acquisition_id: item for item in self.acquisitions}
        if len(order) != len(by_id) or set(order) != set(by_id):
            raise ValueError("Order must contain each acquisition ID exactly once")
        return replace(self, acquisitions=tuple(by_id[acquisition_id] for acquisition_id in order))
