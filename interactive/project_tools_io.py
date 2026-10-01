"""Bounded recovery review and independent project publication on the shared worker."""

import hashlib
import json
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from uuid import UUID

from job_lifecycle import JobResult
from project_io import MAX_RECOVERY_DRAFTS, write_project
from project_state import MAX_PROJECT_BYTES, read_project_document
from sample_state import encoded


@dataclass(frozen=True, slots=True)
class ProjectToolsResult:
    operation: str
    review_json: str
    path: Path | None = None
    sha256: str = ""

    @property
    def nbytes(self):
        return len(self.review_json.encode()) + len(str(self.path).encode()) + 1024


def project_tools_work(argument, control):
    if type(argument) is not bytes or len(argument) > 4 * 1024**2:
        raise ValueError("Project tools request exceeds 4 MiB")
    request = json.loads(argument)
    if request.get("operation") == "duplicate":
        if set(request) != {"operation", "document", "destination", "source_id", "new_id"}:
            raise ValueError("Malformed independent copy request")
        if request["new_id"] == request["source_id"]:
            raise ValueError("Independent copy requires a new project identity")
        from attempt_state import read_history
        from project_state import project_from_document

        destination = Path(request["destination"])
        document = project_from_document(request["document"], destination)
        if (
            str(document.project.project_id) != request["new_id"]
            or str(read_history(document.view.attempts_json).inherited_from) != request["source_id"]
        ):
            raise ValueError("Independent copy differs from its reviewed identities")
        published = write_project(
            encoded(
                {
                    "destination": str(destination),
                    "document": request["document"],
                    "recovery": False,
                    "retire_draft": None,
                    "exclusive": True,
                }
            ).encode(),
            control,
        ).value
        value = ProjectToolsResult(
            "duplicate",
            encoded(
                {
                    "source_id": request["source_id"],
                    "new_id": request["new_id"],
                    "sha256": published.sha256,
                    "sharing": "Exact referenced files remain explicitly shared; inherited history is inspection. Mutable project/recovery state is independent.",
                }
            ),
            destination,
            published.sha256,
        )
    elif request.get("operation") == "recovery":
        if set(request) != {"operation", "root"}:
            raise ValueError("Malformed recovery review request")
        root = Path(request["root"])
        if not root.is_absolute() or root.is_symlink():
            raise ValueError("Recovery root must be an explicit unlinked absolute directory")
        paths = tuple(islice(root.glob("*.slate.json"), MAX_RECOVERY_DRAFTS + 1))
        if len(paths) > MAX_RECOVERY_DRAFTS:
            raise ValueError(
                "Recovery review exceeds 32 drafts; manage the configured folder explicitly"
            )
        rows = []
        for path in sorted(paths):
            if control.canceled:
                raise RuntimeError("Recovery review canceled")
            row = {"path": str(path), "valid": False}
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Linked or non-file recovery candidate")
                UUID(path.name.removesuffix(".slate.json"))
                before = path.stat()
                with path.open("rb") as stream:
                    raw = stream.read(MAX_PROJECT_BYTES + 1)
                document = read_project_document(path, source_bytes=raw)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Recovery candidate changed during review")
                if path.name != f"{document.project.project_id}.slate.json":
                    raise ValueError("Recovery filename differs from project UUID")
                row.update(
                    valid=True,
                    project_id=str(document.project.project_id),
                    name=document.project.name,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    size=len(raw),
                    modified_ns=after.st_mtime_ns,
                    age_seconds=max(0, time.time() - after.st_mtime),
                    sources=[
                        {
                            "acquisition_id": str(a.acquisition_id),
                            "name": a.name,
                            "path": str(a.source_path),
                            "sha256": a.source_sha256,
                        }
                        for a in document.project.acquisitions
                    ],
                )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                row["reason"] = str(exc)[:512]
            rows.append(row)
            control.report(f"Recovery review: {len(rows)}/{len(paths)} candidates")
        text = encoded(
            {
                "root": str(root),
                "drafts": rows,
                "limits": "32 drafts, 1 MiB each. Open rechecks exact bytes and sources; no solver resumes.",
            }
        )
        if len(text.encode()) > 1024**2:
            raise ValueError("Recovery identity review exceeds 1 MiB")
        value = ProjectToolsResult("recovery", text)
    else:
        raise ValueError("Unsupported project tools operation")
    return JobResult(value, len(value.review_json.encode()) + 4096)
