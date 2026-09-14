"""Restore archived native source by Git SHA and verified relative paths, without executing it."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path


def restore(bundle, manifest, worktree, destination):
    bundle, manifest, destination = map(
        lambda p: Path(p).resolve(), (bundle, manifest, destination)
    )
    records = json.loads(manifest.read_bytes())
    matching = [
        record
        for record in records
        if record["worktree"] == worktree or Path(record["worktree"]).name == worktree
    ]
    if len(matching) != 1:
        raise ValueError("worktree must identify exactly one archived source state")
    record = matching[0]
    if destination.exists():
        raise ValueError("restore destination must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_path = manifest.parent / Path(record["archive"]).name

    def target(name):
        result = (destination / name).resolve()
        if result == destination or not result.is_relative_to(destination):
            raise ValueError("archive path escapes the restore destination")
        return result

    with zipfile.ZipFile(archive_path) as archive:
        for entry in record["files"]:
            data = archive.read(entry["path"])
            target(entry["path"])
            if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("working-state archive failed its recorded hash")
        with tempfile.TemporaryDirectory(
            prefix="native-recovery-", dir=destination.parent
        ) as temporary:
            temporary_path = Path(temporary).resolve()
            if not temporary_path.is_relative_to(destination.parent):
                raise ValueError("temporary Git location escaped its declared parent")
            subprocess.run(
                ["git", "clone", "--bare", str(bundle), str(temporary_path / "git")],
                check=True,
                capture_output=True,
            )
            tree = subprocess.check_output(
                ["git", "--git-dir=" + str(temporary_path / "git"), "archive", record["HEAD"]]
            )
            with tarfile.open(fileobj=io.BytesIO(tree)) as source:
                for member in source.getmembers():
                    path = target(member.name)
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(source.extractfile(member).read())
                    else:
                        raise ValueError("source archive contains an unsupported link or device")
        for entry in record["files"]:
            # Environments contain absolute launchers; recreate from pinned metadata.
            if Path(entry["path"]).parts[0] == ".venv":
                continue
            path = target(entry["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(entry["path"]))
        for name in record["deleted"]:
            target(name).unlink(missing_ok=True)
    relocation = dict(
        git_commit=record["HEAD"],
        original_worktree=record["worktree"],
        restored_root=str(destination),
        source_root=str(destination / "src"),
        environment="recreate; original environment bytes remain archived",
    )
    (destination / "RECOVERY.json").write_text(
        json.dumps(relocation, indent=2) + "\n", encoding="utf-8"
    )
    return relocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worktree", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(restore(args.bundle, args.manifest, args.worktree, args.destination), indent=2)
    )


if __name__ == "__main__":
    main()
