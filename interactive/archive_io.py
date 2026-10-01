"""Bounded reviewed archives of delivered desktop state; no scientific execution."""

import hashlib
import json
import os
import shutil
import stat
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from uuid import uuid4

from archive_storage import storage_files, stored_path
from attempt_state import history_text, read_history, retain_results, selected_history
from hbn_io import _hash_file
from job_lifecycle import JobResult
from project_state import (
    MAX_PROJECT_BYTES,
    ProjectDocument,
    StorageOrigin,
    project_from_document,
    project_to_document,
)
from sample_state import encoded, payload_hash
from simulation_io import _external, _stop

MAX_FILES = 256
MAX_FILE_BYTES = 512 * 1024**2
MAX_TOTAL_BYTES = 2 * 1024**3
MAX_MANIFEST_BYTES = 512 * 1024


@dataclass(frozen=True, slots=True)
class ArchiveWorkResult:
    operation: str
    review_json: str | None = None
    path: Path | None = None
    detail: str = ""

    @property
    def nbytes(self):
        return len((self.review_json or "").encode()) + 8192


def product_choices(state):
    rows = [("views", "Masks, calibration, ROI, comparison and physical settings")]
    rows += [
        ("acquisition:" + str(a.acquisition_id), "Acquisition: " + a.name)
        for a in state.project.acquisitions
    ]
    view = state.view
    history = read_history(view.attempts_json)
    for key, label, present in (
        (
            "configured",
            "Configured simulation draft / retained output",
            view.simulation_draft or view.simulation_result or history.configured,
        ),
        (
            "native",
            "Native simulation draft / retained output",
            view.native_simulation_draft or view.native_simulation_result or history.native,
        ),
        ("hbn", "hBN current inputs / retained result history", view.hbn_sessions),
        ("sample", "Sample current inputs / retained result history", view.sample_session),
        ("joint", "Joint captures / retained reports / handoffs", view.joint_session),
        (
            "prepared",
            "Prepared inputs / parameter definitions, history and pending edits",
            view.native_fit_session,
        ),
    ):
        if present:
            rows.append((key, label))
    return tuple(rows)


def _selected(state, selection):
    allowed = {k for k, _ in product_choices(state)}
    if (
        type(selection) is not list
        or len(set(selection)) != len(selection)
        or set(selection) - allowed
    ):
        raise ValueError("Archive selection differs from available products")
    acquisitions = tuple(
        a for a in state.project.acquisitions if "acquisition:" + str(a.acquisition_id) in selection
    )
    ids = {a.acquisition_id for a in acquisitions}
    view = state.view
    if "hbn" in selection and any(s.acquisition_id not in ids for s in view.hbn_sessions):
        raise ValueError("Select the hBN acquisitions with their session dependencies")
    selected = view.selected_acquisition_id if view.selected_acquisition_id in ids else None
    view = replace(
        view,
        selected_acquisition_id=selected,
        detector=view.detector if selected is not None and "views" in selection else None,
        scene=view.scene if selected is not None and "views" in selection else None,
        comparison=view.comparison
        if "views" in selection and len(ids) == len(state.project.acquisitions)
        else None,
        physical_settings_json=view.physical_settings_json if "views" in selection else "{}",
        simulation_draft=view.simulation_draft if "configured" in selection else None,
        simulation_result=view.simulation_result if "configured" in selection else None,
        simulation_detector=view.simulation_detector if "configured" in selection else None,
        native_simulation_draft=view.native_simulation_draft if "native" in selection else None,
        native_simulation_result=view.native_simulation_result if "native" in selection else None,
        hbn_sessions=view.hbn_sessions if "hbn" in selection else (),
        sample_session=view.sample_session if "sample" in selection else None,
        joint_session=view.joint_session if "joint" in selection else None,
        native_fit_session=view.native_fit_session if "prepared" in selection else None,
        attempts_json=selected_history(view.attempts_json, selection),
    )
    numeric = (
        state.numeric_draft
        if state.numeric_draft is not None and state.numeric_draft.acquisition_id in ids
        else None
    )
    return ProjectDocument(replace(state.project, acquisitions=acquisitions), view, numeric)


def _snapshot_draft(path):
    """Read only the bounded numeric snapshot metadata, never its result arrays."""
    from io import BytesIO

    import numpy as np
    from native_simulation_state import native_draft_from_document
    from simulation_state import simulation_draft_from_document

    if path.stat().st_size > 192 * 1024**2:
        raise ValueError("Retained simulation snapshot exceeds 192 MiB")
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("manifest_utf8.npy")
        if info.file_size > 512 * 1024 + 4096:
            raise ValueError("Retained snapshot manifest exceeds its bounded header/data size")
        stream = BytesIO(archive.read(info))
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError("Unsupported retained snapshot manifest array version")
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    if dtype != np.dtype("uint8") or fortran or len(shape) != 1 or not 0 <= shape[0] <= 512 * 1024:
        raise ValueError("Retained snapshot requires a bounded uint8 manifest")
    raw = stream.read()
    if len(raw) != shape[0]:
        raise ValueError("Retained snapshot manifest size differs")
    metadata = json.loads(raw)
    if metadata["schema"] == "slate.configured-snapshot.v1":
        return "configured", simulation_draft_from_document(metadata["draft"])
    if metadata["schema"] == "slate.native-snapshot.v1":
        return "native", native_draft_from_document(metadata["draft"])
    raise ValueError("Unsupported retained snapshot manifest")


def _references(state, storage=()):
    """Inventory explicit delivered formats, including immutable historical launches."""
    rows, qualifications = [], []

    def add(product, path, expected=None, decoded=None, reason=""):
        rows.append(
            dict(
                product=product,
                original=str(Path(path).resolve()),
                expected=expected,
                decoded=decoded,
                reason=reason,
            )
        )

    def exports(product, session):
        for path, digest in session.exports:
            add(product, path, digest)

    def hbn_inputs(product, inputs):
        for path, digest in (
            ("source_path", "source_file_sha256"),
            ("dark_path", "dark_file_sha256"),
            ("configuration_path", "configuration_sha256"),
            ("cif_path", "cif_sha256"),
        ):
            add(product, inputs[path], inputs[digest])

    def sample_inputs(product, inputs):
        for row in inputs["files"]:
            add(product, row["path"], row["sha256"])

    for acquisition in state.project.acquisitions:
        key = "acquisition:" + str(acquisition.acquisition_id)
        add(key, acquisition.source_path, decoded=acquisition.source_sha256)
        m = acquisition.metadata
        if m.configuration_path is not None and m.configuration_cif_path is None:
            add(
                key,
                m.configuration_path,
                m.configuration_sha256,
                reason="No recorded configuration-dependent CIF identity; review the configuration through its current owner",
            )
        for k in ("cif", "configuration", "configuration_cif"):
            path, digest = getattr(m, k + "_path"), getattr(m, k + "_sha256")
            if path is not None:
                if digest is None:
                    raise ValueError("Archive requires an exact identity for " + k)
                add(key, path, digest)
    v = state.view
    if state.numeric_draft is not None:
        d = state.numeric_draft
        add("views", d.configuration_path, d.configuration_sha256)
        add("views", d.cif_path, d.cif_sha256)
    if v.simulation_draft is not None:
        d = v.simulation_draft
        add("configured", d.configuration_path, d.imported_sha256)
        add("configured", d.cif_path, d.cif_sha256)
    if v.native_simulation_draft is not None:
        add(
            "native",
            v.native_simulation_draft.physics_path,
            v.native_simulation_draft.imported_sha256,
        )
    history = read_history(
        retain_results(v.attempts_json, v.simulation_result, v.native_simulation_result)
    )
    for product, ref in (
        *(("configured", ref) for ref in history.configured),
        *(("native", ref) for ref in history.native),
    ):
        if ref is not None:
            add(product, ref.path, ref.sha256)
            actual = stored_path(ref.path, storage, ref.sha256)
            if actual.is_file():
                snapshot_kind, draft = _snapshot_draft(actual)
                if snapshot_kind != product:
                    raise ValueError("Retained snapshot route differs from its project reference")
                if product == "configured":
                    add(product, draft.configuration_path, draft.imported_sha256)
                    add(product, draft.cif_path, draft.cif_sha256)
                else:
                    add(product, draft.physics_path, draft.imported_sha256)
            qualifications.append(
                dict(
                    product=product,
                    state="retained simulation output; not a fit qualification",
                    measure=ref.measure,
                )
            )
    for session in v.hbn_sessions:
        hbn_inputs("hbn", json.loads(session.inputs_json))
        for text in session.results_json:
            result = json.loads(text)
            hbn_inputs("hbn", result["launch"]["inputs"])
            qualifications.append(
                dict(
                    product="hbn",
                    result_id=result["result_id"],
                    selected=session.selected_result_id == result["result_id"],
                    qualification=result.get("qualification"),
                )
            )
        exports("hbn", session)
    if v.sample_session is not None:
        session = v.sample_session
        sample_inputs("sample", json.loads(session.inputs_json))
        for text in session.results_json:
            result = json.loads(text)
            sample_inputs("sample", result["launch"]["inputs"])
            qualifications.append(
                dict(
                    product="sample",
                    result_id=result["result_id"],
                    selected=session.selected_result_id == result["result_id"],
                    state="unqualified geometry record; original fields retained",
                )
            )
        exports("sample", session)
    if v.joint_session is not None:
        session = v.joint_session
        launches = [session.launch]
        for text in session.results_json:
            result = json.loads(text)
            if result["launch"] is not None:
                launches.append(result["launch"])
            if result["report_origin"] is not None:
                add("joint", result["report_origin"]["path"], result["report_origin"]["sha256"])
            qualifications.append(
                dict(
                    product="joint",
                    result_id=result["result_id"],
                    selected=session.selected_result_id == result["result_id"],
                    qualification={
                        k: result["report"][k]
                        for k in ("confidence_qualified", "qualification_failures")
                    },
                    state="original report/qualification retained; storage does not grant launch admission",
                )
            )
        for launch in launches:
            for group, capture in launch["captures"].items():
                inputs = json.loads(capture["session"]["inputs_json"])
                (hbn_inputs if group == "hbn" else sample_inputs)("joint", inputs)
        exports("joint", session)
        for path, digest in session.exports:
            actual = stored_path(path, storage, digest)
            if actual.suffix.lower() == ".json" and actual.is_file():
                if actual.stat().st_size > 2 * 1024**2:
                    raise ValueError("Joint export JSON exceeds 2 MiB")
                data = json.loads(actual.read_bytes())
                if data.get("schema_version") == "rasim-joint-geometry-handoff-v1":
                    for key in (
                        "report",
                        "geometry_manifest",
                        "specimen_config",
                        "detector_base_config",
                        "specimen_cif",
                        "detector_base_cif",
                    ):
                        add("joint", data[key]["path"], data[key]["sha256"])
                    for row in data["osc_images"]:
                        add("joint", row["path"], row["sha256"])

    if v.native_fit_session is not None:
        session = v.native_fit_session
        for text in (session.current_json, *session.history_json):
            d = json.loads(text)
            for row in d["inputs"]["files"]:
                add("prepared", row["path"], row["sha256"])
            qualifications.append(
                dict(
                    product="prepared",
                    description_sha256=d["sha256"],
                    definition_sha256=d["definition"]["sha256"],
                    state="parsed immutable draft; Run/adoption unavailable",
                )
            )
        exports("prepared", session)
    return rows, qualifications


def _verify_osc(path, decoded_sha256, control):
    from osc_import import AXIS_LIMIT, DECODED_LIMIT_BYTES, PIXEL_LIMIT, SOURCE_LIMIT_BYTES

    from rasim_next.io.osc import OscReadLimits, read_osc

    image = read_osc(
        path,
        limits=OscReadLimits(SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, AXIS_LIMIT),
        canceled=lambda: control.canceled,
    )
    if image.decoded_sha256 != decoded_sha256:
        raise ValueError("Decoded OSC differs from acquisition identity")


def _review(request, control):
    anchor = Path(request["anchor"])
    original = project_from_document(request["document"], anchor)
    state = _selected(original, request["selection"])
    snapshot = project_to_document(state, anchor)
    storage = storage_files(original.view.archive_storage_json)
    references, qualifications = _references(state, storage)
    by_path, missing = {}, []
    for reference in references:
        _stop(control)
        key = reference["original"].casefold()
        try:
            if reference["reason"]:
                raise ValueError(reference["reason"])
            path = stored_path(reference["original"], storage, reference["expected"])
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Missing, linked, unsupported or oversized archive input")
            digest = _hash_file(path, control)
            if reference["expected"] is not None and digest != reference["expected"]:
                raise ValueError("SHA256 differs from recorded predecessor")
            if reference["decoded"] is not None:
                _verify_osc(path, reference["decoded"], control)
            row = by_path.get(key)
            if row is not None:
                if row["sha256"] != digest:
                    raise ValueError("Aliased input identity changed")
                row["products"] = sorted(set(row["products"] + [reference["product"]]))
            else:
                by_path[key] = dict(
                    original=reference["original"],
                    source=str(path),
                    sha256=digest,
                    size=path.stat().st_size,
                    products=[reference["product"]],
                )
        except (ValueError, OSError) as exc:
            missing.append({**reference, "reason": str(exc)[:300]})
        control.report(f"Archive inventory: {len(by_path)} files, {len(missing)} unavailable")
    files = list(by_path.values())
    for i, row in enumerate(files):
        source = Path(row["original"])
        suffix = ".osc.gz" if source.name.lower().endswith(".osc.gz") else source.suffix.lower()
        row["name"] = f"files/{i:04d}" + suffix
    total = sum(row["size"] for row in files)
    if len(files) > MAX_FILES or total > MAX_TOTAL_BYTES:
        raise ValueError("Archive exceeds 256 files or 2 GiB expanded data")
    review = dict(
        schema="slate.archive-review.v1",
        snapshot_sha256=payload_hash(request["document"]),
        selection=request["selection"],
        anchor=str(anchor),
        document=snapshot,
        files=files,
        unavailable=missing,
        qualifications=qualifications,
        total_bytes=total,
        project_bytes=len(
            (json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
        future_unavailable=[
            "Native fit Run/adoption (R4)",
            "Stage execution (R5)",
            "New observation preparation (R6)",
        ],
    )
    review["sha256"] = payload_hash(review)
    text = encoded(review)
    if len(text.encode()) > MAX_PROJECT_BYTES + MAX_MANIFEST_BYTES:
        raise ValueError("Archive review exceeds bounded state")
    return ArchiveWorkResult(
        "review",
        text,
        detail="Review complete; unavailable selected predecessors block self-contained export",
    )


def _review_document(text):
    if type(text) is not str or len(text.encode()) > MAX_PROJECT_BYTES + MAX_MANIFEST_BYTES:
        raise ValueError("Archive review exceeds bounded state")
    review = json.loads(text)
    if review["schema"] != "slate.archive-review.v1" or review["sha256"] != payload_hash(
        {k: v for k, v in review.items() if k != "sha256"}
    ):
        raise ValueError("Archive review identity differs")
    if review["unavailable"]:
        raise ValueError("Selected predecessors are unavailable; archive is not self-contained")
    return review


def _copy(source, target, control):
    digest = hashlib.sha256()
    count = 0
    while chunk := source.read(1024 * 1024):
        _stop(control)
        target.write(chunk)
        digest.update(chunk)
        count += len(chunk)
        if count > MAX_FILE_BYTES:
            raise ValueError("Archive member exceeds 512 MiB")
    return digest.hexdigest(), count


def _export(request, control):
    review = _review_document(request["review_json"])
    destination = Path(request["path"]).resolve()
    _external(
        destination,
        protected=[Path(r["source"]) for r in review["files"]] + [Path(review["anchor"])],
    )
    if destination.exists() or destination.suffix.lower() != ".slatezip":
        raise ValueError("Choose a new .slatezip destination")
    for row in review["files"]:
        _stop(control)
        if (
            Path(row["source"]).stat().st_size != row["size"]
            or _hash_file(row["source"], control) != row["sha256"]
        ):
            raise ValueError("Archive source changed since review: " + row["original"])
    project_bytes = (
        json.dumps(review["document"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    manifest = {
        "schema": "slate.portable-archive.v1",
        "review_sha256": review["sha256"],
        "project": {
            "name": "project-original.slate.json",
            "size": len(project_bytes),
            "sha256": hashlib.sha256(project_bytes).hexdigest(),
        },
        "files": [{k: v for k, v in r.items() if k != "source"} for r in review["files"]],
        "qualifications": review["qualifications"],
        "selection": review["selection"],
        "future_unavailable": review["future_unavailable"],
        "original_anchor": review["anchor"],
        "total_bytes": review["total_bytes"],
    }
    manifest_bytes = encoded(manifest).encode()
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("Archive manifest exceeds 512 KiB")
    if (
        shutil.disk_usage(destination.parent).free
        < review["total_bytes"] + MAX_PROJECT_BYTES + MAX_MANIFEST_BYTES + 16 * 1024**2
    ):
        raise ValueError("Archive destination lacks the bounded required free disk space")
    staging = destination.with_name("." + destination.name + "." + uuid4().hex + ".tmp")
    try:
        with staging.open("xb") as stream:
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                archive.writestr("manifest.json", manifest_bytes)
                archive.writestr(manifest["project"]["name"], project_bytes)
                for row in review["files"]:
                    control.report("Archiving " + Path(row["original"]).name)
                    with (
                        Path(row["source"]).open("rb") as source,
                        archive.open(row["name"], "w", force_zip64=True) as target,
                    ):
                        digest, size = _copy(source, target, control)
                    if digest != row["sha256"] or size != row["size"]:
                        raise ValueError("Source changed while archiving")
            stream.flush()
            os.fsync(stream.fileno())
        for row in review["files"]:
            _stop(control)
            if (
                Path(row["source"]).stat().st_size != row["size"]
                or _hash_file(row["source"], control) != row["sha256"]
            ):
                raise ValueError("Archive source changed before publication: " + row["original"])
        _stop(control)
        os.link(staging, destination)
    finally:
        staging.unlink(missing_ok=True)
    return ArchiveWorkResult(
        "export",
        path=destination,
        detail="Self-contained archive published without overwrite; no project state changed",
    )


def _safe_name(name):
    if (
        type(name) is not str
        or not name
        or len(name) > 256
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("Unsafe archive member name")
    parts = PurePosixPath(name).parts
    if (
        name.startswith("/")
        or not parts
        or any(
            p in ("", ".", "..")
            or p.endswith((".", " "))
            or p.split(".")[0].casefold()
            in {
                "con",
                "prn",
                "aux",
                "nul",
                *[f"com{i}" for i in range(1, 10)],
                *[f"lpt{i}" for i in range(1, 10)],
            }
            for p in name.split("/")
        )
    ):
        raise ValueError("Unsafe archive member name")
    return name


def _manifest(archive):
    entries = archive.infolist()
    names = [_safe_name(v.filename) for v in entries]
    if len(entries) > MAX_FILES + 2 or len({v.casefold() for v in names}) != len(names):
        raise ValueError("Archive has duplicate/colliding names or too many files")
    for entry in entries:
        mode = entry.external_attr >> 16
        if (
            entry.is_dir()
            or entry.flag_bits & 1
            or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            or entry.file_size > MAX_FILE_BYTES
        ):
            raise ValueError("Archive has unsupported/linked/encrypted/oversized entries")
    if sum(v.file_size for v in entries) > MAX_TOTAL_BYTES + MAX_PROJECT_BYTES + MAX_MANIFEST_BYTES:
        raise ValueError("Archive expansion exceeds 2 GiB")
    info = archive.getinfo("manifest.json")
    if info.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("Archive manifest exceeds 512 KiB")
    manifest = json.loads(archive.read(info))
    if (
        set(manifest)
        != {
            "schema",
            "review_sha256",
            "project",
            "files",
            "qualifications",
            "selection",
            "future_unavailable",
            "original_anchor",
            "total_bytes",
        }
        or manifest["schema"] != "slate.portable-archive.v1"
    ):
        raise ValueError("Unsupported archive manifest")
    files = manifest["files"]
    if type(files) is not list or len(files) > MAX_FILES:
        raise ValueError("Malformed archive file inventory")
    project = manifest["project"]
    if (
        set(project) != {"name", "size", "sha256"}
        or project["name"] != "project-original.slate.json"
        or project["size"] > MAX_PROJECT_BYTES
    ):
        raise ValueError("Malformed archive project inventory")
    declared = [project, *files]
    if set(names) != {"manifest.json", *[v["name"] for v in declared]} or len(
        {v["name"].casefold() for v in declared}
    ) != len(declared):
        raise ValueError("Archive inventory is incomplete/duplicated")
    for row in declared:
        _safe_name(row["name"])
        if (
            type(row["size"]) is not int
            or row["size"] < 0
            or archive.getinfo(row["name"]).file_size != row["size"]
            or type(row["sha256"]) is not str
            or len(row["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("Malformed archive file identity")
    for row in files:
        if (
            set(row) != {"name", "original", "sha256", "size", "products"}
            or not row["name"].startswith("files/")
            or len(PurePosixPath(row["name"]).parts) != 2
            or not Path(row["original"]).is_absolute()
        ):
            raise ValueError("Malformed archive provenance")
    if manifest["total_bytes"] != sum(v["size"] for v in files):
        raise ValueError("Archive total differs")
    return manifest


def _relocate(state, storage):
    """Change storage references only; hashed scientific launch/draft descriptions stay exact."""
    rows = storage_files(storage)

    def local(path, expected=None):
        return stored_path(path, rows, expected)

    acquisitions = []
    for acquisition in state.project.acquisitions:
        m = acquisition.metadata
        changes = {}
        origins = {v.kind: v for v in acquisition.storage}
        for key in ("cif", "configuration", "configuration_cif"):
            original = getattr(m, key + "_path")
            if original is not None:
                changes[key + "_path"] = local(original, getattr(m, key + "_sha256"))
        for key, original in [
            ("osc", acquisition.source_path),
            *((k, getattr(m, k + "_path")) for k in ("cif", "configuration", "configuration_cif")),
        ]:
            if original is not None:
                row = next(
                    v for v in rows if Path(v["original"]).resolve() == Path(original).resolve()
                )
                prior = origins.get(key)
                provenance = (
                    prior.original_path
                    if prior is not None and prior.raw_sha256 == row["sha256"]
                    else Path(row["original"])
                )
                origins[key] = StorageOrigin(
                    key, "copy", provenance, Path(row["stored"]), row["sha256"]
                )
        acquisitions.append(
            replace(
                acquisition,
                source_path=local(acquisition.source_path),
                metadata=replace(m, **changes),
                storage=tuple(origins.values()),
            )
        )
    v = state.view
    sd = v.simulation_draft
    nd = v.native_simulation_draft
    if nd is not None:
        nd = replace(nd, physics_path=local(nd.physics_path, nd.imported_sha256))
    sr = v.simulation_result
    nr = v.native_simulation_result
    if sr is not None:
        sr = replace(sr, path=local(sr.path, sr.sha256))
    if nr is not None:
        nr = replace(nr, path=local(nr.path, nr.sha256))
    numeric = state.numeric_draft
    if numeric is not None:
        numeric = replace(
            numeric,
            configuration_path=local(numeric.configuration_path, numeric.configuration_sha256),
            cif_path=local(numeric.cif_path, numeric.cif_sha256),
        )
    return ProjectDocument(
        replace(state.project, acquisitions=tuple(acquisitions)),
        replace(
            v,
            simulation_draft=sd,
            simulation_result=sr,
            native_simulation_draft=nd,
            native_simulation_result=nr,
            archive_storage_json=storage,
            attempts_json=history_text(
                replace(
                    read_history(v.attempts_json),
                    configured=tuple(
                        replace(ref, path=local(ref.path, ref.sha256))
                        for ref in read_history(v.attempts_json).configured
                    ),
                    native=tuple(
                        replace(ref, path=local(ref.path, ref.sha256))
                        for ref in read_history(v.attempts_json).native
                    ),
                )
            ),
        ),
        numeric,
    )


def _import(request, control):
    source = Path(request["path"]).resolve(strict=True)
    if source.stat().st_size > MAX_TOTAL_BYTES + 16 * 1024**2:
        raise ValueError("Archive file exceeds bounded size")
    destination = Path(request["destination"]).resolve()
    _external(destination, protected=[source])
    if destination.exists():
        raise ValueError("Import needs a new external directory; existing contents are protected")
    staging = destination.with_name("." + destination.name + "." + uuid4().hex + ".tmp")
    archive_sha = _hash_file(source, control)
    created = False
    try:
        with zipfile.ZipFile(source) as archive:
            manifest = _manifest(archive)
            if (
                shutil.disk_usage(destination.parent).free
                < manifest["total_bytes"] + MAX_PROJECT_BYTES + MAX_MANIFEST_BYTES + 16 * 1024**2
            ):
                raise ValueError("Archive import lacks the bounded required free disk space")
            staging.mkdir()
            created = True
            for row in [manifest["project"], *manifest["files"]]:
                _stop(control)
                target = staging / row["name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                control.report("Restoring " + row["name"])
                with archive.open(row["name"]) as src, target.open("xb") as dst:
                    digest, size = _copy(src, dst, control)
                if digest != row["sha256"] or size != row["size"]:
                    raise ValueError("Archive member SHA256/size differs")
            storage = encoded(
                {
                    "schema": "slate.archive-storage.v1",
                    "archive_sha256": archive_sha,
                    "files": [
                        dict(
                            original=r["original"],
                            stored=str(destination / r["name"]),
                            sha256=r["sha256"],
                            size=r["size"],
                        )
                        for r in manifest["files"]
                    ],
                }
            )
            state = project_from_document(
                json.loads((staging / manifest["project"]["name"]).read_bytes()),
                Path(manifest["original_anchor"]),
            )
            selected = _selected(state, manifest["selection"])
            if project_to_document(
                selected, Path(manifest["original_anchor"])
            ) != project_to_document(state, Path(manifest["original_anchor"])):
                raise ValueError("Archive product selection differs from its original snapshot")
            relocated = _relocate(state, storage)
            document = project_to_document(relocated, destination / "reopened.slate.json")
            # Closure must include every required selected predecessor; manifest tampering cannot remove one.
            staged_storage = storage_files(
                encoded(
                    {
                        "schema": "slate.archive-storage.v1",
                        "archive_sha256": archive_sha,
                        "files": [
                            dict(
                                original=r["original"],
                                stored=str(staging / r["name"]),
                                sha256=r["sha256"],
                                size=r["size"],
                            )
                            for r in manifest["files"]
                        ],
                    }
                )
            )
            refs, qualifications = _references(state, staged_storage)
            if qualifications != manifest["qualifications"]:
                raise ValueError("Archive qualification labels differ from original records")
            available = {
                str(Path(r["original"]).resolve()).casefold(): r for r in manifest["files"]
            }
            for ref in refs:
                row = available.get(ref["original"].casefold())
                if row is None or (
                    ref["expected"] is not None and row["sha256"] != ref["expected"]
                ):
                    raise ValueError("Archive predecessor closure differs from original project")
                if ref["decoded"] is not None:
                    _verify_osc(staging / row["name"], ref["decoded"], control)
            (staging / "reopened.slate.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (staging / "manifest.json").write_text(encoded(manifest), encoding="utf-8")
            _stop(control)
            if _hash_file(source, control) != archive_sha:
                raise ValueError("Archive source changed during import")
            os.rename(staging, destination)
            created = False
    finally:
        if created:
            assert staging.resolve().parent == destination.parent and staging.name.startswith(
                "." + destination.name + "."
            )
            shutil.rmtree(staging)
    return ArchiveWorkResult(
        "import",
        review_json=encoded(manifest),
        path=destination / "reopened.slate.json",
        detail="Archive restored; original bytes/records preserved, explicit storage map saved. Open the reviewed project to inspect; no scientific operation ran.",
    )


def archive_work(argument, control):
    if len(argument) > 4 * 1024**2:
        raise ValueError("Archive request exceeds 4 MiB")
    request = json.loads(argument)
    _stop(control)
    operation = request["operation"]
    if operation == "review":
        value = _review(request, control)
    elif operation == "export":
        value = _export(request, control)
    elif operation == "import":
        value = _import(request, control)
    else:
        raise ValueError("Unsupported archive operation")
    return JobResult(value, value.nbytes)
