"""Read-only acquisition identity review on the desktop's shared worker."""

import json
from dataclasses import dataclass
from pathlib import Path

from hbn_io import _hash_file
from job_lifecycle import JobResult
from osc_import import AXIS_LIMIT, DECODED_LIMIT_BYTES, PIXEL_LIMIT, SOURCE_LIMIT_BYTES
from preparation_state import RECIPES
from sample_state import encoded, payload_hash
from simulation_io import _stop


@dataclass(frozen=True, slots=True)
class PreparationReview:
    review_json: str

    @property
    def nbytes(self):
        return len(self.review_json.encode()) + 1024


def preparation_work(argument, control):
    from rasim_next.io.osc import OscReadLimits, read_osc

    if type(argument) is not bytes or len(argument) > 128 * 1024:
        raise ValueError("Acquisition review request exceeds 128 KiB")
    inputs = json.loads(argument)
    recipe = next((v for v in RECIPES if v[0] == inputs.get("recipe")), None)
    if recipe is None:
        raise ValueError("Unsupported preparation review recipe")
    files = []
    for acquisition in [inputs["acquisition"], *inputs["linked_acquisitions"]]:
        path = Path(acquisition["source_path"])
        control.report("Verifying acquisition identity: " + acquisition["name"])
        _stop(control)
        record = dict(
            kind="osc", acquisition_id=acquisition["id"], path=str(path), state="unverified"
        )
        try:
            before = path.stat()
            if before.st_size > SOURCE_LIMIT_BYTES:
                raise ValueError("OSC source exceeds 64 MiB")
            raw_hash = _hash_file(path, control)
            image = read_osc(
                path,
                limits=OscReadLimits(
                    SOURCE_LIMIT_BYTES, DECODED_LIMIT_BYTES, PIXEL_LIMIT, AXIS_LIMIT
                ),
                canceled=lambda: control.canceled,
            )
            if image.decoded_sha256 != acquisition["source_sha256"]:
                raise ValueError("Decoded OSC identity differs from imported acquisition")
            shape = acquisition["metadata"].get("native_shape")
            if shape is not None and tuple(shape) != image.detector_native_counts.shape:
                raise ValueError("Declared native shape differs from decoded OSC")
            if _hash_file(path, control) != raw_hash or path.stat() != before:
                raise ValueError("OSC changed during review")
            record.update(
                state="verified",
                sha256=raw_hash,
                decoded_sha256=image.decoded_sha256,
                size=before.st_size,
                mtime_ns=before.st_mtime_ns,
                native_shape=list(image.detector_native_counts.shape),
            )
            del image
        except (ValueError, OSError) as exc:
            _stop(control)
            record.update(state="unavailable", reason=str(exc))
        files.append(record)
        metadata = acquisition["metadata"]
        for kind in ("configuration", "configuration_cif", "cif"):
            name = metadata.get(kind + "_path")
            if name is None:
                continue
            _stop(control)
            row = dict(kind=kind, path=name, acquisition_id=acquisition["id"], state="unverified")
            try:
                path = Path(name)
                before = path.stat()
                if before.st_size > 1024 * 1024:
                    raise ValueError("Review reference exceeds 1 MiB")
                digest = _hash_file(path, control)
                if digest != metadata.get(kind + "_sha256") or path.stat() != before:
                    raise ValueError("Reference identity missing, changed or unstable")
                row.update(
                    state="verified",
                    sha256=digest,
                    size=before.st_size,
                    mtime_ns=before.st_mtime_ns,
                )
            except (ValueError, OSError) as exc:
                _stop(control)
                row.update(state="unavailable", reason=str(exc))
            files.append(row)
    _stop(control)
    receipt = dict(
        inputs=inputs,
        files=files,
        support=recipe[2],
        qualification="Read-only source/decoded identity review. No numerical observation preparation, background truth or fit qualification established.",
        coordinate_convention="OSC clockwise conversion once at I/O; arrays [row,column]; continuous (column_px,row_px); radians, metres, angstroms.",
        output_inventory=[
            "Existing frozen observations and full covariance: NativeFitSession typed inspection",
            "Draft stages: existing prepared-plan editor; fitting Run/indexed adoption unavailable",
            "New numerical preparation: unavailable; no output files published",
        ],
    )
    receipt["sha256"] = payload_hash(receipt)
    raw = encoded(receipt)
    if len(raw.encode()) > 32 * 1024:
        raise ValueError("Acquisition review exceeds 32 KiB; current review retained")
    value = PreparationReview(raw)
    return JobResult(value, value.nbytes)
