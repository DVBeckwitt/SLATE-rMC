"""Bounded acquisition review choices; frozen observations stay with NativeFitSession."""

import json
from pathlib import Path
from uuid import UUID

from sample_state import encoded, payload_hash

RECIPES = (
    (
        "raw_native",
        "New raw acquisition",
        "No complete admitted raw-to-observation recipe: fixed signal/control memberships, calibrated background transfer and full shared covariance must be declared and independently qualified.",
    ),
    (
        "frozen_adoption",
        "Verify existing frozen observations",
        "prepare_native.prepare verifies and relocates already frozen observations. Its numerical verification/publication has no independently admitted cooperative desktop boundary; Prepare is unavailable.",
    ),
    (
        "catalog_adoption",
        "Existing experiment catalog",
        "prepare_native.prepare_catalog selects a hash-bound frozen experiment; it is not a new-acquisition recipe. The same preparation execution boundary is unavailable.",
    ),
    (
        "native_projection",
        "Fixed native projection primitive",
        "NativePixelRegionProjection requires an already declared membership and value/variance fields. It supplies no calibrated background-transfer recipe or complete acquisition preparation.",
    ),
)


def read_preparation(text):
    if type(text) is not str or len(text.encode()) > 128 * 1024:
        raise ValueError("Acquisition review state exceeds 128 KiB")
    value = json.loads(text)
    if value == {}:
        return dict(
            schema="slate.preparation-review.v1",
            acquisition_id=None,
            recipe="raw_native",
            notes="",
            selected_description=None,
            reviews=[],
        )
    if type(value) is not dict or set(value) != {
        "schema",
        "acquisition_id",
        "recipe",
        "notes",
        "selected_description",
        "reviews",
    }:
        raise ValueError("Malformed acquisition review state")
    if value["schema"] != "slate.preparation-review.v1" or value["recipe"] not in {
        v[0] for v in RECIPES
    }:
        raise ValueError("Unsupported acquisition review recipe")
    if value["acquisition_id"] is not None:
        if type(value["acquisition_id"]) is not str:
            raise ValueError("Invalid acquisition review UUID")
        UUID(value["acquisition_id"])
    if type(value["notes"]) is not str or len(value["notes"].encode()) > 8192:
        raise ValueError("Pending review notes exceed 8 KiB")
    digest = value["selected_description"]
    if digest is not None and (
        type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Invalid selected frozen description identity")
    reviews = value["reviews"]
    if type(reviews) is not list or len(reviews) > 8:
        raise ValueError(
            "Retain at most eight acquisition reviews; remove an old review explicitly"
        )
    for row in reviews:
        if type(row) is not dict or set(row) != {
            "sha256",
            "inputs",
            "files",
            "support",
            "qualification",
            "coordinate_convention",
            "output_inventory",
        }:
            raise ValueError("Malformed acquisition review receipt")
        if row["sha256"] != payload_hash({k: v for k, v in row.items() if k != "sha256"}):
            raise ValueError("Acquisition review receipt identity differs")
        if (
            type(row["files"]) is not list
            or len(row["files"]) > 12
            or len(encoded(row).encode()) > 32 * 1024
        ):
            raise ValueError("Acquisition review receipt exceeds its bound")
        if type(row["inputs"]) is not dict or type(row["output_inventory"]) is not list:
            raise ValueError("Invalid review inputs or output inventory")
        for file in row["files"]:
            if (
                type(file) is not dict
                or file.get("state") not in ("verified", "unavailable")
                or file.get("kind") not in ("osc", "configuration", "configuration_cif", "cif")
            ):
                raise ValueError("Invalid review file record")
            if (
                type(file.get("path")) is not str
                or not 0 < len(file["path"]) <= 4096
                or not Path(file["path"]).is_absolute()
            ):
                raise ValueError("Review file path must be bounded and absolute")
            if type(file.get("acquisition_id")) is not str:
                raise ValueError("Invalid review file acquisition identity")
            UUID(file["acquisition_id"])
            if file["state"] == "verified":
                digest = file.get("sha256")
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("Review file requires SHA256")
                if any(type(file.get(k)) is not int or file[k] < 0 for k in ("size", "mtime_ns")):
                    raise ValueError("Invalid review file size/time")
            elif type(file.get("reason")) is not str:
                raise ValueError("Unavailable review file requires a reason")
    return value


def review_inputs(document, choices, anchor):
    """Exact pending selection and delivered owners, independent of view presentation."""
    selected = next(
        (a for a in document["project"]["acquisitions"] if a["id"] == choices["acquisition_id"]),
        None,
    )
    if selected is None:
        raise ValueError("Select a current imported acquisition")
    linked = {selected["metadata"].get(k) for k in ("dark_acquisition_id", "mask_acquisition_id")}

    def resolved(acquisition):
        row = json.loads(encoded(acquisition))
        for parent, key in [
            (row, "source_path"),
            *(
                (row["metadata"], k + "_path")
                for k in ("configuration", "configuration_cif", "cif")
            ),
        ]:
            if parent.get(key) is not None:
                path = Path(parent[key])
                parent[key] = str(
                    (path if path.is_absolute() else Path(anchor).parent / path).resolve()
                )
        return row

    selected = resolved(selected)
    view = document["view"]
    geometry = {
        k: view.get(k)
        for k in ("hbn_sessions", "sample_session", "joint_session", "physical_settings_json")
    }
    return dict(
        acquisition=selected,
        linked_acquisitions=[
            resolved(a) for a in document["project"]["acquisitions"] if a["id"] in linked
        ],
        recipe=choices["recipe"],
        notes=choices["notes"],
        geometry={k: payload_hash(v) for k, v in geometry.items()},
        numeric_draft=payload_hash(document.get("numeric_draft")),
    )
