"""Lightweight hBN bundle geometry helpers.

This module intentionally avoids importing the full calibrant fitting stack so
simulation startup paths can reuse bundle geometry metadata cheaply.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import struct
import zipfile

import numpy as np
import yaml

# Simulator detector geometry is rotated 90 degrees clockwise from hBN-fitter
# native coordinates.
SIM_BACKGROUND_ROTATE_K = -1

_CANONICAL_TILT_CORRECTION_KIND = "to_flat"
_CANONICAL_TILT_MODEL = "RzRx"
_CANONICAL_TILT_FRAME = "simulation_background_display"
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X = 1
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y = 1
_HBN_CENTER_CONVERSION = "hbn_fitter_xy_to_row_col_mirror_y_by_image_width"

_BUNDLE_FORMAT_VERSION = 3
_BUNDLE_ARCHIVE_MAX_BYTES = 256 * 1024 * 1024
_BUNDLE_MEMBER_LIMIT = 96
_BUNDLE_MEMBER_MAX_BYTES = 128 * 1024 * 1024
_BUNDLE_TOTAL_MAX_BYTES = 256 * 1024 * 1024
_BUNDLE_DIRECTORY_MAX_BYTES = 1024 * 1024
_BUNDLE_HEADER_MAX_BYTES = 10_000
_BUNDLE_ARRAY_MAX_ELEMENTS = 25_000_000
_BUNDLE_IMAGE_MAX_DIMENSION = 8192
_BUNDLE_RING_LIMIT = 256
_BUNDLE_POINT_LIMIT = 1_000_000
_BUNDLE_JSON_MAX_BYTES = 1024 * 1024
_BUNDLE_JSON_MAX_DEPTH = 16
_BUNDLE_JSON_MAX_NODES = 100_000
_BUNDLE_JSON_MAX_STRING = 32 * 1024
_BUNDLE_MEMBER_NAME = re.compile(r"[A-Za-z0-9_]+\.npy\Z")
_BUNDLE_JSON_KEYS = frozenset(
    {"distance_estimate_m", "expected_peaks", "tilt_correction", "tilt_hint"}
)
_BUNDLE_POINT_PREFIXES = (
    "ell_points_ds",
    "ell_points_raw_ds",
    "ell_points_sigma_px",
    "ell_points_corrected",
)
_BUNDLE_POINT_KEYS = frozenset(
    f"{prefix}_{suffix}"
    for prefix in _BUNDLE_POINT_PREFIXES
    for suffix in ("offsets", "values")
)
_BUNDLE_STRING_KEYS = frozenset(
    {
        "center_source",
        "created_utc",
        "input_dark_path",
        "input_hbn_path",
        "optimizer_kind",
        "point_coord_frame",
        "point_sigma_coord_frame",
        "tilt_correction_kind",
        "tilt_frame",
        "tilt_model",
    }
)
_BUNDLE_CENTER_KEYS = frozenset(
    {"center", "center_initial", "center_prior"}
)
_BUNDLE_FLOAT_VECTOR_KEYS = frozenset(
    {
        "circ_after",
        "circ_before",
        "fit_angular_coverage",
        "fit_confidence_per_ring",
        "fit_residual_px",
        "fit_signal_snr",
        "radii_after",
        "radii_before",
        "ring_snap_sigma_px",
        "ring_weights",
    }
)
_BUNDLE_INT_VECTOR_KEYS = frozenset(
    {
        "ellipse_ring_indices",
        "fit_confidence_ring_indices",
        "fit_points_used",
        "optimizer_ring_ids",
    }
)
_BUNDLE_FLOAT_SCALAR_KEYS = frozenset(
    {
        "center_drift_limit_px",
        "center_prior_sigma_px",
        "cost_final",
        "cost_zero",
        "fit_click_sigma_px",
        "fit_confidence_overall",
        "fit_downsample_factor",
        "fit_downsample_score",
        "projective_distance_px",
        "tilt_x_deg_internal",
        "tilt_y_deg_internal",
    }
)
_BUNDLE_INT_SCALAR_KEYS = frozenset(
    {
        "downsample_factor",
        "npz_format_version",
        "sim_background_rotate_k",
        "simulation_Gamma_sign_from_tilt_y",
        "simulation_gamma_sign_from_tilt_x",
    }
)
_BUNDLE_V3_KEYS = frozenset(
    {
        "ellipse_params",
        "img_bgsub",
        "img_log",
        *_BUNDLE_CENTER_KEYS,
        *_BUNDLE_JSON_KEYS,
        *_BUNDLE_POINT_KEYS,
        *_BUNDLE_FLOAT_VECTOR_KEYS,
        *_BUNDLE_INT_VECTOR_KEYS,
        *_BUNDLE_FLOAT_SCALAR_KEYS,
        *_BUNDLE_INT_SCALAR_KEYS,
        *_BUNDLE_STRING_KEYS,
    }
)


def _dtype_matches(dtype, expected):
    expected = np.dtype(expected)
    return dtype.kind == expected.kind and dtype.itemsize == expected.itemsize


def _bundle_json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _validate_bundle_json_tree(value):
    stack = [(value, 1)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _BUNDLE_JSON_MAX_NODES:
            raise ValueError("Bundle JSON metadata contains too many values.")
        if depth > _BUNDLE_JSON_MAX_DEPTH:
            raise ValueError("Bundle JSON metadata is nested too deeply.")
        if isinstance(current, str):
            if len(current) > _BUNDLE_JSON_MAX_STRING:
                raise ValueError("Bundle JSON metadata contains an oversized string.")
        elif isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str) or len(key) > _BUNDLE_JSON_MAX_STRING:
                    raise ValueError("Bundle JSON metadata has an invalid key.")
                stack.append((child, depth + 1))
        elif isinstance(current, (list, tuple)):
            stack.extend((child, depth + 1) for child in current)
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise ValueError("Bundle JSON metadata contains an unsupported value.")


def _bundle_json_array(value):
    try:
        text = json.dumps(
            value,
            default=_bundle_json_default,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        _validate_bundle_json_tree(json.loads(text))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Bundle metadata is not valid bounded JSON.") from exc
    encoded = text.encode("utf-8")
    if len(encoded) > _BUNDLE_JSON_MAX_BYTES:
        raise ValueError("Bundle JSON metadata exceeds 1 MiB.")
    return np.frombuffer(encoded, dtype=np.uint8)


def _bundle_tilt_json_array(
    value,
    *,
    source_rotate_k,
):
    if value is None:
        return _bundle_json_array(None)
    metadata = dict(value)
    metadata.update(
        {
            "sim_background_rotate_k": int(source_rotate_k),
            "tilt_correction_kind": _CANONICAL_TILT_CORRECTION_KIND,
            "tilt_frame": _CANONICAL_TILT_FRAME,
            "tilt_model": _CANONICAL_TILT_MODEL,
            "simulation_gamma_sign_from_tilt_x": (
                _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X
            ),
            "simulation_Gamma_sign_from_tilt_y": (
                _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y
            ),
        }
    )
    return _bundle_json_array(metadata)


def _bundle_json_value(data, key):
    if key not in data:
        raise KeyError(f"Bundle is missing required key '{key}'.")
    encoded = np.asarray(data[key])
    if not _dtype_matches(encoded.dtype, np.uint8) or encoded.ndim != 1:
        raise ValueError(f"Bundle key '{key}' must contain UTF-8 JSON bytes.")
    if encoded.nbytes > _BUNDLE_JSON_MAX_BYTES:
        raise ValueError(f"Bundle key '{key}' exceeds the JSON size limit.")
    try:
        value = json.loads(encoded.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"Bundle key '{key}' contains invalid JSON.") from exc
    _validate_bundle_json_tree(value)
    return value


def _validate_json_number(value, *, field, integer=False, finite=True):
    valid_type = isinstance(value, int if integer else (int, float))
    if isinstance(value, bool) or not valid_type:
        kind = "integer" if integer else "numeric"
        raise ValueError(f"Bundle metadata field '{field}' must be {kind}.")
    if finite and not math.isfinite(float(value)):
        raise ValueError(f"Bundle metadata field '{field}' must be finite.")


def _validate_tilt_metadata(value, *, key):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"Bundle key '{key}' must be an object or null.")
    tilt_fields = (
        ("tilt_x_deg", "tilt_y_deg")
        if key == "tilt_correction"
        else ("rot1_rad", "rot2_rad")
    )
    required = (
        *tilt_fields,
        "sim_background_rotate_k",
        "tilt_correction_kind",
        "tilt_frame",
        "tilt_model",
        "simulation_gamma_sign_from_tilt_x",
        "simulation_Gamma_sign_from_tilt_y",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(
            f"Bundle key '{key}' is missing required fields: {', '.join(missing)}."
        )
    for field in (
        "cost_final",
        "cost_zero",
        "rot1_rad",
        "rot2_rad",
        "theta_deg",
        "theta_rad",
        "tilt_deg",
        "tilt_rad",
        "tilt_x_deg",
        "tilt_y_deg",
    ):
        if field in value:
            _validate_json_number(value[field], field=field)
    _validate_json_number(
        value["sim_background_rotate_k"],
        field="sim_background_rotate_k",
        integer=True,
    )
    for field in (
        "simulation_gamma_sign_from_tilt_x",
        "simulation_Gamma_sign_from_tilt_y",
    ):
        _validate_json_number(value[field], field=field, integer=True)
        if value[field] not in (-1, 1):
            raise ValueError(f"Bundle metadata field '{field}' must be -1 or 1.")
    if "source" in value and not isinstance(value["source"], str):
        raise ValueError("Bundle metadata field 'source' must be text.")
    for field in ("tilt_correction_kind", "tilt_frame", "tilt_model"):
        if not isinstance(value[field], str):
            raise ValueError(f"Bundle metadata field '{field}' must be text.")
    canonical_strings = {
        "tilt_correction_kind": _CANONICAL_TILT_CORRECTION_KIND,
        "tilt_frame": _CANONICAL_TILT_FRAME,
        "tilt_model": _CANONICAL_TILT_MODEL,
    }
    for field, expected in canonical_strings.items():
        if value[field] != expected:
            raise ValueError(f"Bundle metadata field '{field}' uses an unsupported convention.")
    for field in (
        "circ_after",
        "circ_before",
        "radii_after",
        "radii_after_fit",
        "radii_before",
    ):
        if field not in value:
            continue
        values = value[field]
        if not isinstance(values, list) or len(values) > _BUNDLE_RING_LIMIT:
            raise ValueError(f"Bundle metadata field '{field}' is invalid.")
        for item in values:
            _validate_json_number(item, field=field, finite=False)
    if "center" in value:
        center = value["center"]
        if not isinstance(center, list) or len(center) != 2:
            raise ValueError("Bundle tilt center metadata is invalid.")
        for item in center:
            _validate_json_number(item, field="center")
    if "corrected_points" in value:
        corrected_points = value["corrected_points"]
        if not isinstance(corrected_points, list) or len(corrected_points) > _BUNDLE_RING_LIMIT:
            raise ValueError("Bundle corrected-point metadata is invalid.")
        point_count = 0
        for ring in corrected_points:
            if not isinstance(ring, list):
                raise ValueError("Bundle corrected-point metadata is invalid.")
            point_count += len(ring)
            if point_count > _BUNDLE_POINT_LIMIT:
                raise ValueError("Bundle corrected-point metadata exceeds the point limit.")
            for point in ring:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Bundle corrected-point metadata is invalid.")
                for item in point:
                    _validate_json_number(item, field="corrected_points")


def _pack_bundle_rows(rows, *, width):
    rows = [] if rows is None else list(rows)
    if len(rows) > _BUNDLE_RING_LIMIT:
        raise ValueError(f"Bundle data exceeds {_BUNDLE_RING_LIMIT} rings.")

    packed = []
    offsets = [0]
    for row in rows:
        values = np.asarray(row, dtype=np.float32)
        if width == 1:
            values = values.reshape(-1)
        elif values.size == 0:
            values = np.empty((0, width), dtype=np.float32)
        elif values.ndim != 2 or values.shape[1] != width:
            raise ValueError(f"Bundle point rows must have shape (N, {width}).")
        packed.append(values)
        offsets.append(offsets[-1] + len(values))
        if offsets[-1] > _BUNDLE_POINT_LIMIT:
            raise ValueError(f"Bundle data exceeds {_BUNDLE_POINT_LIMIT} points.")

    shape = (0,) if width == 1 else (0, width)
    combined = np.concatenate(packed) if packed else np.empty(shape, dtype=np.float32)
    return combined, np.asarray(offsets, dtype=np.int32)


def _unpack_bundle_rows(data, prefix, *, width):
    values_key = f"{prefix}_values"
    offsets_key = f"{prefix}_offsets"
    if values_key not in data or offsets_key not in data:
        raise KeyError(f"Bundle is missing packed field '{prefix}'.")

    values = np.asarray(data[values_key])
    offsets = np.asarray(data[offsets_key])
    expected_shape = 1 if width == 1 else 2
    if not _dtype_matches(values.dtype, np.float32) or values.ndim != expected_shape:
        raise ValueError(f"Bundle key '{values_key}' has an invalid dtype or shape.")
    if width != 1 and values.shape[1] != width:
        raise ValueError(f"Bundle key '{values_key}' must have {width} columns.")
    if not _dtype_matches(offsets.dtype, np.int32) or offsets.ndim != 1:
        raise ValueError(f"Bundle key '{offsets_key}' must be a one-dimensional int32 array.")
    if not 1 <= offsets.size <= _BUNDLE_RING_LIMIT + 1:
        raise ValueError(f"Bundle key '{offsets_key}' has too many rings.")
    if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"Bundle key '{offsets_key}' contains invalid offsets.")
    if int(offsets[-1]) != len(values) or len(values) > _BUNDLE_POINT_LIMIT:
        raise ValueError(f"Bundle key '{prefix}' has inconsistent packed data.")
    return [values[start:stop] for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]


def _read_bundle_eocd(handle, file_size):
    if file_size < 22:
        raise ValueError("hBN bundle is not a valid NPZ archive.")
    tail_size = min(file_size, 22 + 65_535)
    handle.seek(file_size - tail_size)
    tail = handle.read(tail_size)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("hBN bundle is missing a valid ZIP directory.")
    (
        _signature,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, offset)
    eocd_offset = file_size - tail_size + offset
    if offset + 22 + comment_size != len(tail):
        raise ValueError("hBN bundle has trailing or malformed ZIP data.")
    if disk_number or directory_disk or disk_entries != total_entries:
        raise ValueError("Multi-disk hBN bundles are not supported.")
    if (
        total_entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        raise ValueError("ZIP64 hBN bundles are not supported.")
    if total_entries > _BUNDLE_MEMBER_LIMIT:
        raise ValueError(
            f"hBN bundle contains more than {_BUNDLE_MEMBER_LIMIT} members."
        )
    if directory_size > _BUNDLE_DIRECTORY_MAX_BYTES:
        raise ValueError("hBN bundle ZIP directory exceeds 1 MiB.")
    if directory_offset + directory_size != eocd_offset:
        raise ValueError("hBN bundle ZIP directory offsets are invalid.")
    return total_entries


def _inspect_bundle_member(archive, member_info):
    try:
        with archive.open(member_info) as member:
            magic = member.read(8)
            if len(magic) != 8 or magic[:6] != b"\x93NUMPY":
                raise ValueError("Missing NPY magic bytes.")
            version = (magic[6], magic[7])
            if version == (1, 0):
                length_bytes = member.read(2)
                length_format = "<H"
            elif version == (2, 0):
                length_bytes = member.read(4)
                length_format = "<I"
            else:
                raise ValueError("Unsupported NPY format version.")
            if len(length_bytes) != struct.calcsize(length_format):
                raise ValueError("Truncated NPY header length.")
            header_length = struct.unpack(length_format, length_bytes)[0]
            if header_length > _BUNDLE_HEADER_MAX_BYTES:
                raise ValueError("NPY header exceeds 10,000 bytes.")
            header_bytes = member.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError("Truncated NPY header.")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("hBN bundle contains an unreadable NPY member.") from exc

    header_reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    try:
        shape, fortran_order, dtype = header_reader(
            io.BytesIO(length_bytes + header_bytes),
        )
        if any(size < 0 for size in shape):
            raise ValueError
    except (EOFError, TypeError, ValueError) as exc:
        raise ValueError("hBN bundle contains an invalid NPY header.") from exc

    element_count = math.prod(shape)
    if element_count > _BUNDLE_ARRAY_MAX_ELEMENTS:
        raise ValueError("hBN bundle array exceeds the element limit.")
    header_prefix = magic + length_bytes + header_bytes
    payload_size = member_info.file_size - len(header_prefix)
    if payload_size < 0:
        raise ValueError("hBN bundle member has an invalid declared size.")
    if not dtype.hasobject and element_count * dtype.itemsize != payload_size:
        raise ValueError("hBN bundle member size does not match its declared array.")
    return {
        "dtype": dtype,
        "fortran_order": fortran_order,
        "header": header_prefix,
        "info": member_info,
        "key": member_info.filename[:-4],
        "payload_size": payload_size,
        "shape": shape,
    }


def _validate_bundle_descriptor(member):
    key = member["key"]
    dtype = member["dtype"]
    shape = member["shape"]
    size = math.prod(shape)
    if size > _BUNDLE_ARRAY_MAX_ELEMENTS:
        raise ValueError(f"Bundle key '{key}' exceeds the element limit.")
    if dtype.fields is not None or dtype.kind not in "biufUS":
        raise ValueError(f"Bundle key '{key}' has an unsupported dtype.")
    if key in ("img_bgsub", "img_log"):
        if (
            not _dtype_matches(dtype, np.float32)
            or len(shape) != 2
            or not all(shape)
            or max(shape) > _BUNDLE_IMAGE_MAX_DIMENSION
        ):
            raise ValueError(f"Bundle key '{key}' has an invalid image shape or dtype.")
    elif key == "ellipse_params":
        if not _dtype_matches(dtype, np.float32) or len(shape) != 2 or shape[1:] != (5,):
            raise ValueError("Bundle ellipse parameters must have float32 shape (N, 5).")
        if shape[0] > _BUNDLE_RING_LIMIT:
            raise ValueError("Bundle contains too many ellipses.")
    elif key in _BUNDLE_JSON_KEYS:
        if not _dtype_matches(dtype, np.uint8) or len(shape) != 1 or size > _BUNDLE_JSON_MAX_BYTES:
            raise ValueError(f"Bundle key '{key}' must contain bounded UTF-8 JSON bytes.")
    elif key.endswith("_offsets"):
        if not _dtype_matches(dtype, np.int32) or len(shape) != 1 or size > _BUNDLE_RING_LIMIT + 1:
            raise ValueError(f"Bundle key '{key}' has invalid packed offsets.")
    elif key.endswith("_values"):
        width = 1 if key == "ell_points_sigma_px_values" else 2
        expected_shape = 1 if width == 1 else 2
        if (
            not _dtype_matches(dtype, np.float32)
            or len(shape) != expected_shape
            or (width == 2 and shape[1:] != (2,))
            or shape[0] > _BUNDLE_POINT_LIMIT
        ):
            raise ValueError(f"Bundle key '{key}' has invalid packed values.")
    elif key in _BUNDLE_CENTER_KEYS:
        allowed_sizes = (0, 2) if key == "center" else (2,)
        if not _dtype_matches(dtype, np.float64) or len(shape) != 1 or size not in allowed_sizes:
            raise ValueError(f"Bundle key '{key}' has an invalid center vector.")
    elif key in _BUNDLE_FLOAT_VECTOR_KEYS:
        if not _dtype_matches(dtype, np.float64) or len(shape) != 1 or size > _BUNDLE_RING_LIMIT:
            raise ValueError(f"Bundle key '{key}' exceeds the ring limit.")
    elif key in _BUNDLE_INT_VECTOR_KEYS:
        if not _dtype_matches(dtype, np.int32) or len(shape) != 1 or size > _BUNDLE_RING_LIMIT:
            raise ValueError(f"Bundle key '{key}' must be a bounded int32 vector.")
    elif key in _BUNDLE_STRING_KEYS:
        if dtype.kind not in "US" or size != 1 or member["payload_size"] > 128 * 1024:
            raise ValueError(f"Bundle key '{key}' has invalid string metadata.")
    elif key in _BUNDLE_FLOAT_SCALAR_KEYS:
        if not _dtype_matches(dtype, np.float64) or size != 1:
            raise ValueError(f"Bundle key '{key}' must be a float64 scalar.")
    elif key in _BUNDLE_INT_SCALAR_KEYS:
        if not _dtype_matches(dtype, np.int32) or size != 1:
            raise ValueError(f"Bundle key '{key}' must be an int32 scalar.")


def _read_bundle_member(archive, member):
    try:
        with archive.open(member["info"]) as source:
            if source.read(len(member["header"])) != member["header"]:
                raise ValueError("NPY header changed during bundle loading.")
            payload = source.read(member["payload_size"] + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ValueError("Failed to read a validated hBN bundle member.") from exc
    if len(payload) != member["payload_size"]:
        raise ValueError("hBN bundle member payload size changed during loading.")
    array = np.frombuffer(payload, dtype=member["dtype"]).copy()
    return array.reshape(member["shape"], order="F" if member["fortran_order"] else "C")


def _validate_bundle_arrays(data):
    required = {
        "center",
        "distance_estimate_m",
        "downsample_factor",
        "ell_points_ds_offsets",
        "ell_points_ds_values",
        "ellipse_params",
        "expected_peaks",
        "img_bgsub",
        "img_log",
        "point_coord_frame",
        "sim_background_rotate_k",
        "simulation_Gamma_sign_from_tilt_y",
        "simulation_gamma_sign_from_tilt_x",
        "tilt_correction",
        "tilt_correction_kind",
        "tilt_frame",
        "tilt_hint",
        "tilt_model",
    }
    missing = sorted(required - data.keys())
    if missing:
        raise KeyError(f"Bundle is missing required keys: {', '.join(missing)}")
    version = int(np.asarray(data["npz_format_version"]).reshape(-1)[0])
    if version != _BUNDLE_FORMAT_VERSION:
        raise ValueError("Only current format-3 hBN bundles are supported.")

    if data["img_bgsub"].shape != data["img_log"].shape:
        raise ValueError("Bundle images must have matching shapes.")
    center_values = np.asarray(data["center"])
    if center_values.size and not np.all(np.isfinite(center_values)):
        raise ValueError("Bundle key 'center' must contain two finite values.")
    downsample_factor = int(np.asarray(data["downsample_factor"]).reshape(-1)[0])
    if downsample_factor < 1:
        raise ValueError("Bundle downsample_factor must be a positive integer.")
    point_coord_frame = str(np.asarray(data["point_coord_frame"]).reshape(-1)[0])
    if point_coord_frame not in ("downsampled", "full"):
        raise ValueError("Bundle point_coord_frame is unsupported.")
    if "point_sigma_coord_frame" in data:
        sigma_frame = str(np.asarray(data["point_sigma_coord_frame"]).reshape(-1)[0])
        if sigma_frame not in ("downsampled", "full"):
            raise ValueError("Bundle point_sigma_coord_frame is unsupported.")
    has_sigma_points = "ell_points_sigma_px_values" in data
    if has_sigma_points != ("point_sigma_coord_frame" in data):
        raise ValueError(
            "Bundle sigma points and their coordinate frame must be provided together."
        )
    source_rotate_k = int(
        np.asarray(data["sim_background_rotate_k"]).reshape(-1)[0]
    )
    for key in (
        "simulation_gamma_sign_from_tilt_x",
        "simulation_Gamma_sign_from_tilt_y",
    ):
        sign = int(np.asarray(data[key]).reshape(-1)[0])
        if sign not in (-1, 1):
            raise ValueError(f"Bundle key '{key}' must be -1 or 1.")
    canonical_strings = {
        "tilt_correction_kind": _CANONICAL_TILT_CORRECTION_KIND,
        "tilt_frame": _CANONICAL_TILT_FRAME,
        "tilt_model": _CANONICAL_TILT_MODEL,
    }
    for key, expected in canonical_strings.items():
        if str(np.asarray(data[key]).reshape(-1)[0]) != expected:
            raise ValueError(f"Bundle key '{key}' uses an unsupported convention.")
    for prefix in _BUNDLE_POINT_PREFIXES:
        values_key = f"{prefix}_values"
        offsets_key = f"{prefix}_offsets"
        if values_key in data or offsets_key in data:
            if values_key not in data or offsets_key not in data:
                raise KeyError(f"Bundle is missing part of packed field '{prefix}'.")
            _unpack_bundle_rows(
                data,
                prefix,
                width=1 if prefix == "ell_points_sigma_px" else 2,
            )

    distance_info = _bundle_json_value(data, "distance_estimate_m")
    if distance_info is not None:
        if not isinstance(distance_info, dict):
            raise ValueError("Bundle distance metadata must be an object or null.")
        for field in ("mean_m", "pixel_size_m"):
            if field in distance_info:
                _validate_json_number(distance_info[field], field=field)
                if distance_info[field] <= 0:
                    raise ValueError(f"Bundle metadata field '{field}' must be positive.")
        if "basis" in distance_info and not isinstance(distance_info["basis"], str):
            raise ValueError("Bundle metadata field 'basis' must be text.")
        if "per_ring_m" in distance_info:
            per_ring_m = distance_info["per_ring_m"]
            if not isinstance(per_ring_m, list) or len(per_ring_m) > _BUNDLE_RING_LIMIT:
                raise ValueError("Bundle distance per_ring_m metadata is invalid.")
            for value in per_ring_m:
                _validate_json_number(value, field="per_ring_m")
                if value <= 0:
                    raise ValueError("Bundle distance per_ring_m values must be positive.")
    for key in ("tilt_correction", "tilt_hint"):
        metadata = _bundle_json_value(data, key)
        _validate_tilt_metadata(metadata, key=key)
        if metadata is None:
            continue
        if int(metadata["sim_background_rotate_k"]) != int(source_rotate_k):
            raise ValueError(
                f"Bundle key '{key}' has inconsistent rotation metadata."
            )
        for sign_key in (
            "simulation_gamma_sign_from_tilt_x",
            "simulation_Gamma_sign_from_tilt_y",
        ):
            top_level_sign = int(np.asarray(data[sign_key]).reshape(-1)[0])
            if int(metadata[sign_key]) != top_level_sign:
                raise ValueError(
                    f"Bundle key '{key}' has inconsistent sign metadata."
                )
    expected_peaks = _bundle_json_value(data, "expected_peaks")
    if expected_peaks is not None:
        if (
            not isinstance(expected_peaks, list)
            or len(expected_peaks) > _BUNDLE_RING_LIMIT
            or any(not isinstance(peak, dict) for peak in expected_peaks)
        ):
            raise ValueError("Bundle expected peaks metadata is invalid.")
        for peak in expected_peaks:
            if set(("hkl", "d_spacing_ang", "two_theta_deg")) - peak.keys():
                raise ValueError("Bundle expected peak metadata is incomplete.")
            hkl = peak["hkl"]
            if not isinstance(hkl, list) or len(hkl) != 3:
                raise ValueError("Bundle expected peak hkl metadata is invalid.")
            for index in hkl:
                _validate_json_number(index, field="hkl", integer=True)
            _validate_json_number(peak["d_spacing_ang"], field="d_spacing_ang")
            _validate_json_number(peak["two_theta_deg"], field="two_theta_deg")
            if peak["d_spacing_ang"] <= 0 or not 0 < peak["two_theta_deg"] < 180:
                raise ValueError("Bundle expected peak values are outside physical bounds.")


def _validate_bundle_payload(data):
    unknown = set(data) - _BUNDLE_V3_KEYS
    if unknown:
        raise ValueError(f"Bundle payload has unknown keys: {', '.join(sorted(unknown))}")
    total_bytes = 0
    for key, value in data.items():
        array = np.asarray(value)
        total_bytes += array.nbytes
        if array.nbytes > _BUNDLE_MEMBER_MAX_BYTES:
            raise ValueError(f"Bundle payload key '{key}' exceeds 128 MiB.")
        member = {
            "dtype": array.dtype,
            "key": key,
            "payload_size": array.nbytes,
            "shape": array.shape,
        }
        _validate_bundle_descriptor(member)
    if total_bytes > _BUNDLE_TOTAL_MAX_BYTES - _BUNDLE_DIRECTORY_MAX_BYTES:
        raise ValueError("Bundle payload exceeds the expanded-size limit.")
    _validate_bundle_arrays(data)


def _load_safe_bundle_arrays(path):
    with open(path, "rb") as handle:
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("hBN bundle must be a regular file.")
        if file_stat.st_size > _BUNDLE_ARCHIVE_MAX_BYTES:
            raise ValueError("hBN bundle exceeds the 256 MiB archive limit.")
        expected_entries = _read_bundle_eocd(handle, file_stat.st_size)
        handle.seek(0)
        try:
            archive = zipfile.ZipFile(handle)
        except zipfile.BadZipFile as exc:
            raise ValueError("hBN bundle is not a valid NPZ archive.") from exc
        with archive:
            infos = archive.infolist()
            if len(infos) != expected_entries:
                raise ValueError("hBN bundle ZIP member count is inconsistent.")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ValueError("hBN bundle contains duplicate members.")
            total_size = 0
            members = {}
            for info in infos:
                if not _BUNDLE_MEMBER_NAME.fullmatch(info.filename):
                    raise ValueError("hBN bundle contains an invalid member name.")
                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted hBN bundle members are not supported.")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("hBN bundle uses an unsupported compression method.")
                if info.file_size > _BUNDLE_MEMBER_MAX_BYTES:
                    raise ValueError("hBN bundle member exceeds the 128 MiB limit.")
                total_size += info.file_size
                if total_size > _BUNDLE_TOTAL_MAX_BYTES:
                    raise ValueError("hBN bundle exceeds the 256 MiB expanded-size limit.")
                member = _inspect_bundle_member(archive, info)
                members[member["key"]] = member

            version_member = members.get("npz_format_version")
            if version_member is None:
                raise ValueError("Only current format-3 hBN bundles are supported.")
            if any(member["dtype"].hasobject for member in members.values()):
                raise ValueError("Current hBN bundles cannot contain pickle-backed data.")
            unknown = set(members) - _BUNDLE_V3_KEYS
            if unknown:
                raise ValueError(
                    "hBN bundle contains unsupported keys: "
                    + ", ".join(sorted(unknown))
                )
            _validate_bundle_descriptor(version_member)
            version_array = _read_bundle_member(archive, version_member)
            if int(version_array.reshape(-1)[0]) != _BUNDLE_FORMAT_VERSION:
                raise ValueError("Only current format-3 hBN bundles are supported.")

            data = {"npz_format_version": version_array}
            for key, member in members.items():
                if key == "npz_format_version":
                    continue
                _validate_bundle_descriptor(member)
                data[key] = _read_bundle_member(archive, member)

    _validate_bundle_arrays(data)
    return data


def _image_shape_hw(image_size):
    if isinstance(image_size, (tuple, list, np.ndarray)):
        if len(image_size) < 2:
            raise ValueError("image_size must provide at least (height, width).")
        height = int(image_size[0])
        width = int(image_size[1])
    else:
        height = int(image_size)
        width = int(image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive.")
    return height, width


def _map_hbn_center_pair(first, second, image_size):
    """Apply the self-inverse hBN/simulation center mapping to one coordinate pair."""

    _, image_width = _image_shape_hw(image_size)
    first_value = float(first)
    second_value = float(second)
    if not (np.isfinite(first_value) and np.isfinite(second_value)):
        raise ValueError("hBN center coordinates must be finite.")
    return first_value, float(image_width - second_value)


def _normalize_sign(value):
    """Return current-format +/-1 sign metadata or reject it."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Sign metadata must be -1 or 1.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sign metadata must be -1 or 1.") from exc
    if not np.isfinite(numeric) or numeric not in (-1.0, 1.0):
        raise ValueError("Sign metadata must be -1 or 1.")
    return int(numeric)


def _rotate_tilt_components(gamma_src, Gamma_src, rotation_deg):
    """Rotate one gamma/Gamma component pair by the requested detector angle."""

    rotation_rad = float(np.deg2rad(rotation_deg))
    cosine = float(np.cos(rotation_rad))
    sine = float(np.sin(rotation_rad))
    return (
        float(cosine * gamma_src + sine * Gamma_src),
        float(-sine * gamma_src + cosine * Gamma_src),
    )


def convert_hbn_bundle_geometry_to_simulation(
    *,
    tilt_x_deg,
    tilt_y_deg,
    center_xy,
    source_rotate_k,
    target_rotate_k,
    image_size,
    simulation_gamma_sign_from_tilt_x=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
    simulation_Gamma_sign_from_tilt_y=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
):
    """Convert hBN correction geometry into simulation detector geometry."""

    tx = float(tilt_x_deg)
    ty = float(tilt_y_deg)

    gamma_sign = _normalize_sign(simulation_gamma_sign_from_tilt_x)
    Gamma_sign = _normalize_sign(simulation_Gamma_sign_from_tilt_y)

    gamma_src = gamma_sign * tx
    Gamma_src = Gamma_sign * ty

    source_k = int(source_rotate_k)
    target_k = int(target_rotate_k)
    k_delta = target_k - source_k
    alpha_deg = 90.0 * k_delta
    gamma_deg, Gamma_deg = _rotate_tilt_components(
        gamma_src,
        Gamma_src,
        alpha_deg,
    )

    center_row = None
    center_col = None
    if center_xy is not None:
        center_row, center_col = _map_hbn_center_pair(
            center_xy[0],
            center_xy[1],
            image_size,
        )

    return {
        "gamma_deg": gamma_deg,
        "Gamma_deg": Gamma_deg,
        "center_row": center_row,
        "center_col": center_col,
        "k_delta": k_delta,
        "conversion_notes": {
            "tilt_correction_kind": _CANONICAL_TILT_CORRECTION_KIND,
            "tilt_model": _CANONICAL_TILT_MODEL,
            "source_rotate_k": source_k,
            "target_rotate_k": target_k,
            "frame_rotation_deg": alpha_deg,
            "component_rotation_applied": True,
            "center_conversion": _HBN_CENTER_CONVERSION,
            "center_row_formula": "center_row = hbn_center_x",
            "center_col_formula": "center_col = image_width - hbn_center_y",
            "simulation_gamma_sign_from_tilt_x": gamma_sign,
            "simulation_Gamma_sign_from_tilt_y": Gamma_sign,
        },
    }






def _load_paths_from_file(paths_file):
    with open(paths_file, encoding="utf-8") as fh:
        text = fh.read()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = json.loads(text)
    return data or {}


def _resolve_file_derived_path_value(value, *, base_dir):
    """Resolve file-derived config values without affecting direct CLI args."""

    if value is None:
        return None

    expanded = os.path.expanduser(value)
    if not expanded:
        return expanded
    if PureWindowsPath(expanded).is_absolute():
        return expanded
    if expanded.startswith(("/", "\\")):
        return expanded

    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())


def resolve_hbn_paths(
    osc_path=None,
    dark_path=None,
    paths_file=None,
):
    """Return resolved paths using CLI args or a YAML/JSON file."""

    resolved = dict(
        osc=os.path.expanduser(osc_path) if osc_path else None,
        dark=os.path.expanduser(dark_path) if dark_path else None,
        click_profile=None,
        beam_center=None,
    )

    search_file = paths_file
    file_value_base_dir = None
    if search_file is None:
        from ra_sim.config import loader as ra_sim_config

        config_dir = ra_sim_config.get_config_dir()
        file_value_base_dir = config_dir.resolve()
        candidate_path = config_dir / "hbn_paths.yaml"
        if candidate_path.exists():
            search_file = str(candidate_path)
    elif search_file:
        file_value_base_dir = Path(search_file).resolve().parent

    if search_file:
        file_data = _load_paths_from_file(search_file)
        for field, key in (
            ("osc", "calibrant"),
            ("dark", "dark"),
            ("click_profile", "click_profile"),
        ):
            if resolved[field] is None:
                resolved[field] = _resolve_file_derived_path_value(
                    file_data.get(key),
                    base_dir=file_value_base_dir,
                )
        beam_x = beam_y = None
        beam_center_from_list = file_data.get("beam_center")
        if isinstance(beam_center_from_list, (list, tuple)) and len(beam_center_from_list) == 2:
            beam_x, beam_y = beam_center_from_list
        if beam_x is not None and beam_y is not None:
            try:
                resolved["beam_center"] = (float(beam_x), float(beam_y))
            except (TypeError, ValueError):
                resolved["beam_center"] = None
    resolved["paths_file"] = search_file if search_file else None
    return resolved


def load_bundle_npz(path, *, verbose=True):
    data = _load_safe_bundle_arrays(path)

    img_bgsub = np.asarray(data["img_bgsub"])
    img_log = np.asarray(data["img_log"])
    ellipse_params = np.asarray(data["ellipse_params"])

    ell_points_ds = [
        [(float(x), float(y)) for x, y in row]
        for row in _unpack_bundle_rows(data, "ell_points_ds", width=2)
    ]
    distance_info = _bundle_json_value(data, "distance_estimate_m")
    tilt_correction = _bundle_json_value(data, "tilt_correction")
    tilt_hint = _bundle_json_value(data, "tilt_hint")
    expected_peaks = _bundle_json_value(data, "expected_peaks")

    ellipses = []
    for row in ellipse_params:
        xc, yc, a, b, theta = [float(v) for v in row]
        ellipses.append(
            dict(
                xc=xc,
                yc=yc,
                a=a,
                b=b,
                theta=theta,
            )
        )

    center_values = np.asarray(data["center"])
    center = (
        tuple(float(value) for value in center_values)
        if center_values.size
        else None
    )

    if verbose:
        print(f"Loaded bundle from:\n  {path}")
        print(f"  image shape: {img_bgsub.shape}")
        print(f"  number of ellipses: {len(ellipses)}")
    if verbose and distance_info:
        print(
            "  distance estimate: "
            f"mean={distance_info.get('mean_m', float('nan')):.4f} m"
        )
    if verbose and center is not None:
        try:
            cx, cy = center
            print(f"  detector center: ({float(cx):.3f}, {float(cy):.3f}) px")
        except Exception:
            pass
    return (
        img_bgsub,
        img_log,
        ell_points_ds,
        ellipses,
        distance_info,
        tilt_correction,
        tilt_hint,
        expected_peaks,
        center,
    )
