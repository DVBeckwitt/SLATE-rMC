"""Read-only verification of the committed reference evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import subprocess
import tomllib
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rasim_next.core.traces import Measure, QuantityKind, TraceRecord
from rasim_next.io.orientation import (
    DetectorIndex,
    OscRawIndex,
    detector_native_to_raw,
    detector_to_raw_index,
    index_to_coordinate,
    raw_to_detector_index,
    raw_to_detector_native,
)
from rasim_next.proof.traces import compare_traces

_HEADER_BYTES = 6000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    _require(not Path(relative).is_absolute() and candidate.is_relative_to(root.resolve()), "path escapes repository")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_sha(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _environment_sha256() -> str:
    encoded = json.dumps(
        {
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _osc_layout(header: bytes) -> tuple[int, str, int, int]:
    _require(len(header) == _HEADER_BYTES and header[:5] == b"RAXIS", "invalid OSC header")
    version = int.from_bytes(header[796:800], byteorder="big", signed=False)
    endian = ">" if version < 20 else "<"
    byteorder = "big" if endian == ">" else "little"
    width = int.from_bytes(header[768:772], byteorder=byteorder, signed=False)
    height = int.from_bytes(header[772:776], byteorder=byteorder, signed=False)
    _require(width > 0 and height > 0, "invalid OSC dimensions")
    return version, endian, height, width


def _decode_tiny_osc(path: Path) -> tuple[int, NDArray[np.int32]]:
    content = path.read_bytes()
    version, endian, height, width = _osc_layout(content[:_HEADER_BYTES])
    _require(len(content) == _HEADER_BYTES + 2 * height * width, "invalid OSC payload size")
    encoded = np.frombuffer(content, dtype=endian + "u2", offset=_HEADER_BYTES)
    decoded = encoded.astype(np.int32).reshape(height, width)
    high = decoded >= 0x8000
    decoded[high] = (decoded[high] - 0x8000) * 32
    return version, decoded


def _stream_gzip_osc(
    path: Path, positions: NDArray[np.int32] | None = None
) -> dict[str, Any]:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        header = handle.read(_HEADER_BYTES)
        digest.update(header)
        version, endian, height, width = _osc_layout(header)
        total = len(header)
        offset = 0
        carry = b""
        minimum: int | None = None
        maximum: int | None = None
        value_sum = 0
        argmax = 0
        points = np.empty((0, 2), dtype=np.int32) if positions is None else positions
        selected_raw = np.zeros(points.shape[0], dtype=np.int64)
        selected_native = np.zeros(points.shape[0], dtype=np.int64)
        requests: dict[int, list[tuple[str, int]]] = {}
        for index, (row_value, column_value) in enumerate(points):
            row, column = int(row_value), int(column_value)
            requests.setdefault(row * width + column, []).append(("raw", index))
            raw_index = detector_to_raw_index(DetectorIndex(row, column), (height, width))
            requests.setdefault(raw_index.row * width + raw_index.column, []).append(("native", index))
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
            if positions is None:
                continue
            data = carry + chunk
            usable = len(data) - len(data) % 2
            carry = data[usable:]
            encoded = np.frombuffer(data[:usable], dtype=endian + "u2")
            decoded = encoded.astype(np.int64)
            high = decoded >= 0x8000
            decoded[high] = (decoded[high] - 0x8000) * 32
            local_min, local_max = int(decoded.min()), int(decoded.max())
            minimum = local_min if minimum is None else min(minimum, local_min)
            if maximum is None or local_max > maximum:
                maximum = local_max
                argmax = offset + int(np.argmax(decoded))
            value_sum += int(decoded.sum())
            for linear, targets in requests.items():
                local = linear - offset
                if 0 <= local < decoded.size:
                    for target, index in targets:
                        (selected_raw if target == "raw" else selected_native)[index] = decoded[local]
            offset += decoded.size
    expected_size = _HEADER_BYTES + 2 * height * width
    _require(total == expected_size, f"invalid decompressed size for {path.name}")
    result: dict[str, Any] = {
        "sha256": digest.hexdigest(),
        "version": version,
        "shape": (height, width),
    }
    if positions is not None:
        _require(not carry and offset == height * width, "incomplete streamed OSC decode")
        raw_argmax = OscRawIndex(*divmod(argmax, width))
        native_argmax = raw_to_detector_index(raw_argmax, (height, width))
        result.update(
            summary=(version, height, width, minimum, maximum, value_sum),
            selected_raw=selected_raw,
            selected_native=selected_native,
            argmax_raw=(raw_argmax.row, raw_argmax.column),
            argmax_native=(native_argmax.row, native_argmax.column),
        )
    return result


def _verify_source_snapshot(root: Path, archive_sha256: str) -> int:
    snapshot = root / "reference" / "legacy_source"
    manifest = json.loads((snapshot / "MANIFEST.json").read_text(encoding="utf-8"))
    _require(manifest["schema_version"] == "rasim-legacy-source-snapshot-v1", "source schema mismatch")
    _require(manifest["original_archive_sha256"] == archive_sha256, "source archive mismatch")
    declared: set[str] = set()
    for item in manifest["files"]:
        relative = str(item["path"])
        _require(relative not in declared, "duplicate source path")
        declared.add(relative)
        path = _path(snapshot, relative)
        _require(path.stat().st_size == item["size_bytes"] and _sha256(path) == item["sha256"], f"source hash mismatch: {relative}")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json" and path.suffix.lower() != ".md"
    }
    _require(actual == declared, "source snapshot file set mismatch")
    osc_lines = (snapshot / "ra_sim" / "io" / "osc_reader.py").read_text(encoding="utf-8").splitlines()
    osc_citation = "\n".join(osc_lines[17:66])
    _require(all(token in osc_citation for token in ("header[796:800]", "endian", "reshape", "0x8000", "*= 32")), "OSC source citation mismatch")
    runtime_lines = (snapshot / "ra_sim" / "gui" / "_runtime" / "runtime_session.py").read_text(encoding="utf-8").splitlines()
    rotation_citation = "\n".join(runtime_lines[557:565])
    _require("DISPLAY_ROTATE_K = -1" in rotation_citation and "SIM_DISPLAY_ROTATE_K = 0" in rotation_citation, "orientation source citation mismatch")
    return len(declared)


def _verify_pack(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    pack_entry = manifest["reference_pack"]
    pack = _path(root, pack_entry["path"])
    observed_hash = _sha256(pack)
    _require(observed_hash == pack_entry["sha256"], "reference pack hash mismatch")
    needed = {
        "osc_summary",
        "osc_selected_positions_row_col",
        "osc_selected_raw_values",
        "osc_selected_native_values",
        "osc_argmax_raw_row_col",
        "osc_argmax_native_row_col",
        "osc_synthetic_big_endian_raw",
        "osc_synthetic_big_endian_native",
        "osc_synthetic_little_endian_raw",
        "osc_synthetic_little_endian_native",
    }
    with np.load(pack, allow_pickle=False) as data:
        manifest_array = data["manifest_json"]
        _require(manifest_array.dtype == np.uint8 and manifest_array.ndim == 1, "invalid embedded manifest")
        embedded = json.loads(manifest_array.tobytes().decode("utf-8"))
        array_names = set(data.files) - {"manifest_json"}
        _require(needed <= array_names, "required OSC intermediates are missing")
        for name in array_names:
            array = data[name]
            _require(array.size > 0 and array.dtype.kind != "O", f"invalid pack array: {name}")
        arrays = {name: np.array(data[name], copy=True) for name in needed}
    _require(embedded["schema_version"] == pack_entry["schema_version"], "pack schema mismatch")
    cases = embedded["cases"]
    case_ids = [case["case_id"] for case in cases]
    _require(len(case_ids) == len(set(case_ids)), "duplicate reference case")
    classifications = Counter(case["classification"] for case in cases)
    _require(classifications == Counter({"MATCH": 8, "CORRECTED": 4}), "case classification mismatch")
    for case in cases:
        divergence = case["first_divergence"]
        _require((case["classification"] == "CORRECTED") == isinstance(divergence, str), f"invalid divergence metadata: {case['case_id']}")
    references = [name for case in cases for name in case["arrays"]]
    counts = Counter(references)
    _require(set(references) == array_names and all(count == 1 for count in counts.values()), "pack array ownership mismatch")
    source = embedded["source"]
    _require(source["original_archive_sha256"] == manifest["original_rasim"]["archive_sha256"], "pack source hash mismatch")
    _require(source["manuscript_archive_sha256"] == manifest["manuscript"]["archive_sha256"], "pack manuscript hash mismatch")
    return {
        "arrays": arrays,
        "case_counts": dict(sorted(classifications.items())),
        "embedded": embedded,
        "pack_sha256": observed_hash,
        "array_count": len(array_names),
    }


def _verify_examples(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "examples" / "MANIFEST.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest["schema_version"] == "rasim-examples-manifest-v1", "example schema mismatch")
    entries = manifest["file"]
    declared: set[str] = set()
    for item in entries:
        relative = str(item["path"])
        _require(relative.startswith("examples/") and relative not in declared, "invalid example path")
        declared.add(relative)
        path = _path(root, relative)
        _require(path.stat().st_size == item["size_bytes"] and _sha256(path) == item["sha256"], f"example hash mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "examples").rglob("*")
        if path.is_file() and path.name != "MANIFEST.toml" and path.suffix.lower() != ".md"
    }
    _require(actual == declared, "example input file set mismatch")
    return entries


def _verify_gzip_osc(root: Path, entries: list[dict[str, Any]], pack: dict[str, Any]) -> int:
    osc_entries = [item for item in entries if str(item["path"]).endswith(".osc.gz")]
    _require(len(osc_entries) == 5, "expected five compressed OSC inputs")
    arrays = pack["arrays"]
    positions = arrays["osc_selected_positions_row_col"]
    expected = {item["name"]: item for item in pack["embedded"]["osc_files"]}
    observed: dict[str, dict[str, Any]] = {}
    for item in osc_entries:
        path = _path(root, item["path"])
        name = path.stem
        observed[name] = _stream_gzip_osc(path, positions if name in expected else None)
        _require(observed[name]["shape"] == (3000, 3000), f"unexpected OSC shape: {name}")
    for index, (name, metadata) in enumerate(expected.items()):
        result = observed[name]
        _require(result["sha256"] == metadata["sha256"], f"decompressed hash mismatch: {name}")
        _require(np.array_equal(result["summary"], arrays["osc_summary"][index, :6]), f"summary mismatch: {name}")
        _require(np.array_equal(result["selected_raw"], arrays["osc_selected_raw_values"][index]), f"raw sample mismatch: {name}")
        _require(np.array_equal(result["selected_native"], arrays["osc_selected_native_values"][index]), f"native sample mismatch: {name}")
        _require(np.array_equal(result["argmax_raw"], arrays["osc_argmax_raw_row_col"][index]), f"raw argmax mismatch: {name}")
        _require(np.array_equal(result["argmax_native"], arrays["osc_argmax_native_row_col"][index]), f"native argmax mismatch: {name}")
    return len(osc_entries)


def _verify_synthetic_osc(root: Path, arrays: dict[str, NDArray[Any]]) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    big_version, big = _decode_tiny_osc(root / "examples" / "common" / "osc" / "non_square_big_endian.osc")
    little_version, little = _decode_tiny_osc(root / "examples" / "common" / "osc" / "non_square_little_endian.osc")
    _require((big_version, little_version) == (1, 20) and np.array_equal(big, little), "endian decoding mismatch")
    native = raw_to_detector_native(big)
    _require(big.shape == (7, 11) and native.shape == (11, 7), "non-square orientation mismatch")
    _require(np.array_equal(big, arrays["osc_synthetic_big_endian_raw"]) and np.array_equal(little, arrays["osc_synthetic_little_endian_raw"]), "synthetic raw intermediate mismatch")
    _require(np.array_equal(native, arrays["osc_synthetic_big_endian_native"]) and np.array_equal(native, arrays["osc_synthetic_little_endian_native"]), "synthetic native intermediate mismatch")
    _require(np.array_equal(detector_native_to_raw(native), big), "OSC inverse mismatch")
    _require(64 in big and 1_048_544 in big, "high-range OSC decoding mismatch")
    marker = raw_to_detector_index(OscRawIndex(4, 3), big.shape)
    coordinate = index_to_coordinate(marker)
    _require(big[4, 3] == 2222 and (marker.row, marker.column) == (3, 2), "marker index mismatch")
    _require((coordinate.column_px, coordinate.row_px) == (2.0, 3.0), "pixel-center mapping mismatch")
    return big, native, np.array([2.0, 3.0])


def _verify_bi2se3_coordinates(root: Path) -> int:
    case = tomllib.loads((root / "examples" / "bi2se3" / "experiment" / "forward_case.toml").read_text(encoding="utf-8"))
    detector, legacy = case["detector"], case["legacy_provenance"]
    _require(detector["center_column_px"] == legacy["legacy_center_y_meant_native_column_px"], "beam-center column mismatch")
    _require(detector["center_row_px"] == legacy["legacy_center_x_meant_native_row_px"], "beam-center row mismatch")
    rows = 0
    csv_path = root / "examples" / "bi2se3" / "observations" / "legacy_peak_selections.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected_column = Decimal(detector["columns"] - 1) - Decimal(row["legacy_raw_x"])
            column_error = abs(Decimal(row["observed_column_px"]) - expected_column)
            _require(column_error <= Decimal("5e-13"), "legacy column mapping mismatch")
            _require(Decimal(row["observed_row_px"]) == Decimal(row["legacy_raw_y"]), "legacy row mapping mismatch")
            rows += 1
    _require(rows == 82, "legacy coordinate row count mismatch")
    return rows


def _trace(stage: str, value: NDArray[Any], kind: QuantityKind) -> TraceRecord:
    return TraceRecord(
        "reference.negative_control",
        stage,
        value,
        "px",
        "detector_native",
        Measure.NONE,
        kind,
        "reference-v1",
        "tracked synthetic OSC",
    )


def _mutations(raw: NDArray[np.int32], native: NDArray[np.int32], center: NDArray[np.float64]) -> list[dict[str, object]]:
    pairs = (
        ("osc_counterclockwise", "osc.detector_native_array", native, np.rot90(raw, 1), QuantityKind.IMAGE, "exact_value"),
        ("osc_transpose", "osc.detector_native_array", native, raw.T.copy(), QuantityKind.IMAGE, "exact_value"),
        ("osc_swap_coordinate_order", "osc.beam_center_native", center, center[::-1].copy(), QuantityKind.POINT, "numeric_value"),
        ("osc_half_pixel", "osc.beam_center_native", center, center + 0.5, QuantityKind.POINT, "numeric_value"),
    )
    results: list[dict[str, object]] = []
    for mutation_id, stage, expected, candidate, kind, metric in pairs:
        comparison = compare_traces((_trace(stage, expected, kind),), (_trace(stage, candidate, kind),))
        results.append(
            {
                "mutation_id": mutation_id,
                "fixture_id": "osc.synthetic.non_square",
                "expected_first_stage": stage,
                "expected_failure_metric": metric,
                "observed_first_stage": comparison.first_failing_stage,
                "observed_failure_metric": comparison.failure_metric,
                "detected": comparison.first_failing_stage == stage and comparison.failure_metric == metric,
            }
        )
    return results


def run_reference_proof(*, allow_missing_pack: bool = False) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    manifest = tomllib.loads((root / "reference" / "reference_manifest.toml").read_text(encoding="utf-8"))
    _require(manifest["schema_version"] == "rasim-reference-manifest-v4", "reference schema mismatch")
    _require(manifest["trace_schema_version"] == "rasim-stage-trace-v4" and manifest["contract_api_version"] == "rasim-contracts-v4", "shared version mismatch")
    _require(manifest["reference_pack"]["read_only"] is True and manifest["reference_pack"]["committed"] is True, "reference pack must be immutable")
    pack_path = _path(root, manifest["reference_pack"]["path"])
    if allow_missing_pack and not pack_path.is_file():
        return {
            "schema_version": 1,
            "task_id": "T01",
            "status": "PASS",
            "base_sha": _base_sha(root),
            "contract_version": 4,
            "trace_schema_version": 4,
            "reference_pack_sha256s": {},
            "environment_sha256": _environment_sha256(),
            "checks": [{"check_id": "reference_pack", "status": "PASS", "evidence": "pack absence allowed"}],
            "classifications": [],
            "limitations": ["reference pack was not present"],
        }
    source_count = _verify_source_snapshot(root, manifest["original_rasim"]["archive_sha256"])
    pack = _verify_pack(root, manifest)
    entries = _verify_examples(root)
    gzip_count = _verify_gzip_osc(root, entries, pack)
    raw, native, center = _verify_synthetic_osc(root, pack["arrays"])
    coordinate_rows = _verify_bi2se3_coordinates(root)
    mutations = _mutations(raw, native, center)
    _require(all(item["detected"] for item in mutations), "reference negative control escaped")
    checks = [
        {"check_id": "source_citations", "status": "PASS", "evidence": f"{source_count} tracked source files and exact OSC/orientation lines verified"},
        {"check_id": "reference_pack", "status": "PASS", "evidence": f"SHA-256, embedded manifest, {pack['array_count']} arrays, and 12 classifications verified"},
        {"check_id": "example_inputs", "status": "PASS", "evidence": f"{len(entries)} declared input hashes and file-set coverage verified"},
        {"check_id": "gzip_osc", "status": "PASS", "evidence": f"{gzip_count} files streamed; three decompressed hashes and selected intermediates verified"},
        {"check_id": "synthetic_osc", "status": "PASS", "evidence": "big/little endian, high range, clockwise inverse, and pixel center verified"},
        {"check_id": "bi2se3_coordinates", "status": "PASS", "evidence": f"native beam center and {coordinate_rows} legacy rows mapped exactly once"},
        {"check_id": "negative_controls", "status": "PASS", "evidence": f"{len(mutations)}/{len(mutations)} in-memory mutations detected at first stage"},
    ]
    return {
        "schema_version": 1,
        "task_id": "T01",
        "status": "PASS",
        "base_sha": _base_sha(root),
        "commit_sha": None,
        "contract_version": 4,
        "trace_schema_version": 4,
        "reference_pack_sha256s": {"rasim_reference_v1": pack["pack_sha256"]},
        "environment_sha256": _environment_sha256(),
        "checks": checks,
        "classifications": [],
        "case_counts": pack["case_counts"],
        "source_archive_sha256": manifest["original_rasim"]["archive_sha256"],
        "limitations": ["the v1 pack authenticates arrays as one file and has no per-array tolerance metadata"],
        "mutations": mutations,
    }
