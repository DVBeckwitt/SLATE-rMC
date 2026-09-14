"""Adopt and relocate a frozen calibrated native experiment after raw-pixel verification."""

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_observations import (
    load_native_background_controls,
    load_native_fit_observations,
)
from rasim_next.io.osc import read_osc


def prepare(observation_path, output_directory, *, raw_path=None, dark_path=None):
    """Keep all calibrated arrays unchanged; paths are relocatable, hashes authoritative."""
    source = Path(observation_path).resolve()
    destination = Path(output_directory).resolve()
    if any((parent / ".git").exists() for parent in (destination, *destination.parents)):
        raise ValueError("prepared inputs must remain outside a repository")
    if destination == source.parent:
        raise ValueError("prepared destination must differ from the original experiment")
    source_bytes = source.read_bytes()
    record = json.loads(source_bytes)
    observations = load_native_fit_observations(source)
    controls = load_native_background_controls(source)
    payloads = []
    snapshots = [(source, hashlib.sha256(source_bytes).hexdigest())]

    def verified(reference, override=None):
        path = Path(override) if override is not None else source.parent / reference["path"]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != reference["sha256"]:
            raise ValueError(f"input SHA256 mismatch: {path}")
        # Hash prefix prevents collisions between different files with the same name.
        name = digest[:16] + "_" + path.name
        payloads.append((name, data))
        snapshots.append((path, digest))
        reference["path"] = name
        return path, data

    _, array_bytes = verified(record["arrays"])
    physics_path, _ = verified(record["physical_input"])
    load_native_fit_physics(physics_path)
    raw_reference = record["raw_acquisition"]
    raw_file, raw_bytes = verified(raw_reference, raw_path)
    if raw_reference["kind"] == "decoded_detector_native_npz":
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as data:
            raw = np.array(data["detector_native_counts"], dtype=float)
    elif raw_reference["kind"] == "osc_clockwise_native":
        raw = np.asarray(read_osc(raw_file).detector_native_counts, dtype=float)
    else:
        raise ValueError("raw acquisition must explicitly declare its native orientation")
    variance = np.maximum(raw, 1.0)
    dark = record["background"]["dark"]
    if dark["kind"] == "scaled_osc":
        dark_file, _ = verified(dark, dark_path)
        dark_counts = np.asarray(read_osc(dark_file).detector_native_counts, dtype=float)
        if dark_counts.shape != raw.shape or not np.isfinite(dark["scale"]) or dark["scale"] < 0:
            raise ValueError("dark image and declared scale must match the acquisition")
        variance += dark["scale"] ** 2 * np.maximum(dark_counts, 1.0)
    elif dark["kind"] != "none" or dark_path is not None:
        raise ValueError("dark override requires an explicitly scaled dark calibration")
    measured, covariance = observations.projection.integrate_field(raw, variance)
    with np.load(io.BytesIO(array_bytes), allow_pickle=False) as arrays:
        if not np.array_equal(measured, arrays["measured"]):
            raise ValueError("raw reprojection differs from frozen measured counts")
        if not np.allclose(covariance, arrays["count_covariance"], rtol=1e-12, atol=1e-8):
            raise ValueError("raw/dark covariance differs from frozen count covariance")
        # The background already owns the chosen dark subtraction convention.
        if not np.allclose(
            measured - arrays["background"], arrays["frozen_net"], rtol=0, atol=1e-8
        ):
            raise ValueError("adopted background changes the frozen objective")
    for control in controls:
        projection = control.projection
        control_variance = np.bincount(
            projection.observation_row,
            weights=projection.detector_area_weight_px2**2
            * variance.ravel()[projection.flat_pixel_index[projection.pixel_column_index]],
            minlength=projection.observation_count,
        )
        if not np.allclose(
            control_variance, control.measurement_variance_count2, rtol=1e-12, atol=1e-8
        ):
            raise ValueError("control measurement variance differs from raw/dark projection")
    record["preparation"] = dict(
        kind="verified_frozen_calibration_adoption",
        source_observation_sha256=hashlib.sha256(source_bytes).hexdigest(),
        measured_convention="raw counts; selected subtraction is carried by background",
    )
    for path, digest in snapshots:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"input changed during preparation: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / source.name
    prepared_bytes = (json.dumps(record, indent=2) + "\n").encode()
    for name, data in (*payloads, (output.name, prepared_bytes)):
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"refusing to replace a different prepared input: {target}")
    # Publish the descriptor last, after every independently verified byte payload.
    for name, data in (*payloads, (output.name, prepared_bytes)):
        target = destination / name
        if target.exists():
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=".prepare-", dir=destination)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--dark", type=Path)
    args = parser.parse_args()
    print(prepare(args.observations, args.output_directory, raw_path=args.raw, dark_path=args.dark))


if __name__ == "__main__":
    main()
