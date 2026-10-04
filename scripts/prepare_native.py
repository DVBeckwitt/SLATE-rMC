"""Adopt and relocate a frozen calibrated native experiment after raw-pixel verification."""

import argparse
import copy
import hashlib
import io
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from rasim_next.fitting.fixed_experiment import (
    apply_fixed_position_to_instrument,
    fixed_position_from_fit_record,
)
from rasim_next.fitting.geometry import ExactTagGeometryModel, IntegerLMarkerKey
from rasim_next.fitting.native_input import load_native_fit_physics
from rasim_next.fitting.native_observations import (
    load_native_background_controls,
    load_native_fit_observations,
)
from rasim_next.io.osc import read_osc
from rasim_next.materials import read_crystal
from rasim_next.pipeline.configured_simulation import (
    build_configured_geometry_inputs,
    load_simulation_config,
)
from rasim_next.selection.osc_series import load_osc_geometry_series


def _geometry_bound_physics(physics_path, position_path, manifest_path, image_id, raw):
    """Adopt one fitted OSC pose without changing native source or specimen physics."""
    position_path = Path(position_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    position_bytes = position_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    position_record = json.loads(position_bytes)
    position, status, selection = fixed_position_from_fit_record(
        position_record,
        expected_manifest_path=manifest_path,
        expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    series = load_osc_geometry_series(manifest_path)
    images = tuple(series.images)
    matches = [index for index, image in enumerate(images) if image.image_id == image_id]
    if len(matches) != 1:
        raise ValueError("geometry image ID must identify exactly one frozen OSC image")
    image_index = matches[0]
    image = images[image_index]
    commanded = tuple(item.axis_rotation_angles_deg[series.incidence_axis_index] for item in images)
    if not np.allclose(
        position.commanded_incidence_angles_rad, np.radians(commanded), rtol=0, atol=2e-14
    ):
        raise ValueError("fitted position and geometry image angles differ")
    if position.incidence_angle_image_ids and tuple(item.image_id for item in images) != (
        position.incidence_angle_image_ids
    ):
        raise ValueError("fitted position and geometry image IDs differ")
    if not np.array_equal(read_osc(image.osc_path).detector_native_counts, raw):
        raise ValueError("native acquisition differs from the selected geometry OSC image")

    config = load_simulation_config(series.config_path)
    original = load_native_fit_physics(physics_path)
    source = config.source
    definition = original.source_definition
    source_pairs = (
        (source.mean_origin_lab_m, definition.mean_origin_lab_m),
        (source.mean_direction_lab, definition.mean_direction_lab),
        (source.transverse_axes_lab, definition.transverse_axes_lab),
        (source.spatial_sigma_m, definition.spatial_sigma_m),
        (source.divergence_sigma_rad, definition.divergence_sigma_rad),
        (source.position_divergence_correlation, definition.position_divergence_correlation),
        (source.line_wavelength_A, definition.line_wavelength_A),
        (source.line_probability, definition.line_probability),
        (source.common_line_sigma_A, definition.common_wavelength_sigma_A),
    )
    if any(not np.allclose(left, right, rtol=0, atol=1e-14) for left, right in source_pairs):
        raise ValueError("native source differs from the geometry configuration")
    if source.polarization_state_id != definition.polarization_state_id:
        raise ValueError("native polarization differs from the geometry configuration")
    crystal = read_crystal(
        config.material.cif_path,
        phase_id=config.material.phase_id,
        expected_sha256=config.cif_sha256,
    )
    reciprocal = 2 * np.pi * np.linalg.inv(crystal.direct_basis_A).T
    if not np.allclose(reciprocal, original.reciprocal_basis_Ainv, rtol=0, atol=1e-14):
        raise ValueError("native reciprocal basis differs from the geometry crystal")
    if len(config.instrument.axis_rotations) != 1 or series.incidence_axis_index != 0:
        raise ValueError("native geometry adoption requires one incidence rotation axis")
    angle_config = replace(
        config,
        instrument=replace(
            config.instrument,
            axis_rotations=(
                replace(
                    config.instrument.axis_rotations[series.incidence_axis_index],
                    angle_deg=float(
                        np.degrees(position.effective_incidence_angles_rad[image_index])
                    ),
                ),
            ),
        ),
    )
    geometry_inputs = build_configured_geometry_inputs(angle_config)
    selected = apply_fixed_position_to_instrument(
        geometry_inputs.instrument, angle_config.instrument.axis_rotations, position
    )
    saved_images = [item for item in position_record["predictions"] if item["image_id"] == image_id]
    if len(saved_images) != 1:
        raise ValueError("position result lacks exactly one selected image prediction")
    saved_sites = saved_images[0]["sites"]
    if not saved_sites:
        raise ValueError("selected geometry image has no frozen markers")
    keys = tuple(
        IntegerLMarkerKey(
            family_m=site["key"]["family_m"],
            integer_L=site["key"]["integer_L"],
            branch=site["key"]["analytic_ewald_branch"],
            root_sign=site["key"]["root_sign"],
            representative_rod_hk=tuple(site["key"]["representative_rod_hk"]),
        )
        for site in saved_sites
    )
    replay = ExactTagGeometryModel(geometry_inputs).predict_integer_l_tags(
        keys, instrument=selected
    )
    if tuple(replay.detector_status) != tuple(site["detector_status"] for site in saved_sites):
        raise ValueError("geometry adoption changed selected marker statuses")
    saved_coordinates = np.asarray(
        [site["predicted_coordinate_px"] for site in saved_sites], dtype=np.float64
    )
    if (
        not np.all(np.isfinite(saved_coordinates))
        or not np.all(np.isfinite(replay.coordinates_px))
        or not np.allclose(replay.coordinates_px, saved_coordinates, rtol=0, atol=2e-9)
    ):
        raise ValueError("geometry adoption changed selected marker coordinates")
    native = original.instrument
    if not np.allclose(
        selected.sample_from_crystal.rotation, original.crystal_to_sample, rtol=0, atol=1e-14
    ) or not np.allclose(
        native.sample_from_crystal.rotation, original.crystal_to_sample, rtol=0, atol=1e-14
    ):
        raise ValueError("crystal-to-sample and native instrument frames differ")
    for name in (
        "detector_shape_rc",
        "detector_row_pitch_m",
        "detector_column_pitch_m",
        "sample_support_model_id",
        "sample_width_m",
        "sample_length_m",
        "film_thickness_A",
        "detector_path_medium_id",
        "detector_path_linear_attenuation_m_inv",
    ):
        left, right = getattr(selected, name), getattr(native, name)
        if left != right and not (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and np.isclose(left, right, rtol=1e-12, atol=0)
        ):
            raise ValueError(f"native and geometry instrument {name} differ")

    record = json.loads(Path(physics_path).read_bytes())
    instrument = record["instrument"]
    for name in ("lab_from_detector", "lab_from_sample"):
        transform = getattr(selected, name)
        instrument[name] = {
            "rotation": transform.rotation.tolist(),
            "translation_m": transform.translation_m.tolist(),
            "source_frame": transform.source_frame,
            "target_frame": transform.target_frame,
        }
    instrument["detector_reference_coordinate_px"] = list(selected.detector_reference_coordinate_px)
    payload = (json.dumps(record, indent=2) + "\n").encode()
    return payload, {
        "position_path": str(position_path),
        "position_sha256": hashlib.sha256(position_bytes).hexdigest(),
        "position_status": status,
        "position_artifact_revision": position.artifact_revision,
        "indexed_selection_revision": selection,
        "geometry_manifest_path": str(manifest_path),
        "geometry_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "simulation_config_sha256": hashlib.sha256(series.config_path.read_bytes()).hexdigest(),
        "image_id": image_id,
        "osc_sha256": hashlib.sha256(image.osc_path.read_bytes()).hexdigest(),
        "marker_replay_count": len(keys),
        "marker_replay_max_abs_error_px": float(
            np.max(np.abs(replay.coordinates_px - saved_coordinates))
        ),
    }


def prepare(
    observation_path,
    output_directory,
    *,
    raw_path=None,
    dark_path=None,
    archived_baseline=None,
    geometry_position=None,
    geometry_manifest=None,
    geometry_image_id=None,
):
    """Keep all calibrated arrays unchanged; paths are relocatable, hashes authoritative."""
    source = Path(observation_path).resolve()
    destination = Path(output_directory).resolve()
    if any((parent / ".git").exists() for parent in (destination, *destination.parents)):
        raise ValueError("prepared inputs must remain outside a repository")
    if destination == source.parent:
        raise ValueError("prepared destination must differ from the original experiment")
    source_bytes = source.read_bytes()
    record = json.loads(source_bytes)
    if "archived_baseline" in record:
        raise ValueError(
            "prepare archived baselines from their catalog source, not a prepared copy"
        )
    original_source_sha256 = record.get("preparation", {}).get(
        "source_observation_sha256", hashlib.sha256(source_bytes).hexdigest()
    )
    if (
        not isinstance(original_source_sha256, str)
        or len(original_source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in original_source_sha256)
    ):
        raise ValueError("prepared source observation SHA256 must be a lowercase digest")
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
    physics_payload_index = len(payloads) - 1
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
    geometry_options = (geometry_position, geometry_manifest, geometry_image_id)
    if any(value is not None for value in geometry_options):
        if not all(value is not None for value in geometry_options):
            raise ValueError("geometry adoption requires position, manifest, and image ID")
        if archived_baseline is not None:
            raise ValueError("an archived baseline cannot be rebound to new geometry")
        bound_bytes, geometry_provenance = _geometry_bound_physics(
            physics_path, geometry_position, geometry_manifest, geometry_image_id, raw
        )
        bound_digest = hashlib.sha256(bound_bytes).hexdigest()
        bound_name = bound_digest[:16] + "_" + physics_path.name
        payloads[physics_payload_index] = (bound_name, bound_bytes)
        record["physical_input"] = {"path": bound_name, "sha256": bound_digest}
        snapshots.extend(
            (
                (Path(geometry_provenance[path_name]), geometry_provenance[digest_name])
                for path_name, digest_name in (
                    ("position_path", "position_sha256"),
                    ("geometry_manifest_path", "geometry_manifest_sha256"),
                )
            )
        )
        geometry_series = load_osc_geometry_series(geometry_manifest)
        snapshots.append(
            (geometry_series.config_path, geometry_provenance["simulation_config_sha256"])
        )
        selected_image = next(
            image for image in geometry_series.images if image.image_id == geometry_image_id
        )
        snapshots.append((selected_image.osc_path, geometry_provenance["osc_sha256"]))
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
    if archived_baseline is not None:
        baseline = copy.deepcopy(archived_baseline)
        if (
            baseline["observation_sha256"] != hashlib.sha256(source_bytes).hexdigest()
            or baseline["physics_sha256"] != record["physical_input"]["sha256"]
        ):
            raise ValueError("archived baseline belongs to a different experiment")
        _, fit_bytes = verified(baseline["parameters"])
        fit = json.loads(fit_bytes)
        if (
            fit["sample"] != record["sample_id"]
            or fit["status"] != "nominally_accepted"
            or fit["acceptance_mode"] != "nominal"
            or fit["numerically_qualified"] is not False
        ):
            raise ValueError("archived baseline must retain its original nominal status")
        _, prediction_bytes = verified(baseline["prediction"])
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as saved:
            prediction = saved[baseline["prediction"]["array"]]
        # The archived vector is already count-scaled. Never reprofile its scale.
        scores = observations.scores(prediction)
        if not scores["guards_pass"] or not np.isclose(
            scores["historical_loss"], fit["cost_selected"], rtol=1e-10, atol=0
        ):
            raise ValueError("archived prediction differs from its historical objective or guards")
        for reference in baseline["figures"].values():
            verified(reference)
        baseline["saved_prediction_scores"] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in scores.items()
        }
        baseline["status"] = "archived_nominal_not_recomputed"
        record["archived_baseline"] = baseline
    previous_adoption = record.get("preparation", {}).get("geometry_adoption")
    record["preparation"] = dict(
        kind="verified_frozen_calibration_adoption",
        source_observation_sha256=original_source_sha256,
        measured_convention="raw counts; selected subtraction is carried by background",
    )
    if any(value is not None for value in geometry_options):
        record["preparation"]["geometry_adoption"] = geometry_provenance
    elif previous_adoption is not None:
        record["preparation"]["geometry_adoption"] = previous_adoption
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


def prepare_catalog(
    sample_id,
    input_root,
    output_directory,
    *,
    with_baseline=False,
    catalog_path=None,
    raw_path=None,
    dark_path=None,
    geometry_position=None,
    geometry_manifest=None,
    geometry_image_id=None,
):
    """Resolve one hash-bound experiment; archived absolute paths are not authority."""
    catalog_path = (
        catalog_path or Path(__file__).resolve().parents[1] / "configs/native_experiments.json"
    )
    catalog = json.loads(Path(catalog_path).read_bytes())
    if catalog["schema"] != "rasim-native-experiment-catalog-v1":
        raise ValueError("unsupported native experiment catalog")
    entries = [entry for entry in catalog["experiments"] if entry["sample_id"] == sample_id]
    if len(entries) != 1:
        raise ValueError("sample must identify exactly one catalog experiment")
    entry = entries[0]
    input_root = Path(input_root).resolve()

    def resolve(reference):
        relative = Path(reference["path"])
        path = (input_root / relative).resolve()
        if relative.is_absolute() or path == input_root or not path.is_relative_to(input_root):
            raise ValueError("catalog input must be a relative file inside the input root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError(f"catalog input SHA256 mismatch: {path}")
        return path

    source = resolve(entry["observations"])
    physics_path = resolve(entry["physics"])
    record = json.loads(source.read_bytes())
    if (
        record["sample_id"] != sample_id
        or record["physical_input"]["sha256"] != entry["physics"]["sha256"]
        or (source.parent / record["physical_input"]["path"]).resolve() != physics_path
    ):
        raise ValueError("catalog and observation physics bindings differ")
    baseline = None
    if with_baseline:
        if "archived_baseline" not in entry:
            raise ValueError("no archived baseline is bound for this sample")
        baseline = copy.deepcopy(entry["archived_baseline"])
        for reference in (
            baseline["parameters"],
            baseline["prediction"],
            *baseline["figures"].values(),
        ):
            reference["path"] = str(resolve(reference))
        baseline["observation_sha256"] = entry["observations"]["sha256"]
        baseline["physics_sha256"] = entry["physics"]["sha256"]
    return prepare(
        source,
        output_directory,
        raw_path=raw_path,
        dark_path=dark_path,
        archived_baseline=baseline,
        geometry_position=geometry_position,
        geometry_manifest=geometry_manifest,
        geometry_image_id=geometry_image_id,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--observations", type=Path)
    selection.add_argument("--sample", help="sample ID from configs/native_experiments.json")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--catalog", type=Path, help="explicit copied/migrated experiment catalog")
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="include the archived nominal parameters, counts and figures",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--dark", type=Path)
    parser.add_argument("--geometry-position", type=Path)
    parser.add_argument("--geometry-manifest", type=Path)
    parser.add_argument("--geometry-image-id")
    args = parser.parse_args()
    if args.sample:
        if args.input_root is None:
            parser.error("--sample requires --input-root")
        output = prepare_catalog(
            args.sample,
            args.input_root,
            args.output_directory,
            with_baseline=args.with_baseline,
            catalog_path=args.catalog,
            raw_path=args.raw,
            dark_path=args.dark,
            geometry_position=args.geometry_position,
            geometry_manifest=args.geometry_manifest,
            geometry_image_id=args.geometry_image_id,
        )
    else:
        if args.input_root is not None or args.with_baseline or args.catalog is not None:
            parser.error("--input-root, --catalog and --with-baseline require --sample")
        output = prepare(
            args.observations,
            args.output_directory,
            raw_path=args.raw,
            dark_path=args.dark,
            geometry_position=args.geometry_position,
            geometry_manifest=args.geometry_manifest,
            geometry_image_id=args.geometry_image_id,
        )
    print(output)


if __name__ == "__main__":
    main()
