"""Hash-bound numeric snapshots and explicit experiment-to-simulator review mappings."""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import yaml
from job_lifecycle import JobResult
from metadata_review import bounded_reference_snapshot
from native_simulation_io import canonical_native
from native_simulation_state import native_draft_from_document
from numeric_fields import description
from parameter_state import _set_at, _source_mapping, configured_draft
from project_state import NumericDraft
from simulation_fields import SIMULATION_FIELDS
from simulation_io import _stop, canonical_configuration
from simulation_state import SimulationDraft, simulation_draft_from_document


def numeric_snapshot_document(draft):
    return {
        **asdict(draft),
        "acquisition_id": str(draft.acquisition_id),
        "configuration_path": str(draft.configuration_path),
        "cif_path": str(draft.cif_path),
    }


def _numeric_snapshot(value):
    if type(value) is not dict or set(value) != set(NumericDraft.__dataclass_fields__):
        raise ValueError("invalid immutable experiment numeric snapshot")
    row = value.copy()
    row["acquisition_id"] = UUID(row["acquisition_id"])
    row["configuration_path"] = Path(row["configuration_path"])
    row["cif_path"] = Path(row["cif_path"])
    row["proposed"] = tuple(tuple(v) for v in row["proposed"])
    return NumericDraft(**row)


@dataclass(frozen=True, slots=True)
class SimulationTransferReview:
    token: str
    target_kind: str
    candidate: object
    source_sha256: str
    rows: tuple[tuple[str, str, str, str, str], ...]

    @property
    def nbytes(self):
        return len(repr(self).encode()) + 8192


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def prepare_simulation_transfer(argument, control):
    request = json.loads(argument)
    source = _numeric_snapshot(request["source"])
    _stop(control)
    # Fresh immutable bytes, rather than the older Relink/copy review receipts.
    raw, _identity = bounded_reference_snapshot(source.configuration_path)
    if hashlib.sha256(raw).hexdigest() != source.configuration_sha256:
        raise ValueError("experiment configuration changed since the numeric snapshot")
    config = configured_draft(source)
    mapping = _source_mapping(source)
    rows = []
    provenance = f"experiment {source.acquisition_id}; numeric revision {source.revision}; source SHA256 {source.source_sha256}; configuration SHA256 {source.configuration_sha256}; CIF SHA256 {source.cif_sha256}"
    for name, value, unit, origin in source.proposed:
        item = description(name)
        _set_at(mapping, item.path, value)
        rows.append(("included", name, str(value), unit, origin))
    mapping["material"]["cif_path"] = str(source.cif_path)
    if request["target_kind"] == "configured":
        old = simulation_draft_from_document(request.get("destination"))
        text = yaml.safe_dump(mapping, sort_keys=False)
        confirmed = simulation_draft_from_document(request.get("confirmed_candidate"))
        candidate = SimulationDraft(
            old.draft_id if old else confirmed.draft_id if confirmed else uuid4(),
            old.revision + 1 if old else 0,
            source.configuration_path,
            source.configuration_sha256,
            text,
            source.cif_path,
            source.cif_sha256,
            old.route if old else "monte_carlo",
            old.position_mode if old else "sampled",
            old.draw_count if old else 8,
            old.detector_seed if old else 1729,
            transfer_provenance=provenance,
        )
        canonical_configuration(candidate)
        for field in SIMULATION_FIELDS:
            value = mapping
            try:
                for key in field.path.split("."):
                    value = value[key]
            except KeyError:
                if field.optional:
                    rows.append(
                        (
                            "effective default",
                            field.path,
                            repr(field.default),
                            field.unit,
                            field.applicability + "; authoritative configured default",
                        )
                    )
                continue
            rows.append(("included", field.path, json.dumps(value), field.unit, provenance))
        rows.append(
            (
                "target retained",
                "execution route / position / draws / detector seed",
                f"{candidate.route}/{candidate.position_mode}/{candidate.draw_count}/{candidate.detector_seed}",
                "declared CPU / MC controls",
                "independent simulator controls; no fit scale",
            )
        )
        rows.append(
            (
                "omitted",
                "acquisition image, masks, exclusions, fits and qualification",
                "",
                "count / provenance",
                "not simulation inputs",
            )
        )
    elif request["target_kind"] == "native":
        from rasim_next.pipeline.configured_simulation import _compile_instrument

        old = native_draft_from_document(request["destination"])
        if old is None or old.recipe != request["recipe"]:
            raise ValueError("load the explicitly selected admitted native recipe before transfer")
        canonical_native(old)
        record = json.loads(old.physics_json)
        instrument = _plain(asdict(_compile_instrument(config.instrument)))
        physical_fields = (
            "lab_from_sample",
            "sample_from_crystal",
            "lab_from_detector",
            "detector_shape_rc",
            "detector_row_pitch_m",
            "detector_column_pitch_m",
            "detector_reference_coordinate_px",
            "sample_support_model_id",
            "sample_width_m",
            "sample_length_m",
            "detector_path_medium_id",
            "detector_path_linear_attenuation_m_inv",
            "detector_path_wavelength_A",
            "detector_path_linear_attenuation_m_inv_by_wavelength",
        )
        for name in physical_fields:
            record["instrument"][name] = instrument[name]
            rows.append(
                (
                    "included",
                    "instrument." + name,
                    json.dumps(instrument[name]),
                    "rigid transform: metre / active rotation; detector chart: column,row px",
                    provenance,
                )
            )
        record["crystal_to_sample"] = instrument["sample_from_crystal"]["rotation"]
        for name in (
            "mean_origin_lab_m",
            "mean_direction_lab",
            "transverse_axes_lab",
            "spatial_sigma_m",
            "divergence_sigma_rad",
            "position_divergence_correlation",
            "polarization_state_id",
        ):
            value = _plain(getattr(config.source, name))
            record["source"][name] = value
            rows.append(
                (
                    "included",
                    "source." + name,
                    json.dumps(value),
                    "metre / radian / declared LAB or source frame",
                    provenance,
                )
            )
        if config.source.wavelength_model_id == "discrete_gaussian_lines.v1":
            for native_name, configured_name in (
                ("line_wavelength_A", "line_wavelength_A"),
                ("line_probability", "line_probability"),
                ("common_wavelength_sigma_A", "common_line_sigma_A"),
            ):
                value = _plain(getattr(config.source, configured_name))
                record["source"][native_name] = value
                rows.append(
                    (
                        "included",
                        "source." + native_name,
                        json.dumps(value),
                        "angstrom / probability",
                        provenance,
                    )
                )
        else:
            rows.append(
                (
                    "incompatible; target retained",
                    "native source spectrum",
                    json.dumps(
                        {
                            k: record["source"][k]
                            for k in (
                                "line_wavelength_A",
                                "line_probability",
                                "common_wavelength_sigma_A",
                            )
                        }
                    ),
                    "angstrom / probability",
                    "configured Gaussian spectrum is not an explicit native line declaration",
                )
            )
        values = dict(zip(old.parameter_names, old.parameter_values, strict=True))
        coherent_extent = old.coherent_repeats * values["c_A"]
        thickness = config.instrument.film_thickness_A
        if thickness >= coherent_extent:
            values["extra_film_thickness_A"] = thickness - coherent_extent
            rows.append(
                (
                    "included",
                    "extra_film_thickness_A",
                    str(thickness - coherent_extent),
                    "angstrom",
                    provenance + "; extra = configured thickness - N*c",
                )
            )
        else:
            rows.append(
                (
                    "incompatible; target retained",
                    "film thickness",
                    str(coherent_extent + values["extra_film_thickness_A"]),
                    "angstrom",
                    "configured thickness is smaller than selected native coherent stack",
                )
            )
        for name, value in (
            ("phase_population_weight", config.weights.phase_population),
            ("polarization_weight", config.weights.polarization),
        ):
            record[name] = value
            rows.append(("included", name, str(value), "1", provenance))
        rows.extend(
            (
                "target retained",
                name,
                json.dumps(record[name]),
                "native declared units",
                "selected admitted native recipe; configured CIF cannot infer this declaration",
            )
            for name in (
                "structure",
                "material",
                "rods",
                "rod_catalog_revision",
                "reciprocal_basis_Ainv",
                "specular_stitch_stack",
                "source_rule",
                "integration_rule",
                "spatial_quadrature_order",
            )
        )
        rows.append(
            (
                "target retained",
                "specimen coordinates / repeats / bin size / proposal",
                json.dumps(
                    {
                        "values": old.parameter_values,
                        "N": old.coherent_repeats,
                        "bin_size_px": old.bin_size_px,
                        "proposal_mosaic": old.proposal_mosaic,
                    }
                ),
                "canonical model units",
                "except explicitly included extra thickness above; no fit scale or model inference",
            )
        )
        rows.append(
            (
                "omitted",
                "configured mosaic / CIF structure / rod families / fit and acquisition arrays",
                "",
                "configured units",
                "native specimen and roster retained explicitly",
            )
        )
        candidate = replace(
            old,
            revision=old.revision + 1,
            physics_json=json.dumps(record, sort_keys=True),
            parameter_values=tuple(values[n] for n in old.parameter_names),
            transfer_provenance=provenance,
        )
        canonical_native(candidate)
    else:
        raise ValueError("unsupported transfer destination")
    if "confirmed_candidate" in request:
        confirmed = (
            simulation_draft_from_document(request["confirmed_candidate"])
            if request["target_kind"] == "configured"
            else native_draft_from_document(request["confirmed_candidate"])
        )
        if candidate != confirmed:
            raise ValueError(
                "confirmed transfer candidate differs from freshly bound source snapshot"
            )
    _stop(control)
    digest = hashlib.sha256(json.dumps(request["source"], sort_keys=True).encode()).hexdigest()
    result = SimulationTransferReview(
        request["token"], request["target_kind"], candidate, digest, tuple(rows)
    )
    return JobResult(result, result.nbytes)
