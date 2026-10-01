"""Explicit geometry edit adapters and bounded canonical forward previews."""

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from uuid import UUID

import numpy as np
import yaml
from hbn_state import HBN_NAMES, hbn_session_document, hbn_session_from_document
from job_lifecycle import JobResult
from joint_state import joint_controls, joint_session_document, joint_session_from_document
from numeric_fields import PARAMETERS, description
from parameter_state import _source_mapping, _value_at, configured_draft, edit_draft
from project_state import NumericDraft
from reciprocal_preview import GRID_AXIS, ReciprocalMapping
from sample_io import _fit_arguments, _images, sample_work_budget
from sample_state import (
    encoded,
    payload_hash,
    sample_session_document,
    sample_session_from_document,
)
from simulation_io import canonical_configuration
from simulation_state import simulation_draft_document, simulation_draft_from_document


@dataclass(frozen=True, slots=True)
class PhysicalField:
    name: str
    value: float | None
    unit: str
    lower: float | None
    upper: float | None
    editable: bool
    frame: str
    role: str
    scope: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class PhysicalView:
    mapping: ReciprocalMapping | None
    coordinates_px: np.ndarray
    feature_ids: tuple[str, ...]
    observable: str
    error: str = ""

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates_px, dtype=np.float64)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 2
            or len(coordinates) != len(self.feature_ids)
            or len(coordinates) > 4096
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Physical preview needs finite bounded native feature coordinates")
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates_px", coordinates)


@dataclass(frozen=True, slots=True)
class PhysicalResult:
    request_sha256: str
    snapshot_sha256: str
    views: tuple[PhysicalView, ...] = ()
    proposal: dict | None = None
    held_fixed: tuple[tuple[str, float | None], ...] = ()
    parameter: str = ""
    step: float | None = None

    @property
    def nbytes(self):
        return (
            16384
            + len(encoded(self.proposal).encode())
            + len(encoded(self.held_fixed).encode())
            + sum(
                v.coordinates_px.nbytes
                + sum(len(n.encode()) + 64 for n in v.feature_ids)
                + (
                    0
                    if v.mapping is None
                    else 8192
                    + sum(
                        a.nbytes
                        for a in (
                            v.mapping.q_sample_Ainv,
                            v.mapping.valid,
                            v.mapping.status,
                            v.mapping.column_px,
                            v.mapping.row_px,
                        )
                    )
                )
                for v in self.views
            )
        )


def numeric_document(draft):
    values = asdict(draft)
    for k in ("acquisition_id", "configuration_path", "cif_path"):
        values[k] = str(values[k])
    return values


def numeric_from_document(values):
    values = dict(values)
    values["acquisition_id"] = UUID(values["acquisition_id"])
    for k in ("configuration_path", "cif_path"):
        values[k] = Path(values[k])
    values["proposed"] = tuple(tuple(v) for v in values["proposed"])
    return NumericDraft(**values)


def snapshot(shell, route):
    """Capture only the existing route's state; no fit or handoff consumption."""
    if route == "configuration":
        a = shell._numeric_validated_acquisition()
        draft = shell._numeric_draft
        if draft is None or draft.acquisition_id != a.acquisition_id:
            raise ValueError("Load the selected acquisition's numeric draft first")
        return numeric_document(draft)
    if route == "simulator":
        if (
            shell.simulator.draft_kind.currentData() != "configured"
            or shell.simulator.draft is None
        ):
            raise ValueError(
                "Native material coordinates have no supported spatial handles; use the native inspector"
            )
        return simulation_draft_document(shell.simulator.draft)
    if route == "hbn":
        if shell.hbn.review_pending():
            raise ValueError("Commit or restore the pending hBN review first")
        return hbn_session_document(shell.hbn.current_inputs())
    if route == "sample":
        p = shell.sample
        if p.session is None or p._draft_dirty or not p.inputs_match():
            raise ValueError("Admit inputs and commit pending sample controls first")
        return sample_session_document(p.session)
    p = shell.joint
    if p.session is None or p.session.controls_json is None or p.session.draft_json is not None:
        raise ValueError("Capture joint inputs and commit pending canonical controls first")
    return joint_session_document(p.session)


def fields(route, data):
    if route in ("configuration", "simulator"):
        if route == "configuration":
            draft = numeric_from_document(data)
            mapping = _source_mapping(draft)
            proposed = {n: v for n, v, _u, _p in draft.proposed}
            scope = (str(draft.acquisition_id),)
        else:
            draft = simulation_draft_from_document(data)
            mapping = yaml.safe_load(draft.yaml_text)
            mapping["instrument"].setdefault(
                "detector_tilt", {"about_column_axis_deg": 0.0, "about_row_axis_deg": 0.0}
            )
            proposed = {}
            scope = ("independent simulator draft " + str(draft.draft_id),)
        rows = []
        for item in PARAMETERS:
            try:
                value = (
                    float(proposed.get(item.field, _value_at(mapping, item.path)))
                    if item.editable
                    else None
                )
                available = item.editable
                reason = item.reason
            except (ValueError, TypeError, KeyError):
                value, available, reason = None, False, "Absent from this configuration"
            rows.append(
                PhysicalField(
                    item.field,
                    value,
                    item.stored_unit,
                    None,
                    None,
                    available,
                    item.frame,
                    "configured initial" if available else "fixed / coupled",
                    scope,
                    reason,
                )
            )
        return tuple(rows)
    if route == "hbn":
        session = hbn_session_from_document(data)
        controls = dict(
            names=HBN_NAMES, initial=session.initial, lower=session.lower, upper=session.upper
        )
        scopes = [(str(session.acquisition_id),)] * 5
        fitted = set(HBN_NAMES)
    elif route == "sample":
        session = sample_session_from_document(data)
        controls = json.loads(session.controls_json)
        scopes = [tuple(v["image_id"] for v in json.loads(session.inputs_json)["images"])] * len(
            controls["names"]
        )
        fitted = set(controls["fitted"])
    else:
        session = joint_session_from_document(data)
        controls = json.loads(session.controls_json)
        captures = json.loads(session.captures_json)
        scopes = []
        for name in controls["names"]:
            groups = [
                g
                for g in captures
                if not name.startswith(("bi2se3_", "bi2te3_", "pbi2_", "hbn_"))
                or name.startswith(g + "_")
            ]
            if name.startswith("goniometer_") or name == "incidence_angle_delta_rad":
                groups = [g for g in groups if g != "hbn"]
            scopes.append(
                tuple(
                    identity
                    for g in groups
                    for identity in (
                        (captures[g]["session"]["acquisition_id"],)
                        if g == "hbn"
                        else tuple(
                            v["image_id"]
                            for v in json.loads(
                                sample_session_from_document(captures[g]["session"]).inputs_json
                            )["images"]
                        )
                    )
                )
            )
        fitted = set(controls["names"]) - {
            "goniometer_axis_pitch_rad",
            "goniometer_pivot_pitch_offset_m",
        }
        fitted = {n for n in fitted if not n.startswith("pbi2_")}
    return tuple(
        PhysicalField(
            n,
            v,
            "rad" if n.endswith("_rad") else "px" if n.endswith("_px") else "m",
            lo,
            hi,
            n in fitted and bool(scope),
            "native detector" if n.endswith("_px") else "canonical owner frame / declared pivot",
            "fit coordinate; initial seed" if n in fitted else "fixed / reduced gauge / unobserved",
            scope,
            ""
            if n in fitted and scope
            else "Fixed reference, inactive coordinate or absent specimen; not independently measured",
        )
        for n, v, lo, hi, scope in zip(
            controls["names"],
            controls["initial"],
            controls["lower"],
            controls["upper"],
            scopes,
            strict=True,
        )
    )


def changed_snapshot(route, data, name, value):
    field = next(f for f in fields(route, data) if f.name == name)
    if not field.editable or not math.isfinite(value):
        raise ValueError(field.reason or "A finite editable initial value is required")
    if field.lower is not None and not field.lower <= value <= field.upper:
        raise ValueError("Proposed initial value lies outside the owner's bounds; no clipping")
    if route == "configuration":
        draft = numeric_from_document(data)
        item = description(name)
        return numeric_document(
            edit_draft(draft, name, format(value / item.display_to_stored, ".17g"))
        )
    if route == "simulator":
        draft = simulation_draft_from_document(data)
        mapping = yaml.safe_load(draft.yaml_text)
        mapping["instrument"].setdefault(
            "detector_tilt", {"about_column_axis_deg": 0.0, "about_row_axis_deg": 0.0}
        )
        from parameter_state import _set_at

        _set_at(mapping, description(name).path, value)
        draft = replace(
            draft, yaml_text=yaml.safe_dump(mapping, sort_keys=False), revision=draft.revision + 1
        )
        canonical_configuration(draft)
        return simulation_draft_document(draft)
    if route == "hbn":
        s = hbn_session_from_document(data)
        values = list(s.initial)
        values[HBN_NAMES.index(name)] = value
        return hbn_session_document(
            replace(
                s,
                initial=tuple(values),
                revision=s.revision + 1,
                frozen_json="",
                center_proposal_json="",
                initial_provenance_json="",
                selected_result_id=None,
            )
        )
    s = (sample_session_from_document if route == "sample" else joint_session_from_document)(data)
    c = json.loads(s.controls_json)
    c["initial"][c["names"].index(name)] = value
    if route == "sample":
        _fit_arguments(c, len(json.loads(s.inputs_json)["images"]))
        return sample_session_document(
            replace(
                s,
                controls_json=encoded(c),
                revision=s.revision + 1,
                initial_provenance_json="",
                selected_result_id=None,
            )
        )
    c = joint_controls(**{k: c[k] for k in ("initial", "lower", "upper")})
    return joint_session_document(
        replace(s, controls_json=encoded(c), revision=s.revision + 1, selected_result_id=None)
    )


def _stop(control):
    if control.canceled:
        raise RuntimeError("Physical preview canceled; no fitted state changed")


def _map(inputs, instrument, control, axis_rotations=None):
    from rasim_next.pipeline.configured_simulation import build_geometry_only_ewald_context

    _stop(control)
    ctx = build_geometry_only_ewald_context(inputs, instrument=instrument)
    rows, cols = instrument.detector_shape_rc
    c, r = np.meshgrid(
        np.linspace(-0.5, cols - 0.5, GRID_AXIS), np.linspace(-0.5, rows - 0.5, GRID_AXIS)
    )
    g = ctx.evaluate_detector_geometry(c, r, include_surface_jacobian=False)
    return ReciprocalMapping(
        instrument,
        ctx.incident,
        ctx.ki_sample_Ainv,
        ctx.incident.states.k_air_sample_Ainv[0],
        g.column_px,
        g.row_px,
        g.q_sample_Ainv,
        g.valid,
        g.status,
        inputs.config.source.mean_wavelength_A,
        tuple(inputs.config.instrument.axis_rotations)
        if axis_rotations is None
        else axis_rotations,
        tuple(inputs.config.source.mean_origin_lab_m),
    )


def _sample_values(session):
    from rasim_next.fitting.indexed_series import zero_sum_helmert_basis

    c = json.loads(session.controls_json)
    args = _fit_arguments(c, len(json.loads(session.inputs_json)["images"]))
    ids = sorted(v["image_id"] for v in json.loads(session.inputs_json)["images"])
    trims = zero_sum_helmert_basis(len(ids)) @ np.asarray(c["initial"][13:])
    return args, dict(zip(ids, trims, strict=True))


def _preview(route, data, image_id, control, prepared=None, parameter=""):
    from rasim_next.pipeline.configured_simulation import (
        build_configured_geometry_inputs,
        load_simulation_config,
    )

    _stop(control)
    if route in ("configuration", "simulator"):
        if route == "configuration":
            draft = numeric_from_document(data)
            for path, digest in (
                (draft.configuration_path, draft.configuration_sha256),
                (draft.cif_path, draft.cif_sha256),
            ):
                from hbn_io import _hash_file

                if _hash_file(path, control) != digest:
                    raise ValueError("Configuration/CIF predecessor changed")
            config = configured_draft(draft)
        else:
            config = canonical_configuration(simulation_draft_from_document(data))
        inputs = build_configured_geometry_inputs(config) if prepared is None else prepared
        if prepared is not None:
            from rasim_next.pipeline.configured_simulation import (
                _compile_instrument,
            )

            # Instrument-only edits preserve the immutable source/material/rod state.
            if inputs.config.source != config.source:
                inputs = build_configured_geometry_inputs(config)
            else:
                inputs = replace(
                    inputs, config=config, instrument=_compile_instrument(config.instrument)
                )
        m = _map(inputs, inputs.instrument, control)
        return PhysicalView(
            m,
            np.column_stack((m.column_px.ravel(), m.row_px.ravel())),
            tuple(str(i) for i in range(GRID_AXIS**2)),
            "13x13 nominal inverse detector Q map",
        ), inputs
    if route == "hbn":
        from hbn_io import _geometry, _inputs_current

        from rasim_next.fitting.hbn import (
            hbn_detector_transform,
            hbn_ring_curves_px,
            hbn_two_theta_rad,
        )

        s = hbn_session_from_document(data)
        inp = json.loads(s.inputs_json)
        if prepared is None:
            _inputs_current(inp, control)
            inputs = build_configured_geometry_inputs(
                load_simulation_config(inp["configuration_path"])
            )
        else:
            inputs = prepared
        angles = hbn_two_theta_rad(
            lattice_a_A=inp["lattice_a_A"],
            lattice_c_A=inp["lattice_c_A"],
            wavelength_A=inp["wavelength_A"],
        )
        curves = hbn_ring_curves_px(s.initial, angles, **_geometry(inp))
        instrument = replace(
            inputs.instrument,
            lab_from_detector=hbn_detector_transform(
                s.initial,
                **_geometry(inp),
                calibrant_origin_lab_m=inputs.instrument.lab_from_sample.translation_m,
                detector_reference_coordinate_px=inputs.instrument.detector_reference_coordinate_px,
            ),
        )
        m = _map(inputs, instrument, control)
        coordinates = np.concatenate(curves)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("hBN curve has an invalid plane intersection")
        return PhysicalView(
            m,
            coordinates,
            tuple(
                f"ring {i} / azimuth {j}"
                for i, curve in enumerate(curves)
                for j in range(len(curve))
            ),
            "hBN private-plane ring curves; inverse Q uses configured material",
        ), inputs
    if route == "sample":
        s = sample_session_from_document(data)
        images = _images(s, control) if prepared is None else prepared
        args, trims = _sample_values(s)
        image = next((im for im in images if im.image_id == image_id), images[0])
        kwargs = dict(
            detector_calibration_corrections=args["initial_detector_calibration_corrections"],
            incidence_angle_delta_rad=args["initial_incidence_angle_delta_rad"],
            incidence_angle_trim_rad=trims[image.image_id],
        )
        instrument = image.corrected_instrument(args["initial"], **kwargs)
        prediction = image.predict_integer_l_tags(
            image.observations.keys, args["initial"], **kwargs
        )
        if not np.all(prediction.active_panel):
            raise ValueError("Frozen sample marker branch/panel topology changed")
        from rasim_next.fitting.indexed_series import corrected_goniometer_axis

        axis = image.model.inputs.config.instrument.axis_rotations[0]
        shifted = replace(
            axis,
            angle_deg=axis.angle_deg
            + math.degrees(
                kwargs["incidence_angle_delta_rad"] + kwargs["incidence_angle_trim_rad"]
            ),
        )
        m = _map(
            image.model.inputs,
            instrument,
            control,
            (corrected_goniometer_axis(shifted, args["initial"]),),
        )
        from sample_io import observation_id

        ids = tuple(observation_id(image.image_id, asdict(k)) for k in image.observations.keys)
        return PhysicalView(
            m, prediction.coordinates_px, ids, "frozen sample integer-L sites: " + image.image_id
        ), images
    from joint_io import _arguments

    from rasim_next.fitting.joint_geometry import (
        JointGeometryState,
        _absolute_instrument_and_model,
        _predict_image,
    )

    s = joint_session_from_document(data)
    cached = {} if prepared is None else prepared
    args = _arguments(s, control) if not cached else dict(cached["arguments"])
    args["initial"] = JointGeometryState.from_array(json.loads(s.controls_json)["initial"])
    captures = json.loads(s.captures_json)
    hbn = hbn_session_from_document(captures["hbn"]["session"])
    if image_id == str(hbn.acquisition_id) or parameter == "hbn_calibrant_distance_m":
        state = args["initial"]
        hbn = replace(
            hbn,
            initial=(
                state.detector_column_tilt_rad,
                state.detector_row_tilt_rad,
                state.beam_center_column_px,
                state.beam_center_row_px,
                state.hbn_calibrant_distance_m,
            ),
            frozen_json="",
            selected_result_id=None,
        )
        view, inputs = _preview(
            "hbn", hbn_session_document(hbn), image_id, control, cached.get("hbn")
        )
        return view, {"arguments": args, "hbn": inputs}
    choices = [(g, im) for g in ("bi2se3", "bi2te3") for im in args[g + "_images"]]
    group, image = next(((g, im) for g, im in choices if im.image_id == image_id), choices[0])
    _raw, errors, _origin = _predict_image(
        group, image, args["initial"], base_detector_rotation=args["base_detector_rotation"]
    )
    model, instrument, _origin = _absolute_instrument_and_model(
        group, image, args["initial"], base_detector_rotation=args["base_detector_rotation"]
    )
    from sample_io import observation_id

    ids = tuple(observation_id(image.image_id, asdict(k)) for k in image.observations.keys)
    from rasim_next.fitting.indexed_series import (
        SharedGeometryCorrections,
        corrected_goniometer_axis,
    )

    state = args["initial"]
    axis_corrections = SharedGeometryCorrections(
        0.0,
        0.0,
        0.0,
        0.0,
        state.goniometer_axis_pitch_rad,
        state.goniometer_axis_yaw_rad,
        0.0,
        state.goniometer_pivot_pitch_offset_m,
        state.goniometer_pivot_yaw_offset_m,
    )
    corrected_axis = corrected_goniometer_axis(
        model.inputs.config.instrument.axis_rotations[0], axis_corrections
    )
    return PhysicalView(
        _map(model.inputs, instrument, control, (corrected_axis,)),
        image.observations.coordinates_px + errors,
        ids,
        "frozen joint sites: " + group + "/" + image.image_id,
    ), {"arguments": args}


def center_proposal(route, data, result_id, image_id, control):
    if route == "hbn":
        from hbn_io import _inputs_current, validate_hbn_result

        s = hbn_session_from_document(data)
        record = next(
            json.loads(t) for t in s.results_json if json.loads(t)["result_id"] == result_id
        )
        validate_hbn_result(record, s)
        inp = json.loads(s.inputs_json)
        _inputs_current(inp, control)
        if record["launch"]["inputs"] != inp:
            raise ValueError("hBN result is historical for the current input identity")
        cal = record["calibration"]
        if not cal["success"]:
            raise ValueError("hBN center unavailable: saved calibration is unqualified")
        center = record["fitted_values"][2:4]
        quality = record["qualification"]
        uncertainty = {
            "type": "marginal standard errors, not full covariance",
            "center_px": cal["standard_error"][2:4],
        }
        pack = record["launch"]["pack"]
        observations = {
            "pack_id": pack["pack_id"],
            "sha256": pack["sha256"],
            "ring_sector_ids": list(zip(pack["ring_index"], pack["angular_sector"], strict=True)),
        }
        affected = [str(s.acquisition_id)]
        coupling = {n: v for n, v in zip(HBN_NAMES, record["fitted_values"], strict=True)}
    elif route == "sample":
        from sample_io import _fit_from_document, validate_sample_result

        from rasim_next.geometry.detector import project_detector_ray

        s = sample_session_from_document(data)
        record = next(
            json.loads(t) for t in s.results_json if json.loads(t)["result_id"] == result_id
        )
        validate_sample_result(record, s, control)
        if record["launch"]["inputs"] != json.loads(s.inputs_json) or record["fit"] is None:
            raise ValueError("Sample center unavailable: no matching admitted geometry result")
        fit = _fit_from_document(record["fit"])
        active = set(fit.fitted_detector_calibration_parameter_names)
        if (
            not {"detector_reference_column_offset_px", "detector_reference_row_offset_px"}
            <= active
            or not fit.success
        ):
            raise ValueError(
                "Sample center unavailable: inactive center calibration or deficient recorded rank/qualification"
            )
        launch = record["launch"]
        saved = replace(
            s,
            inputs_json=encoded(launch["inputs"]),
            controls_json=encoded(launch["controls"]),
            prepared_json=encoded(launch["prepared"]),
            exclusions=tuple(tuple(v) for v in launch["pack"]["review"]),
            frozen_json=encoded(launch["pack"]),
        )
        images = _images(saved, control)
        image = next((im for im in images if im.image_id == image_id), images[0])
        trims = dict(zip(fit.image_ids, fit.incidence_angle_trim_by_image_id_rad, strict=True))
        instrument = image.corrected_instrument(
            fit.corrections,
            detector_calibration_corrections=fit.detector_calibration_corrections,
            incidence_angle_delta_rad=fit.incidence_angle_delta_rad,
            incidence_angle_trim_rad=trims[image.image_id],
        )
        source = image.model.inputs.config.source
        projection = project_detector_ray(
            source.mean_origin_lab_m, source.mean_direction_lab, instrument
        )
        if (
            str(projection.status.value) not in ("VALID", "OUTSIDE_SUPPORT")
            or not np.all(np.isfinite([projection.column_px, projection.row_px]))
            or projection.ray_distance_m <= 0
        ):
            raise ValueError("Sample direct-beam intercept is invalid")
        center = [projection.column_px, projection.row_px]
        inp = launch["inputs"]
        quality = "unqualified initial estimate: standalone sample result has no dataset/parameter-precision/downstream qualification"
        uncertainty = {"type": "unavailable: no saved qualified full covariance"}
        observations = [
            p["observation_id"] for p in record["points"] if p["image_id"] == image.image_id
        ]
        affected = [image.image_id]
        coupling = {
            "corrections": asdict(fit.corrections),
            "calibration": asdict(fit.detector_calibration_corrections),
            "incidence_angle_delta_rad": fit.incidence_angle_delta_rad,
            "trim_rad": trims[image.image_id],
        }
    else:
        raise ValueError("Geometry-derived proposals use admitted hBN or supported sample results")
    seed_updates = None
    if route == "sample":
        seed_updates = {
            "detector_column_tilt_rad": fit.corrections.detector_column_tilt_rad,
            "detector_row_tilt_rad": fit.corrections.detector_row_tilt_rad,
            **asdict(fit.detector_calibration_corrections),
        }
    proposal = dict(
        method=route + " canonical geometric beam intercept",
        center_px=center,
        quality=quality,
        uncertainty=uncertainty,
        result_id=result_id,
        result_sha256=record["sha256"],
        inputs_sha256=payload_hash(inp),
        observation_ids=observations,
        affected_images=affected,
        coupling=coupling,
        sample_initial_updates=seed_updates,
        interpretation="initial estimate only; not an independent observation, ellipse center or detector reference pixel",
    )
    proposal["sha256"] = payload_hash(proposal)
    return proposal


def physical_work(argument, control):
    from threadpoolctl import threadpool_limits

    request = json.loads(argument)
    resources = request["resources"]
    if resources != sample_work_budget(resources["other_cpu_bytes"], resources["other_gpu_bytes"]):
        raise ValueError("Physical preview requires the shared finite CPU admission")
    route, data = request["route"], request["snapshot"]
    baseline_sha = payload_hash(data)
    _stop(control)
    with threadpool_limits(limits=1):
        if request["operation"] == "proposal":
            p = center_proposal(route, data, request["result_id"], request["image_id"], control)
            _stop(control)
            result = PhysicalResult(payload_hash(request), baseline_sha, proposal=p)
        else:
            name = request.get("parameter", "")
            rows = fields(route, data)
            held = tuple((f.name, f.value) for f in rows if f.name != name)
            candidates = [data]
            step = request.get("step")
            if request["operation"] == "sensitivity":
                f = next(f for f in rows if f.name == name)
                if (
                    not f.editable
                    or f.name
                    in (
                        "source.spatial_sigma_x",
                        "source.spatial_sigma_y",
                        "source.divergence_x",
                        "source.divergence_y",
                    )
                    or not math.isfinite(step)
                    or step <= 0
                ):
                    raise ValueError(
                        "Sensitivity needs a positive finite perturbation of an observed editable coordinate"
                    )
                candidates += [f.value + step, f.value - step]
            elif request.get("candidate") is not None:
                candidates += [request["candidate"]]
            views = []
            compiled = None
            for i, candidate in enumerate(candidates):
                _stop(control)
                control.report(
                    "Canonical physical preview " + str(i + 1) + "/" + str(len(candidates))
                )
                try:
                    if isinstance(candidate, (float, int)):
                        candidate = changed_snapshot(route, data, name, candidate)
                    view, compiled = _preview(
                        route, candidate, request["image_id"], control, compiled, name
                    )
                    if views and view.feature_ids != views[0].feature_ids:
                        raise ValueError("Frozen feature identity/topology changed")
                except (ValueError, OSError, RuntimeError) as exc:
                    _stop(control)
                    view = PhysicalView(None, np.empty((0, 2)), (), "invalid preview", str(exc))
                views.append(view)
            result = PhysicalResult(
                payload_hash(request),
                baseline_sha,
                tuple(views),
                held_fixed=held,
                parameter=name,
                step=step,
            )
    _stop(control)
    resident = result.nbytes
    if resident > 2 * 1024**2:
        raise ValueError("Physical preview exceeds its 2 MiB result budget")
    return JobResult(result, resident)
