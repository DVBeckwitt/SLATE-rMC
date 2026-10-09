"""Bounded nominal radial-family geometry for the Simulator detector overlay."""

import json
from dataclasses import dataclass

import numpy as np
from archive_storage import storage_files
from job_lifecycle import JobResult
from native_simulation_state import native_draft_from_document
from simulation_io import _stop, canonical_configuration
from simulation_state import simulation_draft_from_document

MAX_GUIDE_POINTS = 32768
MAX_GUIDE_BYTES = 2 * 1024**2
GUIDE_QR_TOLERANCE_AINV = 0.002


@dataclass(frozen=True, slots=True)
class RodGuides:
    draft: object
    detector_shape_rc: tuple[int, int]
    curves: tuple[tuple[str, np.ndarray], ...]
    families_json: str
    reference_wavelength_A: float
    maximum_vertex_error_Ainv: float

    @property
    def nbytes(self):
        return sum(p.nbytes for _, p in self.curves) + len(self.families_json.encode()) + 192 * 1024


def guide_context(draft, storage=()):
    from rasim_next.pipeline.configured_simulation import (
        GeometryOnlyEwaldContext,
        build_configured_geometry_inputs,
        build_geometry_only_ewald_context,
    )

    if hasattr(draft, "yaml_text"):
        inputs = build_configured_geometry_inputs(canonical_configuration(draft, storage))
        return (
            build_geometry_only_ewald_context(inputs),
            inputs.crystal,
            inputs.config.source.mean_wavelength_A,
        )
    from native_simulation_io import canonical_native

    from rasim_next.geometry.transport import build_incident_states
    from rasim_next.materials import material_optics
    from rasim_next.sampling.source import sample_nominal_mean_geometry_source_ray

    bound = canonical_native(draft)[0]
    source = bound.source_definition
    wavelength = float(np.dot(source.line_wavelength_A, source.line_probability))
    samples = sample_nominal_mean_geometry_source_ray(
        mean_origin_lab_m=source.mean_origin_lab_m,
        mean_direction_lab=source.mean_direction_lab,
        transverse_axes_lab=source.transverse_axes_lab,
        spatial_sigma_m=source.spatial_sigma_m,
        divergence_sigma_rad=source.divergence_sigma_rad,
        reference_wavelength_A=wavelength,
        polarization_state_id=source.polarization_state_id,
        position_divergence_correlation=source.position_divergence_correlation,
    )
    crystal = bound.structure.crystals[0]
    from rasim_next.reciprocal.lattice import ReciprocalLattice

    if not np.allclose(
        ReciprocalLattice.from_crystal(crystal).basis_Ainv,
        bound.reciprocal_basis_Ainv,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("native common rod catalog differs from its bound crystal basis")
    material = material_optics(crystal, samples.wavelength_A)
    incident = build_incident_states(samples, material, bound.instrument)
    context = GeometryOnlyEwaldContext(
        bound.reciprocal_basis_Ainv,
        bound.crystal_to_sample,
        bound.rods,
        incident,
        material,
        bound.instrument,
    )
    return context, crystal, wavelength


def radial_coordinates(context, lattice, column, row):
    geometry = context.evaluate_detector_geometry(column, row, include_surface_jacobian=False)
    q_crystal = context.instrument.sample_from_crystal.inverse().apply_vector(
        geometry.q_sample_Ainv
    )
    hkl = np.linalg.solve(lattice.basis_Ainv, q_crystal.reshape(-1, 3).T).T
    radius = lattice.qr_Ainv(hkl[:, :2]).reshape(geometry.valid.shape)
    return radius, geometry.valid


def family_catalog(context, crystal):
    from rasim_next.reciprocal.rods import build_rod_catalog

    rods = context.rods
    if len(rods) > 2048:
        raise ValueError("nominal guide catalog exceeds the 2048-rod display cap")
    h = [r.h for r in rods]
    k = [r.k for r in rods]
    if (max(h) - min(h) + 1) * (max(k) - min(k) + 1) > 16384:
        raise ValueError("guide catalog bounding rectangle exceeds 16384 entries")
    catalog = build_rod_catalog(crystal, h_bounds=(min(h), max(h)), k_bounds=(min(k), max(k)))
    members = {(r.h, r.k) for r in rods}
    grouped = {}
    for index, (hi, ki) in enumerate(zip(catalog.h, catalog.k, strict=True)):
        pair = int(hi), int(ki)
        if pair not in members:
            continue
        key = catalog.family_id[index]
        record = grouped.setdefault(
            key,
            {
                "family_id": key,
                "family_key": catalog.family_key[index],
                "phase_id": catalog.phase_id[index],
                "qr_Ainv": float(catalog.qr_Ainv[index]),
                "members_hk": [],
            },
        )
        record["members_hk"].append(pair)
    families = sorted(grouped.values(), key=lambda f: (f["qr_Ainv"], f["family_id"]))
    for ordinal, family in enumerate(families):
        family["label"] = f"r{ordinal}"
        family["members_hk"].sort()
    return families


def build_rod_guides(draft, control, storage=()):
    from matplotlib.figure import Figure

    from rasim_next.reciprocal.lattice import ReciprocalLattice

    _stop(control)
    context, crystal, wavelength = guide_context(draft, storage)
    lattice = ReciprocalLattice.from_crystal(crystal)
    families = family_catalog(context, crystal)
    levels = []
    labels = []
    for family in families:
        radius = family["qr_Ainv"]
        if radius == 0:
            continue
        if levels and np.isclose(levels[-1], radius, rtol=0, atol=1e-12):
            labels[-1] += "/" + family["label"]
        else:
            levels.append(radius)
            labels.append(family["label"])
    rows, columns = context.instrument.detector_shape_rc
    curves = []
    maximum_error = 0.0
    for count in (129, 257):
        _stop(control)
        control.report(
            f"Preparing nominal Qz-family geometry ({count}x{count}); no intensity calculation"
        )
        x = np.linspace(-0.5, columns - 0.5, count)
        y = np.linspace(-0.5, rows - 0.5, count)
        xx, yy = np.meshgrid(x, y)
        radius, valid = radial_coordinates(context, lattice, xx, yy)
        if not np.any(valid):
            raise ValueError("nominal ray has no valid top-exit detector geometry")
        figure = Figure()
        contours = figure.add_subplot().contour(
            x, y, np.ma.array(radius, mask=~valid), levels=levels, corner_mask=False
        )
        curves = [
            (label, np.array(line, copy=True))
            for label, segments in zip(labels, contours.allsegs, strict=True)
            for line in segments
            if len(line) > 1
        ]
        figure.clear()
        if sum(len(p) for _, p in curves) > MAX_GUIDE_POINTS:
            raise ValueError("nominal guides exceed the 32768 control-point display cap")
        maximum_error = 0.0
        for label, points in curves:
            _stop(control)
            exact, point_valid = radial_coordinates(context, lattice, points[:, 0], points[:, 1])
            midpoint = (points[:-1] + points[1:]) / 2
            midpoint_radius, midpoint_valid = radial_coordinates(
                context, lattice, midpoint[:, 0], midpoint[:, 1]
            )
            if not np.all(point_valid) or not np.all(midpoint_valid):
                raise ValueError(
                    "guide contour crosses invalid geometry; no disconnected branch is joined"
                )
            target = levels[labels.index(label)]
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(exact - target))),
                float(np.max(np.abs(midpoint_radius - target))),
            )
        if maximum_error <= GUIDE_QR_TOLERANCE_AINV:
            break
    if maximum_error > GUIDE_QR_TOLERANCE_AINV:
        raise ValueError("nominal guide grid cannot meet its 0.002 angstrom^-1 display tolerance")
    for rod in context.rods:
        if rod.h == rod.k == 0:
            mapped = context.map_latent_geometry(
                rod=rod, branch=0, alpha_rad=np.array([0.0]), beta_rad=np.array([0.0])
            )
            if bool(mapped.valid[0]):
                zero = next(f["label"] for f in families if f["qr_Ainv"] == 0)
                curves.append((zero, np.array([[mapped.column_px[0], mapped.row_px[0]]])))
            break
    for _, points in curves:
        points.setflags(write=False)
    result = RodGuides(
        draft,
        (rows, columns),
        tuple(curves),
        json.dumps(families, sort_keys=True),
        wavelength,
        maximum_error,
    )
    if result.nbytes > MAX_GUIDE_BYTES:
        raise ValueError("nominal guide output exceeds its 2 MiB bound")
    _stop(control)
    return result


def prepare_rod_guides(argument, control):
    request = json.loads(argument)
    draft = (
        native_draft_from_document(request["draft"])
        if request["kind"] == "native"
        else simulation_draft_from_document(request["draft"])
    )
    guides = build_rod_guides(draft, control, storage_files(request.get("storage_json", "{}")))
    return JobResult(guides, guides.nbytes)
