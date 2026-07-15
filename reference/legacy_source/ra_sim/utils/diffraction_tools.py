"""Diffraction and CIF helper utilities."""

from __future__ import annotations

import io as pyio
import math
from contextlib import redirect_stdout

import numpy as np

from ra_sim.utils.calculations import d_spacing, two_theta


DEFAULT_PIXEL_SIZE_M = 100e-6


def write_cif_file(cf, output_path) -> None:
    """Write one current PyCifRW document."""

    with open(output_path, "w", encoding="utf-8") as handle:
        with redirect_stdout(pyio.StringIO()):
            handle.write(cf.WriteOut())


def detector_two_theta_max(
    image_size: int,
    center,
    detector_distance: float,
    pixel_size: float = DEFAULT_PIXEL_SIZE_M,
) -> float:
    """Estimate the largest 2θ captured by the detector plane."""

    if image_size is None or image_size <= 0:
        return 180.0
    if not math.isfinite(detector_distance) or detector_distance <= 0:
        return 180.0
    if not math.isfinite(pixel_size) or pixel_size <= 0:
        pixel_size = DEFAULT_PIXEL_SIZE_M

    try:
        centre_row = float(center[0])
        centre_col = float(center[1])
    except (TypeError, ValueError, IndexError):
        centre_row = (image_size - 1) / 2.0
        centre_col = (image_size - 1) / 2.0

    if not math.isfinite(centre_row) or not math.isfinite(centre_col):
        centre_row = (image_size - 1) / 2.0
        centre_col = (image_size - 1) / 2.0

    rows = (0.0, image_size - 1.0)
    cols = (0.0, image_size - 1.0)
    max_radius = 0.0
    for row in rows:
        for col in cols:
            dx = (col - centre_col) * pixel_size
            dy = (centre_row - row) * pixel_size
            radius = math.hypot(dx, dy)
            if radius > max_radius:
                max_radius = radius

    return math.degrees(math.atan2(max_radius, detector_distance))


def _prepare_temp_cif(cif_path: str, occ) -> str:
    """Return path to a temporary CIF with updated occupancies."""
    import os
    import tempfile

    import CifFile

    abs_path = os.path.abspath(cif_path)
    with redirect_stdout(pyio.StringIO()):
        cf = CifFile.ReadCif(abs_path)
    block_names = list(cf.keys())
    if not block_names:
        raise ValueError(f"CIF contains no data blocks: {abs_path}")
    block = cf[block_names[0]]
    occ_field = block.get("_atom_site_occupancy")
    if occ_field is None:
        labels = block.get("_atom_site_label")
        if labels is None:
            raise ValueError(f"CIF contains no atom sites: {abs_path}")
        label_values = [labels] if isinstance(labels, str) else list(labels)
        occ_values = [1.0] * len(label_values)
    else:
        raw_occ_values = [occ_field] if isinstance(occ_field, str) else list(occ_field)
        occ_values = []
        for raw_value in raw_occ_values:
            text = str(raw_value).strip().strip("'\"")
            if "(" in text and text.endswith(")"):
                text = text[: text.index("(")]
            try:
                value = float(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CIF atom occupancy: {raw_value!r}") from exc
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("CIF atom occupancies must be finite and within [0, 1].")
            occ_values.append(value)

    n_sites = len(occ_values)
    if n_sites <= 0:
        raise ValueError(f"CIF contains no atom sites: {abs_path}")
    if not isinstance(occ, (list, tuple, np.ndarray)):
        raise TypeError("occupancies must be a numeric sequence")
    try:
        factors = [float(value) for value in occ]
    except (TypeError, ValueError) as exc:
        raise ValueError("occupancies must be numeric") from exc
    if len(factors) == 1:
        factors *= n_sites
    elif len(factors) != n_sites:
        raise ValueError(f"occupancies require one value or exactly {n_sites} site values")
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in factors):
        raise ValueError("occupancies must be finite and within [0, 1]")
    block["_atom_site_occupancy"] = [
        str(base_occupancy * factor)
        for base_occupancy, factor in zip(occ_values, factors, strict=True)
    ]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
    tmp.close()
    tmp_path = tmp.name
    try:
        write_cif_file(cf, tmp_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return tmp_path


def miller_generator(
    mx,
    cif_file,
    occ,
    lambda_,
    energy=8.047,
    intensity_threshold=1.0,
    two_theta_range=(0, 70),
):
    """Generate filtered Miller indices and normalized intensities."""
    import os

    import Dans_Diffraction as dif

    raw_miller = [
        (h, k, l)
        for h in range(-mx + 1, mx)
        for k in range(-mx + 1, mx)
        for l in range(1, mx)
    ]

    tmp_cif = _prepare_temp_cif(cif_file, occ)
    try:
        xtl = dif.Crystal(tmp_cif)
    finally:
        try:
            os.unlink(tmp_cif)
        except FileNotFoundError:
            pass
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    xtl.Scatter.setup_scatter(scattering_type="xray", energy_kev=energy)
    xtl.Scatter.integer_hkl = True

    kept = []
    for h, k, l in raw_miller:
        d = d_spacing(h, k, l, xtl.Cell.a, xtl.Cell.c)
        tth = two_theta(d, lambda_)
        if tth is None or not (two_theta_range[0] <= tth <= two_theta_range[1]):
            continue
        intensity_val = xtl.Scatter.intensity([h, k, l])
        try:
            intensity_val = float(
                np.asarray(intensity_val, dtype=np.float64).reshape(-1)[0]
            )
        except Exception:
            continue
        if intensity_val < intensity_threshold:
            continue
        kept.append(((h, k, l), float(intensity_val)))

    if not kept:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int32),
            [],
        )

    max_intensity = max(item[1] for item in kept)
    scale = 100.0 / max_intensity if max_intensity > 0 else 0.0

    miller_arr = np.array([item[0] for item in kept], dtype=np.int32)
    intensities_arr = np.array(
        [round(item[1] * scale, 2) for item in kept],
        dtype=np.float64,
    )
    degeneracy_arr = np.ones(len(kept), dtype=np.int32)
    normalized_details = [
        [(item[0], round(item[1] * scale, 2))]
        for item in kept
    ]
    return miller_arr, intensities_arr, degeneracy_arr, normalized_details


def inject_fractional_reflections(miller, intensities, mx, step=0.5, value=0.1):
    """Add fractional Miller indices with constant intensity."""

    offsets = np.array([-step, step])
    candidates = []
    for h, k, l in miller:
        for dl in offsets:
            nl = l + dl
            if (
                -mx + 1 <= h < mx
                and -mx + 1 <= k < mx
                and 1 <= nl < mx
                and not abs(nl - round(nl)) < 1e-8
            ):
                candidates.append((h, k, nl))

    if not candidates:
        return miller.astype(float), intensities

    uniq = np.unique(np.array(candidates, dtype=float), axis=0)
    frac_intens = np.full(len(uniq), value, dtype=float)
    miller_new = np.vstack((miller.astype(float), uniq))
    intensities_new = np.concatenate((intensities, frac_intens))
    return miller_new, intensities_new


