"""Stream accepted angular panels using native-pixel coarse/fine indicators.

Indicators are empirical, not bounds on continuous integration error. Complete
support comes from the existing conditional Ewald bounds and their tail convention.
"""

from concurrent.futures import CancelledError
from dataclasses import replace

import numpy as np
from scipy.special import roots_legendre

from rasim_next.pipeline.fiber_detector import (
    FiberQuadratureNodes,
    _angular_cdf_panel_blocks,
    _angular_inverse_cdf,
    _angular_proposal_parameters,
    _resolved_angular_cdf_panels,
    _wrapped_cauchy_cdf,
    compile_conditional_fiber_batch,
)


def _split(panel, centers, widths):
    i, a, b, pa, pb = panel
    index = int(i)
    offsets = np.array(
        [_wrapped_cauchy_cdf(-c, w) for c, w in zip(centers[index], widths[index], strict=True)]
    )
    pm = pa + (pb - pa) / 2
    middle = _angular_inverse_cdf(pm, a, b, centers[index], widths[index], offsets, 0.2, 0.5)[0]
    if not a < middle < b or not pa < pm < pb:
        raise ValueError("pixel-error panel reached floating-point resolution")
    return np.array([i, a, middle, pa, pm]), np.array([i, middle, b, pm, pb])


def _bytes(value):
    return 0 if value is None else value[1].nbytes


def _flux(value):
    return 0.0 if value is None else float(value[1].sum())


def _difference_l1(coarse, left, right, available_bytes):
    """Tile a patch union, without materializing another detector-sized buffer."""
    selected = [(p, sign) for p, sign in ((left, 1), (right, 1), (coarse, -1)) if p is not None]
    if not selected:
        return 0.0
    r0 = min(p[0][0] for p, _ in selected)
    r1 = max(p[0][1] for p, _ in selected)
    c0 = min(p[0][2] for p, _ in selected)
    c1 = max(p[0][3] for p, _ in selected)
    rows = min(64, max(0, available_bytes // (8 * (c1 - c0))))
    if rows < 1:
        raise MemoryError("pixel-error patch budget exhausted before comparison")
    result = 0.0
    for first in range(r0, r1, rows):
        stop = min(first + rows, r1)
        tile = np.zeros((stop - first, c1 - c0))
        for (bounds, image), sign in selected:
            a, b = max(first, bounds[0]), min(stop, bounds[1])
            if a < b:
                source = image[a - bounds[0] : b - bounds[0]]
                target = tile[a - first : b - first, bounds[2] - c0 : bounds[3] - c0]
                if sign == 1:
                    target += source
                else:
                    target -= source
        np.abs(tile, out=tile)
        result += float(tile.sum())
    return result


def iter_pixel_error_batches(
    axial_rule,
    *,
    coordinate_parameters,
    batch_parameters,
    pixel_patch,
    absolute_budget,
    cancel_requested,
):
    """Prepare once, yield accepted transfers and their already deposited patches.

    W du masses enter the transfer once. Pixel contraction uses canonical S/W
    fractions. No retained evaluated events is an observed zero, not a proof of
    interval emptiness. Caps fail clearly rather than producing a qualified prefix.
    """
    controls = batch_parameters["rule"]
    axial = axial_rule.nodes_Ainv
    c = coordinate_parameters
    centers, widths = _angular_proposal_parameters(
        axial,
        c["radial_Ainv"],
        c["ki_sample_Ainv"],
        c["normal_sample"],
        c["reference_mosaic"].gaussian_sigma_rad,
        c["reference_mosaic"].lorentzian_half_width_rad,
    )
    bounds = np.array(c["source_region_bounds"], copy=True)
    panels = _resolved_angular_cdf_panels(
        np.hypot(c["radial_Ainv"], axial),
        bounds,
        centers,
        widths,
        c["angular_power"],
        controls.pixel_error_initial_width_rad,
        np.empty((0, 5)),
        controls.maximum_angular_panel_nodes,
        np.linalg.norm(c["ki_sample_Ainv"]),
    )
    nodes, weights = roots_legendre(8)
    evaluated = 0
    flux, error = 0.0, 0.0
    for initial in zip(*panels, strict=True):
        stack = [(np.asarray(initial), 0, None)]
        while stack:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            panel, depth, cached = stack.pop()
            pending = sum(_bytes(item[2][1]) for item in stack if item[2] is not None)
            evaluated_panels = []
            child_panels = _split(panel, centers, widths)
            requests = [panel, *child_panels] if cached is None else list(child_panels)
            resident = pending + (0 if cached is None else _bytes(cached[1]))
            if cancel_requested is not None and cancel_requested():
                raise CancelledError
            if evaluated + 8 * len(requests) > controls.maximum_angular_panel_nodes:
                raise ValueError("pixel-error evaluated-node budget exceeded")
            evaluated += 8 * len(requests)
            values = np.column_stack(requests)
            args = (values[0].astype(np.int64), *values[1:])
            index, phi, angular_mass = next(
                _angular_cdf_panel_blocks(args, centers, widths, nodes, weights, 8 * len(requests))
            )
            quadrature = FiberQuadratureNodes(
                axial,
                index,
                phi,
                None,
                axial_rule.strength_weighted_mass[index] * angular_mass,
                axial_rule.signed_fractions,
            )
            batch = compile_conditional_fiber_batch(quadrature, **batch_parameters)
            evaluated_panels = pixel_patch(
                batch,
                panel_count=len(requests),
                maximum_bytes=controls.pixel_error_maximum_bytes - resident,
            )
            resident += sum(_bytes(patch) for _, patch in evaluated_panels)
            coarse, left, right = (
                evaluated_panels if cached is None else [cached, *evaluated_panels]
            )
            delta = _difference_l1(
                coarse[1], left[1], right[1], controls.pixel_error_maximum_bytes - resident
            )
            total = _flux(left[1]) + _flux(right[1])
            if not np.all(np.isfinite([delta, total, flux + total, error + delta])):
                raise FloatingPointError("nonfinite pixel-error indicator")
            if error + delta <= controls.pixel_error_rtol * (flux + total) + absolute_budget:
                flux += total
                error += delta
                remaining_error = delta
                for batch, patch in (left, right):
                    if batch is not None:
                        yield replace(
                            batch, native_pixel_patch=patch, angular_error_indicator=remaining_error
                        )
                        remaining_error = 0.0
            else:
                if depth >= controls.pixel_error_maximum_depth:
                    raise ValueError("pixel-error maximum refinement depth exceeded")
                stack.extend(
                    ((child_panels[1], depth + 1, right), (child_panels[0], depth + 1, left))
                )
            # Release obsolete trial patches before evaluating another panel.
            coarse = left = right = cached = None
            evaluated_panels = []
            batch = patch = None
