#include "cif_guides.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GUIDE_POINTS 250000
#define GUIDE_PATHS 20000

int cif_rod_angles(double wavelength, OscMount m, double rho, double z, OscAngle angles[2]) {
    double k, s, c, x, y2, tolerance, transverse, along, up, side;
    int branch, count;
    if (!osc_mount_valid(m) || !isfinite(wavelength) || wavelength <= 0 || !isfinite(rho) ||
        rho < 0 || !isfinite(z))
        return 0;
    k = 2 * HBN_PI / wavelength;
    s = sin(m.incidence);
    c = cos(m.incidence);
    x = (2 * k * s * z - rho * rho - z * z) / (2 * k * c);
    tolerance = 256 * DBL_EPSILON * fmax(k * k, rho * rho + z * z);
    if (rho == 0) {
        if (z == 0 || fabs(z - 2 * k * s) > 256 * DBL_EPSILON * k)
            return 0;
        x = 0;
    }
    y2 = rho * rho - x * x;
    if (y2 < -tolerance)
        return 0;
    /* Only roundoff-negative radicands are clamped; two positive roots stay distinct. */
    y2 = fmax(0, y2);
    along = 1 + (x * c - z * s) / k;
    up = (x * s + z * c) / k;
    side = sqrt(y2) / k;
    transverse = hypot(up, side);
    if (transverse <= 64 * DBL_EPSILON && along > 0)
        return 0;
    count = y2 == 0 ? 1 : 2;
    for (branch = 0; branch < count; ++branch) {
        angles[branch].theta = atan2(transverse, along);
        angles[branch].phi = osc_wrap_phi(m.normal_phi + atan2(branch ? -side : side, up));
        angles[branch].azimuth_valid = transverse > 64 * DBL_EPSILON;
    }
    return count;
}
void cif_guides_free(CifGuides *g) {
    free(g->points);
    free(g->paths);
    memset(g, 0, sizeof *g);
}
typedef struct {
    const OscGeometry *geometry;
    const CifGuideView *view;
    OscMount mount;
    CifGuides cache;
    int point_capacity, path_capacity, kind, branch;
    double theta, rho, low, high;
} Builder;

static CifGuidePoint evaluate(const Builder *b, double u) {
    CifGuidePoint p = {0};
    if (b->kind == CIF_POWDER) {
        p.theta = b->theta;
        p.phi = osc_wrap_phi(-HBN_PI + 2 * HBN_PI * u);
        p.angle_valid = 1;
    } else {
        OscAngle a[2];
        double z = b->low + (b->high - b->low) * (1 - cos(HBN_PI * u)) / 2;
        int n = cif_rod_angles(b->geometry->settings.wavelength_A, b->mount, b->rho, z, a);
        if (n) {
            p.theta = a[n == 1 ? 0 : b->branch].theta;
            p.phi = a[n == 1 ? 0 : b->branch].phi;
            p.angle_valid = a[0].azimuth_valid;
        }
    }
    if (p.angle_valid)
        p.detector_valid = osc_angle_pixel(b->geometry, p.theta, p.phi, &p.column, &p.row);
    return p;
}
static int append(Builder *b, CifGuidePoint p) {
    if (b->cache.point_count == GUIDE_POINTS)
        return 0;
    if (b->cache.point_count == b->point_capacity) {
        int capacity = b->point_capacity ? b->point_capacity * 2 : 4096;
        CifGuidePoint *points;
        if (capacity > GUIDE_POINTS)
            capacity = GUIDE_POINTS;
        points = (CifGuidePoint *)realloc(b->cache.points, capacity * sizeof *points);
        if (!points)
            return 0;
        b->cache.points = points;
        b->point_capacity = capacity;
    }
    b->cache.points[b->cache.point_count++] = p;
    return 1;
}
static int overlaps(double a, double b, double c, double low, double high) {
    return fmax(a, fmax(b, c)) >= low && fmin(a, fmin(b, c)) <= high;
}
static int segment(Builder *b, double u0, CifGuidePoint p, double u1, CifGuidePoint q, int depth) {
    double um = (u0 + u1) / 2;
    CifGuidePoint m = evaluate(b, um);
    int refine = depth < 4;
    if (p.angle_valid && q.angle_valid && m.angle_valid) {
        double phi_end = p.phi + osc_wrap_phi(q.phi - p.phi);
        double phi_mid = p.phi + osc_wrap_phi(m.phi - p.phi);
        double theta_tolerance = 0.001, phi_tolerance = 0.001;
        const CifGuideView *v = b->view;
        int shift;
        if (v->theta_tolerance > 0 &&
            overlaps(p.theta, q.theta, m.theta, v->theta_min - .002, v->theta_max + .002))
            for (shift = -2; shift <= 2; ++shift) {
                double offset = shift * 2 * HBN_PI;
                if (overlaps(p.phi + offset, phi_mid + offset, phi_end + offset, v->phi_min - .002,
                             v->phi_max + .002)) {
                    theta_tolerance = fmin(theta_tolerance, v->theta_tolerance);
                    phi_tolerance = fmin(phi_tolerance, v->phi_tolerance);
                }
            }
        refine |= fabs(m.theta - (p.theta + q.theta) / 2) > theta_tolerance ||
                  fabs(phi_mid - (p.phi + phi_end) / 2) > phi_tolerance ||
                  fabs(phi_end - p.phi) > HBN_PI / 18;
        if (p.detector_valid && q.detector_valid && m.detector_valid && v->pixel_tolerance > 0 &&
            overlaps(p.column, q.column, m.column, v->column_min - 2, v->column_max + 2) &&
            overlaps(p.row, q.row, m.row, v->row_min - 2, v->row_max + 2))
            refine |= hypot(m.column - (p.column + q.column) / 2, m.row - (p.row + q.row) / 2) >
                      v->pixel_tolerance;
    }
    /* Validity transitions get a bounded refinement; paths break at invalid points. */
    if (depth < 12 && (p.angle_valid != q.angle_valid || p.angle_valid != m.angle_valid ||
                       p.detector_valid != q.detector_valid))
        refine = 1;
    if (refine) {
        if (depth == 18)
            return 0;
        return segment(b, u0, p, um, m, depth + 1) && segment(b, um, m, u1, q, depth + 1);
    }
    return append(b, q);
}
static int start_path(Builder *b, int kind, int group) {
    CifGuidePath *path;
    if (b->cache.path_count == GUIDE_PATHS)
        return 0;
    if (b->cache.path_count == b->path_capacity) {
        int capacity = b->path_capacity ? b->path_capacity * 2 : 128;
        if (capacity > GUIDE_PATHS)
            capacity = GUIDE_PATHS;
        path = (CifGuidePath *)realloc(b->cache.paths, capacity * sizeof *path);
        if (!path)
            return 0;
        b->cache.paths = path;
        b->path_capacity = capacity;
    }
    path = &b->cache.paths[b->cache.path_count++];
    *path = (CifGuidePath){kind, group, b->cache.point_count, 0};
    return 1;
}
static int curve(Builder *b, int kind, int group) {
    CifGuidePoint p, q;
    if (!start_path(b, kind, group))
        return 0;
    b->kind = kind;
    p = evaluate(b, 0);
    q = evaluate(b, 1);
    if (!append(b, p) || !segment(b, 0, p, 1, q, 0))
        return 0;
    b->cache.paths[b->cache.path_count - 1].count =
        b->cache.point_count - b->cache.paths[b->cache.path_count - 1].first;
    return 1;
}
int cif_guides_build(const CifPeaks *peaks, const OscGeometry *geometry, OscMount mount, int rods,
                     const CifGuideView *view, CifGuides *output, char error[256]) {
    Builder b = {0};
    double k = 2 * HBN_PI / peaks->wavelength_A;
    int kind, group;
    if ((rods && !osc_mount_valid(mount)) ||
        fabs(peaks->wavelength_A - geometry->settings.wavelength_A) > 1e-10 * peaks->wavelength_A) {
        strcpy(error,
               "Check wavelength and fiber mounting: |incidence| < 89, normal phi in +/-180 deg.");
        return 0;
    }
    b.geometry = geometry;
    b.view = view;
    b.mount = mount;
    for (kind = CIF_POWDER; kind <= (rods ? CIF_TICK : CIF_POWDER); ++kind) {
        const CifGrouping *set = &peaks->grouping[kind];
        for (group = 0; group < set->count; ++group) {
            const CifPeak *p = &peaks->peaks[set->members[set->groups[group].first]];
            b.theta = p->two_theta_deg * HBN_PI / 180;
            b.rho = p->qr_invA;
            if (kind == CIF_POWDER) {
                if (!curve(&b, kind, group))
                    goto failed;
            } else if (kind == CIF_ROD && b.rho > 0) {
                double center = k * sin(mount.incidence),
                       spread = 2 * k * cos(mount.incidence) * b.rho;
                double lower = center * center - b.rho * b.rho - spread;
                double upper = center * center - b.rho * b.rho + spread;
                int interval;
                if (upper < 0)
                    continue;
                for (interval = 0; interval < (lower > 0 ? 2 : 1); ++interval) {
                    b.low = center + (interval ? sqrt(lower) : -sqrt(upper));
                    b.high = center + (lower > 0 && !interval ? -sqrt(lower) : sqrt(upper));
                    for (b.branch = 0; b.branch < 2; ++b.branch)
                        if (!curve(&b, kind, group))
                            goto failed;
                }
            } else if (kind == CIF_TICK || (kind == CIF_ROD && b.rho == 0)) {
                OscAngle angles[2];
                double z = kind == CIF_ROD ? 2 * k * sin(mount.incidence) : p->qz_invA;
                int i, n = cif_rod_angles(peaks->wavelength_A, mount, p->qr_invA, z, angles);
                for (i = 0; i < n; ++i) {
                    CifGuidePoint point = {
                        angles[i].theta, angles[i].phi, 0, 0, angles[i].azimuth_valid, 0};
                    point.detector_valid = osc_angle_pixel(geometry, point.theta, point.phi,
                                                           &point.column, &point.row);
                    if (!start_path(&b, kind, group) || !append(&b, point))
                        goto failed;
                    b.cache.paths[b.cache.path_count - 1].count = 1;
                }
            }
        }
    }
    cif_guides_free(output);
    *output = b.cache;
    error[0] = 0;
    return 1;
failed:
    cif_guides_free(&b.cache);
    strcpy(error, "Guide display limit reached (250000 points / 20000 paths / curvature). Widen "
                  "view or reduce max 2theta.");
    return 0;
}
