#include "osc_analysis.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double osc_wrap_phi(double phi) {
    return phi - 2 * HBN_PI * floor((phi + HBN_PI) / (2 * HBN_PI));
}
int osc_geometry_compile(OscGeometry *g, char error[256]) {
    const HbnSettings *s = &g->settings;
    int i;
    if (g->rows < 2 || g->columns < 2 || g->rows > 16384 || g->columns > 16384 ||
        (double)g->rows * g->columns > 25000000 || !isfinite(s->pitch_column_m) ||
        !isfinite(s->pitch_row_m) || s->pitch_column_m <= 0 || s->pitch_row_m <= 0 ||
        !isfinite(s->wavelength_A) || s->wavelength_A <= 0 || !isfinite(s->lattice_a_A) ||
        s->lattice_a_A <= 0 || !isfinite(s->lattice_c_A) || s->lattice_c_A <= 0)
        goto invalid;
    for (i = 0; i < 5; ++i)
        if (!isfinite(s->initial[i]))
            goto invalid;
    if (fabs(s->initial[0]) > .15 || fabs(s->initial[1]) > .15 || s->initial[2] < 0 ||
        s->initial[2] > g->columns - 1 || s->initial[3] < 0 || s->initial[3] > g->rows - 1 ||
        s->initial[4] <= 0 || s->initial[4] > 10)
        goto invalid;
    hbn_detector_axes(s->initial, g->beam, g->column, g->row);
    return 1;
invalid:
    strcpy(error, "Invalid analysis geometry: positive pitches/wavelength/distance (at most 10 m), "
                  "on-panel center, tilts within +/-8.594 deg, admitted image dimensions.");
    return 0;
}
OscAngle osc_pixel_angle(const OscGeometry *g, double c, double r) {
    const HbnSettings *s = &g->settings;
    double dc = (c - s->initial[2]) * s->pitch_column_m;
    double dr = (r - s->initial[3]) * s->pitch_row_m;
    /* Dot products with transverse axes remove D*beam exactly at the pole. */
    double x = dc * g->column[0] + dr * g->column[1];
    double y = dc * g->row[0] + dr * g->row[1];
    double z = s->initial[4] + dc * g->beam[0] + dr * g->beam[1];
    double transverse = hypot(x, y);
    OscAngle a;
    a.theta = atan2(transverse, z);
    a.azimuth_valid = transverse > 32 * DBL_EPSILON * hypot(transverse, z);
    a.phi = a.azimuth_valid ? osc_wrap_phi(-HBN_PI / 2 - atan2(y, x)) : 0;
    return a;
}
int osc_angle_pixel(const OscGeometry *g, double theta, double phi, double *c, double *r) {
    const HbnSettings *s = &g->settings;
    double ray[3], chi = -HBN_PI / 2 - phi, length;
    int j;
    for (j = 0; j < 3; ++j)
        ray[j] =
            cos(theta) * g->beam[j] + sin(theta) * (cos(chi) * g->column[j] + sin(chi) * g->row[j]);
    if (ray[2] <= 1e-12)
        return 0;
    length = s->initial[4] * g->beam[2] / ray[2];
    *c = s->initial[2] + (length * ray[0] - s->initial[4] * g->beam[0]) / s->pitch_column_m;
    *r = s->initial[3] + (length * ray[1] - s->initial[4] * g->beam[1]) / s->pitch_row_m;
    return isfinite(*c) && isfinite(*r);
}
int osc_grid_valid(const OscGrid *g, char error[256]) {
    if (!isfinite(g->theta_min) || !isfinite(g->theta_max) || !isfinite(g->phi_min) ||
        !isfinite(g->phi_max) || g->theta_min < 0 || g->theta_max > HBN_PI ||
        g->theta_max <= g->theta_min || g->phi_min < -HBN_PI || g->phi_min >= HBN_PI ||
        g->phi_max <= g->phi_min || g->phi_max - g->phi_min > 2 * HBN_PI + 1e-12 ||
        g->theta_bins < 1 || g->phi_bins < 1 || g->theta_bins > 4096 || g->phi_bins > 1440 ||
        (g->theta_max - g->theta_min) / g->theta_bins < 1e-7 ||
        (g->phi_max - g->phi_min) / g->phi_bins < 1e-7 ||
        (double)g->theta_bins * g->phi_bins > 1048576) {
        strcpy(error, "Choose increasing 2theta limits in 0..180 deg, a phi span up to 360 deg, "
                      "and at most 4096 x 1440 bins / 1,048,576 total bins.");
        return 0;
    }
    return 1;
}
int osc_is_masked(const unsigned char *mask, size_t index) {
    return mask && ((mask[index >> 3] >> (index & 7)) & 1);
}
void osc_mask_box(unsigned char *mask, int columns, int rows, int c0, int r0, int c1, int r1,
                  int exclude) {
    int c, r, swap;
    if (c1 < c0) {
        swap = c0;
        c0 = c1;
        c1 = swap;
    }
    if (r1 < r0) {
        swap = r0;
        r0 = r1;
        r1 = swap;
    }
    for (r = r0 < 0 ? 0 : r0; r <= r1 && r < rows; ++r)
        for (c = c0 < 0 ? 0 : c0; c <= c1 && c < columns; ++c) {
            size_t index = (size_t)r * columns + c;
            unsigned char bit = (unsigned char)(1u << (index & 7));
            if (exclude)
                mask[index >> 3] |= bit;
            else
                mask[index >> 3] &= (unsigned char)~bit;
        }
}

typedef struct {
    double x, y;
} Vertex;
typedef struct {
    int n;
    Vertex p[16];
} Polygon;
static double polygon_area(const Polygon *p) {
    double area = 0;
    int i;
    /* Translation avoids cancellation between large absolute angle products. */
    for (i = 1; i + 1 < p->n; ++i)
        area += (p->p[i].x - p->p[0].x) * (p->p[i + 1].y - p->p[0].y) -
                (p->p[i].y - p->p[0].y) * (p->p[i + 1].x - p->p[0].x);
    return fabs(area) * .5;
}
static void clip(Polygon *p, int axis, double edge, int greater) {
    Polygon q = {0};
    int i;
    Vertex a;
    double av;
    if (!p->n)
        return;
    a = p->p[p->n - 1];
    av = axis ? a.y : a.x;
    for (i = 0; i < p->n; ++i) {
        Vertex b = p->p[i];
        double bv = axis ? b.y : b.x;
        int ai = greater ? av >= edge : av <= edge, bi = greater ? bv >= edge : bv <= edge;
        if (ai != bi) {
            double f = (edge - av) / (bv - av);
            Vertex v = {a.x + f * (b.x - a.x), a.y + f * (b.y - a.y)};
            if (axis)
                v.y = edge;
            else
                v.x = edge;
            q.p[q.n++] = v;
        }
        if (bi)
            q.p[q.n++] = b;
        a = b;
        av = bv;
    }
    *p = q;
}
static int pieces(const OscGeometry *g, int c, int r, const OscAngle a[4], Polygon p[4]) {
    double pc = g->settings.initial[2], pr = g->settings.initial[3];
    double tol = 128 * DBL_EPSILON * fmax(g->columns, g->rows);
    int i, n = 0;
    memset(p, 0, 4 * sizeof *p);
    if (fabs(pc - c) <= .5 + tol && fabs(pr - r) <= .5 + tol) {
        const double cx[] = {-.5, .5, .5, -.5}, cy[] = {-.5, -.5, .5, .5};
        if (fabs(pc - c + .5) <= tol)
            pc = c - .5;
        if (fabs(pc - c - .5) <= tol)
            pc = c + .5;
        if (fabs(pr - r + .5) <= tol)
            pr = r - .5;
        if (fabs(pr - r - .5) <= tol)
            pr = r + .5;
        for (i = 0; i < 4; ++i) {
            int j = (i + 1) % 4;
            double area =
                fabs((c + cx[i] - pc) * (r + cy[j] - pr) - (r + cy[i] - pr) * (c + cx[j] - pc));
            double first, following;
            if (area <= 128 * DBL_EPSILON)
                continue;
            if (!a[i].azimuth_valid || !a[j].azimuth_valid)
                return 0;
            first = osc_wrap_phi(-HBN_PI / 2 - a[i].phi);
            following = first + osc_wrap_phi(osc_wrap_phi(-HBN_PI / 2 - a[j].phi) - first);
            first = -HBN_PI / 2 - first;
            following = -HBN_PI / 2 - following;
            p[n].n = 4;
            p[n].p[0] = (Vertex){0, first};
            p[n].p[1] = (Vertex){a[i].theta, first};
            p[n].p[2] = (Vertex){a[j].theta, following};
            p[n].p[3] = (Vertex){0, following};
            if (polygon_area(p + n) > 0)
                ++n;
        }
    } else {
        const int indices[2][3] = {{0, 1, 2}, {0, 2, 3}};
        for (i = 0; i < 2; ++i) {
            int j;
            p[n].n = 3;
            for (j = 0; j < 3; ++j) {
                int k = indices[i][j];
                if (!a[k].azimuth_valid)
                    return 0;
                p[n].p[j].x = a[k].theta;
                double chi = osc_wrap_phi(-HBN_PI / 2 - a[k].phi);
                if (j)
                    chi = -HBN_PI / 2 - p[n].p[j - 1].y +
                          osc_wrap_phi(chi - osc_wrap_phi(-HBN_PI / 2 - a[indices[i][j - 1]].phi));
                p[n].p[j].y = -HBN_PI / 2 - chi;
            }
            if (polygon_area(p + n) > 0)
                ++n;
        }
    }
    return n;
}
static double deposit(const Polygon *piece, double full, double value, int masked,
                      OscIntegration *result) {
    const OscGrid *g = &result->grid;
    double dt = (g->theta_max - g->theta_min) / g->theta_bins;
    double dp = (g->phi_max - g->phi_min) / g->phi_bins;
    double xmin = piece->p[0].x, xmax = xmin, ymin = piece->p[0].y, ymax = ymin, retained = 0;
    int i, t0, t1, period;
    for (i = 1; i < piece->n; ++i) {
        xmin = fmin(xmin, piece->p[i].x);
        xmax = fmax(xmax, piece->p[i].x);
        ymin = fmin(ymin, piece->p[i].y);
        ymax = fmax(ymax, piece->p[i].y);
    }
    t0 = (int)fmax(0, floor((xmin - g->theta_min) / dt));
    t1 = (int)fmin(g->theta_bins, ceil((xmax - g->theta_min) / dt));
    if (t1 <= t0)
        return 0;
    for (period = -2; period <= 2; ++period) {
        double lo = g->phi_min + period * 2 * HBN_PI;
        int p0 = (int)fmax(0, floor((ymin - lo) / dp));
        int p1 = (int)fmin(g->phi_bins, ceil((ymax - lo) / dp)), p, t;
        for (p = p0; p < p1; ++p)
            for (t = t0; t < t1; ++t) {
                Polygon clipped = *piece;
                double weight;
                size_t index = (size_t)p * g->theta_bins + t;
                clip(&clipped, 0, g->theta_min + t * dt, 1);
                clip(&clipped, 0, g->theta_min + (t + 1) * dt, 0);
                clip(&clipped, 1, lo + p * dp, 1);
                clip(&clipped, 1, lo + (p + 1) * dp, 0);
                weight = polygon_area(&clipped) / full;
                result->panel_area[index] += weight;
                if (!masked) {
                    result->signal[index] += weight * value;
                    result->area[index] += weight;
                }
                retained += weight;
            }
    }
    return retained;
}
void osc_integration_free(OscIntegration *r) {
    free(r->signal);
    free(r->area);
    free(r->panel_area);
    memset(r, 0, sizeof *r);
}
int osc_integrate(const HbnImage *im, int subtract, const unsigned char *mask, const OscGeometry *g,
                  const OscGrid *grid, OscIntegration *output, HbnProgress progress, void *context,
                  char error[256]) {
    OscIntegration result = {0};
    OscAngle *top = NULL, *bottom = NULL, *swap;
    size_t bins;
    int r, c;
    if (!osc_grid_valid(grid, error))
        return 0;
    if (!im->counts || im->rows != g->rows || im->columns != g->columns ||
        (subtract && !im->dark_counts)) {
        strcpy(error, "Applied geometry/image dimensions or dark state do not match.");
        return 0;
    }
    result.grid = *grid;
    bins = (size_t)grid->theta_bins * grid->phi_bins;
    result.signal = (double *)calloc(bins, sizeof(double));
    result.area = (double *)calloc(bins, sizeof(double));
    result.panel_area = (double *)calloc(bins, sizeof(double));
    top = (OscAngle *)malloc((im->columns + 1) * sizeof *top);
    bottom = (OscAngle *)malloc((im->columns + 1) * sizeof *bottom);
    if (!result.signal || !result.area || !result.panel_area || !top || !bottom) {
        strcpy(error, "Not enough memory for angular bins. Increase the bin widths.");
        goto fail;
    }
    for (c = 0; c <= im->columns; ++c)
        top[c] = osc_pixel_angle(g, c - .5, -.5);
    for (r = 0; r < im->rows; ++r) {
        if (progress && progress(context, "Integrating original pixels into angular bins")) {
            strcpy(error, "Canceled.");
            goto fail;
        }
        for (c = 0; c <= im->columns; ++c)
            bottom[c] = osc_pixel_angle(g, c - .5, r + .5);
        for (c = 0; c < im->columns; ++c) {
            OscAngle corners[4] = {top[c], top[c + 1], bottom[c + 1], bottom[c]};
            Polygon polygons[4];
            size_t index = (size_t)r * im->columns + c;
            int count = pieces(g, c, r, corners, polygons), i, masked = osc_is_masked(mask, index);
            double full = 0, retained = 0,
                   value = im->counts[index] - (subtract ? im->dark_counts[index] : 0);
            for (i = 0; i < count; ++i)
                full += polygon_area(polygons + i);
            if (!isfinite(full) || full <= 0) {
                strcpy(error, "A pixel has invalid angular support.");
                goto fail;
            }
            for (i = 0; i < count; ++i)
                retained += deposit(polygons + i, full, value, masked, &result);
            if (!isfinite(retained) || retained < 0 || retained > 1 + 3e-11) {
                strcpy(error, "Angular splitting failed count conservation.");
                goto fail;
            }
            result.input_signal += value;
            if (masked) {
                result.masked_signal += value;
                result.masked_area += 1;
            } else {
                double lost = fmax(0, 1 - retained);
                result.lost_signal += lost * value;
                result.lost_area += lost;
            }
        }
        swap = top;
        top = bottom;
        bottom = swap;
    }
    free(top);
    free(bottom);
    osc_integration_free(output);
    *output = result;
    return 1;
fail:
    free(top);
    free(bottom);
    osc_integration_free(&result);
    return 0;
}

unsigned long osc_mask_crc32(const unsigned char *mask, size_t pixels) {
    unsigned long crc = 0xfffffffful;
    size_t i;
    for (i = 0; i < (pixels + 7) / 8; ++i) {
        int j;
        crc ^= mask ? mask[i] : 0;
        for (j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320ul : 0);
    }
    return crc ^ 0xfffffffful;
}
