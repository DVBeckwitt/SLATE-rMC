#include "osc_profile.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

int osc_mount_valid(OscMount m) {
    return isfinite(m.incidence) && fabs(m.incidence) < 89 * HBN_PI / 180 &&
           isfinite(m.normal_phi) && fabs(m.normal_phi) <= HBN_PI;
}
static void sample_q(double k, double sm, double cm, double a, double st, double cp, double sp,
                     double *qr, double *qz) {
    *qz = k * (sm * a + cm * st * cp);
    *qr = k * hypot(-cm * a + sm * st * cp, st * sp);
}
void osc_angle_q(double wavelength, OscMount m, double theta, double phi, double *qr, double *qz) {
    double half = sin(theta / 2), delta = phi - m.normal_phi;
    sample_q(2 * HBN_PI / wavelength, sin(m.incidence), cos(m.incidence), 2 * half * half,
             sin(theta), cos(delta), sin(delta), qr, qz);
}
void osc_profile_free(OscProfile *p) {
    free(p->signal);
    free(p->area);
    free(p->panel_area);
    memset(p, 0, sizeof *p);
}
int osc_profile_build(const OscIntegration *r, int axis, int q_bins, double wavelength,
                      OscMount mount, OscProfile *output, char error[256]) {
    OscProfile p = {0};
    const OscGrid *g = &r->grid;
    double *trig = NULL, k = 0, sm = 0, cm = 0, dt, dp;
    int t, phi, pass, reciprocal = axis >= PROFILE_QZ;
    size_t i, cells;
    if (axis < PROFILE_THETA || axis > PROFILE_QR || !r->signal || !r->area || !r->panel_area ||
        !osc_grid_valid(g, error)) {
        strcpy(error, "Integrate a valid angular region before plotting a profile.");
        return 0;
    }
    if (reciprocal && (!osc_mount_valid(mount) || !isfinite(wavelength) || wavelength <= 0 ||
                       !isfinite(4 * HBN_PI / wavelength) || q_bins < 1 || q_bins > 4096)) {
        strcpy(error, "Q profiles need a valid sample orientation, wavelength and 1..4096 bins.");
        return 0;
    }
    cells = (size_t)g->theta_bins * g->phi_bins;
    for (i = 0; i < cells; ++i)
        if (!isfinite(r->signal[i]) || !isfinite(r->area[i]) || !isfinite(r->panel_area[i]) ||
            r->area[i] < 0 || r->panel_area[i] < 0 ||
            (r->panel_area[i] == 0 && (r->area[i] != 0 || r->signal[i] != 0))) {
            strcpy(error, "Angular map has invalid signal or support.");
            return 0;
        }
    p.axis = axis;
    p.bins = reciprocal ? q_bins : (axis == PROFILE_THETA ? g->theta_bins : g->phi_bins);
    p.source_grid = *g;
    p.wavelength_A = wavelength;
    p.mount = mount;
    p.signal = (double *)calloc(p.bins, sizeof(double));
    p.area = (double *)calloc(p.bins, sizeof(double));
    p.panel_area = (double *)calloc(p.bins, sizeof(double));
    if (!p.signal || !p.area || !p.panel_area)
        goto memory;
    dt = (g->theta_max - g->theta_min) / g->theta_bins;
    dp = (g->phi_max - g->phi_min) / g->phi_bins;
    if (!reciprocal) {
        p.lower = axis == PROFILE_THETA ? g->theta_min : g->phi_min;
        p.upper = axis == PROFILE_THETA ? g->theta_max : g->phi_max;
        for (phi = 0; phi < g->phi_bins; ++phi)
            for (t = 0; t < g->theta_bins; ++t) {
                int bin = axis == PROFILE_THETA ? t : phi;
                i = (size_t)phi * g->theta_bins + t;
                p.signal[bin] += r->signal[i];
                p.area[bin] += r->area[i];
                p.panel_area[bin] += r->panel_area[i];
            }
    } else {
        /* Trig scales with grid axes, not with the product or number of redraws. */
        trig = (double *)malloc((size_t)2 * (g->theta_bins + g->phi_bins) * sizeof(double));
        if (!trig)
            goto memory;
        k = 2 * HBN_PI / wavelength;
        sm = sin(mount.incidence);
        cm = cos(mount.incidence);
        for (t = 0; t < g->theta_bins; ++t) {
            double theta = g->theta_min + (t + .5) * dt, half = sin(theta / 2);
            trig[2 * t] = 2 * half * half;
            trig[2 * t + 1] = sin(theta);
        }
        for (phi = 0; phi < g->phi_bins; ++phi) {
            double delta = g->phi_min + (phi + .5) * dp - mount.normal_phi;
            trig[2 * (g->theta_bins + phi)] = cos(delta);
            trig[2 * (g->theta_bins + phi) + 1] = sin(delta);
        }
        p.lower = DBL_MAX;
        p.upper = -DBL_MAX;
        for (pass = 0; pass < 2; ++pass) {
            for (phi = 0; phi < g->phi_bins; ++phi)
                for (t = 0; t < g->theta_bins; ++t) {
                    double qr, qz, q;
                    int bin;
                    i = (size_t)phi * g->theta_bins + t;
                    if (r->panel_area[i] == 0)
                        continue;
                    sample_q(k, sm, cm, trig[2 * t], trig[2 * t + 1],
                             trig[2 * (g->theta_bins + phi)], trig[2 * (g->theta_bins + phi) + 1],
                             &qr, &qz);
                    if (!isfinite(qr) || !isfinite(qz)) {
                        strcpy(error, "Q coordinates exceed the finite range. Check wavelength.");
                        goto fail;
                    }
                    q = axis == PROFILE_QZ ? qz : qr;
                    if (!pass) {
                        p.lower = fmin(p.lower, q);
                        p.upper = fmax(p.upper, q);
                        continue;
                    }
                    bin = q == p.upper ? p.bins - 1
                                       : (int)floor((q - p.lower) / (p.upper - p.lower) * p.bins);
                    /* The final edge is inclusive; division may also round a point
                       immediately below it to bins. Never absorb an outside point. */
                    if (bin == p.bins && q >= p.lower && q <= p.upper)
                        bin = p.bins - 1;
                    if (bin < 0 || bin >= p.bins) {
                        strcpy(error, "Q coordinate fell outside its computed profile range.");
                        goto fail;
                    }
                    p.signal[bin] += r->signal[i];
                    p.area[bin] += r->area[i];
                    p.panel_area[bin] += r->panel_area[i];
                }
            if (!pass) {
                if (p.lower == DBL_MAX) {
                    strcpy(error, "This region has no detector support for a Q profile.");
                    goto fail;
                }
                if (p.upper == p.lower) {
                    double width = fmax(1e-12, k * 1e-9);
                    p.lower -= width / 2;
                    if (axis == PROFILE_QR)
                        p.lower = fmax(0, p.lower);
                    p.upper = p.lower + width;
                }
                {
                    double step = (p.upper - p.lower) / p.bins, previous = p.lower;
                    int edge;
                    if (!isfinite(p.lower) || !isfinite(p.upper) || !isfinite(step) || step <= 0)
                        goto range;
                    for (edge = 1; edge <= p.bins; ++edge) {
                        double next = p.lower + edge * step;
                        if (!isfinite(next) || next <= previous)
                            goto range;
                        previous = next;
                    }
                }
            }
        }
    }
    free(trig);
    osc_profile_free(output);
    *output = p;
    return 1;
range:
    strcpy(error, "Q range cannot resolve this many finite bin edges. Use fewer Q bins.");
    goto fail;
memory:
    strcpy(error, "Not enough memory for the 1D profile.");
fail:
    free(trig);
    osc_profile_free(&p);
    return 0;
}
int osc_profile_write(FILE *f, const OscProfile *p) {
    const char *names[] = {"2theta", "phi", "Qz", "Qr"};
    double scale = p->axis < PROFILE_QZ ? 180 / HBN_PI : 1;
    double step = (p->upper - p->lower) / p->bins;
    int i;
    fprintf(
        f, "# axis=%s,unit=%s,method=%s\n", names[p->axis], p->axis < PROFILE_QZ ? "deg" : "A^-1",
        p->axis < PROFILE_QZ ? "angular-map marginal" : "angular-bin-center rebin; approximate");
    fprintf(
        f,
        "# S and N reduced separately; mean=S/N; signed counts retained; no Q-density Jacobian\n"
        "# source_2theta_rad=%.17g,%.17g,%d;source_phi_rad=%.17g,%.17g,%d\n",
        p->source_grid.theta_min, p->source_grid.theta_max, p->source_grid.theta_bins,
        p->source_grid.phi_min, p->source_grid.phi_max, p->source_grid.phi_bins);
    if (p->axis >= PROFILE_QZ)
        fprintf(f,
                "# external-air sample Q; no refraction; auto range includes all panel-supported "
                "centers\n"
                "# wavelength_A=%.17g,incidence_deg=%.17g,normal_phi_deg=%.17g\n"
                "# Refine the source angular grid to assess center-rebin artifacts; Q bins do not "
                "restore subcell data\n",
                p->wavelength_A, p->mount.incidence * 180 / HBN_PI,
                p->mount.normal_phi * 180 / HBN_PI);
    fprintf(f, "x_lower,x_upper,x_center,signal_counts,valid_pixel_area,panel_pixel_area,mean_"
               "counts,valid_fraction_of_panel\n");
    for (i = 0; i < p->bins; ++i) {
        fprintf(f, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,", (p->lower + i * step) * scale,
                (p->lower + (i + 1) * step) * scale, (p->lower + (i + .5) * step) * scale,
                p->signal[i], p->area[i], p->panel_area[i]);
        if (p->area[i] > 0)
            fprintf(f, "%.17g", p->signal[i] / p->area[i]);
        fputc(',', f);
        if (p->panel_area[i] > 0)
            fprintf(f, "%.17g", p->area[i] / p->panel_area[i]);
        fputc('\n', f);
    }
    return !ferror(f);
}
