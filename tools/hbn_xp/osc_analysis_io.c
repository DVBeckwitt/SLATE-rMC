#include "osc_analysis.h"
#include <math.h>
#include <string.h>

int osc_geometry_write(FILE *f, const OscGeometry *g) {
    const HbnSettings *s = &g->settings;
    fprintf(f, "SLATE_ANALYSIS_GEOMETRY,1\n");
    fprintf(f,
            "pitch_column_m,%.17g\npitch_row_m,%.17g\nwavelength_A,%.17g\n"
            "lattice_a_A,%.17g\nlattice_c_A,%.17g\n",
            s->pitch_column_m, s->pitch_row_m, s->wavelength_A, s->lattice_a_A, s->lattice_c_A);
    fprintf(f,
            "column_tilt_rad,%.17g\nrow_tilt_rad,%.17g\ncenter_column_px,%.17g\n"
            "center_row_px,%.17g\nsample_distance_m,%.17g\n",
            s->initial[0], s->initial[1], s->initial[2], s->initial[3], s->initial[4]);
    fprintf(f,
            "rows,%d\ncolumns,%d\nfrom_qualified_hbn,%d\ndistance_from_calibrant,%d\n"
            "calibrant_crc32,%lu\ndark_crc32,%lu\n",
            g->rows, g->columns, g->from_qualified_hbn, g->distance_from_calibrant,
            g->calibrant_crc32, g->dark_crc32);
    return !ferror(f);
}
int osc_geometry_read(const char *path, OscGeometry *output, char error[256]) {
    const char *keys[] = {"pitch_column_m",
                          "pitch_row_m",
                          "wavelength_A",
                          "lattice_a_A",
                          "lattice_c_A",
                          "column_tilt_rad",
                          "row_tilt_rad",
                          "center_column_px",
                          "center_row_px",
                          "sample_distance_m",
                          "rows",
                          "columns",
                          "from_qualified_hbn",
                          "distance_from_calibrant",
                          "calibrant_crc32",
                          "dark_crc32"};
    double v[16];
    unsigned seen = 0;
    char line[256], key[80], tail;
    FILE *f = fopen(path, "r");
    int i, failed = 0;
    OscGeometry g = {0};
    if (!f) {
        strcpy(error, "Cannot open saved analysis geometry.");
        return 0;
    }
    if (!fgets(line, sizeof line, f) || strcmp(line, "SLATE_ANALYSIS_GEOMETRY,1\n"))
        failed = 1;
    while (!failed && fgets(line, sizeof line, f)) {
        double value;
        if (sscanf(line, "%79[^,],%lf %c", key, &value, &tail) != 2 || !isfinite(value)) {
            failed = 1;
            break;
        }
        for (i = 0; i < 16; ++i)
            if (!strcmp(keys[i], key))
                break;
        if (i == 16 || (seen & (1u << i))) {
            failed = 1;
            break;
        }
        seen |= 1u << i;
        v[i] = value;
    }
    failed = failed || ferror(f) || !feof(f) || seen != 65535u;
    fclose(f);
    if (failed)
        goto invalid;
    for (i = 10; i < 16; ++i)
        if (v[i] != floor(v[i]) || v[i] < 0 || v[i] > 4294967295.0)
            goto invalid;
    if (v[10] > 16384 || v[11] > 16384 || v[12] > 1 || v[13] > v[12])
        goto invalid;
    g.settings.pitch_column_m = v[0];
    g.settings.pitch_row_m = v[1];
    g.settings.wavelength_A = v[2];
    g.settings.lattice_a_A = v[3];
    g.settings.lattice_c_A = v[4];
    for (i = 0; i < 5; ++i)
        g.settings.initial[i] = v[i + 5];
    g.rows = (int)v[10];
    g.columns = (int)v[11];
    g.from_qualified_hbn = (int)v[12];
    g.distance_from_calibrant = (int)v[13];
    g.calibrant_crc32 = (unsigned long)v[14];
    g.dark_crc32 = (unsigned long)v[15];
    if (!osc_geometry_compile(&g, error))
        return 0;
    *output = g;
    return 1;
invalid:
    strcpy(error, "Invalid analysis geometry file: expected version 1 and each numeric key once.");
    return 0;
}
static void bin_row(FILE *f, const char *kind, double t0, double t1, double p0, double p1,
                    double signal, double area, double panel) {
    double deg = 180 / HBN_PI;
    fprintf(f, "%s,%.12g,%.12g,%.12g,%.12g,%.17g,%.17g,%.17g,", kind, t0 * deg, t1 * deg, p0 * deg,
            p1 * deg, signal, area, panel);
    if (area > 0)
        fprintf(f, "%.17g", signal / area);
    fputc(',', f);
    if (panel > 0)
        fprintf(f, "%.17g", area / panel);
    fputc('\n', f);
}
int osc_integration_write(FILE *f, const HbnImage *im, int subtract, const OscGeometry *g,
                          const OscIntegration *r, unsigned long mask_crc, HbnProgress progress,
                          void *context) {
    const OscGrid *b = &r->grid;
    double dt = (b->theta_max - b->theta_min) / b->theta_bins,
           dp = (b->phi_max - b->phi_min) / b->phi_bins;
    int t, p;
    fprintf(f, "# SLATE signed angular integration v1; canonical scattering phi, not motor phi\n"
               "# physical-corner TL-BR triangles; pole fans; no cropped-support renormalization\n"
               "# signal=sum(weight*count); area=sum(weight); mean=signal/area after reduction\n"
               "# area is effective detector pixels, not solid angle or complete-ring percentage\n"
               "# blank mean/fraction means no support; negative measurements retained\n"
               "# No solid-angle, polarization, background or exposure correction; dark scale=1\n");
    fprintf(f, "# source_crc32=%08lx dark_crc32=%08lx mask_crc32=%08lx values=%s\n",
            im->source_crc32, subtract ? im->dark_crc32 : 0ul, mask_crc,
            subtract ? "raw-dark" : "raw");
    fprintf(f,
            "# full_input_signal=%.17g masked_signal=%.17g masked_pixels=%.17g "
            "outside_window_signal=%.17g outside_window_area=%.17g\n",
            r->input_signal, r->masked_signal, r->masked_area, r->lost_signal, r->lost_area);
    osc_geometry_write(f, g);
    fprintf(
        f, "kind,two_theta_lower_deg,two_theta_upper_deg,phi_lower_deg,phi_upper_deg,"
           "signal_counts,valid_pixel_area,panel_pixel_area,mean_counts,valid_fraction_of_panel\n");
    for (p = 0; p < b->phi_bins; ++p) {
        if (progress && progress(context, "Exporting angular map and profiles"))
            return 0;
        for (t = 0; t < b->theta_bins; ++t) {
            size_t i = (size_t)p * b->theta_bins + t;
            bin_row(f, "map", b->theta_min + t * dt, b->theta_min + (t + 1) * dt,
                    b->phi_min + p * dp, b->phi_min + (p + 1) * dp, r->signal[i], r->area[i],
                    r->panel_area[i]);
        }
    }
    for (t = 0; t < b->theta_bins; ++t) {
        double s = 0, n = 0, a = 0;
        if (progress && progress(context, "Exporting radial profile"))
            return 0;
        for (p = 0; p < b->phi_bins; ++p) {
            size_t i = (size_t)p * b->theta_bins + t;
            s += r->signal[i];
            n += r->area[i];
            a += r->panel_area[i];
        }
        bin_row(f, "I_vs_2theta", b->theta_min + t * dt, b->theta_min + (t + 1) * dt, b->phi_min,
                b->phi_max, s, n, a);
    }
    for (p = 0; p < b->phi_bins; ++p) {
        double s = 0, n = 0, a = 0;
        if (progress && progress(context, "Exporting azimuthal profile"))
            return 0;
        for (t = 0; t < b->theta_bins; ++t) {
            size_t i = (size_t)p * b->theta_bins + t;
            s += r->signal[i];
            n += r->area[i];
            a += r->panel_area[i];
        }
        bin_row(f, "I_vs_phi", b->theta_min, b->theta_max, b->phi_min + p * dp,
                b->phi_min + (p + 1) * dp, s, n, a);
    }
    return !ferror(f);
}
