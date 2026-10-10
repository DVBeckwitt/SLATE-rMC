#include "hbn.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void hbn_defaults(HbnSettings *s) {
    s->pitch_column_m = s->pitch_row_m = 0.0001;
    s->wavelength_A = 1.5406;
    s->lattice_a_A = 2.504;
    s->lattice_c_A = 6.661;
    s->initial[0] = s->initial[1] = 0;
    s->initial[2] = 1453.12;
    s->initial[3] = 1596.422;
    s->initial[4] = 0.074;
}

int hbn_read_settings(const char *path, HbnSettings *s, char error[256]) {
    const char *keys[] = {"column_pitch_m", "row_pitch_m",     "wavelength_A", "lattice_a_A",
                          "lattice_c_A",    "column_tilt_deg", "row_tilt_deg", "center_column_px",
                          "center_row_px",  "distance_m"};
    double values[10];
    unsigned seen = 0;
    char line[256], key[80], tail;
    FILE *f = fopen(path, "r");
    int i, parse_failed = 0;
    if (!f) {
        snprintf(error, 256, "Cannot open settings: %.180s", path);
        return 0;
    }
    while (fgets(line, sizeof line, f)) {
        double value;
        char *p = line;
        while (*p == ' ' || *p == '\t')
            ++p;
        if (*p == '#' || *p == '\n' || *p == '\r' || !*p)
            continue;
        if (sscanf(p, "%79[^=]=%lf %c", key, &value, &tail) != 2 || !isfinite(value)) {
            parse_failed = 1;
            break;
        }
        for (i = 0; i < 10; ++i)
            if (!strcmp(key, keys[i]))
                break;
        if (i == 10 || (seen & (1u << i))) {
            parse_failed = 1;
            break;
        }
        seen |= 1u << i;
        values[i] = value;
    }
    i = parse_failed || !feof(f) || ferror(f) || seen != 1023u;
    fclose(f);
    if (i) {
        strcpy(error, "Settings require each documented key once, with finite numeric values.");
        return 0;
    }
    s->pitch_column_m = values[0];
    s->pitch_row_m = values[1];
    s->wavelength_A = values[2];
    s->lattice_a_A = values[3];
    s->lattice_c_A = values[4];
    for (i = 0; i < 5; ++i)
        s->initial[i] = values[i + 5];
    s->initial[0] *= HBN_PI / 180;
    s->initial[1] *= HBN_PI / 180;
    return 1;
}

static uint32_t word32(const unsigned char *p, int big) {
    if (big)
        return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
    return ((uint32_t)p[3] << 24) | ((uint32_t)p[2] << 16) | ((uint32_t)p[1] << 8) | p[0];
}
static uint32_t crc_bytes(uint32_t crc, const unsigned char *p, size_t n,
                          const uint32_t table[256]) {
    size_t i;
    for (i = 0; i < n; ++i)
        crc = (crc >> 8) ^ table[(crc ^ p[i]) & 255];
    return crc;
}

int osc_load(const char *path, HbnImage *image, HbnProgress progress, void *context,
             char error[256]) {
    FILE *f = NULL;
    unsigned char header[6000], *row = NULL;
    uint32_t width, height, crc = 0xffffffffu, table[256];
    int big, r, c, i;
    HbnImage next;
    memset(&next, 0, sizeof next);
    for (i = 0; i < 256; ++i) {
        uint32_t v = (uint32_t)i;
        int j;
        for (j = 0; j < 8; ++j)
            v = (v >> 1) ^ ((v & 1) ? 0xedb88320u : 0);
        table[i] = v;
    }
    f = fopen(path, "rb");
    if (!f) {
        snprintf(error, 256, "Cannot open OSC: %.180s", path);
        return 0;
    }
    if (fread(header, 1, sizeof header, f) != sizeof header || memcmp(header, "RAXIS", 5)) {
        strcpy(error, "Expected an uncompressed R-AXIS .osc file with a 6000-byte header.");
        goto fail;
    }
    big = word32(header + 796, 1) < 20;
    width = word32(header + 768, big);
    height = word32(header + 772, big);
    if (!width || !height || width > 16384 || height > 16384 || (double)width * height > 25000000) {
        strcpy(error, "OSC dimensions exceed the 25-million-pixel / 16384-axis admission limit.");
        goto fail;
    }
    next.rows = (int)width;
    next.columns = (int)height;
    next.counts = (int *)malloc((size_t)width * height * sizeof(int));
    /* A small raw-row strip makes the clockwise native writes contiguous.
       At 3000 columns this uses 192 kB, without a second full image. */
    row = (unsigned char *)malloc((size_t)32 * 2 * width);
    if (!next.counts || !row) {
        strcpy(error, "Not enough memory to open this image.");
        goto fail;
    }
    crc = crc_bytes(crc, header, sizeof header, table);
    for (r = 0; r < (int)height; r += 32) {
        int count = (int)height - r, j;
        if (count > 32)
            count = 32;
        for (j = 0; j < count; ++j) {
            unsigned char *raw = row + (size_t)j * 2 * width;
            if (progress && progress(context, "Reading OSC")) {
                strcpy(error, "Canceled.");
                goto fail;
            }
            if (fread(raw, 2, width, f) != width) {
                strcpy(error, "Truncated OSC pixel data.");
                goto fail;
            }
            crc = crc_bytes(crc, raw, 2 * width, table);
        }
        for (c = 0; c < (int)width; ++c) {
            for (j = 0; j < count; ++j) {
                const unsigned char *raw = row + (size_t)j * 2 * width + 2 * c;
                int v = big ? (raw[0] * 256 + raw[1]) : (raw[1] * 256 + raw[0]);
                if (v >= 32768)
                    v = (v - 32768) * 32;
                next.counts[(size_t)c * height + (height - 1 - (unsigned)(r + j))] = v;
            }
        }
    }
    if (fgetc(f) != EOF || ferror(f)) {
        strcpy(error, "OSC file length does not match its header.");
        goto fail;
    }
    next.source_crc32 = (unsigned long)(crc ^ 0xffffffffu);
    free(row);
    fclose(f);
    hbn_free(image);
    *image = next;
    return 1;
fail:
    free(row);
    fclose(f);
    hbn_free(&next);
    return 0;
}

int osc_load_dark(const char *path, HbnImage *image, HbnProgress progress, void *context,
                  char error[256]) {
    HbnImage dark;
    memset(&dark, 0, sizeof dark);
    if (!image->counts) {
        strcpy(error, "Open a measurement image first.");
        return 0;
    }
    if (!osc_load(path, &dark, progress, context, error))
        return 0;
    if (dark.rows != image->rows || dark.columns != image->columns) {
        hbn_free(&dark);
        strcpy(error, "Dark and measurement dimensions differ.");
        return 0;
    }
    free(image->dark_counts);
    image->dark_counts = dark.counts;
    image->dark_crc32 = dark.source_crc32;
    return 1;
}

int hbn_load(const char *source, const char *dark, HbnImage *image, HbnProgress progress,
             void *context, char error[256]) {
    HbnImage next;
    memset(&next, 0, sizeof next);
    if (!osc_load(source, &next, progress, context, error))
        return 0;
    if (!osc_load_dark(dark, &next, progress, context, error)) {
        hbn_free(&next);
        return 0;
    }
    hbn_free(image);
    *image = next;
    return 1;
}
void hbn_free(HbnImage *image) {
    free(image->counts);
    free(image->dark_counts);
    memset(image, 0, sizeof *image);
}

int hbn_report(FILE *f, const HbnSettings *s, const HbnImage *im, const HbnResult *r) {
    const char *names[] = {"column_tilt_rad", "row_tilt_rad", "center_column_px", "center_row_px",
                           "calibrant_distance_m"};
    int i;
    fprintf(f, "# SLATE OSC / hBN standalone v1\n# Coordinates: native clockwise import, "
               "zero-based pixel centers (column,row)\n");
    fprintf(f, "# Tilt convention: base @ Rx(column) @ Ry(row); base=[[1,0,0],[0,0,1],[0,-1,0]], "
               "beam_lab=(0,1,0)\n");
    fprintf(f, "# Distance is calibrant-private along beam. Absolute detector roll is fixed, not "
               "measured.\n");
    fprintf(f, "# Detection uses log1p(max(raw-dark,0)); raw signed measurements are preserved. "
               "Dark scale=1, no exposure normalization.\n");
    fprintf(f, "# Fit is bounded damped Gauss-Newton, soft_l1 scale=1. Conditional standard errors "
               "are not physical confidence intervals.\n");
    fprintf(f, "source_crc32,%08lx\ndark_crc32,%08lx\nrows,%d\ncolumns,%d\n", im->source_crc32,
            im->dark_crc32, im->rows, im->columns);
    fprintf(f,
            "column_pitch_m,%.17g\nrow_pitch_m,%.17g\nwavelength_A,%.17g\nlattice_a_A,%."
            "17g\nlattice_c_A,%.17g\n",
            s->pitch_column_m, s->pitch_row_m, s->wavelength_A, s->lattice_a_A, s->lattice_c_A);
    fprintf(f,
            "qualified,%d\nsolver_converged,%d\nrank,%d\nscaled_condition,%.17g\nrms_px,%.17g\nmax_"
            "px,%.17g\niterations,%d\n",
            r->qualified, r->solver_success, r->rank, r->condition, r->rms, r->maximum,
            r->iterations);
    for (i = 0; i < 5; ++i)
        fprintf(f, "%s,%.17g\nstandard_error_%s,%.17g\ninitial_%s,%.17g\nbound_%s,%d\n", names[i],
                r->values[i], names[i], r->standard_error[i], names[i], s->initial[i], names[i],
                r->bound[i]);
    fprintf(f, "column_tilt_deg,%.12g\nrow_tilt_deg,%.12g\n", r->values[0] * 180 / HBN_PI,
            r->values[1] * 180 / HBN_PI);
    fprintf(f, "ring,count,coverage,rms_px\n");
    for (i = 0; i < 5; ++i)
        fprintf(f, "%d,%d,%.17g,%.17g\n", i, r->count[i], r->coverage[i], r->ring_rms[i]);
    fprintf(f, "ring,sector,column_px,row_px,residual_px\n");
    for (i = 0; i < r->point_count; ++i)
        fprintf(f, "%d,%d,%.17g,%.17g,%.17g\n", r->points[i].ring, r->points[i].sector,
                r->points[i].column, r->points[i].row, r->residual[i]);
    return !ferror(f);
}
