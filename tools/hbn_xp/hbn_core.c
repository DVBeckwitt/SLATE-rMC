#include "hbn.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double clamp(double x, double a, double b) {
    return x < a ? a : (x > b ? b : x);
}
static double angle(const HbnSettings *s, int ring) {
    const int h[] = {0, 1, 1, 1, 0}, l[] = {2, 0, 1, 2, 4};
    double inverse_d2 = 4.0 * h[ring] * h[ring] / (3 * s->lattice_a_A * s->lattice_a_A) +
                        (double)l[ring] * l[ring] / (s->lattice_c_A * s->lattice_c_A);
    return 2 * asin(0.5 * s->wavelength_A * sqrt(inverse_d2));
}
static void beam(const double x[5], double b[3]) {
    b[0] = -sin(x[1]) * cos(x[0]);
    b[1] = sin(x[0]);
    b[2] = cos(x[1]) * cos(x[0]);
}
static int valid_settings(const HbnSettings *s, int rows, int cols, char error[256]) {
    int i;
    if (!isfinite(s->pitch_column_m) || !isfinite(s->pitch_row_m) || !isfinite(s->wavelength_A) ||
        !isfinite(s->lattice_a_A) || !isfinite(s->lattice_c_A) || s->pitch_column_m <= 0 ||
        s->pitch_row_m <= 0 || s->wavelength_A <= 0 || s->lattice_a_A <= 0 || s->lattice_c_A <= 0 ||
        rows < 2 || cols < 2)
        goto invalid;
    for (i = 0; i < 5; ++i)
        if (!isfinite(s->initial[i]) || !isfinite(angle(s, i)))
            goto invalid;
    if (fabs(s->initial[0]) > .15 || fabs(s->initial[1]) > .15 || s->initial[2] < 0 ||
        s->initial[2] > cols - 1 || s->initial[3] < 0 || s->initial[3] > rows - 1 ||
        s->initial[4] < .04 || s->initial[4] > .12)
        goto invalid;
    return 1;
invalid:
    strcpy(error, "Invalid geometry: positive pitches/lattice/wavelength; tilts within +/-8.594 "
                  "deg, center on panel, distance 0.04-0.12 m.");
    return 0;
}

void hbn_detector_axes(const double x[5], double b[3], double u[3], double v[3]) {
    double norm;
    int j;
    beam(x, b);
    for (j = 0; j < 3; ++j)
        u[j] = (j == 0 ? 1 : 0) - b[j] * b[0];
    norm = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (j = 0; j < 3; ++j)
        u[j] /= norm;
    v[0] = b[1] * u[2] - b[2] * u[1];
    v[1] = b[2] * u[0] - b[0] * u[2];
    v[2] = b[0] * u[1] - b[1] * u[0];
}
void hbn_curve(const HbnSettings *s, const double x[5], int ring, double phi, double *c,
               double *r) {
    double b[3], u[3], v[3], t[3], along, theta = angle(s, ring);
    int j;
    hbn_detector_axes(x, b, u, v);
    for (j = 0; j < 3; ++j)
        t[j] = cos(theta) * b[j] + sin(theta) * (u[j] * cos(phi) + v[j] * sin(phi));
    along = x[4] * b[2] / t[2];
    *c = x[2] + (along * t[0] - x[4] * b[0]) / s->pitch_column_m;
    *r = x[3] + (along * t[1] - x[4] * b[1]) / s->pitch_row_m;
}
static double residuals(const HbnSettings *s, const double x[5], const HbnResult *out, double *r) {
    double b[3], factor = x[4] / sqrt(s->pitch_column_m * s->pitch_row_m), cost = 0;
    int i;
    beam(x, b);
    for (i = 0; i < out->point_count; ++i) {
        const HbnPoint *p = out->points + i;
        double sx = x[4] * b[0] + (p->column - x[2]) * s->pitch_column_m;
        double sy = x[4] * b[1] + (p->row - x[3]) * s->pitch_row_m, sz = x[4] * b[2];
        double cosine = (sx * b[0] + sy * b[1] + sz * b[2]) / sqrt(sx * sx + sy * sy + sz * sz);
        r[i] = (acos(clamp(cosine, -1, 1)) - angle(s, p->ring)) * factor;
        cost += r[i] * r[i] / (sqrt(1 + r[i] * r[i]) + 1);
    }
    return cost;
}
static void jacobian(const HbnSettings *s, const double x[5], const double scale[5],
                     const HbnResult *out, double J[HBN_POINTS][5]) {
    int i, j;
    double plus[HBN_POINTS], minus[HBN_POINTS], trial[5];
    for (j = 0; j < 5; ++j) {
        double step = 1e-5 * scale[j];
        memcpy(trial, x, sizeof trial);
        trial[j] += step;
        residuals(s, trial, out, plus);
        trial[j] = x[j] - step;
        residuals(s, trial, out, minus);
        for (i = 0; i < out->point_count; ++i)
            J[i][j] = (plus[i] - minus[i]) / (2e-5);
    }
}
static int solve5(double A[5][5], double b[5], double x[5]) {
    int i, j, k, p;
    for (k = 0; k < 5; ++k) {
        p = k;
        for (i = k + 1; i < 5; ++i)
            if (fabs(A[i][k]) > fabs(A[p][k]))
                p = i;
        if (fabs(A[p][k]) < DBL_MIN)
            return 0;
        for (j = k; j < 5; ++j) {
            double t = A[p][j];
            A[p][j] = A[k][j];
            A[k][j] = t;
        }
        {
            double t = b[p];
            b[p] = b[k];
            b[k] = t;
        }
        for (i = k + 1; i < 5; ++i) {
            double f = A[i][k] / A[k][k];
            for (j = k + 1; j < 5; ++j)
                A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }
    for (i = 4; i >= 0; --i) {
        double v = b[i];
        for (j = i + 1; j < 5; ++j)
            v -= A[i][j] * x[j];
        x[i] = v / A[i][i];
    }
    return 1;
}

/* One-sided Jacobi SVD: keep the small singular directions without squaring
   the Jacobian's condition number. A becomes orthogonal columns. */
static void svd5(double A[HBN_POINTS][5], int n, double singular[5], double V[5][5]) {
    int i, j, p, q, sweep;
    for (i = 0; i < 5; ++i)
        for (j = 0; j < 5; ++j)
            V[i][j] = (i == j);
    for (sweep = 0; sweep < 80; ++sweep) {
        int changed = 0;
        for (p = 0; p < 4; ++p)
            for (q = p + 1; q < 5; ++q) {
                double aa = 0, bb = 0, ab = 0, t, z, c, sn;
                for (i = 0; i < n; ++i) {
                    aa += A[i][p] * A[i][p];
                    bb += A[i][q] * A[i][q];
                    ab += A[i][p] * A[i][q];
                }
                if (fabs(ab) <= 4 * DBL_EPSILON * sqrt(aa * bb))
                    continue;
                z = (bb - aa) / (2 * ab);
                t = (z >= 0 ? 1 : -1) / (fabs(z) + sqrt(1 + z * z));
                c = 1 / sqrt(1 + t * t);
                sn = c * t;
                for (i = 0; i < n; ++i) {
                    double ap = A[i][p], aq = A[i][q];
                    A[i][p] = c * ap - sn * aq;
                    A[i][q] = sn * ap + c * aq;
                }
                for (i = 0; i < 5; ++i) {
                    double vp = V[i][p], vq = V[i][q];
                    V[i][p] = c * vp - sn * vq;
                    V[i][q] = sn * vp + c * vq;
                }
                changed = 1;
            }
        if (!changed)
            break;
    }
    for (j = 0; j < 5; ++j) {
        double v = 0;
        for (i = 0; i < n; ++i)
            v += A[i][j] * A[i][j];
        singular[j] = sqrt(v);
    }
}

int hbn_fit_points(const HbnSettings *s, int rows, int cols, HbnResult *out, HbnProgress progress,
                   void *context, char error[256]) {
    double scale[5] = {.03, .03, 10, 10, 0}, lo[5] = {-.15, -.15, 0, 0, .04},
           hi[5] = {.15, .15, 0, 0, .12};
    double J[HBN_POINTS][5], r[HBN_POINTS], lambda = 1e-3, cost, singular[5], V[5][5];
    int i, j, k, iteration, n = out->point_count, ring_seen[5] = {0}, sectors = 0;
    if (!valid_settings(s, rows, cols, error))
        return 0;
    if (n < 20 || n > HBN_POINTS) {
        strcpy(error, "At least 20 ring points are required.");
        return 0;
    }
    for (i = 0; i < n; ++i) {
        HbnPoint *p = out->points + i;
        if (p->ring < 0 || p->ring >= 5 || p->sector < 0 || p->sector >= 36 ||
            !isfinite(p->column) || !isfinite(p->row)) {
            strcpy(error, "Invalid ring observation.");
            return 0;
        }
        ring_seen[p->ring] = 1;
        if (p->sector >= sectors)
            sectors = p->sector + 1;
    }
    for (i = 0, j = 0; i < 5; ++i)
        j += ring_seen[i];
    if (j < 2) {
        strcpy(error, "At least two sampled rings are required to start.");
        return 0;
    }
    scale[4] = 100 * sqrt(s->pitch_column_m * s->pitch_row_m);
    hi[2] = cols - 1;
    hi[3] = rows - 1;
    for (j = 0; j < 5; ++j)
        if (!isfinite(out->values[j]) || out->values[j] < lo[j] || out->values[j] > hi[j]) {
            strcpy(error, "Fit seed outside admitted bounds.");
            return 0;
        }
    out->solver_success = 0;
    cost = residuals(s, out->values, out, r);
    for (iteration = 0; iteration < 1000; ++iteration) {
        double H[5][5] = {{0}}, g[5] = {0}, gmax = 0;
        int accepted = 0, attempt;
        if (progress && progress(context, "Fitting hBN geometry")) {
            strcpy(error, "Canceled.");
            return 0;
        }
        jacobian(s, out->values, scale, out, J);
        for (i = 0; i < n; ++i) {
            double inv = 1 / sqrt(1 + r[i] * r[i]);
            for (j = 0; j < 5; ++j) {
                g[j] += J[i][j] * r[i] * inv;
                for (k = 0; k < 5; ++k)
                    H[j][k] += J[i][j] * J[i][k] * inv * inv * inv;
            }
        }
        for (j = 0; j < 5; ++j) {
            double projected = g[j];
            if ((out->values[j] <= lo[j] && g[j] > 0) || (out->values[j] >= hi[j] && g[j] < 0))
                projected = 0;
            if (fabs(projected) > gmax)
                gmax = fabs(projected);
        }
        if (gmax < 1e-7) {
            out->solver_success = 1;
            break;
        }
        for (attempt = 0; attempt < 24; ++attempt) {
            double A[5][5], rhs[5], step[5], trial[5], next_r[HBN_POINTS], next_cost, length = 0;
            memcpy(A, H, sizeof A);
            for (j = 0; j < 5; ++j) {
                A[j][j] += lambda * fmax(H[j][j], 1);
                rhs[j] = -g[j];
            }
            if (!solve5(A, rhs, step)) {
                lambda *= 10;
                continue;
            }
            for (j = 0; j < 5; ++j) {
                trial[j] = clamp(out->values[j] + step[j] * scale[j], lo[j], hi[j]);
                length += pow((trial[j] - out->values[j]) / scale[j], 2);
            }
            next_cost = residuals(s, trial, out, next_r);
            if (isfinite(next_cost) && next_cost < cost) {
                double change = cost - next_cost;
                memcpy(out->values, trial, sizeof trial);
                memcpy(r, next_r, (size_t)n * sizeof(double));
                cost = next_cost;
                lambda = fmax(lambda * .3, 1e-12);
                accepted = 1;
                if (change < 1e-12 * (1 + cost) && length < 1e-10)
                    out->solver_success = 1;
                break;
            }
            lambda *= 10;
        }
        if (out->solver_success || !accepted)
            break;
    }
    out->iterations = iteration + 1;
    residuals(s, out->values, out, out->residual);
    jacobian(s, out->values, scale, out, J);
    for (i = 0; i < n; ++i)
        for (j = 0; j < 5; ++j)
            J[i][j] *= pow(1 + out->residual[i] * out->residual[i], -.75);
    {
        double copy[HBN_POINTS][5], largest = 0, smallest = DBL_MAX;
        memcpy(copy, J, sizeof copy);
        svd5(copy, n, singular, V);
        for (j = 0; j < 5; ++j) {
            largest = fmax(largest, singular[j]);
            smallest = fmin(smallest, singular[j]);
        }
        out->rank = 0;
        for (j = 0; j < 5; ++j)
            if (singular[j] > largest * n * DBL_EPSILON)
                ++out->rank;
        out->condition = smallest > 0 ? largest / smallest : INFINITY;
    }
    out->rms = out->maximum = 0;
    for (i = 0; i < n; ++i) {
        out->rms += out->residual[i] * out->residual[i];
        out->maximum = fmax(out->maximum, fabs(out->residual[i]));
    }
    /* pinv(J^T J), retaining the same 1e-15 eigenvalue cutoff as numpy.pinv. */
    for (i = 0; i < n; ++i)
        for (j = 0; j < 5; ++j)
            J[i][j] /= scale[j];
    svd5(J, n, singular, V);
    {
        double largest = 0, variance = out->rms / (n > 5 ? n - 5 : 1);
        for (j = 0; j < 5; ++j)
            largest = fmax(largest, singular[j]);
        for (j = 0; j < 5; ++j) {
            double v = 0;
            for (k = 0; k < 5; ++k)
                if (singular[k] > largest * sqrt(1e-15))
                    v += pow(V[j][k] / singular[k], 2);
            out->standard_error[j] = out->rank == 5 ? sqrt(v * variance) : NAN;
        }
    }
    out->rms = sqrt(out->rms / n);
    out->qualified = out->solver_success && out->rank == 5 && out->condition < 1e8;
    for (j = 0; j < 5; ++j) {
        out->bound[j] = fabs(out->values[j] - lo[j]) <= 1e-8 * (hi[j] - lo[j]) ||
                        fabs(out->values[j] - hi[j]) <= 1e-8 * (hi[j] - lo[j]);
        if (out->bound[j])
            out->qualified = 0;
        out->count[j] = 0;
        out->ring_rms[j] = 0;
        {
            int seen[36] = {0}, distinct = 0;
            for (i = 0; i < n; ++i)
                if (out->points[i].ring == j) {
                    ++out->count[j];
                    out->ring_rms[j] += out->residual[i] * out->residual[i];
                    seen[out->points[i].sector] = 1;
                }
            for (i = 0; i < 36; ++i)
                distinct += seen[i];
            out->coverage[j] = (double)distinct / sectors;
        }
        out->ring_rms[j] = out->count[j] ? sqrt(out->ring_rms[j] / out->count[j]) : INFINITY;
        if (out->count[j] < 8 || out->coverage[j] < .15 || out->ring_rms[j] > 2.5)
            out->qualified = 0;
    }
    return 1;
}

double hbn_detection_pixel(const HbnImage *im, int c, int r) {
    size_t i = (size_t)r * im->columns + c;
    int v = im->counts[i] - im->dark_counts[i];
    return v > 0 ? log1p((double)v) : 0;
}
static double sample(const HbnImage *im, double c, double r) {
    int c0, r0, c1, r1;
    double dx, dy;
    if (!isfinite(c) || !isfinite(r) || c < 0 || r < 0 || c > im->columns - 1 || r > im->rows - 1)
        return 0;
    c0 = (int)c;
    r0 = (int)r;
    c1 = c0 + 1 < im->columns ? c0 + 1 : c0;
    r1 = r0 + 1 < im->rows ? r0 + 1 : r0;
    dx = c - c0;
    dy = r - r0;
    return (1 - dy) *
               ((1 - dx) * hbn_detection_pixel(im, c0, r0) + dx * hbn_detection_pixel(im, c1, r0)) +
           dy * ((1 - dx) * hbn_detection_pixel(im, c0, r1) + dx * hbn_detection_pixel(im, c1, r1));
}
static int reflect(int i, int n) {
    while (i < 0 || i >= n)
        i = i < 0 ? -i - 1 : 2 * n - i - 1;
    return i;
}
typedef struct {
    int radius;
    double weights[193];
} GaussianKernel;
static void gaussian_kernel(double sigma, GaussianKernel *kernel) {
    double sum = 0, *weights = kernel->weights;
    int radius = (int)(4 * sigma + .5), k;
    kernel->radius = radius;
    for (k = -radius; k <= radius; ++k) {
        weights[k + radius] = exp(-.5 * k * k / (sigma * sigma));
        sum += weights[k + radius];
    }
    for (k = 0; k <= 2 * radius; ++k)
        weights[k] /= sum;
}
static void smooth(const double *a, int n, const GaussianKernel *kernel, double *out) {
    int i, k, radius = kernel->radius;
    const double *weights = kernel->weights;
    for (i = 0; i < n; ++i) {
        double v = 0;
        if (i >= radius && i < n - radius) {
            for (k = -radius; k <= radius; ++k)
                v += weights[k + radius] * a[i + k];
        } else {
            for (k = -radius; k <= radius; ++k)
                v += weights[k + radius] * a[reflect(i + k, n)];
        }
        out[i] = v;
    }
}
static void contrast(const double *a, int n, const GaussianKernel *narrow,
                     const GaussianKernel *broad, double *out, double *scratch) {
    int i;
    smooth(a, n, narrow, out);
    smooth(a, n, broad, scratch);
    for (i = 0; i < n; ++i)
        out[i] -= scratch[i];
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double quantile(double *v, int n, double q) {
    double at = (n - 1) * q;
    int i = (int)at;
    qsort(v, (size_t)n, sizeof(double), compare_double);
    return v[i] + (at - i) * (v[i + 1 < n ? i + 1 : i] - v[i]);
}
static int peak(const double *v, int n, double *score, double *snr, double *scratch) {
    double median, mad;
    int i, best = 0;
    for (i = 0; i < n; ++i) {
        scratch[i] = v[i];
        if (v[i] > v[best])
            best = i;
    }
    median = quantile(scratch, n, .5);
    for (i = 0; i < n; ++i)
        scratch[i] = fabs(v[i] - median);
    mad = 1.4826 * quantile(scratch, n, .5) + 1e-6;
    *score = v[best];
    *snr = (*score - median) / mad;
    return best;
}
typedef struct {
    HbnPoint point;
    double score;
    int valid;
} Candidate;
static void balance(Candidate candidates[5][360], int rings, HbnResult *out) {
    int ring, sector, i;
    out->point_count = 0;
    for (ring = 0; ring < rings; ++ring)
        for (sector = 0; sector < 36; ++sector) {
            int best = -1;
            for (i = 0; i < 360; ++i)
                if (candidates[ring][i].valid && candidates[ring][i].point.sector == sector &&
                    (best < 0 || candidates[ring][i].score > candidates[ring][best].score))
                    best = i;
            if (best >= 0)
                out->points[out->point_count++] = candidates[ring][best].point;
        }
}
int hbn_calibrate(const HbnImage *im, const HbnSettings *s, HbnResult *out, HbnProgress progress,
                  void *context, char error[256]) {
    Candidate(*coarse)[360] = NULL, (*current)[360] = NULL;
    GaussianKernel narrow, coarse_kernel, refined_kernel;
    double radii[5], half[] = {45, 45, 28, 45, 55}, start, stop, *memory = NULL, *profile, *score,
                     *scratch, *median;
    int n, i, ring, round, ok = 0;
    if (!im->counts || !im->dark_counts) {
        strcpy(error, "hBN calibration requires both the measurement and a matching dark image.");
        return 0;
    }
    if (!valid_settings(s, im->rows, im->columns, error))
        return 0;
    for (ring = 0; ring < 5; ++ring)
        radii[ring] =
            s->initial[4] * tan(angle(s, ring)) / sqrt(s->pitch_column_m * s->pitch_row_m);
    start = fmax(20, radii[0] - 80);
    stop = radii[4] + 100;
    if (!isfinite(stop) || stop <= start || stop - start > 20000) {
        strcpy(error, "Expected rings fall outside the supported radial search range.");
        return 0;
    }
    n = (int)ceil((stop - start) / .5);
    if (n < 193) {
        strcpy(error, "Ring radii are too small for the declared profile filters.");
        return 0;
    }
    memory = (double *)malloc((size_t)n * 4 * sizeof(double));
    coarse = (Candidate(*)[360])calloc(5, sizeof *coarse);
    current = (Candidate(*)[360])calloc(5, sizeof *current);
    if (!memory || !coarse || !current) {
        strcpy(error, "Not enough memory for ring profiles.");
        goto done;
    }
    profile = memory;
    score = memory + n;
    scratch = memory + 2 * n;
    median = memory + 3 * n;
    gaussian_kernel(2, &narrow);
    gaussian_kernel(24, &coarse_kernel);
    gaussian_kernel(10, &refined_kernel);
    memset(out, 0, sizeof *out);
    memcpy(out->values, s->initial, sizeof out->values);
    for (i = 0; i < 360; ++i) {
        double phi = i * (2 * HBN_PI / 360), cp = cos(phi), sp = sin(phi);
        int j;
        if (progress && progress(context, "Finding hBN rings")) {
            strcpy(error, "Canceled.");
            goto done;
        }
        for (j = 0; j < n; ++j)
            profile[j] = sample(im, s->initial[2] + cp * (start + .5 * j),
                                s->initial[3] + sp * (start + .5 * j));
        contrast(profile, n, &narrow, &coarse_kernel, score, scratch);
        for (ring = 0; ring < 5; ++ring) {
            int low = (int)ceil((radii[ring] - half[ring] - start) / .5),
                high = (int)ceil((radii[ring] + half[ring] - start) / .5), best;
            double strength, snr, radius;
            Candidate *c = &coarse[ring][i];
            low = low < 0 ? 0 : low;
            high = high > n ? n : high;
            if (high <= low) {
                strcpy(error, "Empty expected ring window.");
                goto done;
            }
            best = peak(score + low, high - low, &strength, &snr, median);
            radius = start + .5 * (low + best);
            c->valid = snr >= 3 && strength > .04;
            c->score = snr;
            c->point.column = s->initial[2] + cp * radius;
            c->point.row = s->initial[3] + sp * radius;
            c->point.ring = ring;
            c->point.sector = (int)floor(fmod(phi, 2 * HBN_PI) * 36 / (2 * HBN_PI));
        }
    }
    balance(coarse, 2, out);
    if (!hbn_fit_points(s, im->rows, im->columns, out, progress, context, error))
        goto done;
    for (round = 0; round < 4; ++round) {
        for (ring = 0; ring < 5; ++ring) {
            double curves[360][2], strengths[360], snrs[360], offsets[360], scores_sorted[360];
            int kept = 0;
            for (i = 0; i < 360; ++i)
                hbn_curve(s, out->values, ring, i * (2 * HBN_PI / 360), &curves[i][0],
                          &curves[i][1]);
            for (i = 0; i < 360; ++i) {
                int j, best, prev = (i + 359) % 360, next = (i + 1) % 360;
                double nx = -(curves[next][1] - curves[prev][1]),
                       ny = curves[next][0] - curves[prev][0], norm = sqrt(nx * nx + ny * ny);
                Candidate *c = &current[ring][i];
                nx /= norm;
                ny /= norm;
                if (progress && progress(context, "Refining hBN rings")) {
                    strcpy(error, "Canceled.");
                    goto done;
                }
                for (j = 0; j < 49; ++j)
                    profile[j] = sample(im, curves[i][0] + nx * (-12 + .5 * j),
                                        curves[i][1] + ny * (-12 + .5 * j));
                contrast(profile, 49, &narrow, &refined_kernel, score, scratch);
                best = peak(score, 49, &strengths[i], &snrs[i], median);
                offsets[i] = -12 + .5 * best;
                c->point.column = curves[i][0] + nx * offsets[i];
                c->point.row = curves[i][1] + ny * offsets[i];
                c->point.ring = ring;
                c->point.sector = coarse[ring][i].point.sector;
                c->score = snrs[i];
                scores_sorted[i] = strengths[i];
            }
            {
                double threshold = fmax(.04, quantile(scores_sorted, 360, .15));
                for (i = 0; i < 360; ++i) {
                    current[ring][i].valid =
                        snrs[i] >= 3 && strengths[i] > threshold && fabs(offsets[i]) < 11.5;
                    kept += current[ring][i].valid;
                }
            }
            if (kept < 12)
                memcpy(current[ring], coarse[ring], sizeof current[ring]);
        }
        balance(current, 5, out);
        if (!hbn_fit_points(s, im->rows, im->columns, out, progress, context, error))
            goto done;
    }
    ok = 1;
done:
    free(memory);
    free(coarse);
    free(current);
    return ok;
}
