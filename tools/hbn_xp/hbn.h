#ifndef SLATE_HBN_H
#define SLATE_HBN_H
#include <stdio.h>
#define HBN_RINGS 5
#define HBN_POINTS 180
#define HBN_PI 3.14159265358979323846

/* Native pixel centers: (column,row); active local-x then current-local-y tilts.
   The preset uses the SLATE detector base with beam along its +z.
   Equations and acceptance thresholds are owned by fitting/hbn.py. */
typedef struct {
    double pitch_column_m, pitch_row_m, wavelength_A, lattice_a_A, lattice_c_A;
    double initial[5];
} HbnSettings;
typedef struct {
    int rows, columns;
    int *counts, *dark_counts;
    unsigned long source_crc32, dark_crc32;
} HbnImage;
typedef struct {
    double column, row;
    int ring, sector;
} HbnPoint;
typedef struct {
    double values[5], standard_error[5], residual[HBN_POINTS];
    double ring_rms[HBN_RINGS], coverage[HBN_RINGS], rms, maximum, condition;
    int count[HBN_RINGS], bound[5], rank, solver_success, qualified, iterations;
    int point_count;
    HbnPoint points[HBN_POINTS];
} HbnResult;
/* Return nonzero to cancel at bounded read/tracing/solver boundaries. */
typedef int (*HbnProgress)(void *context, const char *stage);
void hbn_defaults(HbnSettings *settings);
int hbn_read_settings(const char *path, HbnSettings *settings, char error[256]);
int hbn_load(const char *source, const char *dark, HbnImage *image, HbnProgress progress,
             void *context, char error[256]);
int osc_load(const char *path, HbnImage *image, HbnProgress progress, void *context,
             char error[256]);
int osc_load_dark(const char *path, HbnImage *image, HbnProgress progress, void *context,
                  char error[256]);
void hbn_free(HbnImage *image);
int hbn_calibrate(const HbnImage *image, const HbnSettings *settings, HbnResult *result,
                  HbnProgress progress, void *context, char error[256]);
int hbn_fit_points(const HbnSettings *settings, int rows, int columns, HbnResult *result,
                   HbnProgress progress, void *context, char error[256]);
void hbn_curve(const HbnSettings *settings, const double values[5], int ring, double azimuth,
               double *column, double *row);
void hbn_detector_axes(const double values[5], double beam_axis[3], double column_axis[3],
                       double row_axis[3]);
int hbn_report(FILE *stream, const HbnSettings *settings, const HbnImage *image,
               const HbnResult *result);
double hbn_detection_pixel(const HbnImage *image, int column, int row);
#endif
