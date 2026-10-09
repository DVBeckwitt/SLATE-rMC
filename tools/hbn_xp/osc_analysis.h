#ifndef SLATE_OSC_ANALYSIS_H
#define SLATE_OSC_ANALYSIS_H
#include "hbn.h"

/* Applied geometry is independent of hBN seeds/results. Distances and tilts are
   metres and radians; phi follows the detector-oriented SLATE AngleFrame. */
typedef struct {
    HbnSettings settings;
    int rows, columns, from_qualified_hbn, distance_from_calibrant;
    unsigned long calibrant_crc32, dark_crc32;
    double beam[3], column[3], row[3];
} OscGeometry;
typedef struct {
    double theta, phi;
    int azimuth_valid;
} OscAngle;
typedef struct {
    double theta_min, theta_max, phi_min, phi_max;
    int theta_bins, phi_bins;
} OscGrid;
typedef struct {
    OscGrid grid;
    double *signal, *area, *panel_area;
    double input_signal, masked_signal, masked_area, lost_signal, lost_area;
} OscIntegration;

int osc_geometry_compile(OscGeometry *geometry, char error[256]);
OscAngle osc_pixel_angle(const OscGeometry *geometry, double column, double row);
int osc_angle_pixel(const OscGeometry *geometry, double theta, double phi, double *column,
                    double *row);
double osc_wrap_phi(double phi);
int osc_grid_valid(const OscGrid *grid, char error[256]);
int osc_is_masked(const unsigned char *mask, size_t index);
void osc_mask_box(unsigned char *mask, int columns, int rows, int c0, int r0, int c1, int r1,
                  int exclude);
int osc_integrate(const HbnImage *image, int subtract, const unsigned char *mask,
                  const OscGeometry *geometry, const OscGrid *grid, OscIntegration *output,
                  HbnProgress progress, void *context, char error[256]);
void osc_integration_free(OscIntegration *result);
int osc_geometry_write(FILE *stream, const OscGeometry *geometry);
int osc_geometry_read(const char *path, OscGeometry *geometry, char error[256]);
int osc_integration_write(FILE *stream, const HbnImage *image, int subtract,
                          const OscGeometry *geometry, const OscIntegration *result,
                          unsigned long mask_crc32, HbnProgress progress, void *context);
unsigned long osc_mask_crc32(const unsigned char *mask, size_t pixels);
#endif
