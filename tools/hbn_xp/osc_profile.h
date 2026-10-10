#ifndef SLATE_OSC_PROFILE_H
#define SLATE_OSC_PROFILE_H
#include "osc_analysis.h"

enum { PROFILE_THETA, PROFILE_PHI, PROFILE_QZ, PROFILE_QR };
typedef struct {
    OscGrid source_grid;
    OscMount mount;
    double wavelength_A, lower, upper;
    int axis, bins;
    double *signal, *area, *panel_area;
} OscProfile;

/* Angular axes reduce the map exactly. Q axes reassign whole angular-cell masses
   using their centers: a conservative histogram, not direct Q pixel splitting.
   Auto range includes all panel-supported centers, including masked cells. */
int osc_profile_build(const OscIntegration *map, int axis, int q_bins, double wavelength_A,
                      OscMount mount, OscProfile *output, char error[256]);
void osc_profile_free(OscProfile *profile);
int osc_profile_write(FILE *stream, const OscProfile *profile);
#endif
