#ifndef SLATE_CIF_GUIDES_H
#define SLATE_CIF_GUIDES_H
#include "cif_peaks.h"
#include "osc_analysis.h"

typedef struct {
    double theta, phi, column, row;
    int angle_valid, detector_valid;
} CifGuidePoint;
typedef struct {
    int kind, group, first, count;
} CifGuidePath;
typedef struct {
    CifGuidePoint *points;
    CifGuidePath *paths;
    int point_count, path_count;
} CifGuides;
typedef struct {
    double column_min, column_max, row_min, row_max, pixel_tolerance;
    double theta_min, theta_max, phi_min, phi_max, theta_tolerance, phi_tolerance;
} CifGuideView;

/* Returns zero, one or two physical outgoing-ray intersections for a tick. */
int cif_rod_angles(double wavelength_A, OscMount mount, double qr_invA, double qz_invA,
                   OscAngle angles[2]);
void cif_guides_free(CifGuides *guides);
/* Transactional bounded display cache; no reflection or intensity pruning. */
int cif_guides_build(const CifPeaks *peaks, const OscGeometry *geometry, OscMount mount, int rods,
                     const CifGuideView *view, CifGuides *output, char error[256]);
#endif
