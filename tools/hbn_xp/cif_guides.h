#ifndef SLATE_CIF_GUIDES_H
#define SLATE_CIF_GUIDES_H
#include "cif_peaks.h"
#include "osc_analysis.h"

/* External-air fiber geometry. b3 is normal to the direct a1/a2 surface.
   Normal azimuth uses the same phi convention as the image; angles are radians. */
typedef struct {
    double incidence, normal_phi;
} CifMount;
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

int cif_mount_valid(CifMount mount);
/* Returns zero, one or two physical outgoing-ray intersections for a tick. */
int cif_rod_angles(double wavelength_A, CifMount mount, double qr_invA, double qz_invA,
                   OscAngle angles[2]);
void cif_guides_free(CifGuides *guides);
/* Transactional bounded display cache; no reflection or intensity pruning. */
int cif_guides_build(const CifPeaks *peaks, const OscGeometry *geometry, CifMount mount, int rods,
                     const CifGuideView *view, CifGuides *output, char error[256]);
#endif
