#ifndef SLATE_CIF_PEAKS_H
#define SLATE_CIF_PEAKS_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int h, k, l;
    double d_A, two_theta_deg, real_e, imag_e, intensity_e2;
    double qr_invA, qz_invA;
    int group[3];
} CifPeak;
enum { CIF_POWDER, CIF_ROD, CIF_TICK };
typedef struct {
    int first, count;
    double maximum_intensity_e2;
} CifGroup;
typedef struct {
    CifGroup *groups;
    int *members;
    int count;
} CifGrouping;
typedef struct {
    CifPeak *peaks;
    int count, source_sites, expanded_sites, unknown_u_sites;
    double wavelength_A, max_two_theta_deg;
    unsigned long source_crc32, data_crc32;
    char name[128], spacegroup[80], species[1024];
    CifGrouping grouping[3];
} CifPeaks;
typedef int (*CifProgress)(void *context, const char *stage);
/* Transactional output; unknown U=0 is an explicit caller choice. All signed
   hkl within the bound are retained, including extinctions and Friedel mates. */
int cif_calculate(const char *path, const char *data_path, double wavelength_A,
                  double max_two_theta_deg, int unknown_u_zero, CifPeaks *output,
                  CifProgress progress, void *context, char error[256]);
void cif_peaks_free(CifPeaks *result);
int cif_peaks_write(FILE *stream, const CifPeaks *result, CifProgress progress, void *context);
#ifdef __cplusplus
}
#endif
#endif
