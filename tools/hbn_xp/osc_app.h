#ifndef SLATE_OSC_APP_H
#define SLATE_OSC_APP_H
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include "osc_analysis.h"
#include "cif_peaks.h"
#include "cif_guides.h"
#include <commdlg.h>
#include <commctrl.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    OPEN = 100,
    DARK,
    CLEAR_DARK,
    EXPORT_CSV,
    EXPORT_ASC,
    EXPORT_BMP,
    EXPORT_PROFILES,
    LOAD_SETTINGS,
    SAVE_FIT,
    EXIT_APP,
    FIT_VIEW,
    ZOOM_IN,
    ZOOM_OUT,
    AUTO_CONTRAST,
    APPLY_CONTRAST,
    LOG_VIEW,
    SUBTRACT_DARK,
    PICK_CENTER,
    CALIBRATE,
    CANCEL_JOB,
    HELP_APP,
    FIELD = 200,
    BLACK = 220,
    WHITE = 221,
    RESULT = 230,
    STAGE = 231
};
enum {
    JOB_LOAD = 1,
    JOB_DARK,
    JOB_FIT,
    JOB_CSV,
    JOB_ASC,
    JOB_ANGLES,
    JOB_ANGLE_EXPORT,
    JOB_CIF,
    JOB_CIF_EXPORT
};
#define DONE_MESSAGE (WM_APP + 1)
#define STAGE_MESSAGE (WM_APP + 2)
typedef struct {
    int group;
    double angle, intensity;
} CifDisplayRow;

typedef struct {
    HWND window, canvas, result_label, stage_label;
    HINSTANCE instance;
    HFONT font;
    HbnImage image;
    HbnSettings settings;
    HbnResult result;
    HANDLE worker;
    volatile LONG cancel;
    int job, job_ok, closing, has_result, subtract, logarithmic, pick_center;
    char job_path[MAX_PATH], source_path[MAX_PATH], dark_path[MAX_PATH], error[256];
    int view_width, view_height, selected_column, selected_row, dragging, roi_drag;
    int drag_x, drag_y, roi[4], has_roi;
    double zoom, left, top, drag_left, drag_top, black, white;
    float *preview;
    int preview_rows, preview_cols, preview_step;
    double *screen_values;
    unsigned char *screen_bgr;
    int screen_width, screen_height, view_dirty, color_dirty;
    DWORD last_progress;
    HWND angle_canvas;
    int analysis_page, view_mode, mouse_tool, output_mode, geometry_valid, geometry_dirty;
    int updating_analysis, angle_drag, angle_width, angle_height;
    double drag_theta, drag_phi;
    OscGeometry geometry;
    OscGrid selection, drag_grid;
    OscIntegration integration, pending_integration;
    unsigned char *mask;
    unsigned char *angle_bgr;
    int angle_bitmap_width, angle_bitmap_height, angle_color_dirty;
    double *angle_profiles[2];
    int angle_profile_mode[2];
    char calibration_text[160];
    CifPeaks cif, pending_cif;
    int cif_loaded, cif_selected, cif_dirty, cif_unknown_zero, updating_cif;
    double cif_wavelength, cif_maximum;
    char cif_path[MAX_PATH], cif_data_path[MAX_PATH];
    int cif_kind, cif_labels, cif_overlay, cif_mount_applied, cif_mount_dirty, cif_sort_intensity;
    int cif_compact;
    int guide_attempted, guide_rods;
    CifMount cif_mount;
    CifGuides guides;
    OscGeometry guide_geometry;
    CifGuideView guide_view;
    char guide_error[256];
    CifDisplayRow *cif_rows;
} App;

enum {
    ANALYSIS_PAGE = 300,
    CALIBRATION_PAGE,
    CIF_PAGE,
    APPLIED_GEOMETRY = 310,
    APPLY_INPUTS,
    APPLY_HBN,
    SAMPLE_DISTANCE,
    APPLY_DISTANCE,
    THETA_MIN,
    THETA_MAX,
    PHI_MIN,
    PHI_MAX,
    THETA_STEP,
    PHI_STEP,
    VIEW_MODE,
    MOUSE_TOOL,
    OUTPUT_MODE,
    INTEGRATE,
    EXPORT_ANGLES,
    FULL_ANGLES,
    SAVE_GEOMETRY,
    LOAD_GEOMETRY,
    CLEAR_MASK,
    ANALYSIS_LABEL = 350,
    HBN_LABEL = 500
};
enum { PAGE_HBN, PAGE_ANALYSIS, PAGE_CIF };
enum {
    CIF_OPEN = 600,
    CIF_EXPORT,
    CIF_WAVELENGTH,
    CIF_MAXIMUM,
    CIF_CALCULATE,
    CIF_USE_GEOMETRY,
    CIF_UNKNOWN_ZERO,
    CIF_LIST,
    CIF_STATUS,
    CIF_VIEW,
    CIF_SORT,
    CIF_OVERLAY,
    CIF_LABELS,
    CIF_INCIDENCE,
    CIF_NORMAL_PHI,
    CIF_MOUNT,
    CIF_KIND,
    CIF_MEMBERS,
    CIF_COMPACT,
    CIF_LABEL = 640,
    CIF_END = 650
};
enum { VIEW_DETECTOR, VIEW_ANGLES, VIEW_SPLIT };
enum { TOOL_PAN, TOOL_SECTOR, TOOL_MASK, TOOL_UNMASK };
enum { OUTPUT_MEAN, OUTPUT_SUM, OUTPUT_AREA };
int pixel(const App *a, int c, int r);
int app_progress(void *context, const char *stage);
void message(App *a, const char *text);
int choose_path(App *a, char *path, int save, const char *filter, const char *extension);
FILE *begin_output(const char *path, char temporary[MAX_PATH], char error[256]);
int finish_output(FILE *f, const char *temporary, const char *path, int ok, char error[256]);
HWND control(App *a, const char *class_name, const char *text, DWORD style, int id, int x, int y,
             int w, int h);
void set_number(App *a, int id, double value);
int number(App *a, int id, double *value);
int read_settings(App *a);
void start_job(App *a, int job, const char *path);
void make_preview(App *a);
void auto_contrast(App *a);
void fit_view(App *a);
void zoom_at(App *a, double factor, int x, int y);
void render_view(App *a);
int view_coordinate(App *a, int x, int y, int *c, int *r);
void inspect_roi(App *a);
LRESULT CALLBACK canvas_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp);
void export_profiles(App *a, const char *path);
void export_bmp(App *a, const char *path);
void show_result(App *a);
void analysis_controls(App *a);
void analysis_page(App *a, int analysis);
void analysis_layout(App *a);
void analysis_view_controls(App *a);
int analysis_command(App *a, int id, int notification);
void analysis_invalidate(App *a);
int analysis_ready(const App *a);
void analysis_begin(App *a);
void analysis_publish(App *a);
void analysis_selection_fields(App *a);
void analysis_cursor(App *a, int c, int r);
void analysis_overlay(App *a, HDC dc);
int analysis_detector_mouse(App *a, UINT msg, WPARAM wp, LPARAM lp);
int analysis_export(App *a, const char *path);
LRESULT CALLBACK angle_canvas_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp);
void cif_controls(App *a);
void cif_layout(App *a);
void cif_status(App *a);
int cif_command(App *a, int id, int notification);
LRESULT cif_notify(App *a, NMHDR *header);
void cif_publish(App *a);
int cif_export(App *a, const char *path);
int cif_guides_ready(const App *a);
int cif_prepare_guides(App *a);
void cif_group_label(const App *a, int kind, int group, char *text, size_t capacity);
void cif_detector_overlay(App *a, HDC dc);
void cif_angle_marker(App *a, HDC dc, RECT bounds);
void cif_angle_overlay(App *a, HDC dc, RECT bounds);

#endif
