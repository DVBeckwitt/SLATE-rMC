#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include "hbn.h"
#include <commdlg.h>
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
enum { JOB_LOAD = 1, JOB_DARK, JOB_FIT, JOB_CSV, JOB_ASC };
#define DONE_MESSAGE (WM_APP + 1)
#define STAGE_MESSAGE (WM_APP + 2)

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
} App;

static int pixel(const App *a, int c, int r) {
    size_t i = (size_t)r * a->image.columns + c;
    return a->image.counts[i] - (a->subtract && a->image.dark_counts ? a->image.dark_counts[i] : 0);
}
static int progress(void *context, const char *stage) {
    App *a = (App *)context;
    DWORD now = GetTickCount();
    if (now - a->last_progress >= 100) {
        PostMessageA(a->window, STAGE_MESSAGE, 0, (LPARAM)stage);
        a->last_progress = now;
    }
    return InterlockedCompareExchange(&a->cancel, 0, 0) != 0;
}
static void message(App *a, const char *text) {
    MessageBoxA(a->window, text, "SLATE OSC", MB_OK | MB_ICONINFORMATION);
}
static int choose_path(App *a, char *path, int save, const char *filter, const char *extension) {
    OPENFILENAMEA o;
    memset(&o, 0, sizeof o);
    path[0] = 0;
    o.lStructSize = sizeof o;
    o.hwndOwner = a->window;
    o.lpstrFile = path;
    o.nMaxFile = MAX_PATH;
    o.lpstrFilter = filter;
    o.lpstrDefExt = extension;
    o.Flags = OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST |
              (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
    return save ? GetSaveFileNameA(&o) : GetOpenFileNameA(&o);
}
static int safe_output(const char *path) {
    const char *extension = strrchr(path, '.');
    return extension && (!lstrcmpiA(extension, ".csv") || !lstrcmpiA(extension, ".asc") ||
                         !lstrcmpiA(extension, ".bmp"));
}
/* Stage writes in the destination directory, then replace only after a full close.
   The UI confirms replacement; OSC/INI sources cannot be output destinations. */
static FILE *begin_output(const char *path, char temporary[MAX_PATH], char error[256]) {
    char directory[MAX_PATH], *slash;
    FILE *f;
    if (!safe_output(path)) {
        strcpy(error, "Choose a .csv, .asc or .bmp output; source files are protected.");
        return NULL;
    }
    if (strlen(path) >= MAX_PATH - 1) {
        strcpy(error, "Output path is too long.");
        return NULL;
    }
    strcpy(directory, path);
    slash = strrchr(directory, '\\');
    if (!slash)
        slash = strrchr(directory, '/');
    if (slash)
        slash[1] = 0;
    else
        strcpy(directory, ".");
    if (!GetTempFileNameA(directory, "osc", 0, temporary)) {
        strcpy(error, "Cannot create output in that folder.");
        return NULL;
    }
    f = fopen(temporary, "wb");
    if (!f) {
        DeleteFileA(temporary);
        strcpy(error, "Cannot open temporary output.");
    }
    return f;
}
static int finish_output(FILE *f, const char *temporary, const char *path, int ok,
                         char error[256]) {
    if (ferror(f))
        ok = 0;
    if (fclose(f) != 0)
        ok = 0;
    if (ok && MoveFileExA(temporary, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return 1;
    DeleteFileA(temporary);
    if (!*error)
        strcpy(error, "Output was not saved (write, cancellation or rename failure).");
    return 0;
}
static int export_pixels(const App *a, const char *path, int asc, HbnProgress check, void *context,
                         char error[256]) {
    char temporary[MAX_PATH];
    FILE *f = begin_output(path, temporary, error);
    int r, c, ok = 1;
    if (!f)
        return 0;
    fprintf(f,
            "# SLATE native %s matrix: rows=%d columns=%d; column increases right, row down; "
            "zero-based pixel centers\n",
            asc ? "ASC" : "CSV", a->image.rows, a->image.columns);
    fprintf(f, "# Values: %s; source_crc32=%08lx; dark_crc32=%08lx; no display transform\n",
            a->subtract ? "raw-minus-dark (scale=1)" : "raw", a->image.source_crc32,
            a->subtract ? a->image.dark_crc32 : 0ul);
    for (r = 0; r < a->image.rows; ++r) {
        if (check && check(context, "Exporting native pixels")) {
            ok = 0;
            strcpy(error, "Canceled.");
            break;
        }
        for (c = 0; c < a->image.columns; ++c)
            fprintf(f, c ? (asc ? " %d" : ",%d") : "%d", pixel(a, c, r));
        fputc('\n', f);
        if (ferror(f)) {
            ok = 0;
            break;
        }
    }
    return finish_output(f, temporary, path, ok, error);
}
static DWORD WINAPI worker_main(void *context) {
    App *a = (App *)context;
    a->error[0] = 0;
    a->last_progress = 0;
    if (a->job == JOB_LOAD)
        a->job_ok = osc_load(a->job_path, &a->image, progress, a, a->error);
    else if (a->job == JOB_DARK)
        a->job_ok = osc_load_dark(a->job_path, &a->image, progress, a, a->error);
    else if (a->job == JOB_FIT)
        a->job_ok = hbn_calibrate(&a->image, &a->settings, &a->result, progress, a, a->error);
    else
        a->job_ok = export_pixels(a, a->job_path, a->job == JOB_ASC, progress, a, a->error);
    PostMessageA(a->window, DONE_MESSAGE, 0, 0);
    return 0;
}
static HWND control(App *a, const char *class_name, const char *text, DWORD style, int id, int x,
                    int y, int w, int h) {
    HWND hnd = CreateWindowExA(!strcmp(class_name, "EDIT") ? WS_EX_CLIENTEDGE : 0, class_name, text,
                               WS_CHILD | WS_VISIBLE | style, x, y, w, h, a->window,
                               (HMENU)(INT_PTR)id, a->instance, NULL);
    SendMessageA(hnd, WM_SETFONT, (WPARAM)a->font, TRUE);
    return hnd;
}
static void set_number(App *a, int id, double v) {
    char text[64];
    snprintf(text, sizeof text, "%.10g", v);
    SetWindowTextA(GetDlgItem(a->window, id), text);
}
static int number(App *a, int id, double *value) {
    char text[80], *end;
    if (GetWindowTextLengthA(GetDlgItem(a->window, id)) >= (int)sizeof text)
        return 0;
    GetWindowTextA(GetDlgItem(a->window, id), text, sizeof text);
    *value = strtod(text, &end);
    while (*end == ' ' || *end == '\t')
        ++end;
    return end != text && !*end && isfinite(*value);
}
static void show_settings(App *a) {
    set_number(a, FIELD, a->settings.pitch_column_m * 1000);
    set_number(a, FIELD + 1, a->settings.pitch_row_m * 1000);
    set_number(a, FIELD + 2, a->settings.wavelength_A);
    set_number(a, FIELD + 3, a->settings.initial[2]);
    set_number(a, FIELD + 4, a->settings.initial[3]);
    set_number(a, FIELD + 5, a->settings.initial[4] * 1000);
    set_number(a, FIELD + 6, a->settings.initial[0] * 180 / HBN_PI);
    set_number(a, FIELD + 7, a->settings.initial[1] * 180 / HBN_PI);
}
static int read_settings(App *a) {
    double v[8];
    int i;
    for (i = 0; i < 8; ++i)
        if (!number(a, FIELD + i, v + i)) {
            message(a, "Enter finite numbers in all geometry fields.");
            return 0;
        }
    a->settings.pitch_column_m = v[0] / 1000;
    a->settings.pitch_row_m = v[1] / 1000;
    a->settings.wavelength_A = v[2];
    a->settings.initial[2] = v[3];
    a->settings.initial[3] = v[4];
    a->settings.initial[4] = v[5] / 1000;
    a->settings.initial[0] = v[6] * HBN_PI / 180;
    a->settings.initial[1] = v[7] * HBN_PI / 180;
    return 1;
}
static void lock_controls(App *a, int busy) {
    HWND child = GetWindow(a->window, GW_CHILD);
    HMENU menu = GetMenu(a->window);
    int i;
    while (child) {
        EnableWindow(child, !busy || GetDlgCtrlID(child) == CANCEL_JOB);
        child = GetWindow(child, GW_HWNDNEXT);
    }
    EnableWindow(GetDlgItem(a->window, CANCEL_JOB), busy);
    for (i = OPEN; i <= EXIT_APP; ++i)
        EnableMenuItem(menu, i, MF_BYCOMMAND | (busy ? MF_GRAYED : MF_ENABLED));
    DrawMenuBar(a->window);
}
static void start_job(App *a, int job, const char *path) {
    if (a->worker)
        return;
    if (path && strlen(path) >= MAX_PATH) {
        message(a, "Input path exceeds the Windows XP path limit.");
        return;
    }
    if (path)
        strcpy(a->job_path, path);
    a->job = job;
    InterlockedExchange(&a->cancel, 0);
    lock_controls(a, 1);
    SetWindowTextA(a->stage_label, "Working... (Cancel stops at the next processing boundary)");
    a->worker = CreateThread(NULL, 0, worker_main, a, 0, NULL);
    if (!a->worker) {
        lock_controls(a, 0);
        message(a, "Could not start the worker.");
    }
    InvalidateRect(a->canvas, NULL, FALSE);
}
static void make_preview(App *a) {
    int r, c, s;
    size_t total;
    a->view_dirty = 1;
    free(a->preview);
    a->preview = NULL;
    if (!a->image.counts)
        return;
    s = (a->image.rows > a->image.columns ? a->image.rows : a->image.columns);
    s = (s + 749) / 750;
    if (s < 1)
        s = 1;
    a->preview_step = s;
    a->preview_rows = (a->image.rows + s - 1) / s;
    a->preview_cols = (a->image.columns + s - 1) / s;
    total = (size_t)a->preview_rows * a->preview_cols;
    a->preview = (float *)calloc(total, sizeof(float));
    if (!a->preview) {
        message(a, "Preview allocation failed; full-resolution display remains available.");
        return;
    }
    for (r = 0; r < a->image.rows; ++r)
        for (c = 0; c < a->image.columns; ++c)
            a->preview[(size_t)(r / s) * a->preview_cols + c / s] += (float)pixel(a, c, r);
    for (r = 0; r < a->preview_rows; ++r)
        for (c = 0; c < a->preview_cols; ++c) {
            int nr = a->image.rows - r * s, nc = a->image.columns - c * s;
            if (nr > s)
                nr = s;
            if (nc > s)
                nc = s;
            a->preview[(size_t)r * a->preview_cols + c] /= (float)(nr * nc);
        }
    a->view_dirty = 1;
}
static int cmp_float(const void *x, const void *y) {
    float a = *(const float *)x, b = *(const float *)y;
    return (a > b) - (a < b);
}
static void auto_contrast(App *a) {
    size_t n = (size_t)a->preview_rows * a->preview_cols;
    float *v;
    if (!a->preview || !n)
        return;
    v = (float *)malloc(n * sizeof(float));
    if (!v)
        return;
    memcpy(v, a->preview, n * sizeof(float));
    qsort(v, n, sizeof(float), cmp_float);
    a->black = v[(size_t)(.01 * (n - 1))];
    a->white = v[(size_t)(.995 * (n - 1))];
    free(v);
    if (a->white <= a->black)
        a->white = a->black + 1;
    set_number(a, BLACK, a->black);
    set_number(a, WHITE, a->white);
    a->color_dirty = 1;
}
static void fit_view(App *a) {
    if (!a->image.counts || a->view_width < 1 || a->view_height < 1)
        return;
    a->zoom =
        fmin((double)a->view_width / a->image.columns, (double)a->view_height / a->image.rows);
    a->left = -(a->view_width / a->zoom - a->image.columns) * .5 - .5;
    a->top = -(a->view_height / a->zoom - a->image.rows) * .5 - .5;
    a->view_dirty = 1;
}
static int view_coordinate(App *a, int x, int y, int *c, int *r) {
    if (!a->image.counts || a->zoom <= 0)
        return 0;
    *c = (int)floor(a->left + x / a->zoom + .5);
    *r = (int)floor(a->top + y / a->zoom + .5);
    return *c >= 0 && *r >= 0 && *c < a->image.columns && *r < a->image.rows && y < a->view_height;
}
static void render_view(App *a) {
    int x, y, w = a->view_width, h = a->view_height;
    if (w < 1 || h < 1 || !a->image.counts || a->worker)
        return;
    if (a->screen_width != w || a->screen_height != h) {
        free(a->screen_values);
        free(a->screen_bgr);
        a->screen_values = NULL;
        a->screen_bgr = NULL;
        a->screen_values = (double *)malloc((size_t)w * h * sizeof(double));
        a->screen_bgr = (unsigned char *)malloc((size_t)w * h * 4);
        a->screen_width = w;
        a->screen_height = h;
        a->view_dirty = 1;
    }
    if (!a->screen_values || !a->screen_bgr) {
        free(a->screen_values);
        free(a->screen_bgr);
        a->screen_values = NULL;
        a->screen_bgr = NULL;
        a->screen_width = a->screen_height = 0;
        SetWindowTextA(a->stage_label,
                       "Not enough memory for display buffers. Reduce the window size.");
        return;
    }
    if (a->view_dirty) {
        for (y = 0; y < h; ++y)
            for (x = 0; x < w; ++x) {
                int c, r;
                double value = NAN;
                if (view_coordinate(a, x, y, &c, &r)) {
                    if (a->preview && a->zoom < .5)
                        value = a->preview[(size_t)(r / a->preview_step) * a->preview_cols +
                                           c / a->preview_step];
                    else
                        value = pixel(a, c, r);
                }
                a->screen_values[(size_t)y * w + x] = value;
            }
        a->view_dirty = 0;
        a->color_dirty = 1;
    }
    if (a->color_dirty) {
        double denominator = a->logarithmic ? log1p(a->white - a->black) : a->white - a->black;
        for (y = 0; y < h; ++y)
            for (x = 0; x < w; ++x) {
                size_t at = (size_t)y * w + x;
                double v = a->screen_values[at];
                unsigned char gray = 20;
                if (isfinite(v)) {
                    v = fmax(0, fmin(v - a->black, a->white - a->black));
                    gray = (unsigned char)(255 * (a->logarithmic ? log1p(v) : v) / denominator);
                }
                a->screen_bgr[4 * at] = a->screen_bgr[4 * at + 1] = a->screen_bgr[4 * at + 2] =
                    gray;
                a->screen_bgr[4 * at + 3] = 0;
            }
        a->color_dirty = 0;
    }
}
static void line(HDC dc, int x1, int y1, int x2, int y2) {
    MoveToEx(dc, x1, y1, NULL);
    LineTo(dc, x2, y2);
}
static void profile_plot(App *a, HDC dc, RECT area, int vertical) {
    int n = vertical ? a->image.rows : a->image.columns, i, p, w = area.right - area.left - 12,
        h = area.bottom - area.top - 24;
    double min = DBL_MAX, max = -DBL_MAX;
    HPEN pen, old;
    char title[80];
    if (w < 2 || h < 2)
        return;
    for (i = 0; i < n; ++i) {
        double v = vertical ? pixel(a, a->selected_column, i) : pixel(a, i, a->selected_row);
        if (v < min)
            min = v;
        if (v > max)
            max = v;
    }
    if (max <= min)
        max = min + 1;
    snprintf(title, sizeof title, "%s %d   counts %.0f to %.0f", vertical ? "Column" : "Row",
             vertical ? a->selected_column : a->selected_row, min, max);
    TextOutA(dc, area.left + 6, area.top + 2, title, (int)strlen(title));
    pen = CreatePen(PS_SOLID, 1, vertical ? RGB(40, 120, 220) : RGB(180, 70, 0));
    old = (HPEN)SelectObject(dc, pen);
    for (p = 0; p < w; ++p) {
        int begin = p * n / w, end = (p + 1) * n / w;
        double low = DBL_MAX, high = -DBL_MAX;
        if (end <= begin)
            end = begin + 1;
        for (i = begin; i < end && i < n; ++i) {
            double v = vertical ? pixel(a, a->selected_column, i) : pixel(a, i, a->selected_row);
            if (v < low)
                low = v;
            if (v > high)
                high = v;
        }
        line(dc, area.left + 6 + p, area.bottom - 4 - (int)((low - min) / (max - min) * h),
             area.left + 6 + p, area.bottom - 4 - (int)((high - min) / (max - min) * h) - 1);
    }
    SelectObject(dc, old);
    DeleteObject(pen);
}
static void paint_canvas(App *a, HDC dc, RECT bounds) {
    HGDIOBJ old_font = SelectObject(dc, a->font);
    SetBkMode(dc, TRANSPARENT);
    FillRect(dc, &bounds, (HBRUSH)(COLOR_WINDOW + 1));
    if (!a->image.counts || a->worker) {
        const char *text = a->worker ? "Working - image controls are temporarily locked."
                                     : "Open an OSC image to begin.";
        TextOutA(dc, 20, 25, text, (int)strlen(text));
        SelectObject(dc, old_font);
        return;
    }
    render_view(a);
    if (a->screen_bgr) {
        BITMAPINFO b;
        memset(&b, 0, sizeof b);
        b.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        b.bmiHeader.biWidth = a->screen_width;
        b.bmiHeader.biHeight = -a->screen_height;
        b.bmiHeader.biPlanes = 1;
        b.bmiHeader.biBitCount = 32;
        b.bmiHeader.biCompression = BI_RGB;
        SetDIBitsToDevice(dc, 0, 0, a->screen_width, a->screen_height, 0, 0, 0, a->screen_height,
                          a->screen_bgr, &b, DIB_RGB_COLORS);
    }
    SaveDC(dc);
    IntersectClipRect(dc, 0, 0, a->view_width, a->view_height);
    if (a->has_result) {
        int ring, i;
        HPEN pen =
                 CreatePen(PS_SOLID, 1, a->result.qualified ? RGB(0, 255, 100) : RGB(255, 180, 0)),
             old = (HPEN)SelectObject(dc, pen);
        for (ring = 0; ring < 5; ++ring)
            for (i = 0; i <= 360; ++i) {
                double c, r;
                hbn_curve(&a->settings, a->result.values, ring, i * (2 * HBN_PI / 360), &c, &r);
                if (!i)
                    MoveToEx(dc, (int)((c - a->left) * a->zoom), (int)((r - a->top) * a->zoom),
                             NULL);
                else
                    LineTo(dc, (int)((c - a->left) * a->zoom), (int)((r - a->top) * a->zoom));
            }
        SelectObject(dc, old);
        DeleteObject(pen);
        for (i = 0; i < a->result.point_count; ++i) {
            int x = (int)((a->result.points[i].column - a->left) * a->zoom),
                y = (int)((a->result.points[i].row - a->top) * a->zoom);
            RECT dot = {x - 2, y - 2, x + 3, y + 3};
            HBRUSH brush = CreateSolidBrush(RGB(255, 100, 70));
            FillRect(dc, &dot, brush);
            DeleteObject(brush);
        }
    }
    {
        int x = (int)((a->selected_column - a->left) * a->zoom),
            y = (int)((a->selected_row - a->top) * a->zoom);
        HPEN pen = CreatePen(PS_DOT, 1, RGB(0, 180, 255)), old = (HPEN)SelectObject(dc, pen);
        line(dc, x - 9, y, x + 10, y);
        line(dc, x, y - 9, x, y + 10);
        if (a->has_roi) {
            HGDIOBJ brush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
            Rectangle(dc, (int)((a->roi[0] - .5 - a->left) * a->zoom),
                      (int)((a->roi[1] - .5 - a->top) * a->zoom),
                      (int)((a->roi[2] + .5 - a->left) * a->zoom),
                      (int)((a->roi[3] + .5 - a->top) * a->zoom));
            SelectObject(dc, brush);
        }
        SelectObject(dc, old);
        DeleteObject(pen);
    }
    RestoreDC(dc, -1);
    {
        RECT h = {0, a->view_height + 4, a->view_width / 2, bounds.bottom},
             v = {a->view_width / 2, a->view_height + 4, a->view_width, bounds.bottom};
        profile_plot(a, dc, h, 0);
        profile_plot(a, dc, v, 1);
    }
    SelectObject(dc, old_font);
}

static void inspect(App *a, int c, int r) {
    char text[240];
    a->selected_column = c;
    a->selected_row = r;
    snprintf(text, sizeof text, "Column %d, row %d | raw %d | %s %d | zoom %.1f%%", c, r,
             a->image.counts[(size_t)r * a->image.columns + c], a->subtract ? "raw-dark" : "counts",
             pixel(a, c, r), a->zoom * 100);
    SetWindowTextA(a->stage_label, text);
}
static void inspect_roi(App *a) {
    int c, r, c0 = a->roi[0], r0 = a->roi[1], c1 = a->roi[2], r1 = a->roi[3], t;
    double sum = 0, n;
    char text[256];
    if (c0 > c1) {
        t = c0;
        c0 = c1;
        c1 = t;
    }
    if (r0 > r1) {
        t = r0;
        r0 = r1;
        r1 = t;
    }
    a->roi[0] = c0;
    a->roi[1] = r0;
    a->roi[2] = c1;
    a->roi[3] = r1;
    for (r = r0; r <= r1; ++r)
        for (c = c0; c <= c1; ++c)
            sum += pixel(a, c, r);
    n = (double)(c1 - c0 + 1) * (r1 - r0 + 1);
    snprintf(
        text, sizeof text,
        "ROI columns %d..%d, rows %d..%d (inclusive) | %.0f pixels | signed sum %.0f | mean %.6g",
        c0, c1, r0, r1, n, sum, sum / n);
    SetWindowTextA(a->stage_label, text);
}
static void zoom_at(App *a, double factor, int x, int y) {
    double c = a->left + x / a->zoom, r = a->top + y / a->zoom;
    a->zoom = fmax(.02, fmin(32, a->zoom * factor));
    a->left = c - x / a->zoom;
    a->top = r - y / a->zoom;
    a->view_dirty = 1;
    InvalidateRect(a->canvas, NULL, FALSE);
}
static LRESULT CALLBACK canvas_proc(HWND w, UINT message_id, WPARAM wp, LPARAM lp) {
    App *a = (App *)GetWindowLongPtrA(w, GWLP_USERDATA);
    if (message_id == WM_NCCREATE) {
        a = (App *)((CREATESTRUCTA *)lp)->lpCreateParams;
        SetWindowLongPtrA(w, GWLP_USERDATA, (LONG_PTR)a);
    }
    if (!a)
        return DefWindowProcA(w, message_id, wp, lp);
    switch (message_id) {
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        RECT b;
        HDC dc = BeginPaint(w, &ps);
        GetClientRect(w, &b);
        paint_canvas(a, dc, b);
        EndPaint(w, &ps);
        return 0;
    }
    case WM_SIZE:
        a->view_width = LOWORD(lp);
        a->view_height = HIWORD(lp) > 140 ? HIWORD(lp) - 140 : 1;
        if (a->zoom <= 0)
            fit_view(a);
        a->view_dirty = 1;
        return 0;
    case WM_LBUTTONDOWN: {
        int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp), c, r;
        if (a->worker || !view_coordinate(a, x, y, &c, &r))
            return 0;
        SetFocus(w);
        inspect(a, c, r);
        if (a->pick_center) {
            set_number(a, FIELD + 3, c);
            set_number(a, FIELD + 4, r);
            a->pick_center = 0;
            SetWindowTextA(GetDlgItem(a->window, PICK_CENTER), "Pick initial center");
        } else {
            a->dragging = 1;
            a->roi_drag = (wp & MK_SHIFT) != 0;
            a->drag_x = x;
            a->drag_y = y;
            a->drag_left = a->left;
            a->drag_top = a->top;
            if (a->roi_drag) {
                a->has_roi = 1;
                a->roi[0] = a->roi[2] = c;
                a->roi[1] = a->roi[3] = r;
            }
            SetCapture(w);
        }
        InvalidateRect(w, NULL, FALSE);
        return 0;
    }
    case WM_MOUSEMOVE: {
        int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp), c, r;
        if (a->worker || !a->image.counts)
            return 0;
        if (a->dragging) {
            if (a->roi_drag) {
                if (view_coordinate(a, x, y, &c, &r)) {
                    a->roi[2] = c;
                    a->roi[3] = r;
                }
            } else {
                a->left = a->drag_left - (x - a->drag_x) / a->zoom;
                a->top = a->drag_top - (y - a->drag_y) / a->zoom;
                a->view_dirty = 1;
            }
            InvalidateRect(w, NULL, FALSE);
        } else if (view_coordinate(a, x, y, &c, &r)) {
            char text[160];
            snprintf(text, sizeof text,
                     "Cursor: column %d, row %d, counts %d | click: profiles | drag: pan | "
                     "Shift-drag: ROI",
                     c, r, pixel(a, c, r));
            SetWindowTextA(a->stage_label, text);
        }
        return 0;
    }
    case WM_LBUTTONUP:
        if (a->dragging) {
            a->dragging = 0;
            if (a->roi_drag)
                inspect_roi(a);
            ReleaseCapture();
            InvalidateRect(w, NULL, FALSE);
        }
        return 0;
    case WM_CAPTURECHANGED:
        if (a->dragging && a->roi_drag) {
            a->has_roi = 0;
            InvalidateRect(w, NULL, FALSE);
        }
        a->dragging = 0;
        return 0;
    case WM_MOUSEWHEEL: {
        POINT p = {GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
        ScreenToClient(w, &p);
        if (!a->worker && a->image.counts && p.y < a->view_height)
            zoom_at(a, GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 1.25 : .8, p.x, p.y);
        return 0;
    }
    }
    return DefWindowProcA(w, message_id, wp, lp);
}
static void export_profiles(App *a, const char *path) {
    char temporary[MAX_PATH];
    FILE *f;
    int i;
    a->error[0] = 0;
    f = begin_output(path, temporary, a->error);
    if (!f) {
        message(a, a->error);
        return;
    }
    fprintf(f,
            "# native full-resolution single-row/column profiles; selected_column=%d "
            "selected_row=%d; values=%s\n",
            a->selected_column, a->selected_row, a->subtract ? "raw-dark, scale=1" : "raw");
    fprintf(f, "# source_crc32=%08lx dark_crc32=%08lx\naxis,pixel_center,count\n",
            a->image.source_crc32, a->subtract ? a->image.dark_crc32 : 0ul);
    for (i = 0; i < a->image.columns; ++i)
        fprintf(f, "column,%d,%d\n", i, pixel(a, i, a->selected_row));
    for (i = 0; i < a->image.rows; ++i)
        fprintf(f, "row,%d,%d\n", i, pixel(a, a->selected_column, i));
    if (a->has_roi) {
        double sum = 0, n = (double)(a->roi[2] - a->roi[0] + 1) * (a->roi[3] - a->roi[1] + 1);
        int c, r;
        for (r = a->roi[1]; r <= a->roi[3]; ++r)
            for (c = a->roi[0]; c <= a->roi[2]; ++c)
                sum += pixel(a, c, r);
        fprintf(
            f,
            "# ROI inclusive: columns=%d..%d rows=%d..%d pixels=%.0f signed_sum=%.0f mean=%.17g\n",
            a->roi[0], a->roi[2], a->roi[1], a->roi[3], n, sum, sum / n);
    }
    if (!finish_output(f, temporary, path, 1, a->error))
        message(a, a->error);
}
static void export_bmp(App *a, const char *path) {
    char temporary[MAX_PATH];
    FILE *f;
    BITMAPFILEHEADER header;
    BITMAPINFOHEADER info;
    size_t bytes;
    int ok;
    render_view(a);
    if (!a->screen_bgr)
        return;
    bytes = (size_t)a->screen_width * a->screen_height * 4;
    memset(&header, 0, sizeof header);
    memset(&info, 0, sizeof info);
    header.bfType = 0x4d42;
    header.bfOffBits = sizeof header + sizeof info;
    header.bfSize = (DWORD)(header.bfOffBits + bytes);
    info.biSize = sizeof info;
    info.biWidth = a->screen_width;
    info.biHeight = -a->screen_height;
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biCompression = BI_RGB;
    info.biSizeImage = (DWORD)bytes;
    a->error[0] = 0;
    f = begin_output(path, temporary, a->error);
    if (!f) {
        message(a, a->error);
        return;
    }
    ok = fwrite(&header, sizeof header, 1, f) == 1 && fwrite(&info, sizeof info, 1, f) == 1 &&
         fwrite(a->screen_bgr, 1, bytes, f) == bytes;
    if (!finish_output(f, temporary, path, ok, a->error))
        message(a, a->error);
}
static void show_result(App *a) {
    char text[1200];
    const HbnResult *r = &a->result;
    snprintf(text, sizeof text,
             "%s\r\n\r\nTilt about column: %.6f deg\r\nTilt about row: %.6f deg\r\nBeam column: "
             "%.6f px\r\nBeam row: %.6f px\r\nCalibrant distance: %.6f mm\r\n\r\nRMS %.4f px; max "
             "%.4f px\r\n%d points; rank %d; condition %.3g\r\nSolver converged: %s\r\nRing "
             "points: %d / %d / %d / %d / %d\r\n\r\nGreen: qualified curves\r\nOrange: unqualified "
             "curves\r\nRed points: measured selections",
             r->qualified ? "QUALIFIED hBN geometry" : "UNQUALIFIED - inspect support / residuals",
             r->values[0] * 180 / HBN_PI, r->values[1] * 180 / HBN_PI, r->values[2], r->values[3],
             r->values[4] * 1000, r->rms, r->maximum, r->point_count, r->rank, r->condition,
             r->solver_success ? "yes" : "no", r->count[0], r->count[1], r->count[2], r->count[3],
             r->count[4]);
    SetWindowTextA(a->result_label, text);
}
static void handle_command(App *a, int id, int notification) {
    char path[MAX_PATH];
    if (id == CANCEL_JOB) {
        InterlockedExchange(&a->cancel, 1);
        return;
    }
    if (a->worker)
        return;
    if (id >= FIELD && id < FIELD + 8 && notification == EN_CHANGE) {
        a->has_result = 0;
        SetWindowTextA(a->result_label, "Geometry changed. Calculate to obtain a new result.");
        InvalidateRect(a->canvas, NULL, FALSE);
        return;
    }
    if (id == OPEN) {
        if (choose_path(a, path, 0, "R-AXIS OSC\0*.osc\0\0", "osc"))
            start_job(a, JOB_LOAD, path);
        return;
    }
    if (id == EXIT_APP) {
        SendMessageA(a->window, WM_CLOSE, 0, 0);
        return;
    }
    if (id == HELP_APP) {
        message(a, "Open an uncompressed R-AXIS OSC. Coordinates are clockwise detector-native, "
                   "with zero-based pixel centers.\r\n\r\nClick: row/column profiles. Drag: pan. "
                   "Wheel: zoom. Shift-drag: rectangular ROI. Numeric exports retain signed "
                   "original-resolution values; BMP saves the visible display without "
                   "overlays.\r\n\r\nhBN: open matching dark, verify geometry fields, then "
                   "Calculate. Dark scale is 1 (no exposure normalization). The preset assumes the "
                   "SLATE detector base and beam. Distance is calibrant-private. All five rings "
                   "must pass support/rank/residual checks. See README for limits.");
        return;
    }
    if (id == LOAD_SETTINGS) {
        if (choose_path(a, path, 0, "Instrument settings\0*.ini\0\0", "ini")) {
            HbnSettings s;
            if (hbn_read_settings(path, &s, a->error)) {
                a->settings = s;
                show_settings(a);
            } else
                message(a, a->error);
        }
        return;
    }
    if (!a->image.counts) {
        if (id < FIELD)
            message(a, "Open an OSC image first.");
        return;
    }
    switch (id) {
    case DARK:
        if (choose_path(a, path, 0, "Matching dark OSC\0*.osc\0\0", "osc"))
            start_job(a, JOB_DARK, path);
        break;
    case CLEAR_DARK:
        free(a->image.dark_counts);
        a->image.dark_counts = NULL;
        a->image.dark_crc32 = 0;
        a->dark_path[0] = 0;
        a->subtract = 0;
        a->has_result = 0;
        CheckDlgButton(a->window, SUBTRACT_DARK, BST_UNCHECKED);
        make_preview(a);
        auto_contrast(a);
        SetWindowTextA(a->result_label, "Dark cleared; calibration result invalidated.");
        break;
    case FIT_VIEW:
        fit_view(a);
        break;
    case ZOOM_IN:
        zoom_at(a, 1.5, a->view_width / 2, a->view_height / 2);
        break;
    case ZOOM_OUT:
        zoom_at(a, 1 / 1.5, a->view_width / 2, a->view_height / 2);
        break;
    case AUTO_CONTRAST:
        auto_contrast(a);
        break;
    case APPLY_CONTRAST: {
        double low, high;
        if (!number(a, BLACK, &low) || !number(a, WHITE, &high) || high <= low)
            message(a, "White must be a finite number greater than black.");
        else {
            a->black = low;
            a->white = high;
            a->color_dirty = 1;
        }
        break;
    }
    case LOG_VIEW:
        a->logarithmic = IsDlgButtonChecked(a->window, LOG_VIEW) == BST_CHECKED;
        a->color_dirty = 1;
        break;
    case SUBTRACT_DARK:
        if (!a->image.dark_counts) {
            CheckDlgButton(a->window, SUBTRACT_DARK, BST_UNCHECKED);
            message(a, "Load a matching dark image first.");
        } else {
            a->subtract = IsDlgButtonChecked(a->window, SUBTRACT_DARK) == BST_CHECKED;
            make_preview(a);
            auto_contrast(a);
        }
        break;
    case PICK_CENTER:
        a->pick_center = !a->pick_center;
        SetWindowTextA(GetDlgItem(a->window, PICK_CENTER),
                       a->pick_center ? "Click initial center in image" : "Pick initial center");
        break;
    case CALIBRATE:
        if (!a->image.dark_counts)
            message(a, "Open the matching dark OSC before calibration.");
        else if (read_settings(a)) {
            a->has_result = 0;
            SetWindowTextA(a->result_label, "Calculating...");
            start_job(a, JOB_FIT, NULL);
        }
        break;
    case SAVE_FIT:
        if (!a->has_result) {
            message(a, "Calculate hBN geometry first.");
            break;
        }
        if (choose_path(a, path, 1, "Calibration CSV\0*.csv\0\0", "csv")) {
            char temporary[MAX_PATH];
            FILE *f;
            a->error[0] = 0;
            f = begin_output(path, temporary, a->error);
            if (!f)
                message(a, a->error);
            else if (!finish_output(f, temporary, path,
                                    hbn_report(f, &a->settings, &a->image, &a->result), a->error))
                message(a, a->error);
        }
        break;
    case EXPORT_CSV:
    case EXPORT_ASC:
        if (choose_path(a, path, 1,
                        id == EXPORT_CSV ? "Native CSV matrix\0*.csv\0\0"
                                         : "Native ASCII matrix\0*.asc\0\0",
                        id == EXPORT_CSV ? "csv" : "asc"))
            start_job(a, id == EXPORT_CSV ? JOB_CSV : JOB_ASC, path);
        break;
    case EXPORT_PROFILES:
        if (choose_path(a, path, 1, "Profiles and ROI CSV\0*.csv\0\0", "csv"))
            export_profiles(a, path);
        break;
    case EXPORT_BMP:
        if (choose_path(a, path, 1, "Visible display BMP\0*.bmp\0\0", "bmp"))
            export_bmp(a, path);
        break;
    }
    InvalidateRect(a->canvas, NULL, FALSE);
}
static void create_controls(App *a) {
    HMENU bar = CreateMenu(), file = CreatePopupMenu();
    int i;
    const char *labels[] = {
        "Column pitch (mm)",         "Row pitch (mm)",        "Wavelength (A)",
        "Initial center column",     "Initial center row",    "Calibrant distance (mm)",
        "Initial column tilt (deg)", "Initial row tilt (deg)"};
    AppendMenuA(file, MF_STRING, OPEN, "Open OSC...\tCtrl+O");
    AppendMenuA(file, MF_STRING, DARK, "Open matching dark...");
    AppendMenuA(file, MF_STRING, CLEAR_DARK, "Clear dark");
    AppendMenuA(file, MF_SEPARATOR, 0, NULL);
    AppendMenuA(file, MF_STRING, EXPORT_CSV, "Export native CSV matrix...");
    AppendMenuA(file, MF_STRING, EXPORT_ASC, "Export native ASC matrix...");
    AppendMenuA(file, MF_STRING, EXPORT_PROFILES, "Export profiles and ROI...");
    AppendMenuA(file, MF_STRING, EXPORT_BMP, "Save visible view BMP...");
    AppendMenuA(file, MF_SEPARATOR, 0, NULL);
    AppendMenuA(file, MF_STRING, LOAD_SETTINGS, "Load instrument preset...");
    AppendMenuA(file, MF_STRING, SAVE_FIT, "Save hBN calibration...");
    AppendMenuA(file, MF_SEPARATOR, 0, NULL);
    AppendMenuA(file, MF_STRING, EXIT_APP, "Exit");
    AppendMenuA(bar, MF_POPUP, (UINT_PTR)file, "File");
    AppendMenuA(bar, MF_STRING, HELP_APP, "Help");
    SetMenu(a->window, bar);
    control(a, "BUTTON", "Open OSC...", BS_PUSHBUTTON | WS_TABSTOP, OPEN, 10, 10, 112, 27);
    control(a, "BUTTON", "Open dark...", BS_PUSHBUTTON | WS_TABSTOP, DARK, 128, 10, 112, 27);
    control(a, "BUTTON", "Show raw - dark", BS_AUTOCHECKBOX | WS_TABSTOP, SUBTRACT_DARK, 10, 43,
            128, 22);
    control(a, "BUTTON", "Log display", BS_AUTOCHECKBOX | WS_TABSTOP, LOG_VIEW, 140, 43, 100, 22);
    CheckDlgButton(a->window, LOG_VIEW, BST_CHECKED);
    control(a, "STATIC", "Black", 0, -1, 10, 74, 38, 20);
    control(a, "EDIT", "0", WS_TABSTOP | ES_AUTOHSCROLL, BLACK, 48, 70, 67, 24);
    control(a, "STATIC", "White", 0, -1, 126, 74, 39, 20);
    control(a, "EDIT", "1000", WS_TABSTOP | ES_AUTOHSCROLL, WHITE, 168, 70, 72, 24);
    control(a, "BUTTON", "Auto contrast", WS_TABSTOP, AUTO_CONTRAST, 10, 101, 112, 26);
    control(a, "BUTTON", "Apply levels", WS_TABSTOP, APPLY_CONTRAST, 128, 101, 112, 26);
    control(a, "BUTTON", "Fit image", WS_TABSTOP, FIT_VIEW, 10, 134, 90, 26);
    control(a, "BUTTON", "Zoom +", WS_TABSTOP, ZOOM_IN, 105, 134, 65, 26);
    control(a, "BUTTON", "Zoom -", WS_TABSTOP, ZOOM_OUT, 175, 134, 65, 26);
    control(a, "STATIC", "hBN detector calibration", 0, -1, 10, 174, 230, 22);
    for (i = 0; i < 8; ++i) {
        control(a, "STATIC", labels[i], 0, -1, 10, 205 + 26 * i, 140, 20);
        control(a, "EDIT", "", WS_TABSTOP | ES_AUTOHSCROLL, FIELD + i, 155, 201 + 26 * i, 85, 24);
    }
    control(a, "BUTTON", "Pick initial center", WS_TABSTOP, PICK_CENTER, 10, 414, 230, 26);
    control(a, "BUTTON", "Calculate hBN", WS_TABSTOP, CALIBRATE, 10, 448, 143, 28);
    control(a, "BUTTON", "Cancel", WS_TABSTOP, CANCEL_JOB, 160, 448, 80, 28);
    a->result_label =
        control(a, "EDIT", "", ES_MULTILINE | ES_READONLY | WS_VSCROLL, RESULT, 10, 486, 230, 150);
    a->stage_label = control(
        a, "STATIC", "Open OSC to begin. All coordinates are detector-native (column, row).", 0,
        STAGE, 10, 690, 960, 22);
    a->canvas =
        CreateWindowExA(WS_EX_CLIENTEDGE, "SlateOscCanvas", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                        250, 8, 730, 672, a->window, NULL, a->instance, a);
    show_settings(a);
    SetWindowTextA(a->result_label,
                   "Load OSC for viewing.\r\n\r\nFor hBN, also load the matching dark and verify "
                   "the geometry above.\r\n\r\nPreset lattice: a=2.504 A, c=6.661 A.\r\n\r\nClick: "
                   "profiles\r\nDrag: pan\r\nWheel: zoom\r\nShift-drag: rectangular ROI");
    EnableWindow(GetDlgItem(a->window, CANCEL_JOB), FALSE);
}
static LRESULT CALLBACK window_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    App *a = (App *)GetWindowLongPtrA(w, GWLP_USERDATA);
    if (msg == WM_NCCREATE) {
        a = (App *)((CREATESTRUCTA *)lp)->lpCreateParams;
        a->window = w;
        SetWindowLongPtrA(w, GWLP_USERDATA, (LONG_PTR)a);
    }
    if (!a)
        return DefWindowProcA(w, msg, wp, lp);
    switch (msg) {
    case WM_CREATE:
        create_controls(a);
        return 0;
    case WM_GETMINMAXINFO:
        ((MINMAXINFO *)lp)->ptMinTrackSize.x = 850;
        ((MINMAXINFO *)lp)->ptMinTrackSize.y = 680;
        return 0;
    case WM_SIZE:
        if (a->canvas) {
            MoveWindow(a->canvas, 250, 8, LOWORD(lp) > 260 ? LOWORD(lp) - 260 : 1,
                       HIWORD(lp) > 40 ? HIWORD(lp) - 40 : 1, TRUE);
            MoveWindow(a->stage_label, 10, HIWORD(lp) - 26, LOWORD(lp) - 20, 22, TRUE);
            MoveWindow(a->result_label, 10, 486, 230, HIWORD(lp) > 526 ? HIWORD(lp) - 526 : 1,
                       TRUE);
        }
        return 0;
    case WM_COMMAND:
        handle_command(a, LOWORD(wp), HIWORD(wp));
        return 0;
    case STAGE_MESSAGE:
        if (a->worker)
            SetWindowTextA(a->stage_label, (const char *)lp);
        return 0;
    case DONE_MESSAGE:
        WaitForSingleObject(a->worker, INFINITE);
        CloseHandle(a->worker);
        a->worker = NULL;
        lock_controls(a, 0);
        if (a->closing) {
            DestroyWindow(w);
            return 0;
        }
        if (a->job_ok) {
            if (a->job == JOB_LOAD || a->job == JOB_DARK) {
                a->has_result = 0;
                a->has_roi = 0;
                if (a->job == JOB_LOAD) {
                    strcpy(a->source_path, a->job_path);
                    a->dark_path[0] = 0;
                    a->subtract = 0;
                    CheckDlgButton(w, SUBTRACT_DARK, BST_UNCHECKED);
                    a->selected_column = a->image.columns / 2;
                    a->selected_row = a->image.rows / 2;
                } else
                    strcpy(a->dark_path, a->job_path);
                make_preview(a);
                auto_contrast(a);
                if (a->job == JOB_LOAD)
                    fit_view(a);
                SetWindowTextA(a->result_label,
                               a->job == JOB_DARK
                                   ? "Dark loaded. Verify the geometry, then Calculate hBN."
                                   : "Image loaded. Click for profiles; Shift-drag for ROI. hBN "
                                     "calibration additionally needs a matching dark.");
                {
                    char title[380];
                    snprintf(title, sizeof title, "SLATE OSC + hBN - %s [%d columns x %d rows]",
                             a->source_path, a->image.columns, a->image.rows);
                    SetWindowTextA(w, title);
                }
            } else if (a->job == JOB_FIT) {
                a->has_result = 1;
                show_result(a);
            }
            SetWindowTextA(
                a->stage_label,
                a->job == JOB_FIT
                    ? (a->result.qualified
                           ? "hBN fit passed the existing qualification rules. Inspect the overlay "
                             "before use."
                           : "hBN fit is UNQUALIFIED. The result may be saved with this status.")
                    : "Complete.");
        } else {
            SetWindowTextA(a->stage_label, a->error);
            if (a->job == JOB_FIT)
                SetWindowTextA(a->result_label,
                               "Calculation did not complete. No result is available.");
            if (strcmp(a->error, "Canceled."))
                message(a, a->error);
        }
        InvalidateRect(a->canvas, NULL, FALSE);
        return 0;
    case WM_CLOSE:
        if (a->worker) {
            a->closing = 1;
            InterlockedExchange(&a->cancel, 1);
            SetWindowTextA(a->stage_label, "Canceling before closing...");
        } else
            DestroyWindow(w);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(w, msg, wp, lp);
}

static int batch(int argc, char **argv) {
    HbnSettings s;
    HbnImage im = {0};
    HbnResult result = {0};
    char error[256] = {0}, temporary[MAX_PATH];
    FILE *f;
    int ok;
    hbn_defaults(&s);
    if (argc != 5 && argc != 6)
        return 2;
    if (argc == 6 && !hbn_read_settings(argv[5], &s, error))
        return 2;
    if (strcmp(argv[1], "--fit"))
        return 2;
    if (!hbn_load(argv[2], argv[3], &im, NULL, NULL, error) ||
        !hbn_calibrate(&im, &s, &result, NULL, NULL, error)) {
        hbn_free(&im);
        return 2;
    }
    f = begin_output(argv[4], temporary, error);
    if (!f) {
        hbn_free(&im);
        return 2;
    }
    ok = hbn_report(f, &s, &im, &result);
    ok = finish_output(f, temporary, argv[4], ok, error);
    hbn_free(&im);
    return ok ? (result.qualified ? 0 : 3) : 2;
}
int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous, LPSTR command, int show) {
    App app;
    WNDCLASSEXA wc;
    MSG msg;
    ACCEL accelerator = {FCONTROL | FVIRTKEY, 'O', OPEN};
    HACCEL accel;
    (void)previous;
    (void)command;
    if (__argc > 1 && !strcmp(__argv[1], "--fit"))
        return batch(__argc, __argv);
    memset(&app, 0, sizeof app);
    app.instance = instance;
    app.black = 0;
    app.white = 1000;
    app.logarithmic = 1;
    hbn_defaults(&app.settings);
    app.font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    memset(&wc, 0, sizeof wc);
    wc.cbSize = sizeof wc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpfnWndProc = canvas_proc;
    wc.lpszClassName = "SlateOscCanvas";
    if (!RegisterClassExA(&wc))
        return 2;
    wc.lpfnWndProc = window_proc;
    wc.lpszClassName = "SlateOscMain";
    if (!RegisterClassExA(&wc))
        return 2;
    if (!CreateWindowExA(0, wc.lpszClassName, "SLATE OSC + hBN", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                         CW_USEDEFAULT, 1000, 720, NULL, NULL, instance, &app))
        return 2;
    ShowWindow(app.window, show);
    UpdateWindow(app.window);
    accel = CreateAcceleratorTableA(&accelerator, 1);
    if (__argc == 2)
        start_job(&app, JOB_LOAD, __argv[1]);
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        if (!TranslateAcceleratorA(app.window, accel, &msg) &&
            !IsDialogMessageA(app.window, &msg)) {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
    }
    DestroyAcceleratorTable(accel);
    hbn_free(&app.image);
    free(app.preview);
    free(app.screen_values);
    free(app.screen_bgr);
    return 0;
}
