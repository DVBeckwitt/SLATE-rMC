#include "osc_app.h"

int pixel(const App *a, int c, int r) {
    size_t i = (size_t)r * a->image.columns + c;
    return a->image.counts[i] - (a->subtract && a->image.dark_counts ? a->image.dark_counts[i] : 0);
}
int app_progress(void *context, const char *stage) {
    App *a = (App *)context;
    DWORD now = GetTickCount();
    if (now - a->last_progress >= 100) {
        PostMessageA(a->window, STAGE_MESSAGE, 0, (LPARAM)stage);
        a->last_progress = now;
    }
    return InterlockedCompareExchange(&a->cancel, 0, 0) != 0;
}
void message(App *a, const char *text) {
    MessageBoxA(a->window, text, "SLATE OSC", MB_OK | MB_ICONINFORMATION);
}
int choose_path(App *a, char *path, int save, const char *filter, const char *extension) {
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
FILE *begin_output(const char *path, char temporary[MAX_PATH], char error[256]) {
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
int finish_output(FILE *f, const char *temporary, const char *path, int ok, char error[256]) {
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
        a->job_ok = osc_load(a->job_path, &a->image, app_progress, a, a->error);
    else if (a->job == JOB_DARK)
        a->job_ok = osc_load_dark(a->job_path, &a->image, app_progress, a, a->error);
    else if (a->job == JOB_FIT)
        a->job_ok = hbn_calibrate(&a->image, &a->settings, &a->result, app_progress, a, a->error);
    else if (a->job == JOB_ANGLES)
        a->job_ok = osc_integrate(&a->image, a->subtract, a->mask, &a->geometry, &a->selection,
                                  &a->pending_integration, app_progress, a, a->error);
    else if (a->job == JOB_ANGLE_EXPORT)
        a->job_ok = analysis_export(a, a->job_path);
    else if (a->job == JOB_CIF)
        a->job_ok = cif_calculate(a->job_path, a->cif_data_path, a->cif_wavelength, a->cif_maximum,
                                  a->cif_unknown_zero, &a->pending_cif, app_progress, a, a->error);
    else if (a->job == JOB_CIF_EXPORT)
        a->job_ok = cif_export(a, a->job_path);
    else
        a->job_ok = export_pixels(a, a->job_path, a->job == JOB_ASC, app_progress, a, a->error);
    PostMessageA(a->window, DONE_MESSAGE, 0, 0);
    return 0;
}
HWND control(App *a, const char *class_name, const char *text, DWORD style, int id, int x, int y,
             int w, int h) {
    HWND hnd = CreateWindowExA(!strcmp(class_name, "EDIT") ? WS_EX_CLIENTEDGE : 0, class_name, text,
                               WS_CHILD | WS_VISIBLE | style, x, y, w, h, a->window,
                               (HMENU)(INT_PTR)id, a->instance, NULL);
    SendMessageA(hnd, WM_SETFONT, (WPARAM)a->font, TRUE);
    return hnd;
}
void set_number(App *a, int id, double v) {
    char text[64];
    snprintf(text, sizeof text, "%.10g", v);
    SetWindowTextA(GetDlgItem(a->window, id), text);
}
int number(App *a, int id, double *value) {
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
int read_settings(App *a) {
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
    EnableMenuItem(menu, SAVE_GEOMETRY, MF_BYCOMMAND | (busy ? MF_GRAYED : MF_ENABLED));
    EnableMenuItem(menu, LOAD_GEOMETRY, MF_BYCOMMAND | (busy ? MF_GRAYED : MF_ENABLED));
    DrawMenuBar(a->window);
}
void start_job(App *a, int job, const char *path) {
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
    InvalidateRect(a->angle_canvas, NULL, FALSE);
}
void show_result(App *a) {
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
    if (cif_command(a, id, notification) || analysis_command(a, id, notification))
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
        message(
            a, "Open an uncompressed R-AXIS OSC. Coordinates are clockwise detector-native, "
               "with zero-based pixel centers.\r\n\r\nClick: row/column profiles. Drag: pan. "
               "Wheel: zoom. Shift-drag: rectangular ROI. Numeric exports retain signed "
               "original-resolution values; BMP includes mask shading but no curves or "
               "profiles.\r\n\r\nhBN: open matching dark, verify geometry fields, then "
               "Calculate. Dark scale is 1 (no exposure normalization). The preset assumes the "
               "SLATE detector base and beam. Distance is calibrant-private. All five rings "
               "must pass support/rank/residual checks.\r\n\r\nAnalysis: apply the fit or manual "
               "geometry, verify Sample mm, then Integrate. The mouse tool selects angular "
               "sectors or detector masks. Both profiles share the displayed angular limits. "
               "Save/load applied geometry in File. Masks apply only to angular analysis. "
               "\r\n\r\nCIF: set wavelength / maximum 2theta and load a structure. Guides shows "
               "powder arcs or Qz rods with HKL ticks. For rods, enter sample incidence and normal "
               "phi (0 up, +90 left), then Apply a1/a2 fiber: b3 is the surface normal, full fiber "
               "rotation, air geometry without refraction. P/R/T tables group equal angle / rod / "
               "tick positions. Select a group for cyan highlighting and every signed HKL with "
               "its own raw |F|^2 below. +N means additional members; table max is not a sum. "
               "Labels can be hidden independently of guides. Calculate after editing CIF inputs; "
               "apply edited mounting. Unknown Uiso=0 is explicit. See README for conventions and "
               "limits.");
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
        analysis_invalidate(a);
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
        a->angle_color_dirty = 1;
        InvalidateRect(a->angle_canvas, NULL, FALSE);
        break;
    case SUBTRACT_DARK:
        if (!a->image.dark_counts) {
            CheckDlgButton(a->window, SUBTRACT_DARK, BST_UNCHECKED);
            message(a, "Load a matching dark image first.");
        } else {
            a->subtract = IsDlgButtonChecked(a->window, SUBTRACT_DARK) == BST_CHECKED;
            analysis_invalidate(a);
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
    AppendMenuA(file, MF_STRING, SAVE_GEOMETRY, "Save applied analysis geometry...");
    AppendMenuA(file, MF_STRING, LOAD_GEOMETRY, "Load analysis geometry...");
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
    control(a, "STATIC", "hBN seeds / manual geometry inputs", 0, HBN_LABEL, 10, 206, 230, 22);
    for (i = 0; i < 8; ++i) {
        control(a, "STATIC", labels[i], 0, HBN_LABEL + 1 + i, 10, 234 + 26 * i, 140, 20);
        control(a, "EDIT", "", WS_TABSTOP | ES_AUTOHSCROLL, FIELD + i, 155, 230 + 26 * i, 85, 24);
    }
    control(a, "BUTTON", "Pick initial center", WS_TABSTOP, PICK_CENTER, 10, 447, 230, 26);
    control(a, "BUTTON", "Calculate hBN", WS_TABSTOP, CALIBRATE, 10, 482, 230, 28);
    control(a, "BUTTON", "Cancel", WS_TABSTOP, CANCEL_JOB, 178, 174, 62, 26);
    a->result_label =
        control(a, "EDIT", "", ES_MULTILINE | ES_READONLY | WS_VSCROLL, RESULT, 10, 530, 230, 110);
    a->stage_label = control(
        a, "STATIC", "Open OSC to begin. All coordinates are detector-native (column, row).", 0,
        STAGE, 10, 690, 960, 22);
    a->canvas =
        CreateWindowExA(WS_EX_CLIENTEDGE, "SlateOscCanvas", "", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                        250, 8, 730, 672, a->window, NULL, a->instance, a);
    analysis_controls(a);
    cif_controls(a);
    show_settings(a);
    SetWindowTextA(a->result_label,
                   "Load OSC for viewing.\r\n\r\nFor hBN, also load the matching dark and verify "
                   "the geometry above.\r\n\r\nPreset lattice: a=2.504 A, c=6.661 A.\r\n\r\nClick: "
                   "profiles\r\nDrag: pan\r\nWheel: zoom\r\nShift-drag: rectangular ROI");
    EnableWindow(GetDlgItem(a->window, CANCEL_JOB), FALSE);
    analysis_page(a, 0);
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
        ((MINMAXINFO *)lp)->ptMinTrackSize.y = 720;
        return 0;
    case WM_SIZE:
        analysis_layout(a);
        return 0;
    case WM_COMMAND:
        handle_command(a, LOWORD(wp), HIWORD(wp));
        return 0;
    case WM_NOTIFY:
        return cif_notify(a, (NMHDR *)lp);
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
                analysis_invalidate(a);
                if (a->job == JOB_LOAD) {
                    free(a->mask);
                    a->mask = NULL;
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
                analysis_page(a, a->analysis_page);
            } else if (a->job == JOB_FIT) {
                a->has_result = 1;
                show_result(a);
            } else if (a->job == JOB_CIF) {
                cif_publish(a);
            } else if (a->job == JOB_ANGLES) {
                osc_integration_free(&a->integration);
                a->integration = a->pending_integration;
                memset(&a->pending_integration, 0, sizeof a->pending_integration);
                analysis_publish(a);
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
        InvalidateRect(a->angle_canvas, NULL, FALSE);
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
    wc.lpfnWndProc = angle_canvas_proc;
    wc.lpszClassName = "SlateAngleCanvas";
    if (!RegisterClassExA(&wc))
        return 2;
    wc.lpfnWndProc = window_proc;
    wc.lpszClassName = "SlateOscMain";
    if (!RegisterClassExA(&wc))
        return 2;
    if (!CreateWindowExA(0, wc.lpszClassName, "SLATE OSC + hBN", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                         CW_USEDEFAULT, 1000, 740, NULL, NULL, instance, &app))
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
    free(app.mask);
    free(app.angle_bgr);
    osc_integration_free(&app.integration);
    osc_integration_free(&app.pending_integration);
    cif_peaks_free(&app.cif);
    cif_peaks_free(&app.pending_cif);
    cif_guides_free(&app.guides);
    free(app.cif_rows);
    return 0;
}
