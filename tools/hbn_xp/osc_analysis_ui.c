#include "osc_app.h"

int analysis_ready(const App *a) {
    return a->geometry_valid && !a->geometry_dirty && a->image.counts &&
           a->image.rows == a->geometry.rows && a->image.columns == a->geometry.columns;
}
void analysis_invalidate(App *a) {
    osc_integration_free(&a->integration);
    a->angle_color_dirty = 1;
    if (a->analysis_page == PAGE_ANALYSIS && a->result_label)
        SetWindowTextA(a->result_label, "Analysis inputs changed. Click Integrate to recompute.");
    if (a->angle_canvas)
        InvalidateRect(a->angle_canvas, NULL, FALSE);
    if (a->canvas)
        InvalidateRect(a->canvas, NULL, FALSE);
    cif_status(a);
}
static void geometry_status(App *a) {
    const char *text = !a->geometry_valid   ? "Apply geometry to enable angles."
                       : a->geometry_dirty  ? "Distance edited; click Apply."
                       : !analysis_ready(a) ? "Geometry does not match image size."
                                            : a->calibration_text;
    SetWindowTextA(GetDlgItem(a->window, APPLIED_GEOMETRY), text);
}
void analysis_layout(App *a) {
    RECT r;
    int w, h, mode = a->analysis_page ? a->view_mode : VIEW_DETECTOR;
    if (!a->canvas || !a->angle_canvas)
        return;
    GetClientRect(a->window, &r);
    w = r.right - 260;
    h = r.bottom - 40;
    if (w < 2)
        w = 2;
    if (h < 2)
        h = 2;
    ShowWindow(a->canvas, mode == VIEW_ANGLES ? SW_HIDE : SW_SHOW);
    ShowWindow(a->angle_canvas, mode == VIEW_DETECTOR ? SW_HIDE : SW_SHOW);
    MoveWindow(a->canvas, 250, 8, mode == VIEW_SPLIT ? w / 2 : w, h, TRUE);
    MoveWindow(a->angle_canvas, mode == VIEW_SPLIT ? 254 + w / 2 : 250, 8,
               mode == VIEW_SPLIT ? w - w / 2 - 4 : w, h, TRUE);
    MoveWindow(a->stage_label, 10, r.bottom - 26, r.right - 20, 22, TRUE);
    MoveWindow(a->result_label, 10, 530, 230, r.bottom > 570 ? r.bottom - 570 : 1, TRUE);
    a->angle_color_dirty = 1;
    cif_layout(a);
}
void analysis_page(App *a, int enabled) {
    HWND child = GetWindow(a->window, GW_CHILD);
    int changed = a->analysis_page != enabled;
    if (GetCapture() == a->canvas || GetCapture() == a->angle_canvas)
        ReleaseCapture();
    a->analysis_page = enabled;
    a->pick_center = 0;
    SetWindowTextA(GetDlgItem(a->window, PICK_CENTER), "Pick initial center");
    while (child) {
        int id = GetDlgCtrlID(child);
        if ((id >= FIELD && id < FIELD + 8) || id == PICK_CENTER || id == CALIBRATE ||
            (id >= HBN_LABEL && id < HBN_LABEL + 10))
            ShowWindow(child, enabled == PAGE_HBN ? SW_SHOW : SW_HIDE);
        else if (id >= APPLIED_GEOMETRY && id < ANALYSIS_LABEL + 40)
            ShowWindow(child, enabled == PAGE_ANALYSIS ? SW_SHOW : SW_HIDE);
        else if (id >= CIF_OPEN && id < CIF_END)
            ShowWindow(child, enabled == PAGE_CIF ? SW_SHOW : SW_HIDE);
        child = GetWindow(child, GW_HWNDNEXT);
    }
    analysis_layout(a);
    if (changed)
        fit_view(a);
    geometry_status(a);
    ShowWindow(a->result_label, enabled == PAGE_CIF ? SW_HIDE : SW_SHOW);
    SendMessageA(GetDlgItem(a->window, CIF_VIEW), CB_SETCURSEL, a->view_mode, 0);
    cif_status(a);
    if (enabled == PAGE_CIF)
        return;
    if (enabled == PAGE_ANALYSIS) {
        if (a->integration.signal)
            analysis_publish(a);
        else
            SetWindowTextA(a->result_label, "Apply geometry, then Integrate.\r\n\r\n"
                                            "Sector: drag two angular corners in either image.\r\n"
                                            "Mask: drag a detector rectangle.\r\n"
                                            "Phi is scattering azimuth; zero is up, positive "
                                            "toward left in the untilted view.");
    } else if (a->has_result)
        show_result(a);
    else
        SetWindowTextA(a->result_label,
                       "Open hBN and its matching dark, verify seeds, then Calculate hBN.");
}
static void combo(App *a, int id, int y, const char *items, int selected) {
    HWND w =
        control(a, "COMBOBOX", "", WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL, id, 92, y, 148, 180);
    const char *p = items;
    while (*p) {
        SendMessageA(w, CB_ADDSTRING, 0, (LPARAM)p);
        p += strlen(p) + 1;
    }
    SendMessageA(w, CB_SETCURSEL, selected, 0);
}
void analysis_controls(App *a) {
    int i;
    const char *labels[] = {"Sample mm", "2theta deg", "Phi deg",       "Bin width deg",
                            "View",      "Mouse tool", "Map / profiles"};
    control(a, "BUTTON", "Analysis", WS_TABSTOP, ANALYSIS_PAGE, 10, 174, 64, 26);
    control(a, "BUTTON", "hBN", WS_TABSTOP, CALIBRATION_PAGE, 80, 174, 40, 26);
    control(a, "STATIC", "Apply geometry to enable angles.", 0, APPLIED_GEOMETRY, 10, 204, 230, 20);
    control(a, "BUTTON", "Apply inputs", WS_TABSTOP, APPLY_INPUTS, 10, 226, 108, 25);
    control(a, "BUTTON", "Use hBN fit", WS_TABSTOP, APPLY_HBN, 124, 226, 116, 25);
    for (i = 0; i < 7; ++i)
        control(a, "STATIC", labels[i], 0, ANALYSIS_LABEL + i, 10, 258 + i * 30, 82, 21);
    control(a, "EDIT", "74", WS_TABSTOP | ES_AUTOHSCROLL, SAMPLE_DISTANCE, 92, 254, 83, 24);
    control(a, "BUTTON", "Apply", WS_TABSTOP, APPLY_DISTANCE, 181, 254, 59, 24);
    control(a, "EDIT", "0", WS_TABSTOP | ES_AUTOHSCROLL, THETA_MIN, 92, 284, 69, 24);
    control(a, "EDIT", "80", WS_TABSTOP | ES_AUTOHSCROLL, THETA_MAX, 170, 284, 70, 24);
    control(a, "EDIT", "-180", WS_TABSTOP | ES_AUTOHSCROLL, PHI_MIN, 92, 314, 69, 24);
    control(a, "EDIT", "180", WS_TABSTOP | ES_AUTOHSCROLL, PHI_MAX, 170, 314, 70, 24);
    control(a, "EDIT", "0.1", WS_TABSTOP | ES_AUTOHSCROLL, THETA_STEP, 92, 344, 69, 24);
    control(a, "EDIT", "1", WS_TABSTOP | ES_AUTOHSCROLL, PHI_STEP, 170, 344, 70, 24);
    combo(a, VIEW_MODE, 374, "Detector\0Phi vs 2theta\0Side by side\0\0", VIEW_SPLIT);
    combo(a, MOUSE_TOOL, 404, "Inspect / pan\0Angular sector\0Mask rectangle\0Unmask rectangle\0\0",
          TOOL_PAN);
    combo(a, OUTPUT_MODE, 434, "Mean counts\0Total counts\0Valid pixel area\0\0", OUTPUT_MEAN);
    control(a, "BUTTON", "Integrate", WS_TABSTOP, INTEGRATE, 10, 465, 110, 27);
    control(a, "BUTTON", "Full range", WS_TABSTOP, FULL_ANGLES, 128, 465, 112, 27);
    control(a, "BUTTON", "Export angles...", WS_TABSTOP, EXPORT_ANGLES, 10, 499, 110, 25);
    control(a, "BUTTON", "Clear mask", WS_TABSTOP, CLEAR_MASK, 128, 499, 112, 25);
    a->selection = (OscGrid){0, 80 * HBN_PI / 180, -HBN_PI, HBN_PI, 800, 360};
    a->view_mode = VIEW_SPLIT;
    a->angle_canvas = CreateWindowExA(WS_EX_CLIENTEDGE, "SlateAngleCanvas", "",
                                      WS_CHILD | WS_VISIBLE | WS_TABSTOP, 250, 8, 730, 672,
                                      a->window, NULL, a->instance, a);
}
static void applied(App *a, const OscGeometry *g) {
    a->geometry = *g;
    a->geometry_valid = 1;
    a->geometry_dirty = 0;
    snprintf(a->calibration_text, sizeof a->calibration_text, "%s; %d x %d px",
             g->from_qualified_hbn ? "hBN pose applied" : "Manual geometry applied", g->columns,
             g->rows);
    a->updating_analysis = 1;
    set_number(a, SAMPLE_DISTANCE, g->settings.initial[4] * 1000);
    a->updating_analysis = 0;
    analysis_invalidate(a);
    geometry_status(a);
    SetWindowTextA(a->result_label, g->distance_from_calibrant
                                        ? "hBN pose and distance applied. This assumes the sample "
                                          "is at the calibrant position. "
                                          "Set Sample mm for a different distance."
                                        : "Analysis geometry applied. Check sample distance and "
                                          "calibration before integration.");
}
void analysis_selection_fields(App *a) {
    a->updating_analysis = 1;
    set_number(a, THETA_MIN, a->selection.theta_min * 180 / HBN_PI);
    set_number(a, THETA_MAX, a->selection.theta_max * 180 / HBN_PI);
    set_number(a, PHI_MIN, a->selection.phi_min * 180 / HBN_PI);
    set_number(a, PHI_MAX, a->selection.phi_max * 180 / HBN_PI);
    a->updating_analysis = 0;
}
void analysis_begin(App *a) {
    double v[6], nt, np;
    int i;
    OscGrid g;
    if (a->worker)
        return;
    if (a->geometry_dirty)
        analysis_command(a, APPLY_DISTANCE, BN_CLICKED);
    if (!analysis_ready(a)) {
        message(a, "Apply matching analysis geometry first. Use hBN fit or the geometry inputs on "
                   "the hBN page.");
        return;
    }
    for (i = 0; i < 6; ++i)
        if (!number(a, THETA_MIN + i, v + i)) {
            message(a, "Enter finite angular limits and bin widths.");
            return;
        }
    if (v[3] <= v[2])
        v[3] += 360;
    if (v[4] <= 0 || v[5] <= 0 || v[1] <= v[0] || v[3] <= v[2] || v[3] - v[2] > 360) {
        message(a, "Use positive bin widths and increasing ranges. A phi end below the start "
                   "crosses the seam.");
        return;
    }
    nt = ceil((v[1] - v[0]) / v[4]);
    np = ceil((v[3] - v[2]) / v[5]);
    if (nt > 4096 || np > 1440 || nt * np > 1048576) {
        message(a, "Too many bins. Increase bin widths (at most 4096 x 1440 / 1,048,576 total).");
        return;
    }
    g.theta_min = v[0] * HBN_PI / 180;
    g.theta_max = v[1] * HBN_PI / 180;
    g.phi_min = osc_wrap_phi(v[2] * HBN_PI / 180);
    g.phi_max = g.phi_min + (v[3] - v[2]) * HBN_PI / 180;
    g.theta_bins = (int)nt;
    g.phi_bins = (int)np;
    if (!osc_grid_valid(&g, a->error)) {
        message(a, a->error);
        return;
    }
    a->selection = g;
    analysis_selection_fields(a);
    analysis_invalidate(a);
    start_job(a, JOB_ANGLES, NULL);
}
void analysis_publish(App *a) {
    const OscIntegration *r = &a->integration;
    size_t i, n = (size_t)r->grid.theta_bins * r->grid.phi_bins;
    double signal = 0, area = 0;
    char text[700];
    if (!r->signal)
        return;
    for (i = 0; i < n; ++i) {
        signal += r->signal[i];
        area += r->area[i];
    }
    snprintf(
        text, sizeof text,
        "%d x %d angular bins\r\nSigned sum: %.10g\r\nValid pixel area: %.8g\r\n"
        "Masked pixels: %.0f\r\nOutside-window area: %.8g\r\n\r\n"
        "Profiles reduce signal and area before taking the mean. Gray bins have no valid data.\r\n"
        "Coverage is effective detector pixels, not a full-ring percentage.",
        r->grid.theta_bins, r->grid.phi_bins, signal, area, r->masked_area, r->lost_area);
    SetWindowTextA(a->result_label, text);
    a->angle_color_dirty = 1;
    InvalidateRect(a->angle_canvas, NULL, FALSE);
    geometry_status(a);
}
int analysis_export(App *a, const char *path) {
    char temporary[MAX_PATH];
    FILE *f = begin_output(path, temporary, a->error);
    int ok;
    if (!f)
        return 0;
    ok = osc_integration_write(f, &a->image, a->subtract, &a->geometry, &a->integration,
                               osc_mask_crc32(a->mask, (size_t)a->image.rows * a->image.columns),
                               app_progress, a);
    if (InterlockedCompareExchange(&a->cancel, 0, 0)) {
        ok = 0;
        strcpy(a->error, "Canceled.");
    }
    return finish_output(f, temporary, path, ok, a->error);
}
int analysis_command(App *a, int id, int notification) {
    char path[MAX_PATH];
    OscGeometry g;
    if (id == ANALYSIS_PAGE || id == CALIBRATION_PAGE) {
        analysis_page(a, id == ANALYSIS_PAGE);
        return 1;
    }
    if (id < SAMPLE_DISTANCE && id != APPLY_INPUTS && id != APPLY_HBN)
        return 0;
    if (a->updating_analysis)
        return 1;
    if (id >= THETA_MIN && id <= PHI_STEP && notification == EN_CHANGE) {
        analysis_invalidate(a);
        return 1;
    }
    if (id == SAMPLE_DISTANCE && notification == EN_CHANGE) {
        a->geometry_dirty = 1;
        analysis_invalidate(a);
        geometry_status(a);
        return 1;
    }
    switch (id) {
    case VIEW_MODE:
        if (notification == CBN_SELCHANGE) {
            if (GetCapture() == a->canvas || GetCapture() == a->angle_canvas)
                ReleaseCapture();
            a->view_mode = (int)SendDlgItemMessageA(a->window, id, CB_GETCURSEL, 0, 0);
            analysis_layout(a);
            fit_view(a);
        }
        return 1;
    case MOUSE_TOOL:
        if (notification == CBN_SELCHANGE) {
            if (GetCapture() == a->canvas || GetCapture() == a->angle_canvas)
                ReleaseCapture();
            a->mouse_tool = (int)SendDlgItemMessageA(a->window, id, CB_GETCURSEL, 0, 0);
        }
        return 1;
    case OUTPUT_MODE:
        if (notification == CBN_SELCHANGE) {
            a->output_mode = (int)SendDlgItemMessageA(a->window, id, CB_GETCURSEL, 0, 0);
            a->angle_color_dirty = 1;
            InvalidateRect(a->angle_canvas, NULL, FALSE);
        }
        return 1;
    case APPLY_INPUTS:
    case APPLY_HBN:
        if (!a->image.counts) {
            message(a, "Open an image first.");
            return 1;
        }
        memset(&g, 0, sizeof g);
        g.rows = a->image.rows;
        g.columns = a->image.columns;
        if (id == APPLY_HBN) {
            if (!a->has_result || !a->result.qualified) {
                message(a, "A qualified hBN fit is required. Manual geometry remains available "
                           "through Apply inputs.");
                return 1;
            }
            g.settings = a->settings;
            memcpy(g.settings.initial, a->result.values, sizeof a->result.values);
            g.from_qualified_hbn = g.distance_from_calibrant = 1;
            g.calibrant_crc32 = a->image.source_crc32;
            g.dark_crc32 = a->image.dark_crc32;
        } else {
            if (!read_settings(a))
                return 1;
            g.settings = a->settings;
        }
        if (!osc_geometry_compile(&g, a->error))
            message(a, a->error);
        else
            applied(a, &g);
        return 1;
    case APPLY_DISTANCE: {
        double d;
        if (!a->geometry_valid) {
            message(a, "Apply a geometry first.");
            return 1;
        }
        if (!number(a, SAMPLE_DISTANCE, &d)) {
            message(a, "Enter a positive sample distance in mm.");
            return 1;
        }
        g = a->geometry;
        g.settings.initial[4] = d / 1000;
        g.distance_from_calibrant = 0;
        if (!osc_geometry_compile(&g, a->error))
            message(a, a->error);
        else
            applied(a, &g);
        return 1;
    }
    case INTEGRATE:
        analysis_begin(a);
        return 1;
    case FULL_ANGLES:
        a->selection = (OscGrid){0, 80 * HBN_PI / 180, -HBN_PI, HBN_PI, 800, 360};
        analysis_selection_fields(a);
        analysis_invalidate(a);
        return 1;
    case EXPORT_ANGLES:
        if (!a->integration.signal) {
            message(a, "Integrate the current selection first.");
            return 1;
        }
        if (choose_path(a, path, 1, "Angular map and profiles CSV\0*.csv\0\0", "csv"))
            start_job(a, JOB_ANGLE_EXPORT, path);
        return 1;
    case CLEAR_MASK:
        if (a->mask)
            memset(a->mask, 0, ((size_t)a->image.rows * a->image.columns + 7) / 8);
        analysis_invalidate(a);
        a->color_dirty = 1;
        return 1;
    case SAVE_GEOMETRY:
        if (!analysis_ready(a)) {
            message(a, "Apply valid geometry for this image before saving.");
            return 1;
        }
        if (choose_path(a, path, 1, "Analysis geometry CSV\0*.csv\0\0", "csv")) {
            char temporary[MAX_PATH];
            FILE *f;
            a->error[0] = 0;
            f = begin_output(path, temporary, a->error);
            if (!f ||
                !finish_output(f, temporary, path, osc_geometry_write(f, &a->geometry), a->error))
                message(a, a->error);
        }
        return 1;
    case LOAD_GEOMETRY:
        if (choose_path(a, path, 0, "Analysis geometry CSV\0*.csv\0\0", "csv")) {
            if (!osc_geometry_read(path, &g, a->error))
                message(a, a->error);
            else if (a->image.counts && (a->image.rows != g.rows || a->image.columns != g.columns))
                message(a, "Saved geometry dimensions do not match this image. Previous geometry "
                           "is preserved.");
            else
                applied(a, &g);
        }
        return 1;
    }
    return 0;
}
