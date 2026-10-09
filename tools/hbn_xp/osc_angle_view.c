#include "osc_app.h"

void analysis_cursor(App *a, int c, int r) {
    char text[300];
    if (analysis_ready(a)) {
        OscAngle p = osc_pixel_angle(&a->geometry, c, r);
        if (p.azimuth_valid)
            snprintf(text, sizeof text,
                     "Column %d row %d | %s %d | 2theta %.5f deg | phi %.5f deg%s", c, r,
                     a->subtract ? "raw-dark" : "counts", pixel(a, c, r), p.theta * 180 / HBN_PI,
                     p.phi * 180 / HBN_PI,
                     osc_is_masked(a->mask, (size_t)r * a->image.columns + c) ? " | MASKED" : "");
        else
            snprintf(text, sizeof text,
                     "Column %d row %d | counts %d | 2theta 0 deg | phi undefined at direct beam",
                     c, r, pixel(a, c, r));
    } else
        snprintf(text, sizeof text,
                 "Column %d row %d | %s %d | Apply matching analysis geometry for 2theta and phi",
                 c, r, a->subtract ? "raw-dark" : "counts", pixel(a, c, r));
    SetWindowTextA(a->stage_label, text);
}
void analysis_overlay(App *a, HDC dc) {
    int edge, i;
    HPEN pen, old;
    if (a->analysis_page != PAGE_ANALYSIS || !analysis_ready(a))
        return;
    pen = CreatePen(PS_SOLID, 1, RGB(255, 220, 0));
    old = (HPEN)SelectObject(dc, pen);
    for (edge = 0; edge < 4; ++edge) {
        int connected = 0;
        if (edge >= 2 && a->selection.phi_max - a->selection.phi_min >= 2 * HBN_PI - 1e-10)
            continue;
        for (i = 0; i <= 180; ++i) {
            double fraction = i / 180.0, c, r, x, y;
            double t = edge < 2 ? (edge ? a->selection.theta_max : a->selection.theta_min)
                                : a->selection.theta_min +
                                      fraction * (a->selection.theta_max - a->selection.theta_min);
            double p = edge < 2 ? a->selection.phi_min +
                                      fraction * (a->selection.phi_max - a->selection.phi_min)
                                : (edge == 2 ? a->selection.phi_min : a->selection.phi_max);
            if (!osc_angle_pixel(&a->geometry, t, p, &c, &r)) {
                connected = 0;
                continue;
            }
            x = (c - a->left) * a->zoom;
            y = (r - a->top) * a->zoom;
            if (fabs(x) > 100000 || fabs(y) > 100000) {
                connected = 0;
                continue;
            }
            if (connected)
                LineTo(dc, (int)x, (int)y);
            else
                MoveToEx(dc, (int)x, (int)y, NULL);
            connected = 1;
        }
    }
    SelectObject(dc, old);
    DeleteObject(pen);
}
static void select_end(App *a, double theta, double phi) {
    double end = a->drag_phi + osc_wrap_phi(phi - a->drag_phi);
    double lo = fmin(a->drag_phi, end), hi = fmax(a->drag_phi, end);
    a->selection.theta_min = fmin(a->drag_theta, theta);
    a->selection.theta_max = fmax(a->drag_theta, theta);
    a->selection.phi_min = osc_wrap_phi(lo);
    a->selection.phi_max = a->selection.phi_min + hi - lo;
}
int analysis_detector_mouse(App *a, UINT msg, WPARAM wp, LPARAM lp) {
    int c, r;
    (void)wp;
    if (a->analysis_page != PAGE_ANALYSIS || a->mouse_tool == TOOL_PAN || a->worker)
        return 0;
    if (msg == WM_LBUTTONDOWN) {
        SetFocus(a->canvas);
        if (!view_coordinate(a, GET_X_LPARAM(lp), GET_Y_LPARAM(lp), &c, &r))
            return 1;
        if (a->mouse_tool == TOOL_SECTOR) {
            OscAngle p;
            if (!analysis_ready(a)) {
                message(a, "Apply matching geometry before selecting angular regions.");
                return 1;
            }
            p = osc_pixel_angle(&a->geometry, c, r);
            if (!p.azimuth_valid)
                return 1;
            a->drag_theta = p.theta;
            a->drag_phi = p.phi;
            a->drag_grid = a->selection;
            a->angle_drag = 1;
            select_end(a, p.theta, p.phi);
        } else {
            a->angle_drag = 3;
            a->has_roi = 1;
            a->roi[0] = a->roi[2] = c;
            a->roi[1] = a->roi[3] = r;
        }
        SetCapture(a->canvas);
        return 1;
    }
    if (msg == WM_MOUSEMOVE && a->angle_drag) {
        if (view_coordinate(a, GET_X_LPARAM(lp), GET_Y_LPARAM(lp), &c, &r)) {
            if (a->angle_drag == 1) {
                OscAngle p = osc_pixel_angle(&a->geometry, c, r);
                if (p.azimuth_valid)
                    select_end(a, p.theta, p.phi);
            } else {
                a->roi[2] = c;
                a->roi[3] = r;
            }
        }
        InvalidateRect(a->canvas, NULL, FALSE);
        InvalidateRect(a->angle_canvas, NULL, FALSE);
        return 1;
    }
    if (msg == WM_LBUTTONUP && a->angle_drag) {
        int mode = a->angle_drag;
        a->angle_drag = 0;
        ReleaseCapture();
        if (mode == 1) {
            analysis_selection_fields(a);
            analysis_begin(a);
        } else {
            if (!a->mask)
                a->mask =
                    (unsigned char *)calloc(((size_t)a->image.rows * a->image.columns + 7) / 8, 1);
            if (!a->mask)
                message(a, "Not enough memory for an analysis mask.");
            else
                osc_mask_box(a->mask, a->image.columns, a->image.rows, a->roi[0], a->roi[1],
                             a->roi[2], a->roi[3], a->mouse_tool == TOOL_MASK);
            a->has_roi = 0;
            analysis_invalidate(a);
            a->color_dirty = 1;
            SetWindowTextA(a->stage_label, "Analysis mask updated. Click Integrate to recompute; "
                                           "raw pixels remain unchanged.");
        }
        InvalidateRect(a->canvas, NULL, FALSE);
        return 1;
    }
    if (msg == WM_CAPTURECHANGED && a->angle_drag) {
        if (a->angle_drag == 1)
            a->selection = a->drag_grid;
        a->angle_drag = 0;
        a->has_roi = 0;
        InvalidateRect(a->canvas, NULL, FALSE);
        InvalidateRect(a->angle_canvas, NULL, FALSE);
        return 1;
    }
    return 0;
}
static RECT plot_rect(const App *a) {
    RECT r = {42, 28, a->angle_width - 10, a->angle_height - 185};
    return r;
}
static int plot_angle(const App *a, int x, int y, double *theta, double *phi) {
    RECT r = plot_rect(a);
    const OscGrid *g = &a->integration.grid;
    if (!a->integration.signal || r.right <= r.left || r.bottom <= r.top || x < r.left ||
        x >= r.right || y < r.top || y >= r.bottom)
        return 0;
    *theta = g->theta_min + (x - r.left) * (g->theta_max - g->theta_min) / (r.right - r.left);
    *phi = g->phi_max - (y - r.top) * (g->phi_max - g->phi_min) / (r.bottom - r.top);
    return 1;
}
static double bin_value(const App *a, size_t i) {
    const OscIntegration *r = &a->integration;
    if (r->area[i] <= 0)
        return NAN;
    return a->output_mode == OUTPUT_SUM    ? r->signal[i]
           : a->output_mode == OUTPUT_AREA ? r->area[i]
                                           : r->signal[i] / r->area[i];
}
static void map_bitmap(App *a, RECT plot) {
    int w = plot.right - plot.left, h = plot.bottom - plot.top, x, y;
    size_t i, n = (size_t)a->integration.grid.theta_bins * a->integration.grid.phi_bins;
    double low = DBL_MAX, high = -DBL_MAX;
    if (w <= 0 || h <= 0)
        return;
    if (a->angle_bitmap_width != w || a->angle_bitmap_height != h) {
        free(a->angle_bgr);
        a->angle_bgr = NULL;
        a->angle_bitmap_width = a->angle_bitmap_height = 0;
        a->angle_bgr = (unsigned char *)malloc((size_t)w * h * 4);
        if (!a->angle_bgr) {
            SetWindowTextA(a->stage_label,
                           "Not enough memory for the angular display. Reduce the window size.");
            return;
        }
        a->angle_bitmap_width = w;
        a->angle_bitmap_height = h;
        a->angle_color_dirty = 1;
    }
    if (!a->angle_color_dirty)
        return;
    for (i = 0; i < n; ++i) {
        double v = bin_value(a, i);
        if (isfinite(v)) {
            low = fmin(low, v);
            high = fmax(high, v);
        }
    }
    if (low == DBL_MAX)
        low = 0;
    if (high <= low)
        high = low + 1;
    for (y = 0; y < h; ++y)
        for (x = 0; x < w; ++x) {
            int t = x * a->integration.grid.theta_bins / w,
                p = (h - 1 - y) * a->integration.grid.phi_bins / h;
            size_t at = (size_t)y * w + x, index = (size_t)p * a->integration.grid.theta_bins + t;
            double value = bin_value(a, index);
            unsigned char gray = 45;
            if (isfinite(value)) {
                double v = fmax(0, value - low), range = high - low;
                gray = (unsigned char)(255 * fmin(1, a->logarithmic ? log1p(v) / log1p(range)
                                                                    : v / range));
            }
            a->angle_bgr[4 * at] = gray;
            a->angle_bgr[4 * at + 1] = gray;
            a->angle_bgr[4 * at + 2] =
                (!isfinite(value) && a->integration.panel_area[index] > 0) ? 100 : gray;
            a->angle_bgr[4 * at + 3] = 0;
        }
    a->angle_color_dirty = 0;
}
static void angular_profile(App *a, HDC dc, RECT box, int phi) {
    const OscIntegration *r = &a->integration;
    int n = phi ? r->grid.phi_bins : r->grid.theta_bins,
        other = phi ? r->grid.theta_bins : r->grid.phi_bins;
    int i, j, x, w = box.right - box.left - 12, h = box.bottom - box.top - 38;
    double *v, lo = DBL_MAX, hi = -DBL_MAX;
    char text[150];
    HPEN pen, old;
    if (w < 2 || h < 2)
        return;
    v = (double *)malloc(n * sizeof *v);
    if (!v)
        return;
    for (i = 0; i < n; ++i) {
        double s = 0, area = 0;
        for (j = 0; j < other; ++j) {
            size_t at = phi ? (size_t)i * other + j : (size_t)j * n + i;
            s += r->signal[at];
            area += r->area[at];
        }
        v[i] = area > 0 ? (a->output_mode == OUTPUT_SUM    ? s
                           : a->output_mode == OUTPUT_AREA ? area
                                                           : s / area)
                        : NAN;
        if (isfinite(v[i])) {
            lo = fmin(lo, v[i]);
            hi = fmax(hi, v[i]);
        }
    }
    if (lo == DBL_MAX)
        lo = 0;
    if (hi <= lo)
        hi = lo + 1;
    snprintf(text, sizeof text, "%s: %.4g .. %.4g", phi ? "I(phi)" : "I(2theta)", lo, hi);
    TextOutA(dc, box.left + 5, box.top + 2, text, (int)strlen(text));
    pen = CreatePen(PS_SOLID, 1, phi ? RGB(30, 100, 200) : RGB(170, 70, 0));
    old = (HPEN)SelectObject(dc, pen);
    for (x = 0; x < w; ++x) {
        int begin = x * n / w, end = (x + 1) * n / w;
        double min = DBL_MAX, max = -DBL_MAX;
        if (end <= begin)
            end = begin + 1;
        for (i = begin; i < end && i < n; ++i)
            if (isfinite(v[i])) {
                min = fmin(min, v[i]);
                max = fmax(max, v[i]);
            }
        if (min != DBL_MAX) {
            MoveToEx(dc, box.left + 6 + x, box.bottom - 20 - (int)((min - lo) / (hi - lo) * h),
                     NULL);
            LineTo(dc, box.left + 6 + x, box.bottom - 21 - (int)((max - lo) / (hi - lo) * h));
        }
    }
    snprintf(text, sizeof text, "%.4g to %.4g deg",
             (phi ? r->grid.phi_min : r->grid.theta_min) * 180 / HBN_PI,
             (phi ? r->grid.phi_max : r->grid.theta_max) * 180 / HBN_PI);
    TextOutA(dc, box.left + 6, box.bottom - 17, text, (int)strlen(text));
    SelectObject(dc, old);
    DeleteObject(pen);
    if (!phi) {
        RECT marker = {box.left + 6, box.top + 18, box.left + 6 + w, box.bottom - 20};
        cif_angle_marker(a, dc, marker);
    }
    free(v);
}
static void paint_angles(App *a, HDC dc, RECT bounds) {
    RECT p = plot_rect(a);
    HGDIOBJ old = SelectObject(dc, a->font);
    FillRect(dc, &bounds, (HBRUSH)(COLOR_WINDOW + 1));
    SetBkMode(dc, TRANSPARENT);
    if (!a->integration.signal || a->worker) {
        const char *text = a->worker ? "Working... Cancel is available."
                                     : "Apply geometry, choose ranges, then Integrate.";
        TextOutA(dc, 12, 22, text, (int)strlen(text));
    } else {
        char text[150];
        const OscGrid *g = &a->integration.grid;
        BITMAPINFO b = {0};
        RECT left = {0, a->angle_height - 151, a->angle_width / 2, a->angle_height};
        RECT right = {a->angle_width / 2, a->angle_height - 151, a->angle_width, a->angle_height};
        map_bitmap(a, p);
        b.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        b.bmiHeader.biWidth = a->angle_bitmap_width;
        b.bmiHeader.biHeight = -a->angle_bitmap_height;
        b.bmiHeader.biPlanes = 1;
        b.bmiHeader.biBitCount = 32;
        b.bmiHeader.biCompression = BI_RGB;
        if (a->angle_bgr)
            SetDIBitsToDevice(dc, p.left, p.top, a->angle_bitmap_width, a->angle_bitmap_height, 0,
                              0, 0, a->angle_bitmap_height, a->angle_bgr, &b, DIB_RGB_COLORS);
        snprintf(text, sizeof text, "Phi vs 2theta | %s",
                 a->output_mode == OUTPUT_MEAN  ? "mean counts"
                 : a->output_mode == OUTPUT_SUM ? "total counts"
                                                : "valid pixel area");
        TextOutA(dc, 8, 6, text, (int)strlen(text));
        snprintf(text, sizeof text, "%.4g", g->phi_max * 180 / HBN_PI);
        TextOutA(dc, 1, p.top, text, (int)strlen(text));
        snprintf(text, sizeof text, "%.4g", g->phi_min * 180 / HBN_PI);
        TextOutA(dc, 1, p.bottom - 15, text, (int)strlen(text));
        snprintf(text, sizeof text, "2theta %.4g .. %.4g deg; phi increases up",
                 g->theta_min * 180 / HBN_PI, g->theta_max * 180 / HBN_PI);
        TextOutA(dc, p.left, p.bottom + 4, text, (int)strlen(text));
        if (a->angle_drag == 2) {
            HGDIOBJ brush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
            HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 180, 0)),
                 oldpen = (HPEN)SelectObject(dc, pen);
            Rectangle(dc,
                      p.left + (int)((a->selection.theta_min - g->theta_min) /
                                     (g->theta_max - g->theta_min) * (p.right - p.left)),
                      p.bottom - (int)((a->selection.phi_max - g->phi_min) /
                                       (g->phi_max - g->phi_min) * (p.bottom - p.top)),
                      p.left + (int)((a->selection.theta_max - g->theta_min) /
                                     (g->theta_max - g->theta_min) * (p.right - p.left)),
                      p.bottom - (int)((a->selection.phi_min - g->phi_min) /
                                       (g->phi_max - g->phi_min) * (p.bottom - p.top)));
            SelectObject(dc, oldpen);
            DeleteObject(pen);
            SelectObject(dc, brush);
        }
        angular_profile(a, dc, left, 0);
        angular_profile(a, dc, right, 1);
        cif_angle_overlay(a, dc, p);
    }
    SelectObject(dc, old);
}
LRESULT CALLBACK angle_canvas_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    App *a = (App *)GetWindowLongPtrA(w, GWLP_USERDATA);
    if (msg == WM_NCCREATE) {
        a = (App *)((CREATESTRUCTA *)lp)->lpCreateParams;
        SetWindowLongPtrA(w, GWLP_USERDATA, (LONG_PTR)a);
    }
    if (!a)
        return DefWindowProcA(w, msg, wp, lp);
    if (msg == WM_ERASEBKGND)
        return 1;
    if (msg == WM_SIZE) {
        a->angle_width = LOWORD(lp);
        a->angle_height = HIWORD(lp);
        a->angle_color_dirty = 1;
        return 0;
    }
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        RECT b;
        HDC dc = BeginPaint(w, &ps);
        GetClientRect(w, &b);
        paint_angles(a, dc, b);
        EndPaint(w, &ps);
        return 0;
    }
    if (a->worker)
        return DefWindowProcA(w, msg, wp, lp);
    if (msg == WM_LBUTTONDOWN || msg == WM_MOUSEMOVE) {
        double theta, phi;
        if (msg == WM_LBUTTONDOWN)
            SetFocus(w);
        if (plot_angle(a, GET_X_LPARAM(lp), GET_Y_LPARAM(lp), &theta, &phi)) {
            if (msg == WM_LBUTTONDOWN && a->analysis_page == PAGE_ANALYSIS &&
                a->mouse_tool == TOOL_SECTOR) {
                a->angle_drag = 2;
                a->drag_theta = theta;
                a->drag_phi = phi;
                a->drag_grid = a->selection;
                a->selection.theta_min = a->selection.theta_max = theta;
                a->selection.phi_min = a->selection.phi_max = phi;
                SetCapture(w);
            } else if (a->angle_drag == 2) {
                a->selection.theta_min = fmin(a->drag_theta, theta);
                a->selection.theta_max = fmax(a->drag_theta, theta);
                a->selection.phi_min = fmin(a->drag_phi, phi);
                a->selection.phi_max = fmax(a->drag_phi, phi);
                InvalidateRect(w, NULL, FALSE);
                InvalidateRect(a->canvas, NULL, FALSE);
            } else {
                double c, r;
                char text[240];
                if (osc_angle_pixel(&a->geometry, theta, phi, &c, &r))
                    snprintf(text, sizeof text,
                             "2theta %.5f deg | phi %.5f deg | native column %.2f row %.2f%s",
                             theta * 180 / HBN_PI, osc_wrap_phi(phi) * 180 / HBN_PI, c, r,
                             c < -.5 || r < -.5 || c >= a->image.columns - .5 ||
                                     r >= a->image.rows - .5
                                 ? " | outside detector"
                                 : "");
                else
                    strcpy(text, "Angular cursor does not intersect the forward detector.");
                SetWindowTextA(a->stage_label, text);
            }
        }
        return 0;
    }
    if (msg == WM_LBUTTONUP && a->angle_drag == 2) {
        a->angle_drag = 0;
        ReleaseCapture();
        analysis_selection_fields(a);
        analysis_begin(a);
        return 0;
    }
    if (msg == WM_CAPTURECHANGED && a->angle_drag == 2) {
        a->selection = a->drag_grid;
        a->angle_drag = 0;
        InvalidateRect(w, NULL, FALSE);
        InvalidateRect(a->canvas, NULL, FALSE);
        return 0;
    }
    return DefWindowProcA(w, msg, wp, lp);
}
