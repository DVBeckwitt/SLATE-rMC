#include "osc_app.h"

void make_preview(App *a) {
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
void auto_contrast(App *a) {
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
void fit_view(App *a) {
    if (!a->image.counts || a->view_width < 1 || a->view_height < 1)
        return;
    a->zoom =
        fmin((double)a->view_width / a->image.columns, (double)a->view_height / a->image.rows);
    a->left = -(a->view_width / a->zoom - a->image.columns) * .5 - .5;
    a->top = -(a->view_height / a->zoom - a->image.rows) * .5 - .5;
    a->view_dirty = 1;
}
int view_coordinate(App *a, int x, int y, int *c, int *r) {
    if (!a->image.counts || a->zoom <= 0)
        return 0;
    *c = (int)floor(a->left + x / a->zoom + .5);
    *r = (int)floor(a->top + y / a->zoom + .5);
    return *c >= 0 && *r >= 0 && *c < a->image.columns && *r < a->image.rows && y < a->view_height;
}
void render_view(App *a) {
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
                {
                    int c, r;
                    if (view_coordinate(a, x, y, &c, &r) &&
                        osc_is_masked(a->mask, (size_t)r * a->image.columns + c)) {
                        a->screen_bgr[4 * at] = (unsigned char)(gray / 2);
                        a->screen_bgr[4 * at + 1] = (unsigned char)(gray / 2);
                        a->screen_bgr[4 * at + 2] = (unsigned char)(128 + gray / 2);
                    }
                }
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
        HBRUSH brush = CreateSolidBrush(RGB(255, 100, 70));
        for (i = 0; i < a->result.point_count; ++i) {
            int x = (int)((a->result.points[i].column - a->left) * a->zoom),
                y = (int)((a->result.points[i].row - a->top) * a->zoom);
            RECT dot = {x - 2, y - 2, x + 3, y + 3};
            FillRect(dc, &dot, brush);
        }
        DeleteObject(brush);
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
    analysis_overlay(a, dc);
    cif_detector_overlay(a, dc);
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
void inspect_roi(App *a) {
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
void zoom_at(App *a, double factor, int x, int y) {
    double c = a->left + x / a->zoom, r = a->top + y / a->zoom;
    a->zoom = fmax(.02, fmin(32, a->zoom * factor));
    a->left = c - x / a->zoom;
    a->top = r - y / a->zoom;
    a->view_dirty = 1;
    InvalidateRect(a->canvas, NULL, FALSE);
}
LRESULT CALLBACK canvas_proc(HWND w, UINT message_id, WPARAM wp, LPARAM lp) {
    App *a = (App *)GetWindowLongPtrA(w, GWLP_USERDATA);
    if (message_id == WM_NCCREATE) {
        a = (App *)((CREATESTRUCTA *)lp)->lpCreateParams;
        SetWindowLongPtrA(w, GWLP_USERDATA, (LONG_PTR)a);
    }
    if (!a)
        return DefWindowProcA(w, message_id, wp, lp);
    if (analysis_detector_mouse(a, message_id, wp, lp))
        return 0;
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
    case WM_SIZE: {
        int old_width = a->view_width, old_height = a->view_height;
        a->view_width = LOWORD(lp);
        a->view_height = HIWORD(lp) > 140 ? HIWORD(lp) - 140 : 1;
        if (a->zoom <= 0)
            fit_view(a);
        else {
            /* Keep the same native detector point at the viewport center. */
            a->left += (old_width - a->view_width) / (2 * a->zoom);
            a->top += (old_height - a->view_height) / (2 * a->zoom);
        }
        a->view_dirty = 1;
        return 0;
    }
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
            analysis_cursor(a, c, r);
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
