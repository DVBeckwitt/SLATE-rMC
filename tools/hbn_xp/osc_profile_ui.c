#include "osc_app.h"

enum { P_AXIS = 1000, P_MODE, P_EXPORT, P_INCIDENCE, P_NORMAL, P_BINS, P_APPLY, P_NOTE, P_CURSOR };

static void profile_text(App *a, int id, const char *text) {
    SetWindowTextA(GetDlgItem(a->profile_window, id), text);
}
void profile_invalidate(App *a) {
    osc_profile_free(&a->profile);
    if (a->profile_window) {
        EnableWindow(GetDlgItem(a->profile_window, P_EXPORT), FALSE);
        profile_text(a, P_NOTE, "Integration inputs changed. Integrate the region again.");
        profile_text(a, P_CURSOR, "");
        InvalidateRect(a->profile_window, NULL, FALSE);
    }
}
static void sync_mount_fields(App *a) {
    const int source[] = {CIF_INCIDENCE, CIF_NORMAL_PHI}, target[] = {P_INCIDENCE, P_NORMAL};
    int i;
    if (!a->profile_window)
        return;
    a->profile_updating = 1;
    for (i = 0; i < 2; ++i) {
        char draft[64], previous[64];
        GetDlgItemTextA(a->window, source[i], draft, sizeof draft);
        GetDlgItemTextA(a->profile_window, target[i], previous, sizeof previous);
        if (strcmp(draft, previous))
            profile_text(a, target[i], draft);
    }
    a->profile_updating = 0;
}
static void update_profile(App *a) {
    int q = a->profile_axis >= PROFILE_QZ, ok = 0;
    const int ids[] = {P_INCIDENCE, P_NORMAL, P_BINS, P_APPLY};
    size_t i;
    if (!a->profile_window)
        return;
    osc_profile_free(&a->profile);
    a->profile_error[0] = 0;
    sync_mount_fields(a);
    for (i = 0; i < sizeof ids / sizeof ids[0]; ++i)
        EnableWindow(GetDlgItem(a->profile_window, ids[i]), q);
    if (a->worker || !a->integration.signal)
        strcpy(a->profile_error, "Integrate the current angular region to see its 1D profile.");
    else if (q && a->profile_bins_dirty)
        strcpy(a->profile_error, "Q bin count edited. Click Apply to redraw.");
    else if (q && (!a->cif_mount_applied || a->cif_mount_dirty))
        strcpy(a->profile_error,
               "For Qz/Qr, enter the sample orientation and click Apply. No CIF is required.");
    else
        ok = osc_profile_build(&a->integration, a->profile_axis, a->profile_q_bins,
                               a->geometry.settings.wavelength_A, a->cif_mount, &a->profile,
                               a->profile_error);
    EnableWindow(GetDlgItem(a->profile_window, P_EXPORT), ok);
    profile_text(a, P_NOTE,
                 ok ? (q ? "External sample Q: approximate rebin from angular-bin centers. Refine "
                           "angular bins for finer resolution."
                         : "The profile covers the entire integrated region. Mean = total signal / "
                           "valid pixel area; signed counts are retained.")
                    : a->profile_error);
    profile_text(a, P_CURSOR, "Move the mouse over the plot to read a bin.");
    InvalidateRect(a->profile_window, NULL, FALSE);
}
void profile_mount_changed(App *a) {
    sync_mount_fields(a);
    if (a->profile_axis >= PROFILE_QZ)
        update_profile(a);
}
void analysis_mount_apply(App *a, OscMount mount) {
    a->cif_mount = mount;
    a->cif_mount_applied = 1;
    a->cif_mount_dirty = 0;
    a->guide_attempted = 0;
    a->guide_error[0] = 0;
    profile_mount_changed(a);
    cif_status(a);
    InvalidateRect(a->canvas, NULL, FALSE);
    InvalidateRect(a->angle_canvas, NULL, FALSE);
}
static HWND child(App *a, const char *class_name, const char *text, int id, DWORD style, int x,
                  int y, int width, int height) {
    HWND w = CreateWindowExA(!strcmp(class_name, "EDIT") ? WS_EX_CLIENTEDGE : 0, class_name, text,
                             WS_CHILD | WS_VISIBLE | style, x, y, width, height, a->profile_window,
                             (HMENU)(INT_PTR)id, a->instance, NULL);
    SendMessageA(w, WM_SETFONT, (WPARAM)a->font, TRUE);
    return w;
}
static void choices(App *a, int id, const char *items, int selected, int x, int width) {
    HWND w = child(a, "COMBOBOX", "", id, WS_TABSTOP | CBS_DROPDOWNLIST, x, 10, width, 150);
    while (*items) {
        SendMessageA(w, CB_ADDSTRING, 0, (LPARAM)items);
        items += strlen(items) + 1;
    }
    SendMessageA(w, CB_SETCURSEL, selected, 0);
}
static RECT plot_rect(HWND w) {
    RECT box;
    GetClientRect(w, &box);
    box.left = 80;
    box.top = 166;
    box.right -= 24;
    box.bottom -= 66;
    return box;
}
static double ordinate(const App *a, int i) {
    const OscProfile *p = &a->profile;
    if (p->area[i] <= 0)
        return NAN;
    return a->profile_mode == OUTPUT_SUM    ? p->signal[i]
           : a->profile_mode == OUTPUT_AREA ? p->area[i]
                                            : p->signal[i] / p->area[i];
}
static void paint_profile(App *a, HDC dc) {
    const char *names[] = {"2theta (deg)", "phi (deg)", "Qz (A^-1)", "Qr (A^-1)"};
    const char *modes[] = {"Mean counts", "Total counts", "Valid pixel area"};
    RECT whole, box = plot_rect(a->profile_window);
    const OscProfile *p = &a->profile;
    double lo = DBL_MAX, hi = -DBL_MAX, scale = p->axis < PROFILE_QZ ? 180 / HBN_PI : 1;
    int i, connected = 0;
    HGDIOBJ oldfont = SelectObject(dc, a->font);
    char text[160];
    GetClientRect(a->profile_window, &whole);
    FillRect(dc, &whole, (HBRUSH)(COLOR_BTNFACE + 1));
    FillRect(dc, &box, (HBRUSH)(COLOR_WINDOW + 1));
    SetBkMode(dc, TRANSPARENT);
    if (!p->signal) {
        DrawTextA(dc,
                  "Choose an axis above. A completed integration is required; Q axes also need an "
                  "applied sample orientation.",
                  -1, &box, DT_WORDBREAK);
        SelectObject(dc, oldfont);
        return;
    }
    for (i = 0; i < p->bins; ++i) {
        double v = ordinate(a, i);
        if (isfinite(v)) {
            lo = fmin(lo, v);
            hi = fmax(hi, v);
        }
    }
    if (lo == DBL_MAX) {
        const char *text = "No valid pixel support in this region.";
        TextOutA(dc, box.left + 12, box.top + 30, text, (int)strlen(text));
        lo = 0;
        hi = 1;
    } else if (hi == lo) {
        double margin = fmax(1, fabs(lo) * .05);
        lo -= margin;
        hi += margin;
    }
    snprintf(text, sizeof text, "%s versus %s | %d bins", modes[a->profile_mode], names[p->axis],
             p->bins);
    TextOutA(dc, box.left, box.top - 21, text, (int)strlen(text));
    for (i = 0; i <= 4; ++i) {
        int x = box.left + i * (box.right - box.left) / 4;
        int y = box.bottom - i * (box.bottom - box.top) / 4;
        SIZE extent;
        snprintf(text, sizeof text, "%.5g", (p->lower + (p->upper - p->lower) * i / 4) * scale);
        GetTextExtentPoint32A(dc, text, (int)strlen(text), &extent);
        TextOutA(dc, i == 0 ? x : x - (i == 4 ? extent.cx : extent.cx / 2), box.bottom + 6, text,
                 (int)strlen(text));
        snprintf(text, sizeof text, "%.5g", lo + (hi - lo) * i / 4);
        TextOutA(dc, 8, y - 7, text, (int)strlen(text));
    }
    MoveToEx(dc, box.left, box.top, NULL);
    LineTo(dc, box.left, box.bottom);
    LineTo(dc, box.right, box.bottom);
    {
        HPEN pen = CreatePen(PS_SOLID, 1, RGB(30, 85, 175));
        HPEN old = (HPEN)SelectObject(dc, pen);
        for (i = 0; i < p->bins; ++i) {
            double value = ordinate(a, i);
            int x, y;
            if (!isfinite(value)) {
                connected = 0;
                continue;
            }
            x = box.left + (int)((i + .5) * (box.right - box.left) / p->bins);
            y = box.bottom - (int)((value - lo) / (hi - lo) * (box.bottom - box.top));
            if (connected)
                LineTo(dc, x, y);
            else
                MoveToEx(dc, x, y, NULL);
            SetPixel(dc, x, y, RGB(30, 85, 175));
            connected = 1;
        }
        SelectObject(dc, old);
        DeleteObject(pen);
    }
    SelectObject(dc, oldfont);
}
static void export_profile(App *a) {
    char path[MAX_PATH], temporary[MAX_PATH];
    FILE *f;
    int ok, chosen;
    if (!a->profile.signal)
        return;
    EnableWindow(a->profile_window, FALSE);
    chosen = choose_path(a, path, 1, "1D intensity profile CSV\0*.csv\0\0", "csv");
    EnableWindow(a->profile_window, TRUE);
    if (!chosen || !a->profile.signal)
        return;
    f = begin_output(path, temporary, a->error);
    if (!f) {
        message(a, a->error);
        return;
    }
    fprintf(f,
            "# SLATE integrated-region 1D profile; plotted_column=%s\n"
            "# source_crc32=%08lx,dark_crc32=%08lx,mask_crc32=%08lx,values=%s\n"
            "# no exposure, background, polarization or solid-angle correction; dark scale=1\n",
            a->profile_mode == OUTPUT_MEAN  ? "mean_counts"
            : a->profile_mode == OUTPUT_SUM ? "signal_counts"
                                            : "valid_pixel_area",
            a->image.source_crc32, a->subtract ? a->image.dark_crc32 : 0ul,
            osc_mask_crc32(a->mask, (size_t)a->image.rows * a->image.columns),
            a->subtract ? "raw-dark" : "raw");
    ok = osc_geometry_write(f, &a->geometry) && osc_profile_write(f, &a->profile);
    if (!finish_output(f, temporary, path, ok, a->error))
        message(a, a->error);
}
LRESULT CALLBACK profile_window_proc(HWND w, UINT msg, WPARAM wp, LPARAM lp) {
    App *a = (App *)GetWindowLongPtrA(w, GWLP_USERDATA);
    if (msg == WM_NCCREATE) {
        a = (App *)((CREATESTRUCTA *)lp)->lpCreateParams;
        SetWindowLongPtrA(w, GWLP_USERDATA, (LONG_PTR)a);
        a->profile_window = w;
    }
    if (!a)
        return DefWindowProcA(w, msg, wp, lp);
    switch (msg) {
    case WM_CREATE: {
        char bins[20];
        a->profile_updating = 1;
        child(a, "STATIC", "Plot intensity vs", -1, 0, 12, 14, 95, 22);
        choices(a, P_AXIS, "2theta (deg)\0phi (deg)\0Qz (A^-1)\0Qr (A^-1)\0\0", a->profile_axis,
                110, 132);
        choices(a, P_MODE, "Mean counts\0Total counts\0Valid pixel area\0\0", a->profile_mode, 252,
                142);
        child(a, "BUTTON", "Export profile...", P_EXPORT, WS_TABSTOP, 410, 10, 125, 25);
        child(a, "STATIC", "Incidence deg", -1, 0, 12, 49, 82, 21);
        child(a, "EDIT", "0", P_INCIDENCE, WS_TABSTOP | ES_AUTOHSCROLL, 96, 45, 64, 24);
        child(a, "STATIC", "Normal phi deg", -1, 0, 172, 49, 87, 21);
        child(a, "EDIT", "0", P_NORMAL, WS_TABSTOP | ES_AUTOHSCROLL, 260, 45, 64, 24);
        child(a, "STATIC", "Q bins", -1, 0, 337, 49, 42, 21);
        snprintf(bins, sizeof bins, "%d", a->profile_q_bins);
        child(a, "EDIT", bins, P_BINS, WS_TABSTOP | ES_AUTOHSCROLL, 383, 45, 58, 24);
        child(a, "BUTTON", "Apply", P_APPLY, WS_TABSTOP, 454, 45, 81, 24);
        child(a, "STATIC",
              "Sample normal phi: 0 up, +90 left. Positive incidence enters the surface. Shared "
              "with CIF guides.",
              -1, 0, 12, 77, 620, 20);
        child(a, "STATIC", "", P_NOTE, 0, 12, 99, 740, 30);
        child(a, "STATIC", "", P_CURSOR, 0, 12, 460, 740, 22);
        a->profile_updating = 0;
        return 0;
    }
    case WM_GETMINMAXINFO:
        ((MINMAXINFO *)lp)->ptMinTrackSize.x = 660;
        ((MINMAXINFO *)lp)->ptMinTrackSize.y = 420;
        return 0;
    case WM_SIZE: {
        RECT r;
        GetClientRect(w, &r);
        MoveWindow(GetDlgItem(w, P_NOTE), 12, 99, r.right - 24, 32, TRUE);
        MoveWindow(GetDlgItem(w, P_CURSOR), 12, r.bottom - 25, r.right - 24, 22, TRUE);
        InvalidateRect(w, NULL, FALSE);
        return 0;
    }
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(w, &ps);
        paint_profile(a, dc);
        EndPaint(w, &ps);
        return 0;
    }
    case WM_COMMAND: {
        int id = LOWORD(wp), notification = HIWORD(wp);
        if (a->profile_updating || a->worker)
            return 0;
        if (id == P_AXIS && notification == CBN_SELCHANGE) {
            a->profile_axis = (int)SendDlgItemMessageA(w, id, CB_GETCURSEL, 0, 0);
            update_profile(a);
        } else if (id == P_MODE && notification == CBN_SELCHANGE) {
            a->profile_mode = (int)SendDlgItemMessageA(w, id, CB_GETCURSEL, 0, 0);
            profile_text(a, P_CURSOR, "Move the mouse over the plot to read a bin.");
            InvalidateRect(w, NULL, FALSE);
        } else if (id == P_EXPORT)
            export_profile(a);
        else if ((id == P_INCIDENCE || id == P_NORMAL) && notification == EN_CHANGE) {
            char text[64];
            GetDlgItemTextA(w, id, text, sizeof text);
            SetDlgItemTextA(a->window, id == P_INCIDENCE ? CIF_INCIDENCE : CIF_NORMAL_PHI, text);
        } else if (id == P_BINS && notification == EN_CHANGE) {
            a->profile_bins_dirty = 1;
            osc_profile_free(&a->profile);
            EnableWindow(GetDlgItem(w, P_EXPORT), FALSE);
            profile_text(a, P_NOTE, "Q bin count edited. Click Apply to redraw.");
            profile_text(a, P_CURSOR, "");
            InvalidateRect(w, NULL, FALSE);
        } else if (id == P_APPLY) {
            OscMount mount;
            char text[64], tail;
            double bins;
            GetDlgItemTextA(w, P_BINS, text, sizeof text);
            if (sscanf(text, "%lf %c", &bins, &tail) != 1 || !isfinite(bins) || bins < 1 ||
                bins > 4096 || bins != floor(bins) || !number(a, CIF_INCIDENCE, &mount.incidence) ||
                !number(a, CIF_NORMAL_PHI, &mount.normal_phi)) {
                profile_text(a, P_NOTE,
                             "Use 1..4096 whole Q bins and finite sample angles in degrees.");
                return 0;
            }
            mount.incidence *= HBN_PI / 180;
            mount.normal_phi *= HBN_PI / 180;
            if (!osc_mount_valid(mount)) {
                profile_text(a, P_NOTE,
                             "Use |incidence| < 89 deg and normal phi in [-180,180] deg.");
                return 0;
            }
            a->profile_q_bins = (int)bins;
            a->profile_bins_dirty = 0;
            analysis_mount_apply(a, mount);
        } else if (id == IDCANCEL)
            DestroyWindow(w);
        return 0;
    }
    case WM_MOUSEMOVE:
        if (a->profile.signal) {
            RECT box = plot_rect(w);
            int x = GET_X_LPARAM(lp), y = GET_Y_LPARAM(lp);
            if (x >= box.left && x < box.right && y >= box.top && y <= box.bottom) {
                const OscProfile *p = &a->profile;
                int i = (x - box.left) * p->bins / (box.right - box.left);
                double scale = p->axis < PROFILE_QZ ? 180 / HBN_PI : 1;
                char text[220];
                snprintf(text, sizeof text,
                         "x=%.7g %s | signal=%.8g | valid area=%.7g | plotted=%.8g",
                         (p->lower + (i + .5) * (p->upper - p->lower) / p->bins) * scale,
                         p->axis < PROFILE_QZ ? "deg" : "A^-1", p->signal[i], p->area[i],
                         ordinate(a, i));
                profile_text(a, P_CURSOR, text);
            }
        }
        return 0;
    case WM_DESTROY:
        a->profile_window = NULL;
        a->profile_bins_dirty = 0;
        osc_profile_free(&a->profile);
        return 0;
    }
    return DefWindowProcA(w, msg, wp, lp);
}
void profile_show(App *a) {
    if (a->worker || !a->integration.signal) {
        message(a, "Integrate the current angular region first.");
        return;
    }
    if (!a->profile_q_bins)
        a->profile_q_bins = 512;
    if (!a->profile_window &&
        !CreateWindowExA(WS_EX_CONTROLPARENT, "SlateRegionProfile",
                         "Integrated region - 1D intensity", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                         CW_USEDEFAULT, CW_USEDEFAULT, 800, 540, a->window, NULL, a->instance, a)) {
        message(a, "Could not open the 1D profile window.");
        return;
    }
    update_profile(a);
    ShowWindow(a->profile_window, SW_SHOWNORMAL);
    SetForegroundWindow(a->profile_window);
    SetFocus(GetDlgItem(a->profile_window, P_AXIS));
}
