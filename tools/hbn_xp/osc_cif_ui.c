#include "osc_app.h"

static void repaint(App *a) {
    InvalidateRect(a->canvas, NULL, FALSE);
    InvalidateRect(a->angle_canvas, NULL, FALSE);
}
const CifPeak *cif_selected_peak(const App *a) {
    if (!a->cif_loaded || a->cif_dirty || a->worker || !analysis_ready(a) || a->cif_selected < 0 ||
        a->cif_selected >= a->cif.count ||
        fabs(a->cif.wavelength_A - a->geometry.settings.wavelength_A) > 1e-10 * a->cif.wavelength_A)
        return NULL;
    return &a->cif.peaks[a->cif_selected];
}
void cif_status(App *a) {
    char text[1800];
    const char *state;
    if (!a->cif_loaded) {
        SetWindowTextA(GetDlgItem(a->window, CIF_STATUS),
                       "Load CIF to calculate raw |F|^2 (electrons^2). Select a row for a cyan "
                       "2theta marker. No intensity normalization or multiplicity.");
        return;
    }
    state = a->cif_dirty         ? "Inputs edited; Calculate to update. Marker hidden."
            : !analysis_ready(a) ? "Apply image geometry to show a marker."
            : fabs(a->cif.wavelength_A - a->geometry.settings.wavelength_A) >
                    1e-10 * a->cif.wavelength_A
                ? "Wavelength mismatch; marker hidden. Use geometry A, then Calculate."
                : "Select hkl: cyan powder-position marker (no orientation prediction).";
    snprintf(text, sizeof text,
             "%s | %s\r\n%d hkl; %d atoms; lambda %.8g A\r\nUnknown U=0: %d sites. %s\r\n"
             "Factors: %s\r\nRaw |F|^2 in electrons^2; no multiplicity or normalization. "
             "Waasmaier + Chantler; CIF dispersion values not used.",
             a->cif.name, a->cif.spacegroup, a->cif.count, a->cif.expanded_sites,
             a->cif.wavelength_A, a->cif.unknown_u_sites, state, a->cif.species);
    SetWindowTextA(GetDlgItem(a->window, CIF_STATUS), text);
}
void cif_layout(App *a) {
    RECT r;
    GetClientRect(a->window, &r);
    MoveWindow(GetDlgItem(a->window, CIF_LIST), 10, 473, 230, r.bottom > 513 ? r.bottom - 513 : 1,
               TRUE);
}
void cif_controls(App *a) {
    HWND list, view;
    LVCOLUMNA column = {0};
    const char *names[] = {"h k l", "2theta deg", "raw |F|^2"};
    const char *views[] = {"Detector", "Phi vs 2theta", "Side by side"};
    int i;
    INITCOMMONCONTROLSEX controls = {sizeof controls, ICC_LISTVIEW_CLASSES};
    InitCommonControlsEx(&controls);
    control(a, "BUTTON", "CIF", WS_TABSTOP, CIF_PAGE, 126, 174, 46, 26);
    control(a, "BUTTON", "Load CIF...", WS_TABSTOP, CIF_OPEN, 10, 204, 110, 25);
    control(a, "BUTTON", "Export peaks...", WS_TABSTOP, CIF_EXPORT, 128, 204, 112, 25);
    control(a, "STATIC", "Wavelength A", 0, CIF_LABEL, 10, 236, 84, 20);
    control(a, "EDIT", "", WS_TABSTOP | ES_AUTOHSCROLL, CIF_WAVELENGTH, 95, 233, 72, 24);
    control(a, "BUTTON", "Geometry A", WS_TABSTOP, CIF_USE_GEOMETRY, 173, 233, 67, 24);
    control(a, "STATIC", "Max 2theta deg", 0, CIF_LABEL + 1, 10, 266, 84, 20);
    control(a, "EDIT", "80", WS_TABSTOP | ES_AUTOHSCROLL, CIF_MAXIMUM, 95, 263, 72, 24);
    control(a, "BUTTON", "Calculate", WS_TABSTOP, CIF_CALCULATE, 173, 263, 67, 24);
    control(a, "BUTTON", "Unknown Uiso = 0 A^2", WS_TABSTOP | BS_AUTOCHECKBOX, CIF_UNKNOWN_ZERO, 10,
            291, 230, 22);
    CheckDlgButton(a->window, CIF_UNKNOWN_ZERO, BST_CHECKED);
    control(a, "STATIC", "View", 0, CIF_LABEL + 2, 10, 321, 40, 20);
    view = control(a, "COMBOBOX", "", WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL, CIF_VIEW, 55, 318,
                   185, 150);
    for (i = 0; i < 3; ++i)
        SendMessageA(view, CB_ADDSTRING, 0, (LPARAM)views[i]);
    control(a, "EDIT", "", ES_MULTILINE | ES_READONLY | WS_VSCROLL, CIF_STATUS, 10, 349, 230, 90);
    control(a, "BUTTON", "Sort: angle / intensity", WS_TABSTOP, CIF_SORT, 10, 442, 153, 25);
    control(a, "BUTTON", "Clear mark", WS_TABSTOP, CIF_CLEAR, 170, 442, 70, 25);
    list = control(a, WC_LISTVIEWA, "",
                   WS_TABSTOP | LVS_REPORT | LVS_OWNERDATA | LVS_SINGLESEL | LVS_SHOWSELALWAYS,
                   CIF_LIST, 10, 473, 230, 180);
    ListView_SetExtendedListViewStyle(list, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    column.mask = LVCF_TEXT | LVCF_WIDTH;
    for (i = 0; i < 3; ++i) {
        column.pszText = (char *)names[i];
        column.cx = i == 0 ? 68 : (i == 1 ? 70 : 74);
        ListView_InsertColumn(list, i, &column);
    }
    a->cif_selected = -1;
    a->updating_cif = 1;
    set_number(a, CIF_WAVELENGTH, a->settings.wavelength_A);
    a->updating_cif = 0;
    if (GetModuleFileNameA(NULL, a->cif_data_path, MAX_PATH) < MAX_PATH) {
        char *slash = strrchr(a->cif_data_path, '\\');
        if (slash && slash - a->cif_data_path + 20 < MAX_PATH)
            strcpy(slash + 1, "cif_scattering.bin");
        else
            a->cif_data_path[0] = 0;
    } else
        a->cif_data_path[0] = 0;
    cif_status(a);
}
static void calculate(App *a, const char *path) {
    if (!number(a, CIF_WAVELENGTH, &a->cif_wavelength) ||
        !number(a, CIF_MAXIMUM, &a->cif_maximum)) {
        message(a, "Enter finite wavelength and maximum 2theta.");
        return;
    }
    a->cif_unknown_zero = IsDlgButtonChecked(a->window, CIF_UNKNOWN_ZERO) == BST_CHECKED;
    start_job(a, JOB_CIF, path);
}
static int by_angle(const void *left, const void *right) {
    const CifPeak *a = (const CifPeak *)left, *b = (const CifPeak *)right;
    if (a->two_theta_deg != b->two_theta_deg)
        return a->two_theta_deg > b->two_theta_deg ? 1 : -1;
    if (a->h != b->h)
        return a->h - b->h;
    if (a->k != b->k)
        return a->k - b->k;
    return a->l - b->l;
}
static int by_intensity(const void *left, const void *right) {
    const CifPeak *a = (const CifPeak *)left, *b = (const CifPeak *)right;
    if (a->intensity_e2 != b->intensity_e2)
        return a->intensity_e2 < b->intensity_e2 ? 1 : -1;
    return by_angle(left, right);
}
int cif_command(App *a, int id, int notification) {
    char path[MAX_PATH];
    if (id == CIF_PAGE) {
        analysis_page(a, PAGE_CIF);
        return 1;
    }
    if (id < CIF_OPEN || id >= CIF_END)
        return 0;
    if (a->updating_cif)
        return 1;
    if (((id == CIF_WAVELENGTH || id == CIF_MAXIMUM) && notification == EN_CHANGE) ||
        (id == CIF_UNKNOWN_ZERO && notification == BN_CLICKED)) {
        a->cif_dirty = a->cif_loaded;
        cif_status(a);
        repaint(a);
    } else if (id == CIF_OPEN) {
        if (choose_path(a, path, 0, "Crystallographic CIF\0*.cif\0\0", "cif"))
            calculate(a, path);
    } else if (id == CIF_CALCULATE) {
        if (a->cif_loaded)
            calculate(a, a->cif_path);
        else
            message(a, "Load a CIF first.");
    } else if (id == CIF_USE_GEOMETRY) {
        if (a->geometry_valid && !a->geometry_dirty) {
            char wavelength[64];
            snprintf(wavelength, sizeof wavelength, "%.17g", a->geometry.settings.wavelength_A);
            SetWindowTextA(GetDlgItem(a->window, CIF_WAVELENGTH), wavelength);
        } else
            message(a, "Apply analysis geometry first.");
    } else if (id == CIF_VIEW && notification == CBN_SELCHANGE) {
        a->view_mode = (int)SendMessageA(GetDlgItem(a->window, CIF_VIEW), CB_GETCURSEL, 0, 0);
        SendMessageA(GetDlgItem(a->window, VIEW_MODE), CB_SETCURSEL, a->view_mode, 0);
        analysis_layout(a);
        fit_view(a);
    } else if (id == CIF_EXPORT) {
        if (!a->cif_loaded)
            message(a, "Load a CIF first.");
        else if (a->cif_dirty)
            message(a, "Inputs were edited. Calculate before exporting.");
        else if (choose_path(a, path, 1, "CIF peaks CSV\0*.csv\0\0", "csv"))
            start_job(a, JOB_CIF_EXPORT, path);
    } else if (id == CIF_SORT && a->cif_loaded) {
        char text[48];
        GetWindowTextA(GetDlgItem(a->window, CIF_SORT), text, sizeof text);
        if (a->cif.count > 1)
            qsort(a->cif.peaks, a->cif.count, sizeof *a->cif.peaks,
                  !strcmp(text, "Sort by angle") ? by_angle : by_intensity);
        SetWindowTextA(GetDlgItem(a->window, CIF_SORT),
                       !strcmp(text, "Sort by angle") ? "Sort by intensity" : "Sort by angle");
        a->cif_selected = -1;
        ListView_SetItemState(GetDlgItem(a->window, CIF_LIST), -1, 0, LVIS_SELECTED | LVIS_FOCUSED);
        InvalidateRect(GetDlgItem(a->window, CIF_LIST), NULL, TRUE);
        repaint(a);
    } else if (id == CIF_CLEAR) {
        a->cif_selected = -1;
        ListView_SetItemState(GetDlgItem(a->window, CIF_LIST), -1, 0, LVIS_SELECTED | LVIS_FOCUSED);
        repaint(a);
    }
    return 1;
}
LRESULT cif_notify(App *a, NMHDR *header) {
    if (header->idFrom != CIF_LIST)
        return 0;
    if (header->code == LVN_GETDISPINFOA) {
        NMLVDISPINFOA *info = (NMLVDISPINFOA *)header;
        if ((info->item.mask & LVIF_TEXT) && info->item.iItem >= 0 &&
            info->item.iItem < a->cif.count) {
            const CifPeak *p = &a->cif.peaks[info->item.iItem];
            if (info->item.iSubItem == 0)
                snprintf(info->item.pszText, info->item.cchTextMax, "%d %d %d", p->h, p->k, p->l);
            else if (info->item.iSubItem == 1)
                snprintf(info->item.pszText, info->item.cchTextMax, "%.5f", p->two_theta_deg);
            else
                snprintf(info->item.pszText, info->item.cchTextMax, "%.7g", p->intensity_e2);
        }
    } else if (header->code == LVN_ITEMCHANGED && !a->worker) {
        char text[300];
        a->cif_selected = ListView_GetNextItem(header->hwndFrom, -1, LVNI_SELECTED);
        if (a->cif_selected >= 0 && a->cif_selected < a->cif.count) {
            const CifPeak *p = &a->cif.peaks[a->cif_selected];
            snprintf(text, sizeof text,
                     "CIF (%d %d %d) | 2theta %.8f deg | d %.8g A | raw |F|^2 %.10g e^2%s", p->h,
                     p->k, p->l, p->two_theta_deg, p->d_A, p->intensity_e2,
                     cif_selected_peak(a) ? " | cyan position marker"
                                          : " | marker hidden; see CIF status");
            SetWindowTextA(a->stage_label, text);
        }
        repaint(a);
    }
    return 0;
}
void cif_publish(App *a) {
    cif_peaks_free(&a->cif);
    a->cif = a->pending_cif;
    memset(&a->pending_cif, 0, sizeof a->pending_cif);
    strcpy(a->cif_path, a->job_path);
    a->cif_loaded = 1;
    a->cif_selected = -1;
    a->cif_dirty = 0;
    ListView_SetItemState(GetDlgItem(a->window, CIF_LIST), -1, 0, LVIS_SELECTED | LVIS_FOCUSED);
    ListView_SetItemCount(GetDlgItem(a->window, CIF_LIST), a->cif.count);
    SetWindowTextA(GetDlgItem(a->window, CIF_SORT), "Sort by intensity");
    cif_status(a);
    repaint(a);
}
int cif_export(App *a, const char *path) {
    char temporary[MAX_PATH];
    int ok;
    FILE *f = begin_output(path, temporary, a->error);
    if (!f)
        return 0;
    ok = cif_peaks_write(f, &a->cif, app_progress, a);
    if (InterlockedCompareExchange(&a->cancel, 0, 0)) {
        strcpy(a->error, "Canceled.");
        ok = 0;
    }
    return finish_output(f, temporary, path, ok, a->error);
}
void cif_detector_overlay(App *a, HDC dc) {
    const CifPeak *peak = cif_selected_peak(a);
    HPEN pen, old;
    int i, connected = 0;
    if (!peak)
        return;
    pen = CreatePen(PS_SOLID, 2, RGB(0, 220, 240));
    old = (HPEN)SelectObject(dc, pen);
    for (i = 0; i <= 720; ++i) {
        double c, r, x, y;
        if (!osc_angle_pixel(&a->geometry, peak->two_theta_deg * HBN_PI / 180, i * HBN_PI / 360, &c,
                             &r)) {
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
    SelectObject(dc, old);
    DeleteObject(pen);
}
void cif_angle_marker(App *a, HDC dc, RECT bounds) {
    const CifPeak *peak = cif_selected_peak(a);
    const OscGrid *g = &a->integration.grid;
    HPEN pen, old;
    double theta;
    int x;
    if (!peak || !a->integration.signal)
        return;
    theta = peak->two_theta_deg * HBN_PI / 180;
    if (theta < g->theta_min || theta > g->theta_max)
        return;
    x = bounds.left + (int)((theta - g->theta_min) / (g->theta_max - g->theta_min) *
                            (bounds.right - bounds.left));
    pen = CreatePen(PS_SOLID, 2, RGB(0, 200, 220));
    old = (HPEN)SelectObject(dc, pen);
    MoveToEx(dc, x, bounds.top, NULL);
    LineTo(dc, x, bounds.bottom);
    SelectObject(dc, old);
    DeleteObject(pen);
}
