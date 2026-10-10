#include "osc_app.h"

static void repaint(App *a) {
    InvalidateRect(a->canvas, NULL, FALSE);
    InvalidateRect(a->angle_canvas, NULL, FALSE);
}
int cif_guides_ready(const App *a) {
    return a->cif_loaded && !a->cif_dirty && !a->worker && analysis_ready(a) && a->cif_overlay &&
           (a->cif_overlay != 2 || (a->cif_mount_applied && !a->cif_mount_dirty)) &&
           fabs(a->cif.wavelength_A - a->geometry.settings.wavelength_A) <=
               1e-10 * a->cif.wavelength_A;
}
int cif_prepare_guides(App *a) {
    int rods = a->cif_overlay == 2;
    CifGuideView view = {0};
    if (!cif_guides_ready(a))
        return 0;
    view.column_min = a->left;
    view.column_max = a->left + a->view_width / a->zoom;
    view.row_min = a->top;
    view.row_max = a->top + a->view_height / a->zoom;
    view.pixel_tolerance = 0.3 / a->zoom;
    if (a->integration.signal && a->angle_width > 52 && a->angle_height > 213) {
        const OscGrid *g = &a->integration.grid;
        view.theta_min = g->theta_min;
        view.theta_max = g->theta_max;
        view.phi_min = g->phi_min;
        view.phi_max = g->phi_max;
        view.theta_tolerance = 0.2 * (g->theta_max - g->theta_min) / (a->angle_width - 52);
        view.phi_tolerance = 0.2 * (g->phi_max - g->phi_min) / (a->angle_height - 213);
    }
    if (!a->guide_attempted || rods != a->guide_rods ||
        memcmp(&view, &a->guide_view, sizeof view) ||
        memcmp(&a->guide_geometry, &a->geometry, sizeof a->geometry)) {
        a->guide_attempted = 1;
        a->guide_rods = rods;
        a->guide_geometry = a->geometry;
        a->guide_view = view;
        cif_guides_free(&a->guides);
        cif_guides_build(&a->cif, &a->geometry, a->cif_mount, rods, &view, &a->guides,
                         a->guide_error);
        cif_status(a);
    }
    return !a->guide_error[0];
}
void cif_group_label(const App *a, int kind, int group, char *text, size_t capacity) {
    const CifGrouping *set = &a->cif.grouping[kind];
    const CifGroup *g = &set->groups[group];
    const CifPeak *p = &a->cif.peaks[set->members[g->first]];
    char extra[32] = "";
    if (g->count > 1)
        snprintf(extra, sizeof extra, " +%d", g->count - 1);
    if (kind == CIF_ROD)
        snprintf(text, capacity, "R%d (%d %d L)%s", group + 1, p->h, p->k, extra);
    else
        snprintf(text, capacity, "%c%d (%d %d %d)%s", kind == CIF_POWDER ? 'P' : 'T', group + 1,
                 p->h, p->k, p->l, extra);
}
void cif_status(App *a) {
    char text[1900];
    const char *state;
    if (!a->cif_loaded) {
        SetWindowTextA(GetDlgItem(a->window, CIF_STATUS),
                       "Load CIF. P: equal 2theta, R: Qz rods, T: HKL ticks. Select a group to see "
                       "every member below.");
        return;
    }
    state = a->cif_dirty         ? "Inputs edited; Calculate to update. Guides hidden."
            : !a->cif_overlay    ? "Guides off."
            : !analysis_ready(a) ? "Apply matching image geometry to show guides."
            : fabs(a->cif.wavelength_A - a->geometry.settings.wavelength_A) >
                    1e-10 * a->cif.wavelength_A
                ? "Wavelength mismatch; use geometry wavelength, then Calculate."
            : a->cif_overlay == 2 && (!a->cif_mount_applied || a->cif_mount_dirty)
                ? "Enter incidence / normal phi and Apply a1/a2 fiber. Guides hidden."
            : a->guide_error[0] ? a->guide_error
            : a->cif_overlay == 2
                ? "Air guides: a1/a2 surface, b3 normal, full fiber rotation. No refraction."
                : "Powder 2theta guides; no sample orientation assumed.";
    snprintf(
        text, sizeof text,
        "%s\r\n%s | %s | %d hkl; %d P, %d R, %d T groups.\r\n"
        "P = equal Bragg angle; R = equal Qr; T = equal Qr,Qz. +N means more HKLs; all members "
        "below.\r\n"
        "Raw |F|^2 in e^2. Table max is largest individual member, not a sum.\r\n"
        "lambda %.8g A; %d atoms; unknown U=0: %d source sites. Factors: %s\r\n"
        "Waasmaier + Chantler; no intensity normalization/multiplicity; CIF dispersion unused.",
        state, a->cif.name, a->cif.spacegroup, a->cif.count, a->cif.grouping[CIF_POWDER].count,
        a->cif.grouping[CIF_ROD].count, a->cif.grouping[CIF_TICK].count, a->cif.wavelength_A,
        a->cif.expanded_sites, a->cif.unknown_u_sites, a->cif.species);
    SetWindowTextA(GetDlgItem(a->window, CIF_STATUS), text);
}
void cif_layout(App *a) {
    RECT r;
    const int setup[] = {CIF_WAVELENGTH,   CIF_MAXIMUM,   CIF_CALCULATE, CIF_USE_GEOMETRY,
                         CIF_UNKNOWN_ZERO, CIF_OVERLAY,   CIF_INCIDENCE, CIF_NORMAL_PHI,
                         CIF_MOUNT,        CIF_LABELS,    CIF_LABEL,     CIF_LABEL + 1,
                         CIF_LABEL + 3,    CIF_LABEL + 4, CIF_LABEL + 5};
    size_t i;
    int height, top = a->cif_compact ? 338 : 502;
    GetClientRect(a->window, &r);
    for (i = 0; i < sizeof setup / sizeof setup[0]; ++i)
        ShowWindow(GetDlgItem(a->window, setup[i]),
                   a->analysis_page == PAGE_CIF && !a->cif_compact ? SW_SHOW : SW_HIDE);
    SetWindowTextA(GetDlgItem(a->window, CIF_COMPACT), a->cif_compact ? "Show setup" : "More rows");
    MoveWindow(GetDlgItem(a->window, CIF_LABEL + 2), 10, a->cif_compact ? 238 : 321, 40, 20, TRUE);
    MoveWindow(GetDlgItem(a->window, CIF_VIEW), 55, a->cif_compact ? 235 : 318, 275, 150, TRUE);
    MoveWindow(GetDlgItem(a->window, CIF_STATUS), 10, top - 71, 320, 39, TRUE);
    MoveWindow(GetDlgItem(a->window, CIF_KIND), 10, top - 28, 204, 150, TRUE);
    MoveWindow(GetDlgItem(a->window, CIF_SORT), 220, top - 28, 110, 24, TRUE);
    height = (r.bottom - top - 42) / 2;
    if (height < 52)
        height = 52;
    MoveWindow(GetDlgItem(a->window, CIF_LIST), 10, top, 320, height, TRUE);
    MoveWindow(GetDlgItem(a->window, CIF_MEMBERS), 10, top + 6 + height, 320,
               r.bottom > top + 46 + height ? r.bottom - top - 46 - height : 1, TRUE);
    for (i = 0; i < 2; ++i) {
        HWND list = GetDlgItem(a->window, i ? CIF_MEMBERS : CIF_LIST);
        /* Reserve scrollbar space even before owner-data rows have been published. */
        ListView_SetColumnWidth(list, 0, 320 - GetSystemMetrics(SM_CXVSCROLL) - 4 - 68 - 88);
        ListView_SetColumnWidth(list, 1, 68);
        ListView_SetColumnWidth(list, 2, 88);
    }
}
static HWND combo(App *a, int id, int x, int y, int width, const char *items) {
    HWND w = control(a, "COMBOBOX", "", WS_TABSTOP | CBS_DROPDOWNLIST | WS_VSCROLL, id, x, y, width,
                     150);
    for (; *items; items += strlen(items) + 1)
        SendMessageA(w, CB_ADDSTRING, 0, (LPARAM)items);
    SendMessageA(w, CB_SETCURSEL, 0, 0);
    return w;
}
static void list_control(App *a, int id, const char *first, const char *last) {
    const char *names[] = {first, "2theta", last};
    LVCOLUMNA column = {0};
    int i;
    HWND list = control(a, WC_LISTVIEWA, "",
                        WS_TABSTOP | LVS_REPORT | LVS_OWNERDATA | LVS_SINGLESEL | LVS_SHOWSELALWAYS,
                        id, 10, 502, 230, 80);
    ListView_SetExtendedListViewStyle(list,
                                      LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_INFOTIP);
    column.mask = LVCF_TEXT | LVCF_WIDTH;
    for (i = 0; i < 3; ++i) {
        column.pszText = (char *)names[i];
        column.cx = i == 0 ? 95 : (i == 1 ? 54 : 65);
        ListView_InsertColumn(list, i, &column);
    }
}
void cif_controls(App *a) {
    INITCOMMONCONTROLSEX controls = {sizeof controls, ICC_LISTVIEW_CLASSES};
    InitCommonControlsEx(&controls);
    control(a, "BUTTON", "CIF", WS_TABSTOP | BS_PUSHLIKE | BS_CHECKBOX, CIF_PAGE, 126, 174, 46, 26);
    control(a, "BUTTON", "Load CIF...", WS_TABSTOP, CIF_OPEN, 10, 204, 110, 25);
    control(a, "BUTTON", "Export peaks...", WS_TABSTOP, CIF_EXPORT, 128, 204, 112, 25);
    control(a, "BUTTON", "More rows", WS_TABSTOP, CIF_COMPACT, 246, 204, 84, 25);
    control(a, "STATIC", "Wavelength A", 0, CIF_LABEL, 10, 236, 84, 20);
    control(a, "EDIT", "", WS_TABSTOP | ES_AUTOHSCROLL, CIF_WAVELENGTH, 95, 233, 72, 24);
    control(a, "BUTTON", "Use geometry wavelength", WS_TABSTOP, CIF_USE_GEOMETRY, 173, 233, 157,
            24);
    control(a, "STATIC", "Max 2theta deg", 0, CIF_LABEL + 1, 10, 266, 84, 20);
    control(a, "EDIT", "80", WS_TABSTOP | ES_AUTOHSCROLL, CIF_MAXIMUM, 95, 263, 72, 24);
    control(a, "BUTTON", "Calculate", WS_TABSTOP, CIF_CALCULATE, 173, 263, 157, 24);
    control(a, "BUTTON", "Unknown Uiso = 0 A^2", WS_TABSTOP | BS_AUTOCHECKBOX, CIF_UNKNOWN_ZERO, 10,
            291, 230, 22);
    CheckDlgButton(a->window, CIF_UNKNOWN_ZERO, BST_CHECKED);
    control(a, "STATIC", "View", 0, CIF_LABEL + 2, 10, 321, 40, 20);
    combo(a, CIF_VIEW, 55, 318, 185, "Detector\0Phi vs 2theta\0Side by side\0");
    control(a, "STATIC", "Guides", 0, CIF_LABEL + 3, 10, 349, 40, 20);
    combo(a, CIF_OVERLAY, 55, 346, 275, "Off\0Powder 2theta arcs\0Arcs + Qz rods / L ticks\0");
    SendMessageA(GetDlgItem(a->window, CIF_OVERLAY), CB_SETCURSEL, 1, 0);
    control(a, "STATIC", "Incidence deg", 0, CIF_LABEL + 4, 10, 377, 78, 20);
    control(a, "EDIT", "0", WS_TABSTOP | ES_AUTOHSCROLL, CIF_INCIDENCE, 92, 374, 62, 24);
    control(a, "STATIC", "Normal phi deg", 0, CIF_LABEL + 5, 163, 377, 94, 20);
    control(a, "EDIT", "0", WS_TABSTOP | ES_AUTOHSCROLL, CIF_NORMAL_PHI, 263, 374, 67, 24);
    control(a, "BUTTON", "Apply a1/a2 fiber", WS_TABSTOP, CIF_MOUNT, 10, 402, 204, 24);
    control(a, "BUTTON", "Labels", WS_TABSTOP | BS_AUTOCHECKBOX, CIF_LABELS, 225, 402, 105, 24);
    CheckDlgButton(a->window, CIF_LABELS, BST_CHECKED);
    control(a, "EDIT", "", ES_MULTILINE | ES_READONLY | WS_VSCROLL, CIF_STATUS, 10, 431, 230, 39);
    combo(a, CIF_KIND, 10, 474, 150, "P: 2theta groups\0R: Qz rod groups\0T: HKL tick groups\0");
    control(a, "BUTTON", "By strength", WS_TABSTOP, CIF_SORT, 164, 474, 76, 24);
    list_control(a, CIF_LIST, "Group / HKL +N", "max |F|^2");
    list_control(a, CIF_MEMBERS, "Member h k l", "raw |F|^2");
    a->cif_selected = -1;
    a->cif_overlay = 1;
    a->cif_labels = 1;
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
    const CifDisplayRow *a = (const CifDisplayRow *)left, *b = (const CifDisplayRow *)right;
    return a->angle == b->angle ? a->group - b->group : (a->angle > b->angle ? 1 : -1);
}
static int by_intensity(const void *left, const void *right) {
    const CifDisplayRow *a = (const CifDisplayRow *)left, *b = (const CifDisplayRow *)right;
    return a->intensity == b->intensity ? by_angle(left, right)
                                        : (a->intensity < b->intensity ? 1 : -1);
}
static void rows(App *a) {
    const CifGrouping *set = &a->cif.grouping[a->cif_kind];
    int i, selected = a->cif_selected;
    a->updating_cif = 1;
    a->cif_selected = -1;
    ListView_SetItemCount(GetDlgItem(a->window, CIF_MEMBERS), 0);
    ListView_SetItemState(GetDlgItem(a->window, CIF_LIST), -1, 0, LVIS_SELECTED | LVIS_FOCUSED);
    for (i = 0; i < set->count; ++i) {
        const CifPeak *p = &a->cif.peaks[set->members[set->groups[i].first]];
        a->cif_rows[i] = (CifDisplayRow){i, a->cif_kind == CIF_ROD ? p->qr_invA : p->two_theta_deg,
                                         set->groups[i].maximum_intensity_e2};
    }
    if (set->count > 1)
        qsort(a->cif_rows, set->count, sizeof *a->cif_rows,
              a->cif_sort_intensity ? by_intensity : by_angle);
    ListView_SetItemCount(GetDlgItem(a->window, CIF_LIST), set->count);
    InvalidateRect(GetDlgItem(a->window, CIF_LIST), NULL, TRUE);
    a->updating_cif = 0;
    for (i = 0; i < set->count; ++i)
        if (a->cif_rows[i].group == selected) {
            ListView_SetItemState(GetDlgItem(a->window, CIF_LIST), i, LVIS_SELECTED | LVIS_FOCUSED,
                                  LVIS_SELECTED | LVIS_FOCUSED);
            ListView_EnsureVisible(GetDlgItem(a->window, CIF_LIST), i, FALSE);
            break;
        }
    repaint(a);
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
    } else if ((id == CIF_INCIDENCE || id == CIF_NORMAL_PHI) && notification == EN_CHANGE) {
        a->cif_mount_dirty = 1;
        profile_mount_changed(a);
    } else if (id == CIF_MOUNT) {
        OscMount mount;
        if (!number(a, CIF_INCIDENCE, &mount.incidence) ||
            !number(a, CIF_NORMAL_PHI, &mount.normal_phi)) {
            message(a, "Enter finite incidence and normal phi in degrees.");
            return 1;
        }
        mount.incidence *= HBN_PI / 180;
        mount.normal_phi *= HBN_PI / 180;
        if (!osc_mount_valid(mount)) {
            message(a, "Use |incidence| < 89 deg and normal phi in [-180,180] deg. Positive "
                       "incidence means the beam enters the a1/a2 surface. Normal phi is its "
                       "transverse direction (0 up, +90 left). Air geometry; no refraction.");
            return 1;
        }
        analysis_mount_apply(a, mount);
        a->cif_overlay = 2;
        SendMessageA(GetDlgItem(a->window, CIF_OVERLAY), CB_SETCURSEL, 2, 0);
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
    } else if (id == CIF_COMPACT) {
        a->cif_compact = !a->cif_compact;
        cif_layout(a);
    } else if (id == CIF_VIEW && notification == CBN_SELCHANGE) {
        a->view_mode = (int)SendMessageA(GetDlgItem(a->window, CIF_VIEW), CB_GETCURSEL, 0, 0);
        SendMessageA(GetDlgItem(a->window, VIEW_MODE), CB_SETCURSEL, a->view_mode, 0);
        analysis_layout(a);
    } else if (id == CIF_OVERLAY && notification == CBN_SELCHANGE) {
        a->cif_overlay = (int)SendMessageA(GetDlgItem(a->window, CIF_OVERLAY), CB_GETCURSEL, 0, 0);
        a->guide_attempted = 0;
        a->guide_error[0] = 0;
    } else if (id == CIF_LABELS) {
        a->cif_labels = IsDlgButtonChecked(a->window, CIF_LABELS) == BST_CHECKED;
    } else if (id == CIF_KIND && notification == CBN_SELCHANGE) {
        a->cif_selected = -1;
        a->cif_kind = (int)SendMessageA(GetDlgItem(a->window, CIF_KIND), CB_GETCURSEL, 0, 0);
        rows(a);
    } else if (id == CIF_SORT) {
        a->cif_sort_intensity = !a->cif_sort_intensity;
        SetWindowTextA(GetDlgItem(a->window, CIF_SORT),
                       a->cif_sort_intensity ? "By position" : "By strength");
        rows(a);
    } else if (id == CIF_EXPORT) {
        if (!a->cif_loaded)
            message(a, "Load a CIF first.");
        else if (a->cif_dirty)
            message(a, "Inputs were edited. Calculate before exporting.");
        else if (choose_path(a, path, 1, "CIF peaks CSV\0*.csv\0\0", "csv"))
            start_job(a, JOB_CIF_EXPORT, path);
    } else
        return 1;
    cif_status(a);
    repaint(a);
    return 1;
}
LRESULT cif_notify(App *a, NMHDR *header) {
    const CifGrouping *set = &a->cif.grouping[a->cif_kind];
    if (header->idFrom != CIF_LIST && header->idFrom != CIF_MEMBERS)
        return 0;
    if (header->code == LVN_GETINFOTIPA) {
        NMLVGETINFOTIPA *tip = (NMLVGETINFOTIPA *)header;
        int row = tip->iItem;
        if (row < 0)
            return 0;
        if (header->idFrom == CIF_LIST && a->cif_rows && row < set->count) {
            char label[96];
            int id = a->cif_rows[row].group;
            cif_group_label(a, a->cif_kind, id, label, sizeof label);
            snprintf(tip->pszText, tip->cchTextMax,
                     "%s; %d members. Largest individual |F|^2: %.10g e^2", label,
                     set->groups[id].count, set->groups[id].maximum_intensity_e2);
        } else if (header->idFrom == CIF_MEMBERS && a->cif_selected >= 0 &&
                   a->cif_selected < set->count) {
            const CifGroup *group = &set->groups[a->cif_selected];
            if (row < group->count) {
                const CifPeak *p = &a->cif.peaks[set->members[group->first + row]];
                snprintf(tip->pszText, tip->cchTextMax,
                         "(%d %d %d); 2theta %.8g deg; raw |F|^2 %.10g e^2", p->h, p->k, p->l,
                         p->two_theta_deg, p->intensity_e2);
            }
        }
    } else if (header->code == LVN_GETDISPINFOA) {
        NMLVDISPINFOA *info = (NMLVDISPINFOA *)header;
        int row = info->item.iItem;
        const CifPeak *p;
        const CifGroup *group;
        if (!(info->item.mask & LVIF_TEXT) || row < 0)
            return 0;
        if (header->idFrom == CIF_LIST) {
            int id;
            if (row >= set->count || !a->cif_rows)
                return 0;
            id = a->cif_rows[row].group;
            group = &set->groups[id];
            p = &a->cif.peaks[set->members[group->first]];
            if (info->item.iSubItem == 0)
                cif_group_label(a, a->cif_kind, id, info->item.pszText, info->item.cchTextMax);
            else if (info->item.iSubItem == 1)
                snprintf(info->item.pszText, info->item.cchTextMax,
                         a->cif_kind == CIF_ROD ? "varies" : "%.4f", p->two_theta_deg);
            else
                snprintf(info->item.pszText, info->item.cchTextMax, "%.7g",
                         group->maximum_intensity_e2);
        } else {
            if (a->cif_selected < 0 || a->cif_selected >= set->count)
                return 0;
            group = &set->groups[a->cif_selected];
            if (row >= group->count)
                return 0;
            p = &a->cif.peaks[set->members[group->first + row]];
            if (info->item.iSubItem == 0)
                snprintf(info->item.pszText, info->item.cchTextMax, "%d %d %d", p->h, p->k, p->l);
            else if (info->item.iSubItem == 1)
                snprintf(info->item.pszText, info->item.cchTextMax, "%.5f", p->two_theta_deg);
            else
                snprintf(info->item.pszText, info->item.cchTextMax, "%.7g", p->intensity_e2);
        }
    } else if (header->code == LVN_ITEMCHANGED && header->idFrom == CIF_LIST && !a->worker &&
               !a->updating_cif) {
        int row = ListView_GetNextItem(header->hwndFrom, -1, LVNI_SELECTED);
        a->cif_selected = row >= 0 && row < set->count ? a->cif_rows[row].group : -1;
        ListView_SetItemCount(GetDlgItem(a->window, CIF_MEMBERS),
                              a->cif_selected < 0 ? 0 : set->groups[a->cif_selected].count);
        InvalidateRect(GetDlgItem(a->window, CIF_MEMBERS), NULL, TRUE);
        if (a->cif_selected >= 0) {
            char text[240], label[96];
            const CifGroup *g = &set->groups[a->cif_selected];
            const CifPeak *p = &a->cif.peaks[set->members[g->first]];
            cif_group_label(a, a->cif_kind, a->cif_selected, label, sizeof label);
            snprintf(text, sizeof text,
                     "%s | %d members below | representative Qr %.7g, Qz %.7g /A | P%d R%d T%d",
                     label, g->count, p->qr_invA, p->qz_invA, p->group[0] + 1, p->group[1] + 1,
                     p->group[2] + 1);
            SetWindowTextA(a->stage_label, text);
        }
        repaint(a);
    }
    return 0;
}
void cif_publish(App *a) {
    CifDisplayRow *display =
        (CifDisplayRow *)calloc(a->pending_cif.count ? a->pending_cif.count : 1, sizeof *display);
    if (!display) {
        cif_peaks_free(&a->pending_cif);
        message(a, "Not enough memory for CIF display. Previous table retained.");
        return;
    }
    cif_peaks_free(&a->cif);
    free(a->cif_rows);
    a->cif_rows = display;
    a->cif = a->pending_cif;
    memset(&a->pending_cif, 0, sizeof a->pending_cif);
    strcpy(a->cif_path, a->job_path);
    a->cif_loaded = 1;
    a->cif_selected = -1;
    a->cif_dirty = 0;
    a->guide_attempted = 0;
    a->guide_error[0] = 0;
    cif_guides_free(&a->guides);
    rows(a);
    cif_status(a);
}
int cif_export(App *a, const char *path) {
    char temporary[MAX_PATH];
    int ok;
    FILE *f = begin_output(path, temporary, a->error);
    if (!f)
        return 0;
    fprintf(
        f,
        "# fiber_mount_applied=%d,mount_inputs_edited=%d,incidence_deg=%.17g,normal_phi_deg=%.17g\n"
        "# External-air fiber guides; a1/a2 surface, b3 normal; no refraction\n",
        a->cif_mount_applied, a->cif_mount_dirty, a->cif_mount.incidence * 180 / HBN_PI,
        a->cif_mount.normal_phi * 180 / HBN_PI);
    ok = cif_peaks_write(f, &a->cif, app_progress, a);
    if (InterlockedCompareExchange(&a->cancel, 0, 0)) {
        strcpy(a->error, "Canceled.");
        ok = 0;
    }
    return finish_output(f, temporary, path, ok, a->error);
}
