#include "osc_app.h"

typedef struct {
    App *app;
    HDC dc;
    RECT bounds, occupied[512];
    int angular, labels, hidden;
} Overlay;

static int position(const Overlay *o, const CifGuidePoint *p, double *x, double *y) {
    const App *a = o->app;
    if (o->angular) {
        const OscGrid *g = &a->integration.grid;
        double phi;
        if (!p->angle_valid)
            return 0;
        phi = p->phi + 2 * HBN_PI * ceil((g->phi_min - p->phi) / (2 * HBN_PI));
        *x = o->bounds.left + (p->theta - g->theta_min) / (g->theta_max - g->theta_min) *
                                  (o->bounds.right - o->bounds.left);
        *y = o->bounds.bottom -
             (phi - g->phi_min) / (g->phi_max - g->phi_min) * (o->bounds.bottom - o->bounds.top);
    } else {
        if (!p->detector_valid)
            return 0;
        *x = (p->column - a->left) * a->zoom;
        *y = (p->row - a->top) * a->zoom;
    }
    return isfinite(*x) && isfinite(*y);
}
static int inside(RECT b, double x, double y) {
    return x >= b.left && x < b.right && y >= b.top && y < b.bottom;
}
/* Clip before integer GDI conversion, including far-away forward rays. */
static int line(Overlay *o, double x0, double y0, double x1, double y1, POINT *anchor, int draw) {
    double dx = x1 - x0, dy = y1 - y0, lo = 0, hi = 1;
    double p[4] = {-dx, dx, -dy, dy};
    double q[4] = {x0 - o->bounds.left, o->bounds.right - 1 - x0, y0 - o->bounds.top,
                   o->bounds.bottom - 1 - y0};
    int i;
    for (i = 0; i < 4; ++i) {
        if (p[i] == 0) {
            if (q[i] < 0)
                return 0;
        } else if (p[i] < 0)
            lo = fmax(lo, q[i] / p[i]);
        else
            hi = fmin(hi, q[i] / p[i]);
    }
    if (lo > hi)
        return 0;
    anchor->x = (LONG)(x0 + lo * dx);
    anchor->y = (LONG)(y0 + lo * dy);
    if (draw) {
        MoveToEx(o->dc, anchor->x, anchor->y, NULL);
        LineTo(o->dc, (int)(x0 + hi * dx), (int)(y0 + hi * dy));
    }
    return 1;
}
static int label(Overlay *o, POINT anchor, const char *text, SIZE size) {
    int attempt, i;
    if (o->labels == 512 || size.cx + 4 >= o->bounds.right - o->bounds.left)
        return 0;
    for (attempt = 0; attempt < 4; ++attempt) {
        RECT box, overlap;
        box.left = anchor.x + (attempt & 1 ? -size.cx - 6 : 6);
        box.top = anchor.y + (attempt & 2 ? 6 : -size.cy - 4);
        if (box.left < o->bounds.left + 2)
            box.left = o->bounds.left + 2;
        if (box.left + size.cx + 2 > o->bounds.right)
            box.left = o->bounds.right - size.cx - 2;
        if (box.top < o->bounds.top + 2)
            box.top = o->bounds.top + 2;
        if (box.top + size.cy + 2 > o->bounds.bottom)
            box.top = o->bounds.bottom - size.cy - 2;
        box.right = box.left + size.cx + 2;
        box.bottom = box.top + size.cy + 2;
        for (i = 0; i < o->labels; ++i)
            if (IntersectRect(&overlap, &box, &o->occupied[i]))
                break;
        if (i == o->labels) {
            o->occupied[o->labels++] = box;
            SetBkColor(o->dc, RGB(15, 20, 25));
            SetBkMode(o->dc, OPAQUE);
            TextOutA(o->dc, box.left + 1, box.top + 1, text, (int)strlen(text));
            SetBkMode(o->dc, TRANSPARENT);
            return 1;
        }
    }
    return 0;
}
static int selected(const App *a, const CifGuidePath *path) {
    return path->kind == a->cif_kind && path->group == a->cif_selected;
}
static void path_draw(Overlay *o, const CifGuidePath *path, int draw_lines, int draw_label) {
    App *a = o->app;
    int i, visible = 0, placed = 0, attempts = 0;
    POINT anchor = {0};
    POINT last_attempt = {0};
    SIZE size = {0};
    char text[96];
    cif_group_label(a, path->kind, path->group, text, sizeof text);
    if (draw_label)
        GetTextExtentPoint32A(o->dc, text, (int)strlen(text), &size);
    for (i = 0; i < path->count; ++i) {
        const CifGuidePoint *p = &a->guides.points[path->first + i];
        double x, y;
        int hit = 0;
        if (!position(o, p, &x, &y))
            continue;
        if (path->count == 1) {
            if (inside(o->bounds, x, y)) {
                anchor = (POINT){(LONG)x, (LONG)y};
                if (draw_lines) {
                    MoveToEx(o->dc, anchor.x - 4, anchor.y, NULL);
                    LineTo(o->dc, anchor.x + 5, anchor.y);
                    MoveToEx(o->dc, anchor.x, anchor.y - 4, NULL);
                    LineTo(o->dc, anchor.x, anchor.y + 5);
                }
                hit = 1;
            }
        } else if (i) {
            const CifGuidePoint *prev = p - 1;
            double x0, y0;
            if (position(o, prev, &x0, &y0)) {
                if (o->angular) {
                    const OscGrid *g = &a->integration.grid;
                    double period =
                        2 * HBN_PI / (g->phi_max - g->phi_min) * (o->bounds.bottom - o->bounds.top);
                    double dy = -osc_wrap_phi(p->phi - prev->phi) / (2 * HBN_PI) * period;
                    int shift;
                    for (shift = -1; shift <= 1; ++shift)
                        hit |= line(o, x0, y0 + shift * period, x, y0 + dy + shift * period,
                                    &anchor, draw_lines);
                } else
                    hit = line(o, x0, y0, x, y, &anchor, draw_lines);
            }
        }
        visible |= hit;
        if (hit && draw_label && !placed && attempts < 8 && o->labels < 512 &&
            (!attempts || hypot(anchor.x - last_attempt.x, anchor.y - last_attempt.y) >= 40)) {
            placed = label(o, anchor, text, size);
            last_attempt = anchor;
            ++attempts;
        }
    }
    if (draw_label && visible && !placed)
        ++o->hidden;
}
static void overlay(App *a, HDC dc, RECT bounds, int angular) {
    Overlay o = {0};
    HPEN powder, rod, tick, highlight;
    int pass, i, saved;
    if (!cif_prepare_guides(a) || bounds.right <= bounds.left || bounds.bottom <= bounds.top)
        return;
    o.app = a;
    o.dc = dc;
    o.bounds = bounds;
    o.angular = angular;
    saved = SaveDC(dc);
    IntersectClipRect(dc, bounds.left, bounds.top, bounds.right, bounds.bottom);
    powder = CreatePen(PS_SOLID, 1, RGB(135, 170, 185));
    rod = CreatePen(PS_SOLID, 1, RGB(255, 155, 60));
    tick = CreatePen(PS_SOLID, 1, RGB(255, 240, 135));
    highlight = CreatePen(PS_SOLID, 2, RGB(0, 240, 255));
    /* Draw lines, selected lines, selected labels, then other labels. */
    for (pass = 0; pass < 4; ++pass)
        for (i = 0; i < a->guides.path_count; ++i) {
            const CifGuidePath *p = &a->guides.paths[i];
            int chosen = selected(a, p);
            int draw_label = pass >= 2 && a->cif_labels &&
                             (a->cif_overlay == 1 || p->kind != CIF_POWDER || chosen);
            if (chosen != (pass == 1 || pass == 2) || (pass >= 2 && !draw_label))
                continue;
            SelectObject(dc, chosen                  ? highlight
                             : p->kind == CIF_POWDER ? powder
                             : p->kind == CIF_ROD    ? rod
                                                     : tick);
            SetTextColor(dc, chosen                  ? RGB(0, 240, 255)
                             : p->kind == CIF_POWDER ? RGB(175, 215, 230)
                             : p->kind == CIF_ROD    ? RGB(255, 175, 80)
                                                     : RGB(255, 240, 135));
            path_draw(&o, p, pass < 2, draw_label);
        }
    if (o.hidden) {
        char text[120];
        snprintf(text, sizeof text, "%d labels hidden; see CIF groups", o.hidden);
        SetTextColor(dc, RGB(255, 255, 255));
        SetBkColor(dc, RGB(15, 20, 25));
        SetBkMode(dc, OPAQUE);
        TextOutA(dc, bounds.left + 4, bounds.bottom - 18, text, (int)strlen(text));
    }
    RestoreDC(dc, saved);
    DeleteObject(powder);
    DeleteObject(rod);
    DeleteObject(tick);
    DeleteObject(highlight);
}
void cif_detector_overlay(App *a, HDC dc) {
    RECT bounds = {0, 0, a->view_width, a->view_height};
    overlay(a, dc, bounds, 0);
}
void cif_angle_overlay(App *a, HDC dc, RECT bounds) {
    if (a->integration.signal)
        overlay(a, dc, bounds, 1);
}
void cif_angle_marker(App *a, HDC dc, RECT bounds) {
    const OscGrid *g = &a->integration.grid;
    const CifGrouping *set = &a->cif.grouping[CIF_POWDER];
    HPEN pen, selected_pen;
    int i, saved;
    if (!cif_guides_ready(a) || !a->integration.signal)
        return;
    saved = SaveDC(dc);
    IntersectClipRect(dc, bounds.left, bounds.top, bounds.right, bounds.bottom);
    pen = CreatePen(PS_SOLID, 1, RGB(105, 165, 185));
    selected_pen = CreatePen(PS_SOLID, 2, RGB(0, 220, 240));
    for (i = 0; i < set->count; ++i) {
        const CifPeak *p = &a->cif.peaks[set->members[set->groups[i].first]];
        double theta = p->two_theta_deg * HBN_PI / 180;
        int x;
        if (theta < g->theta_min || theta > g->theta_max)
            continue;
        x = bounds.left + (int)((theta - g->theta_min) / (g->theta_max - g->theta_min) *
                                (bounds.right - bounds.left));
        SelectObject(dc, a->cif_kind == CIF_POWDER && a->cif_selected == i ? selected_pen : pen);
        MoveToEx(dc, x, bounds.top, NULL);
        LineTo(dc, x, bounds.bottom);
    }
    RestoreDC(dc, saved);
    DeleteObject(pen);
    DeleteObject(selected_pen);
}
