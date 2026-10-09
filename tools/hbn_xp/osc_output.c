#include "osc_app.h"

void export_profiles(App *a, const char *path) {
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
void export_bmp(App *a, const char *path) {
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
