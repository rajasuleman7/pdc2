/* Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
 * Task 4: fixed-source double-precision Jacobi diffusion.
 * Usage: task4 [N=512] [steps=20000] [threads=4]
 * [scalar|threads|simd|hybrid|persistent|collapse|swapped] [parallel-init=1]
 * [optional field file outside submission]. Five complete runs, minimum kernel time.
 * threads/hybrid open a team every step; persistent reuses one team. */
#include "../23i-0041-CS-D-TASK0-V1.h"
static inline double update(const double *a, int p, int w) {
    return 0.2 * (a[p] + a[p - w] + a[p + w] + a[p - 1] + a[p + 1]);
}
__attribute__((optimize("no-tree-vectorize"))) static double row_scalar(const double *a, double *b,
                                                                        int i, int n) {
    int w = n + 2, lo = n / 4, hi = lo + 7;
    double delta = 0;
    /* Segment the source rows: no source branch inside either column loop. */
    int left = (i >= 3 * n / 4 && i <= 3 * n / 4 + 7) ? lo - 1 : n;
    for (int j = 1; j <= left; j++) {
        int p = i * w + j;
        double v = update(a, p, w);
        b[p] = v;
        double e = fabs(v - a[p]);
        if (e > delta)
            delta = e;
    }
    if (left < n)
        for (int j = hi + 1; j <= n; j++) {
            int p = i * w + j;
            double v = update(a, p, w);
            b[p] = v;
            double e = fabs(v - a[p]);
            if (e > delta)
                delta = e;
        }
    return delta;
}
static double row_simd(const double *restrict a, double *restrict b, int i, int n) {
    int w = n + 2, lo = n / 4, hi = lo + 7;
    double delta = 0;
    int left = (i >= 3 * n / 4 && i <= 3 * n / 4 + 7) ? lo - 1 : n;
#pragma omp simd reduction(max : delta)
    for (int j = 1; j <= left; j++) {
        int p = i * w + j;
        double v = update(a, p, w);
        b[p] = v;
        double e = fabs(v - a[p]);
        if (e > delta)
            delta = e;
    }
    if (left < n) {
#pragma omp simd reduction(max : delta)
        for (int j = hi + 1; j <= n; j++) {
            int p = i * w + j;
            double v = update(a, p, w);
            b[p] = v;
            double e = fabs(v - a[p]);
            if (e > delta)
                delta = e;
        }
    }
    return delta;
}
static void initialise(double *a, double *b, int n, int par) {
    int w = n + 2;
#pragma omp parallel for if (par) schedule(static)
    for (int i = 0; i < w; i++)
        for (int j = 0; j < w; j++) {
            double v = i == 0 ? 100 : 0;
            if (i >= 3 * n / 4 && i <= 3 * n / 4 + 7 && j >= n / 4 && j <= n / 4 + 7)
                v = 200;
            a[(size_t)i * w + j] = b[(size_t)i * w + j] = v;
        }
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 512, S = argc > 2 ? atoi(argv[2]) : 20000,
        t = threads(argc > 3 ? atoi(argv[3]) : 4), init = argc > 5 ? atoi(argv[5]) : 1;
    const char *v = argc > 4 ? argv[4] : "persistent";
    if (n < 32 || n > 10000 || S < 1)
        return 1;
    int scalar = !strcmp(v, "scalar"), simd = !strcmp(v, "simd"), hybrid = !strcmp(v, "hybrid"),
        persist = !strcmp(v, "persistent"), collapse = !strcmp(v, "collapse"),
        swapped = !strcmp(v, "swapped");
    if (!scalar && !simd && !hybrid && !persist && !collapse && !swapped && strcmp(v, "threads")) {
        fprintf(stderr, "unknown variant\n");
        return 1;
    }
    int w = n + 2;
    size_t count = (size_t)w * w;
    double *a = mem(count, 8), *b = mem(count, 8), best = 1e99, bestinit = 1e99, delta = 0,
           centre = 0, mean = 0;
    for (int repeat = 0; repeat < 5; repeat++) {
        double tm = omp_get_wtime();
        initialise(a, b, n, init);
        tm = omp_get_wtime() - tm;
        if (tm < bestinit)
            bestinit = tm;
        double start = omp_get_wtime();
        if (persist) {
#pragma omp parallel shared(a, b, delta)
            {
                for (int s = 0; s < S; s++) {
#pragma omp single
                    delta = 0;
#pragma omp for reduction(max : delta) schedule(static)
                    for (int i = 1; i <= n; i++) {
                        double e = row_simd(a, b, i, n);
                        if (e > delta)
                            delta = e;
                    }
#pragma omp single
                    {
                        double *tmp = a;
                        a = b;
                        b = tmp;
                    }
                }
            }
        } else
            for (int s = 0; s < S; s++) {
                delta = 0;
                if (collapse) {
#pragma omp parallel for collapse(2) reduction(max : delta) schedule(static)
                    for (int i = 1; i <= n; i++)
                        for (int j = 1; j <= n; j++)
                            if (!(i >= 3 * n / 4 && i <= 3 * n / 4 + 7 && j >= n / 4 &&
                                  j <= n / 4 + 7)) {
                                int p = i * w + j;
                                double u = update(a, p, w);
                                b[p] = u;
                                double e = fabs(u - a[p]);
                                if (e > delta)
                                    delta = e;
                            }
                } else if (swapped) {
#pragma omp parallel for reduction(max : delta) schedule(static)
                    for (int j = 1; j <= n; j++)
                        for (int i = 1; i <= n; i++)
                            if (!(i >= 3 * n / 4 && i <= 3 * n / 4 + 7 && j >= n / 4 &&
                                  j <= n / 4 + 7)) {
                                int p = i * w + j;
                                double u = update(a, p, w);
                                b[p] = u;
                                double e = fabs(u - a[p]);
                                if (e > delta)
                                    delta = e;
                            }
                } else if (scalar) {
                    for (int i = 1; i <= n; i++) {
                        double e = row_scalar(a, b, i, n);
                        if (e > delta)
                            delta = e;
                    }
                } else if (simd) {
                    for (int i = 1; i <= n; i++) {
                        double e = row_simd(a, b, i, n);
                        if (e > delta)
                            delta = e;
                    }
                } else {
#pragma omp parallel for reduction(max : delta) schedule(static)
                    for (int i = 1; i <= n; i++) {
                        double e = hybrid ? row_simd(a, b, i, n) : row_scalar(a, b, i, n);
                        if (e > delta)
                            delta = e;
                    }
                }
                double *tmp = a;
                a = b;
                b = tmp;
            }
        double dt = omp_get_wtime() - start;
        if (dt < best)
            best = dt;
        centre = a[(n / 2) * w + n / 2];
        mean = 0;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= n; j++)
                mean += a[i * w + j];
        mean /= (double)n * n;
    }
    printf("TASK4 n=%d steps=%d cells=%zu threads=%d init_parallel=%d\n", n, S, (size_t)n * n,
           (scalar || simd) ? 1 : t, init);
    printf("RESULT variant=%s threads=%d seconds=%.9f init_seconds=%.9f centre=%.13g mean=%.13g "
           "delta=%.13g\n",
           v, (scalar || simd) ? 1 : t, best, bestinit, centre, mean, delta);
    if (argc > 6) {
        FILE *f = fopen(argv[6], "w");
        if (!f)
            return 2;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++)
                fprintf(f, "%.9g%c", a[i * w + j], j == n ? '\n' : ' ');
        }
        fclose(f);
    }
    /* The printed assignment reference is rounded to 5-6 significant digits. */
    int fail = n == 512 && S == 20000 &&
               (fabs(centre - 2.4557) > 5e-5 || fabs(mean - 18.1928) > 5e-5 ||
                fabs(delta - .00120) > 5e-6);
    free(a);
    free(b);
    return fail ? 3 : 0;
}
