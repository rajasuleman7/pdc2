/* Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
 * Task 1: corrected scalar, threads, SIMD and hybrid Taylor kernels.
 * Usage: task1 [N=16777216] [threads=4] [terms=12] [schedule=static|dynamic1|dynamic4096]
 * Exactly `terms` corrections are used, matching Listing 2's j <= terms. */
#include "../23i-0041-CS-D-TASK0-V1.h"
/* The minimum numerical fix is float denom instead of int denom. */
__attribute__((noinline)) static float series(float x, int terms) {
    float value = x, numer = x * x * x, denom = 6.0f;
    int sign = -1;
    for (int j = 1; j <= terms; j++) {
        value += sign * numer / denom;
        numer *= x * x;
        denom *= (2 * j + 2) * (2 * j + 3);
        sign *= -1;
    }
    return value;
}
/* Fixed-count specialisations preserve operation order while allowing GCC
 * to unroll the dependent inner recurrence before vectorising across inputs. */
#define FIXED_SERIES(K)                                                                            \
    static inline __attribute__((always_inline)) float series##K(float x) {                        \
        float value = x, numer = x * x * x, denom = 6.0f;                                          \
        int sign = -1;                                                                             \
        _Pragma("GCC unroll 12") for (int j = 1; j <= K; j++) {                                    \
            value += sign * numer / denom;                                                         \
            numer *= x * x;                                                                        \
            denom *= (2 * j + 2) * (2 * j + 3);                                                    \
            sign *= -1;                                                                            \
        }                                                                                          \
        return value;                                                                              \
    }
FIXED_SERIES(4)
FIXED_SERIES(8)
FIXED_SERIES(12)
#undef FIXED_SERIES
/* Defined two's-complement wrapping model of the faulty int recurrence;
 * the original signed-overflow C has undefined behaviour. */
static float broken(float x, int terms) {
    float value = x, numer = x * x * x;
    uint32_t bits = 6;
    int sign = -1;
    for (int j = 1; j <= terms; j++) {
        int32_t denom;
        memcpy(&denom, &bits, sizeof denom);
        value += sign * numer / denom;
        numer *= x * x;
        bits *= (uint32_t)((2 * j + 2) * (2 * j + 3));
        sign *= -1;
    }
    return value;
}
__attribute__((noinline, optimize("no-tree-vectorize"))) static void
scalar(int n, int terms, const float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = series(x[i], terms);
}
__attribute__((noinline, optimize("no-tree-vectorize"))) static void
threaded(int n, int terms, const float *x, float *y) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++)
        y[i] = series(x[i], terms);
}
static void simd(int n, int terms, const float *restrict x, float *restrict y) {
#define SIMD_CASE(K)                                                                               \
    case K:                                                                                        \
        _Pragma("omp simd") for (int i = 0; i < n; i++) y[i] = series##K(x[i]);                    \
        break;
    switch (terms) { SIMD_CASE(4) SIMD_CASE(8) SIMD_CASE(12) }
#undef SIMD_CASE
}
static void hybrid(int n, int terms, const float *restrict x, float *restrict y) {
#define HYBRID_CASE(K)                                                                             \
    case K:                                                                                        \
        _Pragma("omp parallel for simd schedule(runtime)") for (int i = 0; i < n; i++) y[i] =      \
            series##K(x[i]);                                                                       \
        break;
    switch (terms) { HYBRID_CASE(4) HYBRID_CASE(8) HYBRID_CASE(12) }
#undef HYBRID_CASE
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1 << 24, t = threads(argc > 2 ? atoi(argv[2]) : 4),
        terms = argc > 3 ? atoi(argv[3]) : 12;
    const char *sched = argc > 4 ? argv[4] : "static";
    if (n < 1 || (terms != 4 && terms != 8 && terms != 12))
        return 1;
    omp_set_schedule(!strcmp(sched, "static") ? omp_sched_static : omp_sched_dynamic,
                     !strcmp(sched, "dynamic1") ? 1 : 4096);
    float *x = mem(n, 4), *ref = mem(n, 4), *y = mem(n, 4);
    const double pi = acos(-1.0);
    for (int i = 0; i < n; i++)
        x[i] = (float)(-pi + 2 * pi * i / n);
    printf("TASK1 N=%d threads=%d terms=%d schedule=%s\n", n, t, terms, sched);
    for (int q = 4; q <= 12; q += 4) {
        double e = 0, b = 0;
        for (int i = 0; i < n; i++) {
            e = fmax(e, fabs((double)series(x[i], q) - sinf(x[i])));
            b = fmax(b, fabs((double)broken(x[i], q) - sinf(x[i])));
        }
        printf("ERROR terms=%d corrected=%.12g wrapping_original=%.12g\n", q, e, b);
    }
    scalar(n, terms, x, ref);
    void (*fn[])(int, int, const float *, float *) = {scalar, threaded, simd, hybrid};
    const char *names[] = {"scalar", "threads", "simd", "hybrid"};
    double base = 0;
    for (int v = 0; v < 4; v++) {
        double best = 1e99;
        for (int r = 0; r < 5; r++) {
            double a = omp_get_wtime();
            fn[v](n, terms, x, y);
            double dt = omp_get_wtime() - a;
            if (dt < best)
                best = dt;
        }
        if (!v)
            base = best;
        double err = 0;
        int diff = 0;
        for (int i = 0; i < n; i++) {
            err = fmax(err, fabs((double)y[i] - ref[i]));
            diff += memcmp(y + i, ref + i, 4) != 0;
        }
        printf("RESULT variant=%s threads=%d seconds=%.9f speedup=%.6f efficiency=%.6f "
               "maxdiff=%.12g bitdiff=%d\n",
               names[v], (v == 0 || v == 2) ? 1 : t, best, base / best,
               base / best / ((v == 0 || v == 2) ? 1 : t), err, diff);
        if (err > 2e-6)
            return 2;
    }
    free(x);
    free(y);
    free(ref);
    return 0;
}
