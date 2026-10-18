/* Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
 * Task 3: CSR conversions and AVX2 SpMV. Usage: task3 [threads=4] [grid=1024] [optional graph]
 * Tables are single precision; columns/offsets are 32-bit. Duplicate
 * coordinates are not allowed in the conversion input. Graph estimates only
 * inspect CSR, so duplicate edges do not affect those estimates. */
#include "../23i-0041-CS-D-TASK0-V1.h"
#include <immintrin.h>
typedef struct {
    int n, nnz, *row_ptr, *col;
    float *val;
} csr_t;
typedef struct {
    int n, nnz, *row, *col;
    float *val;
} coo_t;
typedef struct {
    int n, W, *col;
    float *val;
} ell_t;
typedef struct {
    int n, nd, *off;
    float *val;
} dia_t;
static int use_parallel = 1;
coo_t csr_to_coo(const csr_t *A) {
    coo_t C = {A->n, A->nnz, mem(A->nnz, 4), mem(A->nnz, 4), mem(A->nnz, 4)};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A->n; i++)
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            C.row[p] = i;
            C.col[p] = A->col[p];
            C.val[p] = A->val[p];
        }
    return C;
}
ell_t csr_to_ell(const csr_t *A) {
    int W = 0, n = A->n;
#pragma omp parallel for reduction(max : W)
    for (int i = 0; i < n; i++) {
        int len = A->row_ptr[i + 1] - A->row_ptr[i];
        if (len > W)
            W = len;
    }
    ell_t E = {n, W, mem((size_t)n * W, 4), mem((size_t)n * W, 4)};
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            size_t k = (size_t)(p - A->row_ptr[i]) * n + i;
            E.col[k] = A->col[p];
            E.val[k] = A->val[p];
        }
    /* calloc padding: zero value AND valid column zero (gather still reads it). */
    return E;
}
int csr_to_dia(const csr_t *A, dia_t *D, size_t max_bytes) {
    int n = A->n, nd = 0;
    *D = (dia_t){0};
    int *flag = mem((size_t)2 * n - 1, 4);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++) {
            int o = A->col[p] - i + n - 1;
#pragma omp atomic write
            flag[o] = 1;
        }
    for (int k = 0; k < 2 * n - 1; k++)
        if (flag[k])
            nd++;
    size_t unit = ((size_t)n + 1) * 4;
    /* Scratch flags are necessary for counting; no DIA output allocation on rejection. */
    if ((size_t)nd > max_bytes / unit) {
        free(flag);
        return -1;
    }
    D->n = n;
    D->nd = nd;
    D->off = mem(nd, 4);
    D->val = mem((size_t)nd * n, 4);
    int d = 0;
    for (int k = 0; k < 2 * n - 1; k++)
        if (flag[k]) {
            D->off[d] = k - n + 1;
            flag[k] = d++;
        }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++)
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
            D->val[(size_t)flag[A->col[p] - i + n - 1] * n + i] = A->val[p];
    free(flag);
    return 0;
}
__attribute__((optimize("no-tree-vectorize"))) void spmv_csr(const csr_t *A, const float *x,
                                                             float *y) {
#pragma omp parallel for if (use_parallel) schedule(static)
    for (int i = 0; i < A->n; i++) {
        float s = 0;
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
            s += A->val[p] * x[A->col[p]];
        y[i] = s;
    }
}
void spmv_ell_avx2(const ell_t *E, const float *x, float *y) {
    int n = E->n;
#pragma omp parallel for if (use_parallel) schedule(static)
    for (int b = 0; b < n / 8; b++) {
        int i = 8 * b;
        __m256 sum = _mm256_setzero_ps();
        for (int k = 0; k < E->W; k++) {
            size_t p = (size_t)k * n + i;
            __m256i c = _mm256_loadu_si256((const __m256i *)(E->col + p));
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(E->val + p), _mm256_i32gather_ps(x, c, 4), sum);
        }
        _mm256_storeu_ps(y + i, sum);
    }
    for (int i = n / 8 * 8; i < n; i++) {
        float s = 0;
        for (int k = 0; k < E->W; k++) {
            size_t p = (size_t)k * n + i;
            s = fmaf(E->val[p], x[E->col[p]], s);
        }
        y[i] = s;
    }
}
void spmv_dia_avx2(const dia_t *D, const float *x, float *y) {
    int n = D->n;
/* Threads own row blocks; parallelising diagonals would race on y. */
#pragma omp parallel for if (use_parallel) schedule(static)
    for (int b = 0; b < (n + 7) / 8; b++) {
        int i = b * 8, end = i + 8 < n ? i + 8 : n;
        __m256 sum = _mm256_setzero_ps();
        float tail[8] = {0};
        for (int d = 0; d < D->nd; d++) {
            int o = D->off[d];
            const float *v = D->val + (size_t)d * n;
            if (end - i == 8 && i + o >= 0 && i + 7 + o < n) {
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(v + i), _mm256_loadu_ps(x + i + o), sum);
            } else {
                for (int j = i; j < end; j++)
                    if (j + o >= 0 && j + o < n)
                        tail[j - i] = fmaf(v[j], x[j + o], tail[j - i]);
            }
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, sum);
        for (int j = i; j < end; j++)
            y[j] = lanes[j - i] + tail[j - i];
    }
}
static void release(csr_t *a, coo_t *c, ell_t *e, dia_t *d) {
    free(a->row_ptr);
    free(a->col);
    free(a->val);
    free(c->row);
    free(c->col);
    free(c->val);
    free(e->col);
    free(e->val);
    free(d->off);
    free(d->val);
}
static csr_t small(int B) {
    /* Explicit sparse arrays, never dense matrices. */
    int ra[] = {0, 2, 4, 7, 9}, ca[] = {0, 1, 1, 2, 0, 2, 3, 1, 3};
    float va[] = {1, 7, 2, 8, 5, 3, 9, 6, 4};
    int rb[] = {0, 2, 4, 7, 8, 11, 13}, cb[] = {0, 1, 1, 4, 0, 2, 3, 3, 2, 4, 5, 3, 5};
    float vb[] = {4, 1, 4, 9, 6, 4, 5, 4, 8, 4, 1, 6, 4};
    csr_t a = {B ? 6 : 4, B ? 13 : 9, NULL, NULL, NULL};
    a.row_ptr = mem(a.n + 1, 4);
    a.col = mem(a.nnz, 4);
    a.val = mem(a.nnz, 4);
    memcpy(a.row_ptr, B ? rb : ra, (a.n + 1) * 4);
    memcpy(a.col, B ? cb : ca, a.nnz * 4);
    memcpy(a.val, B ? vb : va, a.nnz * 4);
    return a;
}
static int validate(int B) {
    csr_t a = small(B);
    coo_t c = csr_to_coo(&a);
    ell_t e = csr_to_ell(&a);
    dia_t d;
    if (csr_to_dia(&a, &d, SIZE_MAX))
        return 1;
    float x[6] = {1, 2, 3, 4, 5, 6}, y[6], refA[] = {15, 28, 50, 28},
          refB[] = {6, 53, 38, 16, 50, 48};
    /* Figure A is copied literally in the report; validate its sparse values. */
    for (int v = 0; v < 3; v++) {
        if (v == 0)
            spmv_csr(&a, x, y);
        else if (v == 1)
            spmv_ell_avx2(&e, x, y);
        else
            spmv_dia_avx2(&d, x, y);
        printf("VALID matrix=%c kernel=%d y=", B ? 'B' : 'A', v);
        for (int i = 0; i < a.n; i++) {
            printf("%.0f%s", y[i], i == a.n - 1 ? "\n" : ",");
            if (y[i] != (B ? refB[i] : refA[i]))
                return 2;
        }
    }
    dia_t reject;
    if (csr_to_dia(&a, &reject, 1) != -1 || reject.val || reject.off)
        return 3;
    for (int i = 0; i < a.n; i++)
        for (int p = a.row_ptr[i]; p < a.row_ptr[i + 1]; p++)
            if (c.row[p] != i || c.col[p] != a.col[p] || c.val[p] != a.val[p])
                return 4;
    release(&a, &c, &e, &d);
    return 0;
}
static csr_t laplace(int g) {
    int n = g * g, nnz = 5 * n - 4 * g;
    csr_t a = {n, nnz, mem(n + 1, 4), mem(nnz, 4), mem(nnz, 4)};
    int p = 0;
    for (int i = 0; i < n; i++) {
        a.row_ptr[i] = p;
        int r = i / g, c = i % g;
        if (r) {
            a.col[p] = i - g;
            a.val[p++] = -1;
        }
        if (c) {
            a.col[p] = i - 1;
            a.val[p++] = -1;
        }
        a.col[p] = i;
        a.val[p++] = 4;
        if (c + 1 < g) {
            a.col[p] = i + 1;
            a.val[p++] = -1;
        }
        if (r + 1 < g) {
            a.col[p] = i + g;
            a.val[p++] = -1;
        }
    }
    a.row_ptr[n] = p;
    return a;
}
int main(int argc, char **argv) {
    int t = threads(argc > 1 ? atoi(argv[1]) : 4), g = argc > 2 ? atoi(argv[2]) : 1024;
    if (g < 2 || g > 10000)
        return 1;
    printf("TASK3 threads=%d grid=%d\n", t, g);
    if (validate(0) || validate(1))
        return 2;
    csr_t a = laplace(g);
    coo_t c = csr_to_coo(&a);
    ell_t e = csr_to_ell(&a);
    dia_t d;
    if (csr_to_dia(&a, &d, (size_t)1 << 30))
        return 3;
    float *x = mem(a.n, 4), *y = mem(a.n, 4), *ref = mem(a.n, 4);
    for (int i = 0; i < a.n; i++)
        x[i] = (float)(i % 17) / 17;
    use_parallel = 0;
    spmv_csr(&a, x, ref);
    double base = 0;
    printf("MEM n=%d nnz=%d csr=%zu coo=%zu ell=%zu dia=%zu W=%d nd=%d\n", a.n, a.nnz,
           (size_t)8 * a.nnz + 4 * ((size_t)a.n + 1), (size_t)12 * a.nnz, (size_t)8 * a.n * e.W,
           (size_t)4 * d.nd * (a.n + 1), e.W, d.nd);
    const char *names[] = {"scalar", "threads", "ell_simd", "ell_hybrid", "dia_simd", "dia_hybrid"};
    for (int v = 0; v < 6; v++) {
        use_parallel = (v == 1 || v == 3 || v == 5);
        double best = 1e99;
        for (int repeat = 0; repeat < 5; repeat++) {
            double begin = omp_get_wtime();
            for (int r = 0; r < 50; r++) {
                if (v < 2)
                    spmv_csr(&a, x, y);
                else if (v < 4)
                    spmv_ell_avx2(&e, x, y);
                else
                    spmv_dia_avx2(&d, x, y);
            }
            double dt = (omp_get_wtime() - begin) / 50;
            if (dt < best)
                best = dt;
        }
        if (v == 0)
            base = best;
        double err = 0;
        for (int i = 0; i < a.n; i++)
            err = fmax(err, fabs(y[i] - ref[i]));
        printf("RESULT variant=%s threads=%d seconds=%.9f speedup=%.6f efficiency=%.6f "
               "maxdiff=%.12g\n",
               names[v], use_parallel ? t : 1, best, base / best,
               base / best / (use_parallel ? t : 1), err);
        if (err > 2e-6)
            return 4;
    }
    if (argc > 3) {
        graph_t G = graph_read(argv[3]);
        int W = 0, nd = 0;
        unsigned char *flags = mem((size_t)2 * G.n - 1, 1);
        for (int i = 0; i < G.n; i++) {
            int len = G.rp[i + 1] - G.rp[i];
            if (len > W)
                W = len;
            for (int p = G.rp[i]; p < G.rp[i + 1]; p++)
                flags[G.col[p] - i + G.n - 1] = 1;
        }
        for (int i = 0; i < 2 * G.n - 1; i++)
            nd += flags[i];
        printf("GRAPH n=%d nnz=%d W=%d nd=%d csr=%zu ell=%zu dia=%zu\n", G.n, G.m, W, nd,
               (size_t)8 * G.m + 4 * ((size_t)G.n + 1), (size_t)8 * G.n * W,
               (size_t)4 * nd * (G.n + 1));
        free(flags);
        graph_free(&G);
    }
    free(x);
    free(y);
    free(ref);
    release(&a, &c, &e, &d);
    return 0;
}
