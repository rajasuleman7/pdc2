/* Muhammad Ayaan Naveed | 23i-0041 | CS-D | Assignment 2
 * Shared checked allocation and sparse directed-graph input. */
#ifndef PDC_COMMON_H
#define PDC_COMMON_H
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static inline void *mem(size_t n, size_t s) {
    if (s && n > SIZE_MAX / s) {
        fprintf(stderr, "allocation overflow\n");
        exit(1);
    }
    void *p = calloc(n ? n : 1, s);
    if (!p) {
        perror("calloc");
        exit(1);
    }
    return p;
}
static inline int threads(int t) {
    if (t < 1) {
        fprintf(stderr, "threads must be positive\n");
        exit(1);
    }
    omp_set_dynamic(0);
    omp_set_num_threads(t);
    return t;
}
typedef struct {
    int n, m, *src, *dst, *out, *rp, *col;
    long long *id;
    double *val;
} graph_t;
static int cmp_id(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}
static int id_index(const long long *a, int n, long long v) {
    int l = 0, h = n;
    while (l < h) {
        int m = l + (h - l) / 2;
        if (a[m] < v)
            l = m + 1;
        else
            h = m;
    }
    return l;
}
static graph_t graph_read(const char *path) {
    graph_t g = {0};
    size_t cap = 1024, m = 0;
    long long *s = mem(cap, sizeof(*s)), *d = mem(cap, sizeof(*d));
    if (!strcmp(path, "g6")) {
        long long ss[] = {0, 0, 1, 2, 2, 3, 4, 4, 5, 5}, dd[] = {1, 2, 2, 0, 3, 4, 2, 5, 0, 4};
        memcpy(s, ss, sizeof ss);
        memcpy(d, dd, sizeof dd);
        m = 10;
    } else {
        FILE *f = fopen(path, "r");
        if (!f) {
            perror(path);
            exit(1);
        }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            char *p = line;
            while (*p == ' ' || *p == '\t')
                p++;
            if (*p == '#' || *p == '\n' || !*p)
                continue;
            long long a, b;
            char extra;
            if (sscanf(p, "%lld %lld %c", &a, &b, &extra) != 2 || a < 0 || b < 0) {
                fprintf(stderr, "invalid edge line\n");
                exit(1);
            }
            if (m >= INT_MAX / 2) {
                fprintf(stderr, "32-bit graph limit\n");
                exit(1);
            }
            if (m == cap) {
                cap *= 2;
                void *ss = realloc(s, cap * sizeof(*s)), *dd = realloc(d, cap * sizeof(*d));
                if (!ss || !dd)
                    exit(1);
                s = ss;
                d = dd;
            }
            s[m] = a;
            d[m++] = b;
        }
        if (ferror(f)) {
            perror("read");
            exit(1);
        }
        fclose(f);
    }
    if (!m) {
        fprintf(stderr, "empty graph\n");
        exit(1);
    }
    /* Sort endpoint IDs rather than allocating by the largest original ID. */
    long long *ids = mem(2 * m, sizeof(*ids));
    for (size_t k = 0; k < m; k++) {
        ids[2 * k] = s[k];
        ids[2 * k + 1] = d[k];
    }
    qsort(ids, 2 * m, sizeof(*ids), cmp_id);
    int n = 0;
    for (size_t k = 0; k < 2 * m; k++)
        if (!k || ids[k] != ids[k - 1])
            ids[n++] = ids[k];
    g.n = n;
    g.m = (int)m;
    g.id = ids;
    g.src = mem(m, sizeof(int));
    g.dst = mem(m, sizeof(int));
    g.out = mem(n, sizeof(int));
    g.rp = mem(n + 1, sizeof(int));
    g.col = mem(m, sizeof(int));
    g.val = mem(m, sizeof(double));
    for (int k = 0; k < g.m; k++) {
        g.src[k] = id_index(ids, n, s[k]);
        g.dst[k] = id_index(ids, n, d[k]);
        g.out[g.src[k]]++;
        g.rp[g.dst[k] + 1]++;
    }
    /* Counting construction of incoming CSR: counts -> prefix sum -> scatter. */
    for (int i = 0; i < n; i++)
        g.rp[i + 1] += g.rp[i];
    int *cursor = mem(n, sizeof(int));
    memcpy(cursor, g.rp, n * sizeof(int));
    for (int k = 0; k < g.m; k++) {
        int p = cursor[g.dst[k]]++;
        g.col[p] = g.src[k];
        g.val[p] = 1.0 / g.out[g.src[k]];
    }
    free(cursor);
    free(s);
    free(d);
    return g;
}
static void graph_free(graph_t *g) {
    free(g->src);
    free(g->dst);
    free(g->out);
    free(g->rp);
    free(g->col);
    free(g->id);
    free(g->val);
}
#endif
