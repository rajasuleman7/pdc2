/* Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
 * Task 2: row-oriented sparse PageRank; directed original-ID input.
 * Usage: task2 [g6|edge-list] [threads] [static|dynamic1024|guided] [omit-dangling=0]
 * The optional final flag demonstrates loss of rank mass, not a correct algorithm. */
#include "../23i-0041-CS-D-TASK0-V1.h"
__attribute__((optimize("no-tree-vectorize"))) static int
pagerank(const graph_t *g, double *r, double *work, int t, int omit, double *seconds) {
    int n = g->n;
    double *a = r, *b = work;
    for (int i = 0; i < n; i++)
        a[i] = 1.0 / n;
    int it;
    double start = omp_get_wtime();
    for (it = 1; it <= 200; it++) {
        double dangling = 0, delta = 0;
#pragma omp parallel for if (t > 0) reduction(+ : dangling) schedule(static)
        for (int j = 0; j < n; j++)
            if (!g->out[j])
                dangling += a[j];
        if (omit)
            dangling = 0;
#pragma omp parallel for if (t > 0) reduction(+ : delta) schedule(runtime)
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int p = g->rp[i]; p < g->rp[i + 1]; p++)
                sum += g->val[p] * a[g->col[p]];
            b[i] = 0.15 / n + 0.85 * (sum + dangling / n);
            delta += fabs(b[i] - a[i]);
        }
        double *tmp = a;
        a = b;
        b = tmp;
        if (delta < 1e-8)
            break;
    }
    *seconds = omp_get_wtime() - start;
    /* Only final extraction is copied, outside timing; iterations swap pointers. */
    if (a != r)
        memcpy(r, a, n * sizeof(double));
    return it > 200 ? 200 : it;
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "g6", *sched = argc > 3 ? argv[3] : "static";
    int t = threads(argc > 2 ? atoi(argv[2]) : 4), omit = argc > 4 ? atoi(argv[4]) : 0;
    graph_t g = graph_read(path);
    int n = g.n, dangling = 0, maxrow = 0;
    double row2 = 0;
    for (int i = 0; i < n; i++) {
        dangling += !g.out[i];
        int len = g.rp[i + 1] - g.rp[i];
        if (len > maxrow)
            maxrow = len;
        row2 += (double)len * len;
    }
    printf(
        "TASK2 n=%d nnz=%d dangling=%d threads=%d schedule=%s meanrow=%.6f maxrow=%d row_sd=%.6f\n",
        n, g.m, dangling, t, sched, (double)g.m / n, maxrow,
        sqrt(row2 / n - pow((double)g.m / n, 2)));
    omp_set_schedule(!strcmp(sched, "guided")        ? omp_sched_guided
                     : !strcmp(sched, "dynamic1024") ? omp_sched_dynamic
                                                     : omp_sched_static,
                     !strcmp(sched, "dynamic1024") ? 1024 : 0);
    double *r = mem(n, 8), *work = mem(n, 8), *ref = mem(n, 8), dt, base = 1e99, best = 1e99;
    int it = 0;
    for (int k = 0; k < 5; k++) {
        pagerank(&g, ref, work, 0, omit, &dt);
        if (dt < base)
            base = dt;
    }
    for (int k = 0; k < 5; k++) {
        it = pagerank(&g, r, work, t, omit, &dt);
        if (dt < best)
            best = dt;
    }
    double sum = 0, err = 0;
    for (int i = 0; i < n; i++) {
        sum += r[i];
        err = fmax(err, fabs(r[i] - ref[i]));
    }
    printf("RESULT variant=threads threads=%d seconds=%.9f baseline=%.9f speedup=%.6f "
           "efficiency=%.6f iterations=%d mass=%.12f maxdiff=%.12g omitted=%d\n",
           t, best, base, base / best, base / best / t, it, sum, err, omit);
    int *used = mem(n, 4);
    for (int k = 0; k < (n < 10 ? n : 10); k++) {
        int b = -1;
        for (int i = 0; i < n; i++)
            if (!used[i] && (b < 0 || r[i] > r[b]))
                b = i;
        used[b] = 1;
        printf("TOP position=%d id=%lld rank=%.12f\n", k + 1, g.id[b], r[b]);
    }
    if (!strcmp(path, "g6")) {
        double expected[] = {.186037, .104066, .273216, .141117, .189870, .105695};
        for (int i = 0; i < 6; i++)
            if (fabs(r[i] - expected[i]) > 1e-6)
                return 2;
    }
    int fail = err > 1e-10 || (!omit && fabs(sum - 1) > 1e-8);
    free(used);
    free(r);
    free(work);
    free(ref);
    graph_free(&g);
    return fail ? 2 : 0;
}
