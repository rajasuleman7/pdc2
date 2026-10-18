/* Muhammad Ayaan Naveed | Student ID: 23i-0041 | CS-D | Assignment 2
 * Task 5: SuiteSparse:GraphBLAS C API, out-degrees and masked BFS.
 * Usage: task5 [g6|edge-list] [threads=4] [source-original-ID=0]
 * GraphBLAS blocking mode makes kernel timing include completed operations. */
#include "../23i-0041-CS-D-TASK0-V1.h"
#include <GraphBLAS.h>
#define CHECK_GB(call)                                                                             \
    do {                                                                                           \
        GrB_Info status = (call);                                                                  \
        if (status != GrB_SUCCESS) {                                                               \
            fprintf(stderr, "GraphBLAS error %d at line %d\n", (int)status, __LINE__);             \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)
static void example(void) {
    GrB_Index src[] = {0, 0, 1, 2, 3}, dst[] = {1, 2, 3, 3, 0};
    bool one[] = {1, 1, 1, 1, 1};
    GrB_Matrix A;
    GrB_Vector q, next;
    CHECK_GB(GrB_Matrix_new(&A, GrB_BOOL, 4, 4));
    CHECK_GB(GrB_Matrix_build_BOOL(A, src, dst, one, 5, GrB_LOR));
    CHECK_GB(GrB_Vector_new(&q, GrB_BOOL, 4));
    CHECK_GB(GrB_Vector_new(&next, GrB_BOOL, 4));
    CHECK_GB(GrB_Vector_setElement_BOOL(q, true, 0));
    CHECK_GB(GrB_vxm(next, NULL, NULL, GrB_LOR_LAND_SEMIRING_BOOL, q, A, NULL));
    GrB_Index k = 4, idx[4];
    bool val[4];
    CHECK_GB(GrB_Vector_extractTuples_BOOL(idx, val, &k, next));
    printf("Vertices one hop from 0:");
    for (GrB_Index j = 0; j < k; j++)
        printf(" %lu", (unsigned long)idx[j]);
    printf("\n");
    if (k != 2 || idx[0] != 1 || idx[1] != 2)
        exit(3);
    CHECK_GB(GrB_Matrix_free(&A));
    CHECK_GB(GrB_Vector_free(&q));
    CHECK_GB(GrB_Vector_free(&next));
}
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "g6";
    int t = threads(argc > 2 ? atoi(argv[2]) : 4);
    long long source = argc > 3 ? atoll(argv[3]) : 0;
    graph_t g = graph_read(path);
    int n = g.n, s = id_index(g.id, n, source);
    if (s >= n || g.id[s] != source) {
        fprintf(stderr, "source ID absent\n");
        return 1;
    }
    CHECK_GB(GrB_init(GrB_BLOCKING));
    CHECK_GB(GxB_Global_Option_set_INT32(GxB_NTHREADS, t));
    unsigned int major, minor;
    CHECK_GB(GrB_getVersion(&major, &minor));
    printf("GraphBLAS C API version: %u.%u ; SuiteSparse implementation %d.%d.%d\n", major, minor,
           GxB_IMPLEMENTATION_MAJOR, GxB_IMPLEMENTATION_MINOR, GxB_IMPLEMENTATION_SUB);
    example();
    GrB_Matrix A;
    CHECK_GB(GrB_Matrix_new(&A, GrB_BOOL, n, n));
    GrB_Index *src = mem(g.m, sizeof(*src)), *dst = mem(g.m, sizeof(*dst));
    bool *one = mem(g.m, sizeof(bool));
    for (int k = 0; k < g.m; k++) {
        src[k] = g.src[k];
        dst[k] = g.dst[k];
        one[k] = true;
    }
    CHECK_GB(GrB_Matrix_build_BOOL(A, src, dst, one, g.m, GrB_LOR));
    free(src);
    free(dst);
    free(one);
    GrB_Vector degree;
    CHECK_GB(GrB_Vector_new(&degree, GrB_UINT64, n));
    CHECK_GB(GrB_Matrix_reduce_Monoid(degree, NULL, NULL, GrB_PLUS_MONOID_UINT64, A, NULL));
    if (!strcmp(path, "g6")) {
        printf("OUTDEGREE");
        for (int i = 0; i < n; i++) {
            uint64_t d = 0;
            CHECK_GB(GrB_Vector_extractElement_UINT64(&d, degree, i));
            printf(" %lu", (unsigned long)d);
            if (d != (uint64_t)g.out[i])
                return 4;
        }
        printf("\n");
    }
    GrB_Vector q, next, level;
    CHECK_GB(GrB_Vector_new(&q, GrB_BOOL, n));
    CHECK_GB(GrB_Vector_new(&next, GrB_BOOL, n));
    CHECK_GB(GrB_Vector_new(&level, GrB_INT32, n));
    double best = 1e99;
    for (int r = 0; r < 5; r++) {
        CHECK_GB(GrB_Vector_clear(q));
        CHECK_GB(GrB_Vector_clear(next));
        CHECK_GB(GrB_Vector_clear(level));
        CHECK_GB(GrB_Vector_setElement_BOOL(q, true, s));
        GrB_Index count = 1;
        int32_t depth = 0;
        double begin = omp_get_wtime();
        while (count) {
            /* Store level zero too: the visited MASK MUST be structural. */
            CHECK_GB(GrB_Vector_assign_INT32(level, q, NULL, depth, GrB_ALL, n, GrB_DESC_S));
            CHECK_GB(GrB_vxm(next, level, NULL, GrB_LOR_LAND_SEMIRING_BOOL, q, A, GrB_DESC_RSC));
            GrB_Vector tmp = q;
            q = next;
            next = tmp;
            CHECK_GB(GrB_Vector_nvals(&count, q));
            depth++;
        }
        double dt = omp_get_wtime() - begin;
        if (dt < best)
            best = dt;
    }
    GrB_Index reached;
    CHECK_GB(GrB_Vector_nvals(&reached, level));
    GrB_Index *idx = mem(reached, sizeof(*idx));
    int32_t *val = mem(reached, sizeof(*val));
    CHECK_GB(GrB_Vector_extractTuples_INT32(idx, val, &reached, level));
    int *levels = mem(n, 4), *counts = mem(n, 4);
    for (int i = 0; i < n; i++)
        levels[i] = -1;
    int max = 0;
    for (GrB_Index k = 0; k < reached; k++) {
        levels[idx[k]] = val[k];
        counts[val[k]]++;
        if (val[k] > max)
            max = val[k];
    }
    /* Independent scalar queue BFS verifies every level, not just reachability. */
    int *rp = mem(n + 1, 4), *col = mem(g.m, 4), *cur = mem(n, 4), *queue = mem(n, 4),
        *ref = mem(n, 4);
    for (int k = 0; k < g.m; k++)
        rp[g.src[k] + 1]++;
    for (int i = 0; i < n; i++)
        rp[i + 1] += rp[i];
    memcpy(cur, rp, n * 4);
    for (int k = 0; k < g.m; k++)
        col[cur[g.src[k]]++] = g.dst[k];
    for (int i = 0; i < n; i++)
        ref[i] = -1;
    int head = 0, tail = 1;
    queue[0] = s;
    ref[s] = 0;
    while (head < tail) {
        int u = queue[head++];
        for (int p = rp[u]; p < rp[u + 1]; p++) {
            int v = col[p];
            if (ref[v] < 0) {
                ref[v] = ref[u] + 1;
                queue[tail++] = v;
            }
        }
    }
    for (int i = 0; i < n; i++)
        if (ref[i] != levels[i]) {
            fprintf(stderr, "BFS mismatch\n");
            return 5;
        }
    printf("TASK5 n=%d edges=%d threads=%d source=%lld reached=%lu maxlevel=%d\n", n, g.m, t,
           source, (unsigned long)reached, max);
    printf("RESULT variant=graphblas threads=%d seconds=%.9f validation=PASS\n", t, best);
    for (int i = 0; i <= max; i++)
        printf("LEVEL depth=%d count=%d\n", i, counts[i]);
    if (!strcmp(path, "g6")) {
        int expected[] = {0, 1, 1, 2, 3, 4};
        printf("BFS");
        for (int i = 0; i < n; i++) {
            printf(" %d", levels[i]);
            if (levels[i] != expected[i])
                return 6;
        }
        printf("\n");
    }
    free(idx);
    free(val);
    free(levels);
    free(counts);
    free(rp);
    free(col);
    free(cur);
    free(queue);
    free(ref);
    CHECK_GB(GrB_Vector_free(&degree));
    CHECK_GB(GrB_Vector_free(&q));
    CHECK_GB(GrB_Vector_free(&next));
    CHECK_GB(GrB_Vector_free(&level));
    CHECK_GB(GrB_Matrix_free(&A));
    CHECK_GB(GrB_finalize());
    graph_free(&g);
    return 0;
}
