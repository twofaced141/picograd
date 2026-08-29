#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/core/tensor.h"
#include "../src/backend/cpu/gemm.h"
#include "../src/ops/common.h"
#include "../src/ops/matmul.h"
#include "../src/ops/elementwise.h"
#include "../src/ops/reduce.h"
#include "../src/ops/index.h"
#include "../src/ops/scan.h"

static int fails = 0;

#define CHECK(c)                                                       \
    do {                                                               \
        if (!(c)) {                                                    \
            printf("FAIL:%d %s\n", __LINE__, #c);                      \
            fails++;                                                   \
        }                                                              \
    } while (0)

#define CHECKF(t, i, v)                                                \
    do {                                                               \
        if (!closef((t)->data[i], (v))) {                              \
            printf("FAIL:%d %s[%zu]=%f want %f\n", __LINE__, #t,       \
                   (size_t)(i), (t)->data[i], (double)(v));            \
            fails++;                                                   \
        }                                                              \
    } while (0)

static bool closef(float x, float y)
{
    return fabsf(x - y) <= 1e-4f * (1.0f + fabsf(y));
}

static bool eq_shape(const pg_tensor *t, size_t nd, const size_t *s)
{
    return pg_shape_equal(t->ndim, t->shape, nd, s);
}

static void ref_gemm(size_t m, size_t n, size_t k,
                     const float *a, const float *b, float *c)
{
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0;
            for (size_t t = 0; t < k; t++)
                s += (double)a[i * k + t] * (double)b[t * n + j];
            c[i * n + j] = (float)s;
        }
}

static void test_creation(void)
{
    pg_tensor *e = pg_tensor_eye(3);
    CHECK(e && eq_shape(e, 2, (size_t[]){3, 3}));
    for (size_t i = 0; i < 9; i++)
        CHECKF(e, i, i % 4 == 0 ? 1.0f : 0.0f);

    pg_tensor *l = pg_tensor_linspace(0.0f, 1.0f, 5);
    CHECK(l && l->numel == 5);
    CHECKF(l, 0, 0.0f);
    CHECKF(l, 1, 0.25f);
    CHECKF(l, 2, 0.5f);
    CHECKF(l, 3, 0.75f);
    CHECKF(l, 4, 1.0f);

    pg_tensor *l1 = pg_tensor_linspace(2.5f, 100.0f, 1);
    CHECK(l1 && l1->numel == 1 && l1->data[0] == 2.5f);

    pg_tensor_free(e);
    pg_tensor_free(l);
    pg_tensor_free(l1);
}

static void test_gemm(void)
{
    size_t m = 13, n = 9, k = 7;
    pg_seed(42);
    pg_tensor *A = pg_tensor_uniform(2, (size_t[]){m, k}, -1.0f, 1.0f);
    pg_tensor *B = pg_tensor_uniform(2, (size_t[]){k, n}, -1.0f, 1.0f);
    pg_tensor *C = pg_tensor_zeros(2, (size_t[]){m, n});
    pg_tensor *R = pg_tensor_zeros(2, (size_t[]){m, n});

    pg_cpu_gemm(m, n, k, A->data, k, B->data, n, C->data, n);
    ref_gemm(m, n, k, A->data, B->data, R->data);
    CHECK(pg_tensor_allclose(C, R, 1e-4f, 1e-4f));

    size_t m2 = 16, n2 = 16, k2 = 300;
    pg_tensor *A2 = pg_tensor_uniform(2, (size_t[]){m2, k2}, -1.0f, 1.0f);
    pg_tensor *B2 = pg_tensor_uniform(2, (size_t[]){k2, n2}, -1.0f, 1.0f);
    pg_tensor *C2 = pg_tensor_zeros(2, (size_t[]){m2, n2});
    pg_tensor *R2 = pg_tensor_zeros(2, (size_t[]){m2, n2});
    pg_cpu_gemm(m2, n2, k2, A2->data, k2, B2->data, n2, C2->data, n2);
    ref_gemm(m2, n2, k2, A2->data, B2->data, R2->data);
    CHECK(pg_tensor_allclose(C2, R2, 1e-3f, 1e-3f));

    pg_tensor_free(A); pg_tensor_free(B); pg_tensor_free(C); pg_tensor_free(R);
    pg_tensor_free(A2); pg_tensor_free(B2); pg_tensor_free(C2); pg_tensor_free(R2);
}

static void test_elementwise(void)
{
    size_t shp[2] = {2, 3};
    float raw[6] = {1, 2, 3, 4, 5, 6};
    pg_tensor *a = pg_tensor_from_data(2, shp, raw);
    float braw[3] = {10, 20, 30};
    pg_tensor *b = pg_tensor_from_data(1, (size_t[]){3}, braw);

    pg_tensor *s = pg_add(a, b);
    CHECK(s && eq_shape(s, 2, shp));
    CHECKF(s, 0, 11); CHECKF(s, 1, 22); CHECKF(s, 2, 33);
    CHECKF(s, 3, 14); CHECKF(s, 4, 25); CHECKF(s, 5, 36);

    pg_tensor *col = pg_tensor_from_data(2, (size_t[]){2, 1}, (float[]){100, 200});
    pg_tensor *row = pg_tensor_from_data(2, (size_t[]){1, 3}, (float[]){1, 2, 3});
    pg_tensor *pr = pg_mul(col, row);
    CHECK(pr && eq_shape(pr, 2, shp));
    CHECKF(pr, 0, 100); CHECKF(pr, 1, 200); CHECKF(pr, 2, 300);
    CHECKF(pr, 3, 200); CHECKF(pr, 4, 400); CHECKF(pr, 5, 600);

    pg_tensor *d = pg_div(a, b);
    CHECK(d);
    CHECKF(d, 0, 0.1f); CHECKF(d, 4, 0.25f);

    pg_tensor *twos = pg_tensor_full(2, shp, 2.0f);
    pg_tensor *pw = pg_pow(a, twos);
    CHECK(pw);
    CHECKF(pw, 0, 1); CHECKF(pw, 3, 16); CHECKF(pw, 5, 36);

    pg_tensor *tmp_ex_in = pg_tensor_zeros(1, (size_t[]){1});
    pg_tensor *ex = pg_exp(tmp_ex_in);
    pg_tensor_free(tmp_ex_in);
    CHECK(ex && ex->data[0] == 1.0f);
    pg_tensor *one = pg_tensor_ones(1, (size_t[]){1});
    pg_tensor *lg = pg_log(one);
    CHECK(lg && lg->data[0] == 0.0f);

    float pi6 = 3.14159265f / 6.0f;
    pg_tensor *ang = pg_tensor_full(1, (size_t[]){1}, pi6);
    pg_tensor *sn = pg_sin(ang);
    pg_tensor *cs = pg_cos(ang);
    CHECK(sn && closef(sn->data[0], sinf(pi6)));
    CHECK(cs && closef(cs->data[0], cosf(pi6)));

    pg_tensor *onef = pg_tensor_ones(1, (size_t[]){4});
    pg_tensor *ef = pg_erf(onef);
    CHECK(ef && closef(ef->data[3], erff(1.0f)));

    pg_tensor *tmp_sc_in = pg_tensor_ones(1, (size_t[]){3});
    pg_tensor *sc = pg_add(b, tmp_sc_in);
    pg_tensor_free(tmp_sc_in);
    CHECK(sc && sc->data[2] == 31.0f);

    {
        pg_tensor *tmp_bad = pg_tensor_ones(1, (size_t[]){4});
        pg_tensor *bad_res = pg_add(a, tmp_bad);
        CHECK(bad_res == NULL);
        pg_tensor_free(bad_res);
        pg_tensor_free(tmp_bad);
    }

    pg_tensor_free(a); pg_tensor_free(b); pg_tensor_free(s); pg_tensor_free(col);
    pg_tensor_free(row); pg_tensor_free(pr); pg_tensor_free(d); pg_tensor_free(twos);
    pg_tensor_free(pw); pg_tensor_free(ex); pg_tensor_free(one); pg_tensor_free(lg);
    pg_tensor_free(ang); pg_tensor_free(sn); pg_tensor_free(cs); pg_tensor_free(onef);
    pg_tensor_free(ef); pg_tensor_free(sc);
}

static void test_reduce(void)
{
    size_t shp[3] = {3, 4, 5};
    pg_seed(7);
    pg_tensor *t = pg_tensor_uniform(3, shp, -2.0f, 2.0f);

    for (size_t axis = 0; axis < 3; axis++) {
        size_t outer = 1, len = shp[axis], inner = 1;
        pg_axis_split(3, shp, axis, &outer, &len, &inner);

        for (int kd = 0; kd < 2; kd++) {
            bool keep = kd == 1;
            size_t osh[3];
            if (keep) {
                for (size_t d = 0; d < 3; d++)
                    osh[d] = d == axis ? 1 : shp[d];
            } else {
                size_t j = 0;
                for (size_t d = 0; d < 3; d++)
                    if (d != axis)
                        osh[j++] = shp[d];
            }

            pg_tensor *sm = pg_sum(t, axis, keep);
            pg_tensor *mn = pg_mean(t, axis, keep);
            pg_tensor *mx = pg_max(t, axis, keep);
            pg_tensor *mi = pg_min(t, axis, keep);
            pg_tensor *v0 = pg_var(t, axis, keep, 0);
            pg_tensor *v1 = pg_var(t, axis, keep, 1);
            pg_tensor *s0 = pg_std(t, axis, keep, 0);

            CHECK(eq_shape(sm, keep ? 3 : 2, osh));
            CHECK(eq_shape(v0, keep ? 3 : 2, osh));

            for (size_t o = 0; o < outer; o++) {
                for (size_t ii = 0; ii < inner; ii++) {
                    double sum = 0, mxv = -1e30, mnv = 1e30;
                    for (size_t v = 0; v < len; v++) {
                        double x = t->data[o * len * inner + v * inner + ii];
                        sum += x;
                        if (x > mxv) mxv = x;
                        if (x < mnv) mnv = x;
                    }
                    double mean = sum / len;
                    double ss = 0;
                    for (size_t v = 0; v < len; v++) {
                        double x = t->data[o * len * inner + v * inner + ii];
                        ss += (x - mean) * (x - mean);
                    }
                    size_t oi = o * (keep ? (keep ? 0 : 0) : 0);
                    (void)oi;
                    size_t base = keep ? o * (axis == 0 ? len * inner : (axis == 1 ? inner : 1)) : o * inner;
                    size_t off = base + ii;
                    CHECK(closef(sm->data[off], (float)sum));
                    CHECK(closef(mn->data[off], (float)(sum / len)));
                    CHECK(closef(mx->data[off], (float)mxv));
                    CHECK(closef(mi->data[off], (float)mnv));
                    CHECK(closef(v0->data[off], (float)(ss / len)));
                    CHECK(closef(v1->data[off], (float)(ss / (len - 1))));
                    CHECK(closef(s0->data[off], (float)sqrt(ss / len)));
                }
            }

            pg_tensor_free(sm); pg_tensor_free(mn); pg_tensor_free(mx);
            pg_tensor_free(mi); pg_tensor_free(v0); pg_tensor_free(v1);
            pg_tensor_free(s0);
        }
    }

    pg_tensor *am = pg_argmax(t, 1, false);
    pg_tensor *an = pg_argmin(t, 1, true);
    CHECK(am && an);
    for (size_t o = 0; o < 3; o++) {
        for (size_t ii = 0; ii < 5; ii++) {
            float best = -INFINITY;
            size_t bi = 0;
            float worst = INFINITY;
            size_t wi = 0;
            for (size_t v = 0; v < 4; v++) {
                float x = t->data[o * 20 + v * 5 + ii];
                if (x > best) { best = x; bi = v; }
                if (x < worst) { worst = x; wi = v; }
            }
            CHECK(am->data[o * 5 + ii] == (float)bi);
            CHECK(an->data[o * 5 + ii] == (float)wi);
        }
    }

    float ties[4] = {1, 3, 3, 2};
    pg_tensor *tt = pg_tensor_from_data(2, (size_t[]){1, 4}, ties);
    pg_tensor *ta = pg_argmax(tt, 1, false);
    CHECK(ta && ta->data[0] == 1.0f);

    pg_tensor_free(t); pg_tensor_free(am); pg_tensor_free(an); pg_tensor_free(tt);
    pg_tensor_free(ta);
}

static void test_index(void)
{
    float raw[6] = {10, 20, 30, 40, 50, 60};
    pg_tensor *t = pg_tensor_from_data(2, (size_t[]){2, 3}, raw);

    pg_tensor *sel_idx = pg_tensor_from_data(1, (size_t[]){2}, (float[]){1, 0});
    pg_tensor *sel = pg_index_select(t, 0, sel_idx);
    pg_tensor_free(sel_idx);
    CHECK(sel && eq_shape(sel, 2, (size_t[]){2, 3}));
    CHECKF(sel, 0, 40); CHECKF(sel, 1, 50); CHECKF(sel, 2, 60);
    CHECKF(sel, 3, 10); CHECKF(sel, 4, 20); CHECKF(sel, 5, 30);

    pg_tensor *cols_idx = pg_tensor_from_data(1, (size_t[]){2}, (float[]){2, 0});
    pg_tensor *cols = pg_index_select(t, 1, cols_idx);
    pg_tensor_free(cols_idx);
    CHECK(cols && eq_shape(cols, 2, (size_t[]){2, 2}));
    CHECKF(cols, 0, 30); CHECKF(cols, 1, 10); CHECKF(cols, 2, 60); CHECKF(cols, 3, 40);

    float iraw[4] = {0, 2, 1, 1};
    pg_tensor *gi = pg_tensor_from_data(2, (size_t[]){2, 2}, iraw);
    pg_tensor *g = pg_gather(t, 1, gi);
    CHECK(g && eq_shape(g, 2, (size_t[]){2, 2}));
    CHECKF(g, 0, 10); CHECKF(g, 1, 30); CHECKF(g, 2, 50); CHECKF(g, 3, 50);

    float sraw[4] = {-1, -2, -3, -4};
    pg_tensor *sv = pg_tensor_from_data(2, (size_t[]){2, 2}, sraw);
    pg_tensor *sc = pg_scatter(t, 1, gi, sv);
    CHECK(sc && eq_shape(sc, 2, (size_t[]){2, 3}));
    CHECKF(sc, 0, -1); CHECKF(sc, 1, 20); CHECKF(sc, 2, -2);
    CHECKF(sc, 3, 40); CHECKF(sc, 4, -4); CHECKF(sc, 5, 60);

    float mraw[6] = {1, 0, 0, 1, 1, 0};
    pg_tensor *mask = pg_tensor_from_data(2, (size_t[]){2, 3}, mraw);
    pg_tensor *ms = pg_masked_select(t, mask);
    CHECK(ms && ms->numel == 3);
    CHECKF(ms, 0, 10); CHECKF(ms, 1, 40); CHECKF(ms, 2, 50);

    pg_tensor *bad_idx = pg_tensor_from_data(2, (size_t[]){1, 1}, (float[]){3});
    pg_tensor *bad = pg_gather(t, 1, bad_idx);
    pg_tensor_free(bad_idx);
    CHECK(bad == NULL);
    pg_tensor_free(bad);

    pg_tensor_free(t); pg_tensor_free(sel); pg_tensor_free(cols); pg_tensor_free(gi);
    pg_tensor_free(g); pg_tensor_free(sv); pg_tensor_free(sc); pg_tensor_free(mask);
    pg_tensor_free(ms);
}

static void test_scan(void)
{
    float raw[6] = {1, 2, 3, 4, 5, 6};
    pg_tensor *t = pg_tensor_from_data(2, (size_t[]){2, 3}, raw);

    pg_tensor *cs = pg_cumsum(t, 1);
    CHECK(cs && eq_shape(cs, 2, (size_t[]){2, 3}));
    CHECKF(cs, 0, 1); CHECKF(cs, 1, 3); CHECKF(cs, 2, 6);
    CHECKF(cs, 3, 4); CHECKF(cs, 4, 9); CHECKF(cs, 5, 15);

    pg_tensor *cs0 = pg_cumsum(t, 0);
    CHECKF(cs0, 0, 1); CHECKF(cs0, 1, 2); CHECKF(cs0, 2, 3);
    CHECKF(cs0, 3, 5); CHECKF(cs0, 4, 7); CHECKF(cs0, 5, 9);

    pg_tensor *cp = pg_cumprod(t, 1);
    CHECKF(cp, 0, 1); CHECKF(cp, 1, 2); CHECKF(cp, 2, 6);
    CHECKF(cp, 3, 4); CHECKF(cp, 4, 20); CHECKF(cp, 5, 120);

    float rraw[5] = {3, -1, 4, -1, 5};
    pg_tensor *r = pg_tensor_from_data(2, (size_t[]){1, 5}, rraw);

    pg_kv sa = pg_sort(r, 1, false);
    CHECK(sa.values && sa.indices);
    float want_v[5] = {-1, -1, 3, 4, 5};
    float want_i[5] = {1, 3, 0, 2, 4};
    for (size_t i = 0; i < 5; i++) {
        CHECKF(sa.values, i, want_v[i]);
        CHECK(sa.indices->data[i] == want_i[i]);
    }

    pg_kv sd = pg_sort(r, 1, true);
    float wantd_v[5] = {5, 4, 3, -1, -1};
    float wantd_i[5] = {4, 2, 0, 1, 3};
    for (size_t i = 0; i < 5; i++) {
        CHECKF(sd.values, i, wantd_v[i]);
        CHECK(sd.indices->data[i] == wantd_i[i]);
    }

    pg_kv tk = pg_topk(r, 1, 2);
    CHECK(tk.values && tk.indices);
    CHECK(eq_shape(tk.values, 2, (size_t[]){1, 2}));
    CHECKF(tk.values, 0, 5); CHECKF(tk.values, 1, 4);
    CHECK(tk.indices->data[0] == 4 && tk.indices->data[1] == 2);

    pg_tensor_free(t); pg_tensor_free(cs); pg_tensor_free(cs0); pg_tensor_free(cp);
    pg_tensor_free(r);
    pg_tensor_free(sa.values); pg_tensor_free(sa.indices);
    pg_tensor_free(sd.values); pg_tensor_free(sd.indices);
    pg_tensor_free(tk.values); pg_tensor_free(tk.indices);
}

static void test_matmul(void)
{
    pg_seed(99);

    size_t m = 3, k = 4, n = 5;
    pg_tensor *a = pg_tensor_uniform(2, (size_t[]){m, k}, -1, 1);
    pg_tensor *b = pg_tensor_uniform(2, (size_t[]){k, n}, -1, 1);
    float ref[15];
    ref_gemm(m, n, k, a->data, b->data, ref);

    pg_tensor *c = pg_matmul(a, b);
    CHECK(c && eq_shape(c, 2, (size_t[]){3, 5}));
    {
        pg_tensor *ref_t = pg_tensor_from_data(2, (size_t[]){3, 5}, ref);
        CHECK(pg_tensor_allclose(c, ref_t, 1e-4f, 1e-4f));
        pg_tensor_free(ref_t);
    }

    size_t M = 13, K = 7, N = 9;
    pg_tensor *ta = pg_tensor_uniform(2, (size_t[]){M, K}, -1, 1);
    pg_tensor *tb = pg_tensor_uniform(2, (size_t[]){K, N}, -1, 1);
    float tref[13 * 9];
    ref_gemm(M, N, K, ta->data, tb->data, tref);
    pg_tensor *tc = pg_matmul(ta, tb);
    {
        pg_tensor *tref_t = pg_tensor_from_data(2, (size_t[]){M, N}, tref);
        CHECK(tc && pg_tensor_allclose(tc, tref_t, 1e-4f, 1e-4f));
        pg_tensor_free(tref_t);
    }

    pg_tensor *v = pg_tensor_uniform(1, (size_t[]){4}, -1, 1);
    pg_tensor *vv = pg_matmul(v, v);
    CHECK(vv && vv->numel == 1);
    double dot = 0;
    for (size_t i = 0; i < 4; i++)
        dot += (double)v->data[i] * v->data[i];
    CHECK(closef(vv->data[0], (float)dot));

    pg_tensor *vm = pg_matmul(v, b);
    CHECK(vm && eq_shape(vm, 1, (size_t[]){5}));
    for (size_t j = 0; j < 5; j++) {
        double s = 0;
        for (size_t t = 0; t < 4; t++)
            s += (double)v->data[t] * b->data[t * 5 + j];
        CHECK(closef(vm->data[j], (float)s));
    }

    pg_tensor *mv = pg_matmul(a, v);
    CHECK(mv && eq_shape(mv, 1, (size_t[]){3}));

    size_t bt = 2, bm = 3, bk = 4, bn = 5;
    pg_tensor *ba = pg_tensor_uniform(3, (size_t[]){bt, bm, bk}, -1, 1);
    pg_tensor *bb = pg_tensor_uniform(3, (size_t[]){bt, bk, bn}, -1, 1);
    pg_tensor *bc = pg_bmm(ba, bb);
    CHECK(bc && eq_shape(bc, 3, (size_t[]){bt, bm, bn}));
    for (size_t s = 0; s < bt; s++) {
        float r2[15];
        ref_gemm(bm, bn, bk, ba->data + s * 12, bb->data + s * 20, r2);
        for (size_t i = 0; i < 15; i++)
            CHECK(closef(bc->data[s * 15 + i], r2[i]));
    }

    pg_tensor *mbc = pg_matmul(ba, bb);
    CHECK(mbc && pg_tensor_allclose(mbc, bc, 0, 0));

    pg_tensor *w = pg_tensor_uniform(2, (size_t[]){bk, bn}, -1, 1);
    pg_tensor *shared = pg_matmul(ba, w);
    CHECK(shared && eq_shape(shared, 3, (size_t[]){bt, bm, bn}));
    for (size_t s = 0; s < bt; s++) {
        float r2[15];
        ref_gemm(bm, bn, bk, ba->data + s * 12, w->data, r2);
        for (size_t i = 0; i < 15; i++)
            CHECK(closef(shared->data[s * 15 + i], r2[i]));
    }

    pg_tensor *inp = pg_tensor_uniform(2, (size_t[]){bm, bn}, -1, 1);
    float r2[15];
    ref_gemm(bm, bn, bk, a->data, w->data, r2);
    pg_tensor *am3 = pg_addmm(inp, a, w, 2.0f, 0.5f);
    CHECK(am3 && eq_shape(am3, 2, (size_t[]){bm, bn}));
    for (size_t i = 0; i < 15; i++)
        CHECK(closef(am3->data[i], 0.5f * inp->data[i] + 2.0f * r2[i]));

    size_t ad4[4] = {2, 3, 4, 5}, bd4[4] = {3, 5, 4, 6};
    pg_tensor *da = pg_tensor_uniform(4, ad4, -1, 1);
    pg_tensor *db = pg_tensor_uniform(4, bd4, -1, 1);
    size_t axa[2] = {1, 3}, axb[2] = {0, 1};
    pg_tensor *td = pg_tensordot(da, db, 2, axa, axb);
    CHECK(td && eq_shape(td, 4, (size_t[]){2, 4, 4, 6}));
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 4; j++)
            for (size_t jp = 0; jp < 4; jp++)
                for (size_t l = 0; l < 6; l++) {
                    double s = 0;
                    for (size_t p = 0; p < 3; p++)
                        for (size_t q = 0; q < 5; q++)
                            s += (double)da->data[((i * 3 + p) * 4 + j) * 5 + q] *
                                 (double)db->data[((p * 5 + q) * 4 + jp) * 6 + l];
                    CHECK(closef(td->data[((i * 4 + j) * 4 + jp) * 6 + l], (float)s));
                }

    size_t axa1[1] = {1}, axb1[1] = {1};
    pg_tensor *dc = pg_tensor_uniform(4, (size_t[]){4, 3, 5, 6}, -1, 1);
    pg_tensor *td1 = pg_tensordot(da, dc, 1, axa1, axb1);
    CHECK(td1 && eq_shape(td1, 6, (size_t[]){2, 4, 5, 4, 5, 6}));
    for (size_t i = 0; i < 2; i++)
        for (size_t j = 0; j < 4; j++)
            for (size_t l = 0; l < 5; l++)
                for (size_t p = 0; p < 4; p++)
                    for (size_t q = 0; q < 5; q++)
                        for (size_t r = 0; r < 6; r++) {
                            double s = 0;
                            for (size_t c = 0; c < 3; c++)
                                s += (double)da->data[((i * 3 + c) * 4 + j) * 5 + l] *
                                     (double)dc->data[((p * 3 + c) * 5 + q) * 6 + r];
                            CHECK(closef(td1->data[((((i * 4 + j) * 5 + l) * 4 + p) * 5 + q) * 6 + r], (float)s));
                        }

    pg_tensor_free(a); pg_tensor_free(b); pg_tensor_free(c);
    pg_tensor_free(ta); pg_tensor_free(tb); pg_tensor_free(tc);
    pg_tensor_free(v); pg_tensor_free(vv); pg_tensor_free(vm); pg_tensor_free(mv);
    pg_tensor_free(ba); pg_tensor_free(bb); pg_tensor_free(bc); pg_tensor_free(mbc);
    pg_tensor_free(w); pg_tensor_free(shared); pg_tensor_free(inp); pg_tensor_free(am3);
    pg_tensor_free(da); pg_tensor_free(db); pg_tensor_free(dc); pg_tensor_free(td); pg_tensor_free(td1);
}

int main(void)
{
    test_creation();
    test_gemm();
    test_elementwise();
    test_reduce();
    test_index();
    test_scan();
    test_matmul();

    if (fails == 0)
        printf("test_ops: all passed\n");
    else
        printf("test_ops: %d failures\n", fails);
    return fails != 0;
}
