// gemv_bt.h - batched thin-M GEMV (Y = X @ B, M=2..16 rows): stream B once, keep MROWS
// dot-products in registers. TRANS_B=col-major B; TRANS_A=col-major X; EPILOGUE=bias; BATCHED=bmm.
#ifdef MB_BUILD_GEMV_BT
#include <metal_stdlib>
using namespace metal;

#define IN_T     __IN_T__
#define ACC_T    __ACC_T__
#define OUT_T    __OUT_T__
#define BLOCK_N  __BLOCK_N__
#define NWARPS   __NWARPS__
#define VEC      __VEC__
#define MROWS    __MROWS__
#define NCOLS    __NCOLS__

#ifndef BATCHED
#define BATCHED 0
#endif
#ifndef TRANS_B
#define TRANS_B 0
#endif
#ifndef TRANS_A
#define TRANS_A 0
#endif
#if EPILOGUE
#define MB_BATCH_BUF 8
#else
#define MB_BATCH_BUF 4
#endif
#if BATCHED && EPILOGUE
#define MB_BBAT _bbat
#else
#define MB_BBAT 0
#endif

// X element (m,k): row-major X is m*ldx+k (k unit); col-major X (TRANS_A) is m+k*ldx (m unit).
#if TRANS_A
#define X_AT(m, k) X[(m) + (k) * gLdx]
#define LOADX(dst, m, k) do { for (int _i = 0; _i < VEC; ++_i) (dst).v[_i] = X[(m) + ((k) + _i) * gLdx]; } while (0)
#else
#define X_AT(m, k) X[(m) * gLdx + (k)]
#define LOADX(dst, m, k) do { (dst) = *((const device VecT_BT*)(&X[(m) * gLdx + (k)])); } while (0)
#endif

// Lanes own VEC cols (warp reads a BLOCK_N=32*VEC line); each lane keeps MROWS x VEC
// accs. Picker caps MROWS*VEC (registers) and NWARPS*MROWS*VEC (partials tile < 32KB).
struct alignas(sizeof(IN_T) * VEC) VecT_BT { IN_T v[VEC]; };
struct MBGemvBtDims { int N, K, ldb, ldx, ldy; };

kernel void gemv_bt(
    device const IN_T   *B   [[buffer(0)]],   // TRANS_B=0: K x N row-major; =1: K-contiguous cols (ldb)
    device const IN_T   *X   [[buffer(1)]],   // M x K; TRANS_A selects row-/col-major (ldx)
    device       OUT_T  *Y   [[buffer(2)]],   // M x N row-major (ldy)
    constant MBGemvBtDims& gP [[buffer(3)]],
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride  [[buffer(5)]],
    constant ACC_T& beta     [[buffer(6)]],
    constant ACC_T& alpha    [[buffer(7)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(MB_BATCH_BUF)]],   // (sB, sX, sY, sBias) per-batch strides
#endif
    uint3        tgid        [[threadgroup_position_in_grid]],
    uint         sgid        [[simdgroup_index_in_threadgroup]],
    uint         lane        [[thread_index_in_simdgroup]])
{
    static_assert(BLOCK_N == 32 * VEC, "BLOCK_N must equal 32*VEC");
    int gN = gP.N, gK = gP.K, gLdb = gP.ldb, gLdx = gP.ldx, gLdy = gP.ldy;
#if BATCHED
    B += (int64_t)tgid.z * (int64_t)batch.x;
    X += (int64_t)tgid.z * (int64_t)batch.y;
    Y += (int64_t)tgid.z * (int64_t)batch.z;
  #if EPILOGUE
    int _bbat = (int)tgid.z * batch.w;
  #endif
#endif

#if TRANS_B
    // Column-major B (B's columns are K-contiguous, e.g. W.t() for W=(N,K) row-major):
    // a warp reduces over K via simd_sum. NCOLS>1 owns NCOLS columns, loading each X-row
    // once and reusing it across them (cuts the X re-read that bounds high-M decode).
    const int KS = 32 * VEC;
#if NCOLS == 1
    int n = int(tgid.x) * NWARPS + int(sgid);
    if (n >= gN) return;                              // divergent: no tgmem barrier below
    const device IN_T *Bn = &B[(int64_t)n * gLdb];    // column n, K-contiguous
    ACC_T acc[MROWS];
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) acc[m] = (ACC_T)0;
    int k = int(lane) * VEC;
    for (; k + 3 * KS + VEC <= gK; k += 4 * KS) {     // 4x unrolled (low-M is load-latency bound)
        VecT_BT w0 = *((const device VecT_BT*)(&Bn[k]));
        VecT_BT w1 = *((const device VecT_BT*)(&Bn[k + KS]));
        VecT_BT w2 = *((const device VecT_BT*)(&Bn[k + 2 * KS]));
        VecT_BT w3 = *((const device VecT_BT*)(&Bn[k + 3 * KS]));
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            VecT_BT x0, x1, x2, x3;
            LOADX(x0, m, k); LOADX(x1, m, k + KS); LOADX(x2, m, k + 2 * KS); LOADX(x3, m, k + 3 * KS);
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
                acc[m] += (ACC_T)w0.v[i] * (ACC_T)x0.v[i] + (ACC_T)w1.v[i] * (ACC_T)x1.v[i]
                        + (ACC_T)w2.v[i] * (ACC_T)x2.v[i] + (ACC_T)w3.v[i] * (ACC_T)x3.v[i];
        }
    }
    for (; k + VEC <= gK; k += KS) {
        VecT_BT wv = *((const device VecT_BT*)(&Bn[k]));
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            VecT_BT xv; LOADX(xv, m, k);
            #pragma unroll
            for (int i = 0; i < VEC; ++i) acc[m] += (ACC_T)wv.v[i] * (ACC_T)xv.v[i];
        }
    }
    if (lane == 0) {                                  // scalar K%VEC tail; simd_sum folds it in
        int kk = (gK / VEC) * VEC;
        for (; kk < gK; ++kk) {
            ACC_T wk = (ACC_T)Bn[kk];
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) acc[m] += wk * (ACC_T)X_AT(m, kk);
        }
    }
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) {
        ACC_T s = simd_sum(acc[m]);
        if (lane == 0)
#if EPILOGUE
            Y[m * gLdy + n] = mb_epi<OUT_T, ACC_T, ACC_T>(
                s, bias, MB_BBAT + m * bstride.x + n * bstride.y, beta, alpha);
#else
            Y[m * gLdy + n] = (OUT_T)s;
#endif
    }
#else  // NCOLS > 1: register-blocked over output columns (one X load feeds NCOLS columns)
    int nbase = (int(tgid.x) * NWARPS + int(sgid)) * NCOLS;
    if (nbase >= gN) return;
    int ncv = min(NCOLS, gN - nbase);                 // valid columns in this block (edge)
    ACC_T acc[MROWS][NCOLS];
    #pragma unroll
    for (int m = 0; m < MROWS; ++m)
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) acc[m][c] = (ACC_T)0;
    for (int k = int(lane) * VEC; k + VEC <= gK; k += KS) {
        VecT_BT w[NCOLS];
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c)
            if (c < ncv) w[c] = *((const device VecT_BT*)(&B[(int64_t)(nbase + c) * gLdb + k]));
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            VecT_BT xv; LOADX(xv, m, k);
            #pragma unroll
            for (int c = 0; c < NCOLS; ++c)
                #pragma unroll
                for (int i = 0; i < VEC; ++i) acc[m][c] += (ACC_T)w[c].v[i] * (ACC_T)xv.v[i];
        }
    }
    if (lane == 0) {                                  // scalar K%VEC tail
        int kk = (gK / VEC) * VEC;
        for (; kk < gK; ++kk) {
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                ACC_T xk = (ACC_T)X_AT(m, kk);
                #pragma unroll
                for (int c = 0; c < NCOLS; ++c)
                    if (c < ncv) acc[m][c] += (ACC_T)B[(int64_t)(nbase + c) * gLdb + kk] * xk;
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MROWS; ++m)
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            ACC_T s = simd_sum(acc[m][c]);
            if (lane == 0 && c < ncv) {
                int n = nbase + c;
#if EPILOGUE
                Y[m * gLdy + n] = mb_epi<OUT_T, ACC_T, ACC_T>(
                    s, bias, MB_BBAT + m * bstride.x + n * bstride.y, beta, alpha);
#else
                Y[m * gLdy + n] = (OUT_T)s;
#endif
            }
        }
#endif  // NCOLS
#else  // !TRANS_B (row-major B)
    threadgroup ACC_T partials[NWARPS][MROWS][BLOCK_N];

    int col0 = int(tgid.x) * BLOCK_N;
    int n0   = col0 + int(lane) * VEC;        // first column this lane owns

    // Distribute K across warps: warp sgid handles k in [k_start, k_end).
    int k_per_warp = (gK + NWARPS - 1) / NWARPS;
    int k_start    = int(sgid) * k_per_warp;
    int k_end      = min(gK, k_start + k_per_warp);

    ACC_T acc[MROWS][VEC];
    #pragma unroll
    for (int m = 0; m < MROWS; ++m)
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc[m][i] = (ACC_T)0;

    bool full = (n0 + VEC) <= gN;
    if (full) {
        // Full-line path: aligned VEC-wide B loads, 4x unrolled over k. Each B vector
        // feeds all MROWS rows (one stream of B, MROWS MACs) -> bandwidth-bound on B.
        int k = k_start;
        for (; k + 4 <= k_end; k += 4) {
            VecT_BT b0 = *((const device VecT_BT*)(&B[(k+0) * gLdb + n0]));
            VecT_BT b1 = *((const device VecT_BT*)(&B[(k+1) * gLdb + n0]));
            VecT_BT b2 = *((const device VecT_BT*)(&B[(k+2) * gLdb + n0]));
            VecT_BT b3 = *((const device VecT_BT*)(&B[(k+3) * gLdb + n0]));
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                ACC_T x0 = (ACC_T)X_AT(m, k+0), x1 = (ACC_T)X_AT(m, k+1);
                ACC_T x2 = (ACC_T)X_AT(m, k+2), x3 = (ACC_T)X_AT(m, k+3);
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    acc[m][i] += (ACC_T)b0.v[i] * x0;
                    acc[m][i] += (ACC_T)b1.v[i] * x1;
                    acc[m][i] += (ACC_T)b2.v[i] * x2;
                    acc[m][i] += (ACC_T)b3.v[i] * x3;
                }
            }
        }
        for (; k < k_end; ++k) {
            VecT_BT bv = *((const device VecT_BT*)(&B[k * gLdb + n0]));
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                ACC_T xk = (ACC_T)X_AT(m, k);
                #pragma unroll
                for (int i = 0; i < VEC; ++i) acc[m][i] += (ACC_T)bv.v[i] * xk;
            }
        }
    } else {
        // Edge block: scalar with per-column bounds for non-VEC-aligned N.
        for (int k = k_start; k < k_end; ++k) {
            #pragma unroll
            for (int m = 0; m < MROWS; ++m) {
                ACC_T xk = (ACC_T)X_AT(m, k);
                #pragma unroll
                for (int i = 0; i < VEC; ++i) {
                    int n = n0 + i;
                    if (n < gN) acc[m][i] += (ACC_T)B[k * gLdb + n] * xk;
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MROWS; ++m)
        #pragma unroll
        for (int i = 0; i < VEC; ++i) partials[sgid][m][int(lane) * VEC + i] = acc[m][i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp aggregates the NWARPS K-partials and writes Y for every row.
    if (sgid == 0) {
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int c = int(lane) * VEC + i;
                int n = col0 + c;
                if (n < gN) {
                    ACC_T s = (ACC_T)0;
                    #pragma unroll
                    for (int w = 0; w < NWARPS; ++w) s += partials[w][m][c];
#if EPILOGUE
                    Y[m * gLdy + n] = mb_epi<OUT_T, ACC_T, ACC_T>(
                        s, bias, MB_BBAT + m * bstride.x + n * bstride.y, beta, alpha);
#else
                    Y[m * gLdy + n] = (OUT_T)s;
#endif
                }
            }
        }
    }
#endif  // TRANS_B
}
#endif  // MB_BUILD_GEMV_BT
