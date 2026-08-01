// gemv_nt.h - GEMV  y = A @ x  (A row-major); cache-line-wide coalesced row loads.
#ifdef MB_BUILD_GEMV_NT
#include <metal_stdlib>
using namespace metal;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define ROWS_PER_SG __ROWS_PER_SG__
#define NWARPS      __NWARPS__
#define VEC         __VEC__
// RED_TG=1 reduces per-warp partials via threadgroup mem (int64: no simd_sum(long)).
#define RED_TG      __RED_TG__

// y = A @ x (A is M x K row-major). Each warp does ROWS_PER_SG rows; lanes split K
// then simd_sum, each reading VEC elements so a warp spans a full cache line.
struct alignas(sizeof(IN_T) * VEC) VecT_NT { IN_T v[VEC]; };

kernel void gemv_nt(
    device const IN_T   *A   [[buffer(0)]],
    device const IN_T   *x   [[buffer(1)]],
    device       OUT_T  *y   [[buffer(2)]],
    constant int4&  gP       [[buffer(3)]],   // packed (gM, gK, gLda)
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstep = its stride per output row
    constant int&   bstep    [[buffer(5)]],
    constant ACC_T& beta     [[buffer(6)]],
    constant ACC_T& alpha    [[buffer(7)]],
#endif
    uint3        tgid        [[threadgroup_position_in_grid]],
    uint         sgid        [[simdgroup_index_in_threadgroup]],
    uint         lane        [[thread_index_in_simdgroup]])
{
    int gM = gP.x, gK = gP.y, gLda = gP.z;
#if RED_TG
    threadgroup ACC_T part[NWARPS][32];   // per-warp lane partials (no simd_sum for long)
#endif
    const int rows_per_tg = NWARPS * ROWS_PER_SG;
    int row0_tg = int(tgid.x) * rows_per_tg;
    const int K_STRIDE = 32 * VEC;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_SG; ++r) {
        int row = row0_tg + int(sgid) * ROWS_PER_SG + r;
        if (row >= gM) return;
        const device IN_T *Arow = &A[row * gLda];
        ACC_T acc = (ACC_T)0;
        int k = int(lane) * VEC;
        // 4-way unrolled; last per-lane read at k+3*K_STRIDE+(VEC-1) must be in range.
        for (; k + 3 * K_STRIDE + VEC <= gK; k += 4 * K_STRIDE) {
            VecT_NT a0 = *((const device VecT_NT*)(&Arow[k + 0 * K_STRIDE]));
            VecT_NT a1 = *((const device VecT_NT*)(&Arow[k + 1 * K_STRIDE]));
            VecT_NT a2 = *((const device VecT_NT*)(&Arow[k + 2 * K_STRIDE]));
            VecT_NT a3 = *((const device VecT_NT*)(&Arow[k + 3 * K_STRIDE]));
            VecT_NT x0 = *((const device VecT_NT*)(&x[k + 0 * K_STRIDE]));
            VecT_NT x1 = *((const device VecT_NT*)(&x[k + 1 * K_STRIDE]));
            VecT_NT x2 = *((const device VecT_NT*)(&x[k + 2 * K_STRIDE]));
            VecT_NT x3 = *((const device VecT_NT*)(&x[k + 3 * K_STRIDE]));
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc += (ACC_T)a0.v[i] * (ACC_T)x0.v[i];
                acc += (ACC_T)a1.v[i] * (ACC_T)x1.v[i];
                acc += (ACC_T)a2.v[i] * (ACC_T)x2.v[i];
                acc += (ACC_T)a3.v[i] * (ACC_T)x3.v[i];
            }
        }
        // Single-stride tail: per-lane k+VEC<=gK keeps trailing lanes working when
        // k+K_STRIDE would overrun gK (else lane 31 skips its last block at small K).
        for (; k + VEC <= gK; k += K_STRIDE) {
            VecT_NT av = *((const device VecT_NT*)(&Arow[k]));
            VecT_NT xv = *((const device VecT_NT*)(&x[k]));
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
                acc += (ACC_T)av.v[i] * (ACC_T)xv.v[i];
        }
        // Scalar tail for K % VEC != 0; only lane 0 runs it, simd_sum picks it up.
        if (lane == 0) {
            int kk = (gK / VEC) * VEC;
            for (; kk < gK; ++kk)
                acc += (ACC_T)Arow[kk] * (ACC_T)x[kk];
        }
#if RED_TG
        // simdgroup_barrier, not threadgroup_barrier: the out-of-range-row return above
        // is divergent across warps and would deadlock a full threadgroup barrier.
        simdgroup_barrier(mem_flags::mem_threadgroup);
        part[sgid][lane] = acc;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            ACC_T s = (ACC_T)0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) s += part[sgid][i];
#if EPILOGUE
            y[row] = mb_epi<OUT_T, ACC_T, ACC_T>(s, bias, row * bstep, beta, alpha);
#else
            y[row] = (OUT_T)s;
#endif
        }
#else
        acc = simd_sum(acc);
        if (lane == 0)
#if EPILOGUE
            y[row] = mb_epi<OUT_T, ACC_T, ACC_T>(acc, bias, row * bstep, beta, alpha);
#else
            y[row] = (OUT_T)acc;
#endif
#endif
    }
}
#endif  // MB_BUILD_GEMV_NT
