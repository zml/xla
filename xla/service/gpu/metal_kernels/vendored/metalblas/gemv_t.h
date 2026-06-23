// gemv_t.h - GEMV  y = x @ B  (B row-major); K split across simdgroups, tgroup reduce.
#ifdef MB_BUILD_GEMV_T
#include <metal_stdlib>
using namespace metal;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define BLOCK_N     __BLOCK_N__
#define NWARPS      __NWARPS__
#define VEC         __VEC__

// Bandwidth-bound GEMV (B is K x N row-major): y[n] = sum_k B[k,n]*x[k]. Lanes own
// VEC cols (warp reads BLOCK_N=32*VEC, one coalesced line); NWARPS split K, reduce in tgmem.
struct alignas(sizeof(IN_T) * VEC) VecT { IN_T v[VEC]; };

kernel void gemv_t(
    device const IN_T   *B   [[buffer(0)]],
    device const IN_T   *x   [[buffer(1)]],
    device       OUT_T  *y   [[buffer(2)]],
    constant int4&  gP       [[buffer(3)]],   // packed (gN, gK, gLdb, gXs)
#if EPILOGUE
    device const OUT_T *bias [[buffer(4)]],   // addmm input; bstep = its stride per output col
    constant int&   bstep    [[buffer(5)]],
    constant ACC_T& beta     [[buffer(6)]],
    constant ACC_T& alpha    [[buffer(7)]],
#endif
    uint3        tgid        [[threadgroup_position_in_grid]],
    uint         sgid        [[simdgroup_index_in_threadgroup]],
    uint         lane        [[thread_index_in_simdgroup]])
{
    static_assert(BLOCK_N == 32 * VEC, "BLOCK_N must equal 32*VEC");
    int gN = gP.x, gK = gP.y, gLdb = gP.z, gXs = gP.w;
    threadgroup ACC_T partials[NWARPS][BLOCK_N];

    int col0 = int(tgid.x) * BLOCK_N;
    int n0   = col0 + int(lane) * VEC;     // first column this lane owns

    // Distribute K across warps: warp sgid handles k in [start, end).
    int k_per_warp = (gK + NWARPS - 1) / NWARPS;
    int k_start    = int(sgid) * k_per_warp;
    int k_end      = min(gK, k_start + k_per_warp);

    ACC_T acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = (ACC_T)0;

    bool full = (n0 + VEC) <= gN;
    if (full) {
        // Full-line path: one aligned VEC-wide load per k, 4x unrolled.
        int k = k_start;
        for (; k + 4 <= k_end; k += 4) {
            VecT b0 = *((const device VecT*)(&B[(k+0) * gLdb + n0]));
            VecT b1 = *((const device VecT*)(&B[(k+1) * gLdb + n0]));
            VecT b2 = *((const device VecT*)(&B[(k+2) * gLdb + n0]));
            VecT b3 = *((const device VecT*)(&B[(k+3) * gLdb + n0]));
            ACC_T x0 = (ACC_T)x[(k+0)*gXs], x1 = (ACC_T)x[(k+1)*gXs];
            ACC_T x2 = (ACC_T)x[(k+2)*gXs], x3 = (ACC_T)x[(k+3)*gXs];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                acc[i] += (ACC_T)b0.v[i] * x0;
                acc[i] += (ACC_T)b1.v[i] * x1;
                acc[i] += (ACC_T)b2.v[i] * x2;
                acc[i] += (ACC_T)b3.v[i] * x3;
            }
        }
        for (; k < k_end; ++k) {
            VecT bv = *((const device VecT*)(&B[k * gLdb + n0]));
            ACC_T xk = (ACC_T)x[k*gXs];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) acc[i] += (ACC_T)bv.v[i] * xk;
        }
    } else {
        // Edge block: scalar with per-column bounds for non-VEC-aligned N.
        for (int k = k_start; k < k_end; ++k) {
            ACC_T xk = (ACC_T)x[k*gXs];
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int n = n0 + i;
                if (n < gN) acc[i] += (ACC_T)B[k * gLdb + n] * xk;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < VEC; ++i) partials[sgid][int(lane) * VEC + i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp aggregates partials; each lane writes its VEC columns.
    if (sgid == 0) {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int c = int(lane) * VEC + i;
            ACC_T s = (ACC_T)0;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) s += partials[w][c];
            int n = col0 + c;
            if (n < gN)
#if EPILOGUE
                y[n] = mb_epi<OUT_T, ACC_T, ACC_T>(s, bias, n * bstep, beta, alpha);
#else
                y[n] = (OUT_T)s;
#endif
        }
    }
}
#endif  // MB_BUILD_GEMV_T
