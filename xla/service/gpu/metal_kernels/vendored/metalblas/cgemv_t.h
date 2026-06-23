// cgemv_t.h - complex GEMV  y = x @ B  (B is K x N row-major complex).
// Reads B once as interleaved C2 (vs torch's ~3x matrix re-reads for complex), K
// split across NWARPS simdgroups and reduced in threadgroup memory. Bandwidth-bound.
#ifdef MB_BUILD_CGEMV_T
#include <metal_stdlib>
using namespace metal;

#define C2      __C2__      // interleaved complex element: float2 / half2
#define ACC2    __ACC2__    // accumulator (float2): fp32 accum for both complex dtypes
#define R       __R__       // output real-component scalar: float / half
#define BLOCK_N __BLOCK_N__ // columns per threadgroup (== 32, one per lane)
#define NWARPS  __NWARPS__

kernel void cgemv_t(
    device const C2 *B  [[buffer(0)]],
    device const C2 *x  [[buffer(1)]],
    device       C2 *y  [[buffer(2)]],
    constant int4&  gP  [[buffer(3)]],   // packed (gN, gK, gLdb, gXs)
#if EPILOGUE
    device const C2* bias    [[buffer(4)]],   // addmm input; bstep = its stride per output col
    constant int&   bstep    [[buffer(5)]],
    constant float& beta_re  [[buffer(6)]],
    constant float& beta_im  [[buffer(7)]],
    constant float& alpha_re [[buffer(8)]],
    constant float& alpha_im [[buffer(9)]],
#endif
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    int gN = gP.x, gK = gP.y, gLdb = gP.z, gXs = gP.w;
    threadgroup ACC2 partials[NWARPS][BLOCK_N];

    int col0 = int(tgid.x) * BLOCK_N;
    int n    = col0 + int(lane);              // column this lane owns

    // Split K across the NWARPS simdgroups; warp sgid handles [k_start, k_end).
    int k_per_warp = (gK + NWARPS - 1) / NWARPS;
    int k_start    = int(sgid) * k_per_warp;
    int k_end      = min(gK, k_start + k_per_warp);

    ACC2 acc = ACC2(0);
    if (n < gN) {
        for (int k = k_start; k < k_end; ++k) {
            C2 b = B[k * gLdb + n];
            C2 xk = x[k * gXs];
            // (b.r + i b.i)(x.r + i x.i)
            acc.x += (float)b.x * (float)xk.x - (float)b.y * (float)xk.y;
            acc.y += (float)b.x * (float)xk.y + (float)b.y * (float)xk.x;
        }
    }
    partials[sgid][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First warp sums the per-warp partials and writes its column.
    if (sgid == 0) {
        ACC2 s = ACC2(0);
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) s += partials[w][lane];
        if (n < gN) {
#if EPILOGUE
            // out = alpha*(s) + beta*bias as a complex multiply (mirrors complex_combine).
            float outr = 0.0f, outi = 0.0f;
#if ALPHA_NZ
            outr += alpha_re * s.x - alpha_im * s.y;
            outi += alpha_re * s.y + alpha_im * s.x;
#endif
#if BETA_NZ
            C2 bv = bias[n * bstep];
            outr += beta_re * (float)bv.x - beta_im * (float)bv.y;
            outi += beta_re * (float)bv.y + beta_im * (float)bv.x;
#endif
            y[n] = C2((R)outr, (R)outi);
#else
            y[n] = C2((R)s.x, (R)s.y);
#endif
        }
    }
}
#endif  // MB_BUILD_CGEMV_T
