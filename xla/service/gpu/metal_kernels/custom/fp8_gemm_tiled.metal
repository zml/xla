#include <metal_stdlib>
using namespace metal;

// Decode one OCP f8e4m3fn byte to float (exponent bias 7, 3-bit mantissa, no
// inf; 0x7f/0xff encode NaN, which block-quantized weights never produce).
static inline float decode_e4m3fn(uchar b) {
    int s = (b >> 7) & 0x1;
    int e = (b >> 3) & 0xf;
    int m = b & 0x7;
    float v = (e == 0) ? (float(m) * 1.953125e-3f)
                       : (float(8 + m) * exp2(float(e) - 10.0f));
    return s ? -v : v;
}

constant constexpr int MAX_M = 16;

// Batched-decode block-FP8 GEMV. Identical to fp8_gemv -- one threadgroup (256
// threads = 8 SIMD groups) per output column `n`, cooperatively reducing over
// K, with the 256-entry f8-decode LUT and N-threadgroup occupancy -- except it
// computes ALL M rows at once: the weight row w[n,:] is read from DRAM ONCE and
// reused across the M activation rows (held in M per-thread accumulators),
// instead of fp8_gemv re-reading w[n,:] for every (n, row) and so paying M-x the
// weight bandwidth at batch M. The M activation rows are tiny (M*K bf16) and
// stay resident in L2 across the N threadgroups, so DRAM traffic is ~one pass
// over the 1-byte weight regardless of M.
//
//   x:     bfloat [M, K]            (row-major; K is the contraction dim)
//   w:     uchar  [N, K]            (f8e4m3fn, row-major)
//   scale: bfloat [N/128, K/128]    (DeepSeek-style 128x128 block scales)
//   out:   bfloat [M, N]            (row-major)
//   dims = (M, K, N, Kb=K/128)
kernel void fp8_gemm_tiled(
    device const bfloat *x      [[buffer(0)]],
    device const uchar  *w      [[buffer(1)]],
    device const bfloat *scale  [[buffer(2)]],
    device       bfloat *out    [[buffer(3)]],
    constant int4&       dims   [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const int M = dims.x, K = dims.y, N = dims.z, Kb = dims.w;

    threadgroup float lut[256];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ni = int(tgid.x);
    if (ni >= N) return;

    const device uchar  *wrow = w + (long)ni * K;
    const device bfloat *srow = scale + (long)(ni >> 7) * Kb;

    float acc[MAX_M];
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    // Reduce over K. Each lane reads w[ni, k..k+3] ONCE (coalesced), dequantizes
    // via the LUT + block scale, then accumulates it into all M rows.
    for (int k = int(tid) * 4; k + 4 <= K; k += 256 * 4) {
        uchar4 wv = *(const device uchar4 *)(wrow + k);
        float s = float(srow[k >> 7]);
        float w0 = s * lut[wv.x], w1 = s * lut[wv.y];
        float w2 = s * lut[wv.z], w3 = s * lut[wv.w];
        for (int m = 0; m < M; ++m) {
            bfloat4 xv = *(const device bfloat4 *)(x + (long)m * K + k);
            acc[m] += float(xv.x) * w0 + float(xv.y) * w1 +
                      float(xv.z) * w2 + float(xv.w) * w3;
        }
    }

    // Reduce each row's 256 partials -> one output (8 SIMD groups then a final
    // cross-group simd_sum), reusing the 8-slot scratch per row.
    threadgroup float part[8];
    for (int m = 0; m < M; ++m) {
        float a = simd_sum(acc[m]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) part[sgid] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == 0) {
            float t = (lane < 8) ? part[lane] : 0.0f;
            t = simd_sum(t);
            if (lane == 0) out[(long)m * N + ni] = bfloat(t);
        }
    }
}
