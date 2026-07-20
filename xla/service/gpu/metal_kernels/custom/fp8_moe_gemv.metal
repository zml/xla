#include <metal_stdlib>
using namespace metal;

// Decode one OCP f8e4m3fn byte to float (exponent bias 7, 3-bit mantissa, no
// inf; 0x7f/0xff encode NaN, which block-quantized weights never produce).
//   e==0 : subnormal,  value = m * 2^-9
//   e!=0 : normal,     value = (1 + m/8) * 2^(e-7) = (8+m) * 2^(e-10)
static inline float decode_e4m3fn(uchar b) {
    int s = (b >> 7) & 0x1;
    int e = (b >> 3) & 0xf;
    int m = b & 0x7;
    float v = (e == 0) ? (float(m) * 1.953125e-3f)
                       : (float(8 + m) * exp2(float(e) - 10.0f));
    return s ? -v : v;
}

// Grouped FP8 (block-scaled) GEMV for mixture-of-experts expert projections,
// x-caching tiled variant. Each output row picks its own expert weight matrix:
//   e = expert_id[ri]
//   out[ri, ni] = sum_k  x[ri, k] * dequant(w[e, ni, k])
//   dequant(w[e, ni, k]) = decode_e4m3fn(w[e, ni, k]) * scale[e, ni/128, k/128]
//
//   x:         bfloat [R, K]              (row-major; K is the contraction dim)
//   w:         uchar  [E, N, K]           (f8e4m3fn, row-major; expert-major)
//   scale:     bfloat [E, N/128, K/128]   (DeepSeek-style 128x128 block scales)
//   expert_id: int    [R]                 (which expert each output row uses)
//   out:       bfloat [R, N]              (row-major)
//   dims = (R, K, N, E)                   (4th word unused; the scale strides
//                                          are derived from K and N)
//
// The per-(ni, ri) predecessor re-read the full x row once per output column
// (N times per row) -- at b16 that x re-read was ~2/3 of the kernel's DRAM
// traffic. Here ONE threadgroup owns a block of TN=8 columns for a single row
// (TN simdgroups, one column each) and streams x in BK=128 tiles through
// threadgroup memory: each x tile is loaded from DRAM once and reused by all TN
// columns, so x is read ~N/TN times per row instead of N. The f8 weight column
// is still read once (coalesced) per (ni, ri); the 256-entry LUT turns the f8
// decode into a threadgroup load.
kernel void fp8_moe_gemv(
    device const bfloat *x          [[buffer(0)]],
    device const uchar  *w          [[buffer(1)]],
    device const bfloat *scale      [[buffer(2)]],
    device const int    *expert_id  [[buffer(3)]],
    device       bfloat *out        [[buffer(4)]],
    constant int4&       dims        [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const int R = max(dims.x, 0), K = max(dims.y, 0), N = max(dims.z, 0);
    const int Nb = N / 128;     // scale rows per expert
    const int Kb = K / 128;     // scale cols per expert (= scale tiles over K)
    constexpr int TN = 8;       // simdgroups == output columns per threadgroup
    constexpr int BK = 128;     // K tile == one 128-element scale block

    // 256 threads (TN*32) each decode one of the 256 possible f8 byte values.
    threadgroup float  lut[256];
    threadgroup bfloat Xs[BK];
    lut[tid] = decode_e4m3fn(uchar(tid));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ri = int(tgid.y);
    if (ri >= R) return;
    const int ni = int(tgid.x) * TN + int(sgid);   // this simdgroup's column
    const bool active = ni < N;

    const int e = expert_id[ri];
    const device bfloat *xrow = x + (long)ri * K;
    // wrow/srow are only dereferenced when active (ni < N).
    const device uchar  *wrow = w + ((long)e * N + ni) * K;
    const device bfloat *srow = scale + ((long)e * Nb + (ni >> 7)) * Kb;

    // K-tiled reduction: load each x tile into Xs once (shared by all TN
    // columns), then each simdgroup's 32 lanes cover the 128-wide tile (4 k
    // each, vectorized). All threads cross the barriers uniformly (the loop
    // bound K is uniform); only the math/store are guarded by `active`.
    float acc = 0.0f;
    for (int kt = 0; kt < K; kt += BK) {
        if (tid < BK) Xs[tid] = xrow[kt + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            const int k = int(lane) * 4;           // 32 lanes * 4 = 128 = BK
            uchar4 wv = *(const device uchar4 *)(wrow + kt + k);
            bfloat4 xv = *(const threadgroup bfloat4 *)(Xs + k);
            float s = float(srow[kt >> 7]);         // one block scale per tile
            acc += s * (float(xv.x) * lut[wv.x] + float(xv.y) * lut[wv.y] +
                        float(xv.z) * lut[wv.z] + float(xv.w) * lut[wv.w]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each simdgroup owns one full column: a single simd_sum over its 32 lanes.
    acc = simd_sum(acc);
    if (ni < N && lane == 0) {
        out[(long)ri * N + ni] = bfloat(acc);
    }
}
